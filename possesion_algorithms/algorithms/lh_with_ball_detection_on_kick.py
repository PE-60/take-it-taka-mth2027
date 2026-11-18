"""LH-alt with ball_detection fallback when a kick has nobody inside t_p.

Keep / switch rules (self-contained in this module):

* Keep possession while the ball 1 s ahead stays within ``t_a`` of the holder.
* A team change needs a kick peak within +/- ``kick_win`` frames, or the candidate
  staying nearest within ``t_p`` for ``min_hold_s``.
* Same-team gains go through without a kick.

Extra rule:

* On a kick-window frame where the nearest player is **outside** ``t_p`` (or missing),
  if Stage C ``touch_attributions.json`` names a team for that kick with enough confidence,
  adopt that team as possession even though no ATD player is in the control radius.

Kick source is selectable via ``kick_method``:

* ``export`` (default) — use ``track.kick`` from the match-viewer export
* ``vector_peak`` — recompute ‖Δv‖·fps local maxima (same as Stage B default)
* ``extrema`` — local min/max of signed a_x / a_y with prominence + optional width filters

Regardless of source, kicks are then passed through ``filter_kicks_require_rise``: a peak
is kept only if ‖Δv‖ *rises* by at least ``kick_min_rise`` (m/s²) from the valley before it
up into the peak. Plateau shoulders (almost no climb into the second tick) are dropped.

Possession switches are recorded in ``.decisions.json`` next to the results CSV so the match
viewer can show *why* a change happened (nearest-in-t_p vs balldet fallback). The on-pitch
ball_detection overlay is **not** the same signal as these decisions — low overlay confidence
with a switch usually means the switch came from LH nearest/kick/hold, not from ball_detection.

``lh_balldet_gk_dead`` adds corrected keepers and paints D on ``dead_ball.json`` frames, but
labels the **whole** covered video in one pass (``dead_ball_continuous``) instead of restarting
on every alive gap.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np

from ball_detection.kicks import KICK_METHODS, detect_kicks
from ball_detection.paths import EXPORT_DIR as BALLDET_EXPORT
from possesion_algorithms.base import TACTICAL, PossessionAlgorithm, Tracking
from possesion_algorithms.paths import GOALKEEPER_CSV

DEFAULT_ATTRIBUTIONS_JSON = os.path.join(BALLDET_EXPORT, "touch_attributions.json")

# kick_method values: export (match-viewer track.kick) or a ball_detection.kicks method.
KICK_METHOD_EXPORT = "export"
KICK_METHOD_CHOICES = (KICK_METHOD_EXPORT, *KICK_METHODS)

# Base LH-alt defaults (inlined so this module does not import other algorithms).
_LH_ALT_PARAMS = {
    "t_p": 1.0,
    "t_a": 2.5,
    "horizon_s": 1.0,
    "kick_win": 4,
    "min_hold_s": 0.5,
}


def _ball_distances(ball: np.ndarray, xy: np.ndarray, pitch: tuple) -> np.ndarray:
    """(n, S) ball-to-slot distance in metres, inf where the ball or the slot is missing."""
    m = np.asarray(pitch, dtype=np.float32)
    d = np.hypot(*np.moveaxis((xy - ball[:, None, :]) * m, -1, 0))
    return np.where(np.isfinite(d), d, np.inf).astype(np.float32)


def load_derived_goalkeepers(path: str, start: int, n: int) -> np.ndarray:
    """Home/away keeper pitch fractions for absolute frames start .. start + n - 1.

    Returns (n, 2, 2) with axis-1 order (home, away). Missing or unassigned frames are NaN.
    """
    if n <= 0:
        return np.full((0, 2, 2), np.nan, dtype=np.float32)
    if not os.path.isfile(path):
        raise IOError(
            f"no derived goalkeeper tracks at {path}\n"
            "  make them:  python data_correction/goalkeeper_trackers/run.py"
        )
    out = np.full((n, 2, 2), np.nan, dtype=np.float32)
    side_index = {"home": 0, "away": 1}
    end = start + n
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "frame" not in reader.fieldnames:
            raise ValueError(f"{path}: expected a goalkeeper_tracks.csv header")
        for row in reader:
            side = (row.get("team_side") or "").strip().lower()
            if side not in side_index:
                continue
            frame = int(float(row["frame"]))
            if frame < start or frame >= end:
                continue
            x, y = row.get("x", ""), row.get("y", "")
            if x == "" or y == "":
                continue
            out[frame - start, side_index[side]] = (float(x), float(y))
    return out


def with_derived_goalkeepers(track: Tracking, derived: np.ndarray,
                             raw_tracker_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Drop the raw keeper slots and append corrected home/away keepers as the last two slots.

    `derived` is (track.n, 2, 2) for this stretch of frames. Returns (xy, team).
    """
    if track.slot_ids is None:
        raise ValueError("tracking has no slot_ids; cannot locate raw goalkeeper trackers")
    raw = {str(t) for t in raw_tracker_ids}
    keep = np.array([str(sid) not in raw for sid in track.slot_ids], dtype=bool)
    missing = sorted(raw - {str(sid) for sid in track.slot_ids})
    if missing:
        raise ValueError(f"raw goalkeeper tracker(s) not in tracking: {missing}")
    if derived.shape != (track.n, 2, 2):
        raise ValueError(f"derived keepers shape {derived.shape} does not match track length {track.n}")
    xy = np.concatenate((track.xy[:, keep], derived.astype(np.float32, copy=False)), axis=1)
    team = np.concatenate((track.team[keep], np.asarray(["H", "A"], dtype="<U1")))
    return xy, team


def filter_kicks_require_rise(
        kick_mask: np.ndarray,
        accel_m_s2: np.ndarray,
        min_rise: float,
        *,
        lookback: int | None = None,
) -> np.ndarray:
    """Keep a kick peak only if ‖a‖ climbs by ≥ ``min_rise`` into that peak.

    For each candidate peak ``p``, take the lowest finite ‖a‖ on ``(left, p)`` where
    ``left`` is the previous *kept* peak (or ``p - lookback`` for the first). The rise is
    ``accel[p] - valley``. Plateaus / shoulders have almost no climb from that valley, so
    they are dropped; a real second kick after accel has fallen and climbed again is kept.

    ``min_rise ≤ 0`` leaves the mask unchanged. ``lookback`` defaults to the full stretch
    before the first peak.

    Default min_rise is 3.0
    """
    kick = np.asarray(kick_mask, dtype=bool)
    accel = np.asarray(accel_m_s2, dtype=np.float64)
    if kick.shape != accel.shape:
        raise ValueError(f"kick_mask and accel must match, got {kick.shape} vs {accel.shape}")
    min_rise = float(min_rise)
    if min_rise <= 0 or not kick.any():
        return kick.copy()

    n = len(accel)
    peaks = [int(i) for i in np.flatnonzero(kick)]
    kept: list[int] = []
    for p in peaks:
        ap = float(accel[p]) if np.isfinite(accel[p]) else np.nan
        if not np.isfinite(ap):
            continue
        if kept:
            left = kept[-1] + 1
        elif lookback is None:
            left = 0
        else:
            left = max(0, p - int(lookback))
        segment = accel[left:p]
        if segment.size == 0 or not np.isfinite(segment).any():
            # No samples to measure a rise from — keep (isolated / first sample).
            kept.append(p)
            continue
        valley = float(np.nanmin(segment))
        if not np.isfinite(valley):
            kept.append(p)
            continue
        if (ap - valley) >= min_rise:
            kept.append(p)

    out = np.zeros(n, dtype=bool)
    out[kept] = True
    return out


def load_kick_team_map(path: str) -> dict[int, tuple[str, float]]:
    """Absolute video kick_frame -> (team, confidence) from Stage C JSON.

    Missing / unreadable file -> empty map (algorithm falls back to plain LH-alt).
    """
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    out: dict[int, tuple[str, float]] = {}
    for row in doc.get("attributions") or []:
        team = str(row.get("team") or "").strip()
        if team not in ("H", "A"):
            continue
        try:
            frame = int(row["kick_frame"])
            conf = float(row.get("confidence") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        prev = out.get(frame)
        if prev is None or conf >= prev[1]:
            out[frame] = (team, conf)
    return out


def build_kick_team_hints(
        n: int,
        track_start: int,
        kick: np.ndarray,
        kick_teams: dict[int, tuple[str, float]],
        *,
        kick_win: int,
        min_confidence: float,
        match_tol: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(hints, confs)`` each shape (n,); hints are '' / 'H' / 'A', confs float."""
    hints = np.full(n, "", dtype="<U1")
    confs = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or not kick_teams:
        return hints, confs
    kick = np.asarray(kick, dtype=bool)
    tol = max(0, int(match_tol))
    win = max(0, int(kick_win))
    abs_keys = np.asarray(sorted(kick_teams), dtype=np.int64)
    for rel in np.flatnonzero(kick):
        abs_f = int(track_start) + int(rel)
        j = int(np.argmin(np.abs(abs_keys - abs_f)))
        keyed = int(abs_keys[j])
        if abs(keyed - abs_f) > tol:
            continue
        team, conf = kick_teams[keyed]
        if conf < float(min_confidence) or team not in ("H", "A"):
            continue
        a = max(0, int(rel) - win)
        b = min(n, int(rel) + win + 1)
        hints[a:b] = team
        confs[a:b] = conf
    return hints, confs


def label_lh_alt_balldet(
        ball: np.ndarray,
        xy: np.ndarray,
        team: np.ndarray,
        kick: np.ndarray,
        pitch: tuple,
        fps: float,
        params: dict,
        kick_team_hint: np.ndarray | None = None,
        kick_team_conf: np.ndarray | None = None,
        *,
        frame_offset: int = 0,
        decisions: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """LH-alt labelling with optional per-frame kick team hints from ball_detection.

    When ``decisions`` is a list, append one record for every possession-team change.
    """
    n = len(ball)
    out = np.full(n, "", dtype="<U1")
    if n == 0 or xy.shape[1] == 0:
        return out

    t_p = float(params["t_p"])
    t_a = float(params["t_a"])
    kick_win = int(params["kick_win"])
    min_hold = int(params["min_hold_s"] * fps)
    h = int(params["horizon_s"] * fps)
    bd_min = float(params.get("bd_min_confidence") or 0.0)

    d = _ball_distances(ball, xy, pitch)
    nearest = np.argmin(d, axis=1)
    near_d = d[np.arange(n), nearest]
    seen = np.isfinite(ball[:, 0])
    present = np.isfinite(xy[:, :, 0])

    kick_near = np.zeros(n, dtype=bool)
    for kick_frame in np.flatnonzero(kick):
        kick_near[max(0, kick_frame - kick_win):min(n, kick_frame + kick_win + 1)] = True

    if kick_team_hint is None:
        hints = np.full(n, "", dtype="<U1")
    else:
        hints = np.asarray(kick_team_hint)
        if hints.shape != (n,):
            raise ValueError(f"kick_team_hint must have shape ({n},), got {hints.shape}")
    if kick_team_conf is None:
        confs = np.full(n, np.nan, dtype=np.float64)
    else:
        confs = np.asarray(kick_team_conf, dtype=np.float64)
        if confs.shape != (n,):
            raise ValueError(f"kick_team_conf must have shape ({n},), got {confs.shape}")

    holder, last_team = -1, ""
    cand, streak = -1, 0
    for r in range(n):
        team_before = last_team
        reason = None
        bd_used = False

        if not seen[r]:
            holder = -1
            cand, streak = -1, 0
            out[r] = last_team
            continue

        k, dk = int(nearest[r]), float(near_d[r])
        nearest_team = str(team[k]) if 0 <= k < len(team) else "?"

        if holder >= 0 and present[r, holder]:
            r2 = min(n - 1, r + h)
            ahead = d[r2, holder] if np.isfinite(d[r2, holder]) else d[r, holder]
            if ahead > t_a or (kick_near[r] and dk <= t_p and k != holder):
                holder = -1
        else:
            holder = -1

        if holder < 0 and dk <= t_p:
            switch = last_team != "" and team[k] != last_team
            streak = streak + 1 if k == cand else 1
            cand = k
            if not switch:
                holder = k
                if team_before == "":
                    reason = "nearest_bootstrap"
            elif kick_near[r]:
                holder = k
                reason = "nearest_kick"
            elif streak >= min_hold:
                holder = k
                reason = "nearest_hold"
        else:
            cand, streak = -1, 0
            # Kick registered, but nobody inside t_p → adopt ball_detection team if present.
            if holder < 0 and kick_near[r] and dk > t_p:
                bd = str(hints[r])
                if bd in ("H", "A"):
                    if bd != last_team:
                        reason = "balldet_fallback"
                        bd_used = True
                    last_team = bd

        if holder >= 0:
            last_team = str(team[holder])

        out[r] = last_team

        if (decisions is not None and last_team in ("H", "A") and last_team != team_before
                and reason is not None):
            bd_conf = float(confs[r]) if np.isfinite(confs[r]) else None
            detail = _switch_detail(
                reason, team_before, last_team, dk, t_p, nearest_team,
                bool(kick_near[r]), streak, min_hold, hints[r], bd_conf, bd_min, bd_used)
            decisions.append({
                "f": int(frame_offset + r),
                "from": team_before or "",
                "to": last_team,
                "reason": reason,
                "detail": detail,
                "nearest_dist_m": float(dk) if np.isfinite(dk) else None,
                "nearest_team": nearest_team,
                "kick_near": bool(kick_near[r]),
                "streak": int(streak),
                "min_hold_frames": int(min_hold),
                "t_p": t_p,
                "bd_hint": str(hints[r]) if str(hints[r]) in ("H", "A") else "",
                "bd_conf": bd_conf,
                "bd_min_confidence": bd_min,
                "bd_used": bool(bd_used),
            })

    return out


def _switch_detail(reason, team_before, last_team, dk, t_p, nearest_team,
                   kick_near, streak, min_hold, hint, bd_conf, bd_min, bd_used) -> str:
    prev = team_before or "∅"
    if reason == "balldet_fallback":
        conf_s = "n/a" if bd_conf is None else f"{bd_conf:.2f}"
        return (f"{prev}→{last_team} via ball_detection: nearest {dk:.2f} m > t_p={t_p:.2f}, "
                f"kick_near, hint {hint} conf={conf_s} ≥ {bd_min:.2f}")
    if reason == "nearest_kick":
        return (f"{prev}→{last_team} via LH nearest+kick: {nearest_team} at {dk:.2f} m ≤ t_p={t_p:.2f} "
                f"(ball_detection NOT used; overlay conf is unrelated)")
    if reason == "nearest_hold":
        return (f"{prev}→{last_team} via LH min_hold: {nearest_team} at {dk:.2f} m, "
                f"streak {streak}/{min_hold} (ball_detection NOT used)")
    if reason == "nearest_bootstrap":
        return (f"∅→{last_team} via nearest bootstrap: {nearest_team} at {dk:.2f} m ≤ t_p={t_p:.2f}")
    return f"{prev}→{last_team} ({reason})"


class LinkHoernigAltBallDet(PossessionAlgorithm):
    """LH-alt plus Stage C kick-team hints when no player is within t_p at a kick."""

    name = "lh_balldet"
    description = ("lh_alt, but when a kick has no player within t_p, adopt the team from "
                   "ball_detection touch_attributions.json if confidence is high enough")
    dead_ball = False
    source = TACTICAL
    params = {
        **_LH_ALT_PARAMS,
        "attributions_json": "",          # empty -> ball_detection/export/touch_attributions.json
        "bd_min_confidence": 0.5,
        "bd_match_tol": 12,                # frames: match ATD kick to Stage C kick_frame
        # Kick source: export = track.kick from the match viewer; vector_peak / extrema recompute.
        "kick_method": KICK_METHOD_EXPORT,
        "kick_accel_m_s2": 4.0,           # vector_peak threshold
        "kick_prominence": 3.0,           # extrema: min vertical separation (m/s²)
        "kick_min_width_s": 0.0,          # extrema: min half-prominence width (s)
        "kick_max_width_s": 0.40,         # extrema: max width (s); <0 disables
        "kick_smooth_s": 0.08,            # extrema: pre-smooth window (s)
        "kick_refractory_s": 0.3,
        # Min ‖a‖ climb (m/s²) from the valley before a peak into that peak; rejects plateaus. 0 disables.
        "kick_min_rise": 3.0,
    }

    def __init__(self, **params):
        super().__init__(**params)
        nonnegative = ("t_p", "t_a", "horizon_s", "kick_win", "min_hold_s")
        bad = [name for name in nonnegative if self.params[name] < 0]
        if bad:
            raise ValueError(f"{self.name}: parameters must be non-negative: {bad}")
        if self.params["bd_min_confidence"] < 0:
            raise ValueError(f"{self.name}: bd_min_confidence must be non-negative")
        if self.params["bd_match_tol"] < 0:
            raise ValueError(f"{self.name}: bd_match_tol must be non-negative")
        method = str(self.params["kick_method"] or KICK_METHOD_EXPORT).strip().lower()
        if method not in KICK_METHOD_CHOICES:
            raise ValueError(f"{self.name}: kick_method must be one of {KICK_METHOD_CHOICES}, "
                             f"got {self.params['kick_method']!r}")
        self.params["kick_method"] = method
        for key in ("kick_accel_m_s2", "kick_prominence", "kick_min_width_s",
                    "kick_smooth_s", "kick_refractory_s", "kick_min_rise"):
            if self.params[key] < 0:
                raise ValueError(f"{self.name}: {key} must be non-negative")
        self._kick_teams: dict[int, tuple[str, float]] | None = None
        self._decisions: list[dict[str, Any]] = []

    def _attributions_path(self) -> str:
        path = str(self.params["attributions_json"] or "").strip()
        return path or DEFAULT_ATTRIBUTIONS_JSON

    def settings(self) -> dict:
        out = super().settings()
        out["params"] = {**out["params"], "attributions_json": self._attributions_path()}
        return out

    def prepare(self, track: Tracking, live: np.ndarray | None = None) -> None:
        self._kick_teams = load_kick_team_map(self._attributions_path())
        self._decisions = []

    def _kick_for(self, track: Tracking) -> np.ndarray:
        """Kick mask for this stretch: export / recompute, then rise-filter on ‖Δv‖."""
        method = str(self.params["kick_method"])
        max_width = self.params["kick_max_width_s"]
        max_width_s = None if float(max_width) < 0 else float(max_width)
        # Always need ‖a‖ for the rise filter; for export we still recompute accel only.
        det_method = "vector_peak" if method == KICK_METHOD_EXPORT else method
        _kick_recomputed, accel = detect_kicks(
            track.ball, track.pitch, track.fps,
            threshold=float(self.params["kick_accel_m_s2"]),
            refractory_s=float(self.params["kick_refractory_s"]),
            method=det_method,
            prominence=float(self.params["kick_prominence"]),
            min_width_s=float(self.params["kick_min_width_s"]),
            max_width_s=max_width_s,
            smooth_s=float(self.params["kick_smooth_s"]),
        )
        if method == KICK_METHOD_EXPORT:
            kick = np.asarray(track.kick, dtype=bool)
        else:
            kick = _kick_recomputed
        return filter_kicks_require_rise(kick, accel, float(self.params["kick_min_rise"]))

    def _hints_for(self, track: Tracking, kick: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self._kick_teams is None:
            self.prepare(track)
        assert self._kick_teams is not None
        kick_mask = track.kick if kick is None else kick
        return build_kick_team_hints(
            track.n, track.start, kick_mask, self._kick_teams,
            kick_win=int(self.params["kick_win"]),
            min_confidence=float(self.params["bd_min_confidence"]),
            match_tol=int(self.params["bd_match_tol"]),
        )

    def label(self, track: Tracking) -> np.ndarray:
        kick = self._kick_for(track)
        hints, confs = self._hints_for(track, kick)
        return label_lh_alt_balldet(
            track.ball, track.xy, track.team, kick, track.pitch, track.fps,
            self.params, kick_team_hint=hints, kick_team_conf=confs,
            frame_offset=track.start, decisions=self._decisions,
        )

    def take_decisions(self) -> dict[str, Any]:
        """Export switch log for the match viewer (absolute video frames)."""
        return {
            "version": 1,
            "algorithm": self.name,
            "params": {
                "t_p": self.params["t_p"],
                "t_a": self.params["t_a"],
                "kick_win": self.params["kick_win"],
                "min_hold_s": self.params["min_hold_s"],
                "bd_min_confidence": self.params["bd_min_confidence"],
                "bd_match_tol": self.params["bd_match_tol"],
                "kick_method": self.params["kick_method"],
                "kick_accel_m_s2": self.params["kick_accel_m_s2"],
                "kick_prominence": self.params["kick_prominence"],
                "kick_min_width_s": self.params["kick_min_width_s"],
                "kick_max_width_s": self.params["kick_max_width_s"],
                "kick_smooth_s": self.params["kick_smooth_s"],
                "kick_refractory_s": self.params["kick_refractory_s"],
                "kick_min_rise": self.params["kick_min_rise"],
                "attributions_json": self._attributions_path(),
            },
            "note": ("Overlay ball_detection confidence is independent of these switches. "
                     "reason=nearest_* means LH used a player inside t_p; "
                     "reason=balldet_fallback means Stage C was used."),
            "n_switches": len(self._decisions),
            "switches": list(self._decisions),
        }


class LinkHoernigAltBallDetGkDead(LinkHoernigAltBallDet):
    """LH-balldet on corrected keepers; whole-video label then D on dead-ball frames."""

    name = "lh_balldet_gk_dead"
    description = ("lh_balldet with corrected goalkeeper tracks from goalkeeper_tracks.csv; "
                   "one continuous pass over the covered video, then D on dead_ball.json; tactical")
    dead_ball = True
    dead_ball_continuous = True
    params = {
        **LinkHoernigAltBallDet.params,
        "goalkeeper_csv": "",
        "raw_gk_trackers": "24,46",
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._gk_full: np.ndarray | None = None
        self._gk_start: int | None = None

    def _csv_path(self) -> str:
        path = str(self.params["goalkeeper_csv"] or "").strip()
        return path or GOALKEEPER_CSV

    def _raw_trackers(self) -> tuple[str, ...]:
        text = str(self.params["raw_gk_trackers"])
        ids = tuple(p.strip() for p in text.split(",") if p.strip())
        if len(ids) != 2:
            raise ValueError(f"{self.name}: raw_gk_trackers must be two ids, got {text!r}")
        return ids

    def settings(self) -> dict:
        out = super().settings()
        out["params"] = {**out["params"], "goalkeeper_csv": self._csv_path()}
        return out

    def prepare(self, track: Tracking, live: np.ndarray | None = None) -> None:
        super().prepare(track, live)
        self._gk_full = load_derived_goalkeepers(self._csv_path(), track.start, track.n)
        self._gk_start = track.start

    def _derived_for(self, track: Tracking) -> np.ndarray:
        if self._gk_full is None or self._gk_start is None:
            self.prepare(track)
        assert self._gk_full is not None and self._gk_start is not None
        a = track.start - self._gk_start
        b = a + track.n
        if a < 0 or b > len(self._gk_full):
            raise ValueError(f"{self.name}: track frames {track.start}..{track.start + track.n - 1} "
                             f"are outside the prepared goalkeeper range starting at {self._gk_start}")
        return self._gk_full[a:b]

    def label(self, track: Tracking) -> np.ndarray:
        xy, team = with_derived_goalkeepers(track, self._derived_for(track), self._raw_trackers())
        kick = self._kick_for(track)
        hints, confs = self._hints_for(track, kick)
        return label_lh_alt_balldet(
            track.ball, xy, team, kick, track.pitch, track.fps,
            self.params, kick_team_hint=hints, kick_team_conf=confs,
            frame_offset=track.start, decisions=self._decisions,
        )

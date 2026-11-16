"""Link & Hoernig with kick-windowed team switches and a sustained-control fallback.

Same keep rule as the original heuristic (ball 1 s ahead within t_a), but a gain that changes
the team needs a kick peak within +/- kick_win frames, or the candidate staying nearest within
t_p for min_hold_s (fallback for missed kicks). Same-team gains go through without a kick.
``track.kick`` must mark local peaks of ball acceleration, not every frame above threshold.

``lh_alt_gk_dead`` swaps the raw ATD keeper slots for the corrected home/away tracks from
``data_correction/goalkeeper_trackers/goalkeeper_tracks.csv``, labels each alive interval of
``dead_ball_investigation/dead_ball.json`` on its own, and writes D on the dead-ball frames.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from possesion_algorithms.base import TACTICAL, PossessionAlgorithm, Tracking
from possesion_algorithms.paths import GOALKEEPER_CSV


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


def label_lh_alt(ball: np.ndarray, xy: np.ndarray, team: np.ndarray, kick: np.ndarray,
                 pitch: tuple, fps: float, params: dict) -> np.ndarray:
    """Core LH-alt labelling on explicit candidate arrays."""
    n = len(ball)
    out = np.full(n, "", dtype="<U1")
    if n == 0 or xy.shape[1] == 0:
        return out

    t_p = float(params["t_p"])
    t_a = float(params["t_a"])
    kick_win = int(params["kick_win"])
    min_hold = int(params["min_hold_s"] * fps)
    h = int(params["horizon_s"] * fps)

    d = _ball_distances(ball, xy, pitch)
    nearest = np.argmin(d, axis=1)
    near_d = d[np.arange(n), nearest]
    seen = np.isfinite(ball[:, 0])
    present = np.isfinite(xy[:, :, 0])

    kick_near = np.zeros(n, dtype=bool)
    for kick_frame in np.flatnonzero(kick):
        kick_near[max(0, kick_frame - kick_win):min(n, kick_frame + kick_win + 1)] = True

    holder, last_team = -1, ""
    cand, streak = -1, 0
    for r in range(n):
        if not seen[r]:
            holder = -1
            cand, streak = -1, 0
            out[r] = last_team
            continue

        k, dk = int(nearest[r]), float(near_d[r])

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
            elif kick_near[r] or streak >= min_hold:
                holder = k
        else:
            cand, streak = -1, 0

        if holder >= 0:
            last_team = str(team[holder])
        out[r] = last_team

    return out


class LinkHoernigAlt(PossessionAlgorithm):
    """LH with gated team switches: kick window or sustained control."""

    name = "lh_alt"
    description = ("LH alt: keep while ball 1 s ahead is within t_a; team change needs a kick "
                   "within +/- kick_win or min_hold_s of control; whole video, tactical")
    dead_ball = False
    source = TACTICAL
    params = {
        "t_p": 1.0,
        "t_a": 2.5,
        "horizon_s": 1.0,
        "kick_win": 3,
        "min_hold_s": 0.5,
    }

    def __init__(self, **params):
        super().__init__(**params)
        nonnegative = ("t_p", "t_a", "horizon_s", "kick_win", "min_hold_s")
        bad = [name for name in nonnegative if self.params[name] < 0]
        if bad:
            raise ValueError(f"{self.name}: parameters must be non-negative: {bad}")

    def label(self, track: Tracking) -> np.ndarray:
        return label_lh_alt(track.ball, track.xy, track.team, track.kick,
                            track.pitch, track.fps, self.params)


class LinkHoernigAltGkDead(LinkHoernigAlt):
    """LH alt on corrected keepers, dead-ball intervals masked to D."""

    name = "lh_alt_gk_dead"
    description = ("lh_alt with corrected goalkeeper tracks from goalkeeper_tracks.csv; "
                   "alive intervals of dead_ball.json, D on dead balls; tactical")
    dead_ball = True
    params = {
        **LinkHoernigAlt.params,
        "goalkeeper_csv": "",           # empty -> data_correction/goalkeeper_trackers/goalkeeper_tracks.csv
        "raw_gk_trackers": "24,46",     # ATD slot ids replaced by the derived home/away keepers
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._gk_full: np.ndarray | None = None   # (full_track.n, 2, 2) on prepare()'s track
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
        return label_lh_alt(track.ball, xy, team, track.kick, track.pitch, track.fps, self.params)

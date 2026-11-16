"""Katya's control model: possession from cleaned ball visits and inferred touches.

Implements the method in ``Katya_algo.md`` for the ``PossessionAlgorithm`` contract: clean the
ball track, join near-player visits, score each visit as a touch or pass-by from velocity change
and dwell time, then walk touches to assign sticky team possession.

``katya`` runs over the whole covered video. ``katya_dead`` labels each alive interval of
``dead_ball.json`` on its own (restart taker ignored) and writes D on dead-ball frames.
``katya_gk_dead`` does the same with corrected home/away keepers from
``data_correction/goalkeeper_trackers/goalkeeper_tracks.csv``.
"""

from __future__ import annotations

import numpy as np

from possesion_algorithms.algorithms.lh_alternative_V2 import (
    load_derived_goalkeepers,
    with_derived_goalkeepers,
)
from possesion_algorithms.base import TACTICAL, PossessionAlgorithm, Tracking
from possesion_algorithms.paths import GOALKEEPER_CSV


# Savitzky-Golay quadratic smoother, window 7 (Vidal-Codina et al.).
_SAVGOL7 = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0], dtype=np.float64) / 21.0


def _ball_distances(ball: np.ndarray, xy: np.ndarray, pitch: tuple) -> np.ndarray:
    m = np.asarray(pitch, dtype=np.float32)
    d = np.hypot(*np.moveaxis((xy - ball[:, None, :]) * m, -1, 0))
    return np.where(np.isfinite(d), d, np.inf).astype(np.float32)


def _savgol_stretch(values: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Convolve a finite 1-D stretch; short stretches are returned unchanged."""
    n, w = len(values), len(coeffs)
    if n < w:
        return values.copy()
    pad = w // 2
    ext = np.empty(n + 2 * pad, dtype=np.float64)
    ext[pad:-pad] = values
    ext[:pad] = values[0]
    ext[-pad:] = values[-1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.dot(coeffs, ext[i:i + w])
    return out


def clean_ball_track(ball: np.ndarray, pitch: tuple, fps: float,
                     max_speed_ms: float, savgol_window: int) -> np.ndarray:
    """Drop points unreachable at ``max_speed_ms``, then Savitzky-Golay each seen stretch."""
    out = np.asarray(ball, dtype=np.float64).copy()
    n = len(out)
    if n == 0:
        return out.astype(np.float32)

    pitch_m = np.asarray(pitch, dtype=np.float64)
    max_step = float(max_speed_ms) / float(fps)
    last = -1
    for i in range(n):
        if not np.isfinite(out[i, 0]) or not np.isfinite(out[i, 1]):
            out[i] = np.nan
            continue
        if last < 0:
            last = i
            continue
        dist = float(np.hypot(*(out[i] - out[last]) * pitch_m))
        # Drop unreachable spikes. A sustained track at a new place is reacquired once
        # enough frames have passed that the jump is feasible at max_speed_ms.
        if dist > max_step * (i - last):
            out[i] = np.nan
        else:
            last = i

    window = int(savgol_window)
    if window < 3:
        return out.astype(np.float32)
    if window % 2 == 0:
        window += 1
    coeffs = _SAVGOL7 if window == 7 else None

    i = 0
    while i < n:
        if not np.isfinite(out[i, 0]):
            i += 1
            continue
        j = i + 1
        while j < n and np.isfinite(out[j, 0]):
            j += 1
        if coeffs is not None and j - i >= window:
            for axis in (0, 1):
                out[i:j, axis] = _savgol_stretch(out[i:j, axis], coeffs)
        i = j
    return out.astype(np.float32)


def _find_visits(ball: np.ndarray, xy: np.ndarray, pitch: tuple,
                 zone_m: float, gap_frames: int) -> list[tuple[int, int, int]]:
    """Visits as (frame_start, frame_end, slot), inclusive ends; gaps up to ``gap_frames``."""
    n = len(ball)
    if n == 0 or xy.shape[1] == 0:
        return []

    d = _ball_distances(ball, xy, pitch)
    nearest = np.argmin(d, axis=1)
    near_d = d[np.arange(n), nearest]
    at = near_d <= float(zone_m)

    visits: list[tuple[int, int, int]] = []
    i = 0
    gap = int(gap_frames)
    while i < n:
        if not at[i]:
            i += 1
            continue
        slot = int(nearest[i])
        start = end = i
        i += 1
        while i < n:
            if at[i] and int(nearest[i]) == slot:
                end = i
                i += 1
                continue
            if at[i] and int(nearest[i]) != slot:
                break
            # Ball not at anyone (or missing): allow a short gap before the same player.
            found = -1
            for j in range(i, min(n, i + gap + 1)):
                if at[j] and int(nearest[j]) == slot:
                    found = j
                    break
                if at[j] and int(nearest[j]) != slot:
                    break
            if found < 0:
                break
            end = found
            i = found + 1
        visits.append((start, end, slot))
    return visits


def _mean_velocity(ball_m: np.ndarray, fps: float, lo: int, hi: int) -> np.ndarray | None:
    """Mean inter-frame velocity (m/s) from consecutive finite points in [lo, hi)."""
    lo = max(0, lo)
    hi = min(len(ball_m), hi)
    if hi - lo < 2:
        return None
    total = np.zeros(2, dtype=np.float64)
    count = 0
    for i in range(lo, hi - 1):
        a, b = ball_m[i], ball_m[i + 1]
        if np.isfinite(a).all() and np.isfinite(b).all():
            total += (b - a) * fps
            count += 1
    if count == 0:
        return None
    return total / count


def _visit_confidence(ball: np.ndarray, xy: np.ndarray, team: np.ndarray, pitch: tuple,
                      fps: float, start: int, end: int, slot: int, params: dict) -> tuple[float, float, bool]:
    """Return (p, dv_ms, duel) for one visit."""
    pitch_m = np.asarray(pitch, dtype=np.float64)
    ball_m = np.asarray(ball, dtype=np.float64) * pitch_m
    look = max(1, int(round(float(params["vel_look_s"]) * fps)))

    v_before = _mean_velocity(ball_m, fps, start - look, start)
    v_after = _mean_velocity(ball_m, fps, end + 1, end + 1 + look)
    if v_before is None or v_after is None:
        dv = 0.0
    else:
        dv = float(np.linalg.norm(v_after - v_before))

    duration = (end - start + 1) / float(fps)
    dv0 = float(params["dv_touch_ms"])
    hold0 = float(params["hold_s"])
    # p = 0.5 at either threshold alone; both evidences add; clip to [0, 1].
    p = min(1.0, 0.5 * (dv / dv0 + duration / hold0)) if dv0 > 0 and hold0 > 0 else 0.0

    zone = float(params["zone_m"])
    d = _ball_distances(ball[start:end + 1], xy[start:end + 1], pitch)
    holder_team = str(team[slot])
    duel = False
    for r in range(end - start + 1):
        for k in range(xy.shape[1]):
            if k == slot or not np.isfinite(d[r, k]):
                continue
            if d[r, k] <= zone and str(team[k]) != holder_team and str(team[k]) in ("H", "A"):
                duel = True
                break
        if duel:
            break
    if duel:
        p *= float(params["duel_factor"])
    return float(p), dv, duel


def _taker_slot(ball: np.ndarray, xy: np.ndarray, pitch: tuple,
                zone_m: float, fps: float, taker_s: float) -> int:
    """Slot already at the ball in the first ``taker_s`` seconds, or -1."""
    n = len(ball)
    if n == 0 or xy.shape[1] == 0:
        return -1
    last = min(n, max(1, int(round(float(taker_s) * fps))))
    d = _ball_distances(ball[:last], xy[:last], pitch)
    nearest = np.argmin(d, axis=1)
    near_d = d[np.arange(last), nearest]
    for r in range(last):
        if near_d[r] <= float(zone_m):
            return int(nearest[r])
    return -1


def label_katya(ball: np.ndarray, xy: np.ndarray, team: np.ndarray, pitch: tuple,
                fps: float, params: dict, apply_taker: bool = False) -> np.ndarray:
    """Core Katya labelling on explicit arrays. ``apply_taker`` enables the restart taker rule."""
    n = len(ball)
    out = np.full(n, "", dtype="<U1")
    if n == 0 or xy.shape[1] == 0:
        return out

    cleaned = clean_ball_track(
        ball, pitch, fps,
        float(params["max_ball_speed_ms"]),
        int(params["savgol_window"]),
    )
    visits = _find_visits(
        cleaned, xy, pitch,
        float(params["zone_m"]),
        int(params["visit_gap_frames"]),
    )

    p_touch = float(params["p_touch"])
    touches: list[tuple[int, int, int, float]] = []
    for start, end, slot in visits:
        p, _dv, _duel = _visit_confidence(cleaned, xy, team, pitch, fps, start, end, slot, params)
        if p >= p_touch:
            touches.append((start, end, slot, p))

    taker = _taker_slot(cleaned, xy, pitch, float(params["zone_m"]), fps,
                        float(params["taker_s"])) if apply_taker else -1

    last_team = ""
    first_possession = True
    for start, _end, slot, _p in touches:
        if apply_taker and slot == taker:
            continue
        t = str(team[slot])
        if t not in ("H", "A"):
            continue
        if t != last_team:
            fill_from = 0 if (apply_taker and first_possession) else start
            last_team = t
            first_possession = False
            out[fill_from:] = last_team
        # same-team keep: possession already sticky
    return out


class Katya(PossessionAlgorithm):
    """Control-model possession from visit confidence (dv + dwell)."""

    name = "katya"
    description = ("Katya control model: clean ball, score visits as touches via dv/dwell, "
                   "sticky team of last touch; whole video, tactical")
    dead_ball = False
    source = TACTICAL
    params = {
        "zone_m": 1.0,
        "max_ball_speed_ms": 60.0,
        "savgol_window": 7,
        "visit_gap_frames": 3,
        "dv_touch_ms": 3.0,
        "hold_s": 0.5,
        "vel_look_s": 1.0,
        "duel_factor": 0.7,
        "p_touch": 0.5,
        "taker_s": 1.0,
    }

    def __init__(self, **params):
        super().__init__(**params)
        nonnegative = ("zone_m", "max_ball_speed_ms", "savgol_window", "visit_gap_frames",
                       "dv_touch_ms", "hold_s", "vel_look_s", "duel_factor", "p_touch", "taker_s")
        bad = [name for name in nonnegative if self.params[name] < 0]
        if bad:
            raise ValueError(f"{self.name}: parameters must be non-negative: {bad}")
        if int(self.params["savgol_window"]) < 3:
            raise ValueError(f"{self.name}: savgol_window must be >= 3")

    def label(self, track: Tracking) -> np.ndarray:
        return label_katya(track.ball, track.xy, track.team, track.pitch, track.fps,
                           self.params, apply_taker=bool(self.dead_ball))


class KatyaDead(Katya):
    """Katya on each alive dead-ball interval; D on dead balls."""

    name = "katya_dead"
    description = ("katya on each alive interval of dead_ball.json on its own, "
                   "D on dead balls; restart taker ignored; tactical")
    dead_ball = True


class KatyaGkDead(Katya):
    """Katya with corrected keepers, dead-ball intervals masked to D."""

    name = "katya_gk_dead"
    description = ("katya with corrected goalkeeper tracks from goalkeeper_tracks.csv; "
                   "alive intervals of dead_ball.json, D on dead balls; tactical")
    dead_ball = True
    params = {
        **Katya.params,
        "goalkeeper_csv": "",       # empty -> data_correction/goalkeeper_trackers/goalkeeper_tracks.csv
        "raw_gk_trackers": "24,46",  # ATD slot ids replaced by the derived home/away keepers
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
        return label_katya(track.ball, xy, team, track.pitch, track.fps,
                           self.params, apply_taker=True)

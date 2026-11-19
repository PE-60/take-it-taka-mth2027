"""Which frames / slots are safe to sample for appearance banks.

Ball appearance must be learned away from kicks and player contact (ATD is unreliable there
and crops would be contaminated). Kit appearance excludes goalkeeper slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ball_detection.geometry import pitch_distance_m
from ball_detection.gk import default_goalkeeper_slot_ids, outfield_slot_mask


@dataclass(frozen=True)
class BallHarvestParams:
    """Filters for clean ball crops (Stage A)."""
    clearance_m: float = 2.5          # ball must be this far from every outfield player (shoes!)
    max_reproj_error: float = 1.0     # soft trust on homography fit
    pitch_margin: float = 0.02        # stay inside [margin, 1-margin] in normalised pitch
    max_ball_speed_ms: float | None = 25.0  # drop extreme / teleport samples; None = off
    require_valid_H: bool = True


@dataclass(frozen=True)
class KitHarvestParams:
    """Filters for outfield kit crops (Stage A)."""
    ball_clearance_m: float = 1.0     # player not overlapping the ball crop
    pitch_margin: float = 0.02
    max_reproj_error: float = 1.0
    teams: tuple[str, ...] = ("H", "A")
    require_valid_H: bool = True


def _finite_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])


def ball_speed_ms(ball: np.ndarray, pitch_m: tuple[float, float], fps: float) -> np.ndarray:
    """Per-frame speed from finite differences; NaN where either endpoint is missing."""
    ball = np.asarray(ball, dtype=np.float64)
    n = len(ball)
    speed = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return speed
    delta = pitch_distance_m(ball[1:], ball[:-1], pitch_m) * float(fps)
    speed[1:] = delta
    speed[0] = delta[0] if n > 1 else np.nan
    return speed


def ball_harvest_mask(
        ball: np.ndarray,
        players_xy: np.ndarray,
        pitch_m: tuple[float, float],
        *,
        slot_ids=None,
        gk_ids=None,
        valid_H: np.ndarray | None = None,
        reproj_error: np.ndarray | None = None,
        fps: float = 25.0,
        params: BallHarvestParams | None = None,
) -> np.ndarray:
    """True on frames safe for ball appearance harvest.

    ball: (n, 2) normalised
    players_xy: (n, S, 2) normalised
    """
    params = params or BallHarvestParams()
    ball = np.asarray(ball, dtype=np.float64)
    players_xy = np.asarray(players_xy, dtype=np.float64)
    n = len(ball)
    if players_xy.shape[0] != n:
        raise ValueError("players_xy and ball must share the frame axis")

    ok = _finite_xy(ball)
    margin = float(params.pitch_margin)
    ok &= (ball[:, 0] >= margin) & (ball[:, 0] <= 1.0 - margin)
    ok &= (ball[:, 1] >= margin) & (ball[:, 1] <= 1.0 - margin)

    if params.require_valid_H:
        if valid_H is None:
            raise ValueError("valid_H required when require_valid_H is True")
        ok &= np.asarray(valid_H, dtype=bool)
    if reproj_error is not None and params.max_reproj_error is not None:
        err = np.asarray(reproj_error, dtype=np.float64)
        ok &= np.isfinite(err) & (err <= float(params.max_reproj_error))

    if params.max_ball_speed_ms is not None:
        speed = ball_speed_ms(ball, pitch_m, fps)
        # keep samples with unknown speed; reject only known extremes
        ok &= ~np.isfinite(speed) | (speed <= float(params.max_ball_speed_ms))

    S = players_xy.shape[1]
    if slot_ids is None:
        outfield = np.ones(S, dtype=bool)
    else:
        outfield = outfield_slot_mask(slot_ids, gk_ids if gk_ids is not None
                                      else default_goalkeeper_slot_ids())

    # Distance to nearest outfield player with a finite position.
    clear = np.ones(n, dtype=bool)
    if outfield.any() and params.clearance_m > 0:
        dists = np.full((n, int(outfield.sum())), np.inf, dtype=np.float64)
        cols = np.flatnonzero(outfield)
        for j, col in enumerate(cols):
            ply = players_xy[:, col, :]
            finite = _finite_xy(ply) & ok
            d = np.full(n, np.inf, dtype=np.float64)
            if finite.any():
                d[finite] = pitch_distance_m(ball[finite], ply[finite], pitch_m)
            dists[:, j] = d
        nearest = dists.min(axis=1)
        clear = nearest >= float(params.clearance_m)
    return ok & clear


def kit_harvest_mask(
        players_xy: np.ndarray,
        team: np.ndarray,
        ball: np.ndarray | None = None,
        pitch_m: tuple[float, float] = (105.3, 68.0),
        *,
        slot_ids=None,
        gk_ids=None,
        valid_H: np.ndarray | None = None,
        reproj_error: np.ndarray | None = None,
        params: KitHarvestParams | None = None,
) -> np.ndarray:
    """(n, S) mask: True where that outfield slot may contribute a kit crop."""
    params = params or KitHarvestParams()
    players_xy = np.asarray(players_xy, dtype=np.float64)
    team = np.asarray(team)
    n, S = players_xy.shape[:2]
    if team.shape not in ((S,), (n, S)):
        raise ValueError("team must be (S,) per-slot or (n, S) per-frame")

    if team.ndim == 1:
        team_frame = np.broadcast_to(team.reshape(1, S), (n, S))
    else:
        team_frame = team

    ok = _finite_xy(players_xy)
    margin = float(params.pitch_margin)
    ok &= (players_xy[..., 0] >= margin) & (players_xy[..., 0] <= 1.0 - margin)
    ok &= (players_xy[..., 1] >= margin) & (players_xy[..., 1] <= 1.0 - margin)

    allowed = np.isin(team_frame.astype(str), list(params.teams))
    ok &= allowed

    if slot_ids is None:
        outfield = np.ones(S, dtype=bool)
    else:
        outfield = outfield_slot_mask(slot_ids, gk_ids if gk_ids is not None
                                      else default_goalkeeper_slot_ids())
    ok &= outfield.reshape(1, S)

    if params.require_valid_H:
        if valid_H is None:
            raise ValueError("valid_H required when require_valid_H is True")
        ok &= np.asarray(valid_H, dtype=bool).reshape(n, 1)

    if reproj_error is not None and params.max_reproj_error is not None:
        err = np.asarray(reproj_error, dtype=np.float64).reshape(n, 1)
        ok &= np.isfinite(err) & (err <= float(params.max_reproj_error))

    if ball is not None and params.ball_clearance_m > 0:
        ball = np.asarray(ball, dtype=np.float64)
        ball_ok = _finite_xy(ball)
        for s in range(S):
            d = np.full(n, np.inf, dtype=np.float64)
            both = ball_ok & ok[:, s]
            if both.any():
                d[both] = pitch_distance_m(players_xy[both, s], ball[both], pitch_m)
            # If ball missing, allow the crop; if present, require clearance.
            ok[:, s] &= ~ball_ok | (d >= float(params.ball_clearance_m))

    return ok


def subsample_indices(mask: np.ndarray, stride: int, limit: int | None = None) -> np.ndarray:
    """Frame indices where mask is True, taking every ``stride``-th hit, optionally capped."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if stride > 1:
        idx = idx[:: int(stride)]
    if limit is not None and len(idx) > int(limit):
        idx = idx[: int(limit)]
    return idx

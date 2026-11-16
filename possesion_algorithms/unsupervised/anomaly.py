"""Tracking that cannot be right: ghost balls, teleporting slots, mis-filed goalkeepers.

Anomalies are masked *before* features are built. The possession model then treats those frames /
slots as missing, which is safer for spellF1 than trusting a ball nobody is playing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possesion_algorithms.base import Tracking
from possesion_algorithms.unsupervised.features import central_velocity


@dataclass
class AnomalyReport:
    ball_mask: np.ndarray       # (n,) True = ignore this ball observation
    player_mask: np.ndarray     # (n, S) True = ignore this slot on this frame
    team: np.ndarray            # (S,) team labels after goalkeeper repair
    n_ball: int
    n_player: int
    gk_flips: list              # [(slot, old, new), ...]


def _ball_anomalies(track: Tracking, team: np.ndarray, max_speed: float = 45.0,
                    far_m: float = 18.0, still_m_s: float = 1.5,
                    players_moving_m_s: float = 1.2,
                    converge_max: float = 0.5) -> np.ndarray:
    """Frames whose ball observation is physically or tactically implausible."""
    n, S = track.n, track.n_slots
    pitch = np.asarray(track.pitch, dtype=np.float32)
    ball = track.ball.astype(np.float32) * pitch
    xy = track.xy.astype(np.float32) * pitch
    seen = np.isfinite(ball).all(axis=1)
    is_team = (team == "H") | (team == "A")
    present = np.isfinite(xy).all(axis=2) & is_team[None, :]

    v = central_velocity(ball, float(track.fps))
    speed = np.hypot(v[:, 0], v[:, 1])
    diff = ball[:, None, :] - xy
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(present, dist, np.inf)
    near = dist.min(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        approach = (central_velocity(xy, float(track.fps)) * diff).sum(axis=2) / dist
    weight = np.exp(-np.clip(dist, 0, 80) / 15.0)
    w = np.where(present & np.isfinite(approach), weight, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        converge = np.nansum(np.where(w > 0, approach * w, 0.0), axis=1) / np.where(
            w.sum(1) > 0, w.sum(1), np.nan)

    p_speed = np.hypot(*np.moveaxis(central_velocity(xy, float(track.fps)), -1, 0))
    p_count = present.sum(axis=1)
    p_sum = np.where(present, np.where(np.isfinite(p_speed), p_speed, 0.0), 0.0).sum(axis=1)
    mean_p = np.where(p_count > 0, p_sum / p_count, np.nan)

    mask = np.zeros(n, bool)
    mask |= seen & np.isfinite(speed) & (speed > max_speed)
    # Ghost ball: far from everyone and nearly still. A real ball that far is almost always in
    # flight (fast) or about to be contested (someone converging). Still + isolated => tracker error.
    # The players-moving / converge checks cut stoppages where everyone stands away from a restart.
    isolated = seen & np.isfinite(near) & (near > far_m) & np.isfinite(speed) & (speed < still_m_s)
    active = ((np.isfinite(mean_p) & (mean_p > players_moving_m_s))
              | (np.isfinite(converge) & (converge < converge_max) & (converge > -0.5)))
    # Also flag isolated still balls when nobody at all is approaching (converge ~ 0 / NaN).
    nobody = np.isnan(converge) | (np.abs(converge) < converge_max)
    mask |= isolated & (active | (nobody & (near > far_m + 2.0)))
    return mask


def _player_teleports(track: Tracking, team: np.ndarray, max_speed: float = 14.0,
                      pad: int = 1) -> np.ndarray:
    """Mask slots that jump impossibly far between consecutive frames."""
    n, S = track.n, track.n_slots
    pitch = np.asarray(track.pitch, dtype=np.float32)
    xy = track.xy.astype(np.float32) * pitch
    is_team = (team == "H") | (team == "A")
    present = np.isfinite(xy).all(axis=2) & is_team[None, :]
    mask = np.zeros((n, S), bool)
    if n < 2:
        return mask
    dxy = xy[1:] - xy[:-1]
    delta = np.hypot(dxy[..., 0], dxy[..., 1])  # (n-1, S)
    step = delta * float(track.fps)
    both = present[1:] & present[:-1]
    bad = both & np.isfinite(step) & (step > max_speed)
    # pad a frame either side so the broken sample does not leak into velocities
    for off in range(-pad, pad + 1):
        rows = np.flatnonzero(bad.any(axis=1))
        for r in rows:
            rr = r + 1 + off
            if 0 <= rr < n:
                mask[rr, bad[r]] = True
    return mask


def _repair_goalkeepers(track: Tracking, team: np.ndarray, min_s: float = 30.0,
                        goal_band_m: float = 18.0) -> tuple[np.ndarray, list]:
    """Reassign a slot that lives by one goal to the team defending that goal.

    Track slots have no identities; the deepest player near each goal for long enough is treated
    as that side's goalkeeper. If its team label disagrees, flip it for feature computation.
    """
    out = np.array(team, copy=True)
    flips = []
    n, S = track.n, track.n_slots
    if S == 0 or n == 0:
        return out, flips
    pitch = np.asarray(track.pitch, dtype=np.float32)
    length = float(pitch[0])
    xy = track.xy.astype(np.float32) * pitch
    present = np.isfinite(xy).all(axis=2)
    min_frames = max(1, int(round(min_s * float(track.fps))))

    # Which goal each slot spends most time near, among frames it is present.
    near_left = present & (xy[..., 0] < goal_band_m)
    near_right = present & (xy[..., 0] > length - goal_band_m)
    left_count = near_left.sum(axis=0)
    right_count = near_right.sum(axis=0)
    present_count = present.sum(axis=0)

    for goal, counts, defender in (("left", left_count, "H"), ("right", right_count, "A")):
        # Prefer the slot that is deepest for longest; require it lives there most of its life.
        candidates = np.flatnonzero((counts >= min_frames)
                                    & (counts >= 0.5 * np.maximum(present_count, 1)))
        if not len(candidates):
            continue
        # Attack direction is unknown a priori; left goal is defended by whoever's outfield
        # block sits to the right of it on average. Use mean x of labelled H vs A.
        mean_h = np.nanmean(xy[:, out == "H", 0])
        mean_a = np.nanmean(xy[:, out == "A", 0])
        if np.isfinite(mean_h) and np.isfinite(mean_a):
            # The team with the smaller mean x defends the left goal.
            if mean_h <= mean_a:
                left_def, right_def = "H", "A"
            else:
                left_def, right_def = "A", "H"
            defender = left_def if goal == "left" else right_def
        slot = int(candidates[np.argmax(counts[candidates])])
        if out[slot] != defender and out[slot] in ("H", "A", "?"):
            flips.append((slot, str(out[slot]), defender))
            out[slot] = defender
    return out, flips


def detect_anomalies(track: Tracking, repair_gk: bool = True) -> AnomalyReport:
    """Ball / player masks and (optionally) repaired team labels for `track`."""
    team = np.asarray(track.team, dtype="<U1").copy()
    flips: list = []
    if repair_gk:
        team, flips = _repair_goalkeepers(track, team)
    ball_mask = _ball_anomalies(track, team)
    player_mask = _player_teleports(track, team)
    return AnomalyReport(
        ball_mask=ball_mask,
        player_mask=player_mask,
        team=team,
        n_ball=int(ball_mask.sum()),
        n_player=int(player_mask.any(axis=0).sum()) if player_mask.size else 0,
        gk_flips=flips,
    )

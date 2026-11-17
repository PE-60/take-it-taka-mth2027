"""Per-frame ball features for the ball-aware model of gap_lgbm_double.

Only meaningful where the ball is seen (that is the only place the ball-aware model is used);
NaN elsewhere. Kept in the same H / A layout as features.TeamFeatures so a row can be mirrored:

    per team (block, swapped when T changes):   nearest player's distance to the ball now and in
                                                1 s, players within 1 / 3 / 5 m of the ball
    H-signed (negated when T = A):              ball position and velocity in H's attacking
                                                frame, which team's player is closest now / in 1 s,
                                                the nearest-distance gap, H's attacking direction
    neutral:                                    ball speed, kick now / within +/- KICK_WIN frames,
                                                seconds since the last kick / until the next one
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possesion_algorithms.base import Tracking
from possesion_algorithms.unsupervised.features import central_velocity

BLOCK = ("near_m", "near_1s_m", "n_1m", "n_3m", "n_5m")
SIGNED = ("ball_x", "ball_y", "ball_vx", "ball_vy", "closest_team", "closest_team_1s", "near_gap_m",
          "attack_dir")
NEUTRAL = ("ball_speed", "kick", "kick_near", "since_kick_s", "to_kick_s")
KICK_WIN = 3
MAX_BALL_SPEED = 40.0
MAX_KICK_S = 10.0


@dataclass
class BallFeatures:
    H: np.ndarray        # (n, len(BLOCK))
    A: np.ndarray        # (n, len(BLOCK))
    signed: np.ndarray   # (n, len(SIGNED)) from H's side
    neutral: np.ndarray  # (n, len(NEUTRAL))


def ball_feature_names() -> list[str]:
    return ([f"T_{c}" for c in BLOCK] + [f"O_{c}" for c in BLOCK] + list(SIGNED) + list(NEUTRAL))


def build_ball_X(bf: BallFeatures, rows: np.ndarray, t_is_h: bool) -> np.ndarray:
    mine, theirs = (bf.H, bf.A) if t_is_h else (bf.A, bf.H)
    s = 1.0 if t_is_h else -1.0
    return np.concatenate([mine[rows], theirs[rows], bf.signed[rows] * s, bf.neutral[rows]], axis=1)


def _since_kick(kick: np.ndarray, fps: float, reverse: bool = False) -> np.ndarray:
    k = kick[::-1] if reverse else kick
    idx = np.where(k, np.arange(len(k)), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, (np.arange(len(k)) - last) / fps, MAX_KICK_S)
    out = np.minimum(out, MAX_KICK_S).astype(np.float32)
    return out[::-1] if reverse else out


def compute_ball_features(track: Tracking, direction: np.ndarray, chunk: int = 20000) -> BallFeatures:
    n, fps = track.n, float(track.fps)
    pitch = np.asarray(track.pitch, np.float32)
    length, width = float(pitch[0]), float(pitch[1])
    is_h, is_a = track.team == "H", track.team == "A"
    ball = track.ball.astype(np.float32) * pitch
    seen = np.isfinite(ball).all(axis=1)

    H = np.full((n, len(BLOCK)), np.nan, np.float32)
    A = np.full((n, len(BLOCK)), np.nan, np.float32)
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        xy = track.xy[c0:c1].astype(np.float32) * pitch
        d = np.hypot(xy[..., 0] - ball[c0:c1, None, 0], xy[..., 1] - ball[c0:c1, None, 1])
        for mask, out in ((is_h, H), (is_a, A)):
            dm = d[:, mask]
            fin = np.isfinite(dm)
            near = np.where(fin, dm, np.inf).min(axis=1) if dm.shape[1] else np.full(c1 - c0, np.inf)
            out[c0:c1, 0] = np.where(np.isfinite(near), near, np.nan)
            for j, r in ((2, 1.0), (3, 3.0), (4, 5.0)):
                out[c0:c1, j] = (fin & (dm <= r)).sum(axis=1)
    h = max(1, int(round(fps)))
    for out in (H, A):
        out[:-h, 1] = out[h:, 0]
        out[~seen, 2:] = np.nan

    v = central_velocity(ball, fps)
    speed = np.hypot(v[:, 0], v[:, 1])
    fast = ~(speed <= MAX_BALL_SPEED)
    v[fast] = np.nan
    speed[fast] = np.nan
    d = np.asarray(direction, np.float32)
    signed = np.full((n, len(SIGNED)), np.nan, np.float32)
    signed[:, 0] = (ball[:, 0] - length / 2) * d
    signed[:, 1] = (ball[:, 1] - width / 2) * d
    signed[:, 2] = v[:, 0] * d
    signed[:, 3] = v[:, 1] * d
    gap_now = A[:, 0] - H[:, 0]
    gap_1s = A[:, 1] - H[:, 1]
    signed[:, 4] = np.sign(gap_now)
    signed[:, 5] = np.sign(gap_1s)
    signed[:, 6] = np.clip(gap_now, -30, 30)
    signed[:, 7] = d

    kick = np.asarray(track.kick, bool)
    near = np.zeros(n, bool)
    for kf in np.flatnonzero(kick):
        near[max(0, kf - KICK_WIN):kf + KICK_WIN + 1] = True
    neutral = np.column_stack([speed, kick, near, _since_kick(kick, fps), _since_kick(kick, fps, True)]
                              ).astype(np.float32)
    return BallFeatures(H, A, signed, neutral)

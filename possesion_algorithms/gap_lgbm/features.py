"""Per-frame team features, computed without the ball.

No per-frame feature reads the ball, so a frame inside a real gap and a frame inside a pretend
gap are described the same way. (TeamFeatures carries the ball only so that model.edge_features
can read it at the seen frames just outside each gap.) Each team gets one block of columns in its own attacking frame:
x points from its own goal towards the opponent's (metres from the centre line), y is turned
with it, so the H block and the A block mean the same thing and can be swapped to mirror a
sample (see model.py).

Slots are tracklets, not people, so players are never columns: per frame the present players
of each team are packed into at most MAX_PER_TEAM places and summarised (order statistics,
means, pressure counts). The tracking is processed in chunks so that a composed tactical +
broadcast Tracking with a few hundred slots stays small in memory.

The attacking direction is the one unsupervised.features uses: over a rolling window, H
attacks towards the side where A's centroid is.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from possesion_algorithms.base import Tracking
from possesion_algorithms.unsupervised.features import central_velocity, nan_moving_mean

MAX_PER_TEAM = 16
MAX_SPEED_M_S = 12.0

# one block per team, in that team's attacking frame
BASE = (
    "n_players",        # present players
    "cen_x", "cen_y",   # centroid
    "adv_x", "adv2_x", "adv3_x",   # three most advanced players
    "line_x",           # second deepest player (the deepest is usually the keeper)
    "length",           # most advanced minus second deepest (m)
    "width",            # y range (m)
    "hull",             # convex hull area without the deepest player (m²)
    "mean_speed", "max_speed",     # m/s
    "vel_x",            # mean velocity towards the opponent's goal (m/s)
    "adv_speed",        # speed of the most advanced player
    "p3_mean", "p5_mean",          # mean number of opponents within 3 m / 5 m of each player
    "p3_max",           # most opponents within 3 m of one player
    "p3_frac",          # share of players with an opponent within 3 m
    "nn_opp_mean", "nn_opp_min",   # distance to the nearest opponent: mean and min over players
)
DELTA_OF = ("cen_x", "cen_y", "length", "width", "hull", "adv_x", "line_x", "p5_mean")
DELTA_S = (0.5, 1.0)
DIFF_OF = ("cen_x", "length", "width", "hull", "mean_speed", "vel_x", "line_x", "p3_mean", "p5_mean")
SYM = (
    "all_mean_speed",   # mean speed of everyone (m/s)
    "cen_dist",         # distance between the two centroids (m)
    "n_total",          # present players
    # stoppage cues, for the dead-vs-alive model (gap_lgbm_dead)
    "still_frac",       # share of players slower than STILL_M_S
    "speed_std",        # spread of player speeds (m/s)
    "near_touch",       # players within NEAR_TOUCH_M of a touchline
    "near_goalline",    # players within NEAR_GOAL_M of a goal line
    "dspeed_1s",        # all_mean_speed now minus 1 s ago (negative = slowing down)
)
STILL_M_S = 1.0
NEAR_TOUCH_M = 3.0
NEAR_GOAL_M = 6.0


def block_names() -> tuple[str, ...]:
    return BASE + tuple(f"d{str(s).replace('.', '')}_{c}" for s in DELTA_S for c in DELTA_OF)


@dataclass
class TeamFeatures:
    H: np.ndarray           # (n, FB) H block, in H's attacking frame
    A: np.ndarray           # (n, FB) A block, in A's attacking frame
    sym: np.ndarray         # (n, len(SYM)) unchanged when H and A swap
    direction: np.ndarray   # (n,) +1 where H attacks towards x = 1
    block: tuple            # names of the block columns
    fps: float = 25.0
    pitch: tuple = (105.0, 68.0)
    # (n, 2) ball in metres, NaN where not seen. Never used per frame — only at the edges of a
    # gap (last seen before, first seen after), see model.edge_features. None = edge features NaN.
    ball: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.sym)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull_area(points) -> float:
    """Convex hull area of a few 2-D points (Andrew's monotone chain), 0 below 3 points."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    s = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _pack(p: np.ndarray, v: np.ndarray, k: int):
    """Present players first, at most k of them: (m, k, 2) positions and velocities, NaN padded."""
    present = np.isfinite(p).all(axis=2)
    order = np.argsort(~present, axis=1, kind="stable")[:, :k]
    pp = np.take_along_axis(p, order[..., None], axis=1)
    vv = np.take_along_axis(v, order[..., None], axis=1)
    keep = np.take_along_axis(present, order, axis=1)
    if pp.shape[1] < k:
        pad = k - pp.shape[1]
        pp = np.concatenate([pp, np.full((len(pp), pad, 2), np.nan, np.float32)], axis=1)
        vv = np.concatenate([vv, np.full((len(vv), pad, 2), np.nan, np.float32)], axis=1)
        keep = np.concatenate([keep, np.zeros((len(keep), pad), bool)], axis=1)
    pp[~keep] = np.nan
    vv[~keep] = np.nan
    return pp, vv, keep


def _team_block(p, v, keep, opp, opp_keep, length, width, d):
    """BASE columns for one team in a chunk. p, v (m, k, 2) metres, d (m,) its attack sign."""
    m = len(p)
    out = np.full((m, len(BASE)), np.nan, np.float32)
    col = {name: i for i, name in enumerate(BASE)}
    x = (p[..., 0] - length / 2) * d[:, None]
    y = (p[..., 1] - width / 2) * d[:, None]
    vx = v[..., 0] * d[:, None]
    speed = np.minimum(np.hypot(v[..., 0], v[..., 1]), MAX_SPEED_M_S)
    n = keep.sum(axis=1)
    out[:, col["n_players"]] = n
    out[:, col["cen_x"]] = np.nanmean(x, axis=1)
    out[:, col["cen_y"]] = np.nanmean(y, axis=1)
    desc = -np.sort(np.where(keep, -x, np.inf), axis=1)            # most advanced first, -inf pad
    for i, name in enumerate(("adv_x", "adv2_x", "adv3_x")):
        out[:, col[name]] = np.where(n > i, desc[:, i], np.nan)
    asc = np.sort(np.where(keep, x, np.inf), axis=1)
    out[:, col["line_x"]] = np.where(n >= 2, asc[:, 1], np.nan)
    out[:, col["length"]] = np.where(n >= 3, desc[:, 0] - asc[:, 1], np.nan)
    out[:, col["width"]] = np.where(n >= 2, np.nanmax(y, axis=1) - np.nanmin(y, axis=1), np.nan)
    out[:, col["mean_speed"]] = np.nanmean(speed, axis=1)
    out[:, col["max_speed"]] = np.nanmax(speed, axis=1)
    out[:, col["vel_x"]] = np.nanmean(np.clip(vx, -MAX_SPEED_M_S, MAX_SPEED_M_S), axis=1)
    adv_idx = np.argmax(np.where(keep, x, -np.inf), axis=1)
    out[:, col["adv_speed"]] = np.where(n > 0, speed[np.arange(m), adv_idx], np.nan)

    dist = np.hypot(p[:, :, None, 0] - opp[:, None, :, 0], p[:, :, None, 1] - opp[:, None, :, 1])
    valid = keep[:, :, None] & opp_keep[:, None, :]
    c3 = (valid & (dist <= 3.0)).sum(axis=2).astype(np.float32)
    c5 = (valid & (dist <= 5.0)).sum(axis=2).astype(np.float32)
    c3[~keep], c5[~keep] = np.nan, np.nan
    out[:, col["p3_mean"]] = np.nanmean(c3, axis=1)
    out[:, col["p5_mean"]] = np.nanmean(c5, axis=1)
    out[:, col["p3_max"]] = np.nanmax(c3, axis=1)
    out[:, col["p3_frac"]] = np.nanmean(np.where(keep, (c3 > 0).astype(np.float32), np.nan), axis=1)
    nn = np.where(valid, dist, np.inf).min(axis=2)
    nn = np.where(np.isfinite(nn) & keep, nn, np.nan)
    out[:, col["nn_opp_mean"]] = np.nanmean(nn, axis=1)
    out[:, col["nn_opp_min"]] = np.nanmin(nn, axis=1)

    # hull without the deepest player
    deepest = np.argmin(np.where(keep, x, np.inf), axis=1)
    hc = col["hull"]
    for r in range(m):
        if n[r] < 4:
            continue
        pts = [(float(a), float(b)) for j, (a, b, ok) in enumerate(zip(x[r], y[r], keep[r]))
               if ok and j != deepest[r]]
        out[r, hc] = hull_area(pts)
    return out


def _deltas(base: np.ndarray, fps: float) -> np.ndarray:
    """Rates of change (per second) of DELTA_OF over each DELTA_S, on a 5-frame moving mean."""
    idx = [BASE.index(c) for c in DELTA_OF]
    sm = nan_moving_mean(base[:, idx], 5)
    cols = []
    for s in DELTA_S:
        k = max(1, int(round(s * fps)))
        d = np.full_like(sm, np.nan)
        d[k:] = (sm[k:] - sm[:-k]) / s
        cols.append(d)
    return np.concatenate(cols, axis=1).astype(np.float32)


def attack_direction_chunked(track: Tracking, is_h, is_a, window_s: float, chunk: int) -> np.ndarray:
    """unsupervised.features.attack_direction, without materialising all slots at once."""
    n = track.n
    mh = np.full(n, np.nan, np.float32)
    ma = np.full(n, np.nan, np.float32)
    length = float(track.pitch[0])
    for c0 in range(0, n, chunk):
        x = track.xy[c0:c0 + chunk, :, 0] * length
        for mask, outv in ((is_h, mh), (is_a, ma)):
            xs = x[:, mask]
            cnt = np.isfinite(xs).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(np.isfinite(xs), xs, 0).sum(axis=1) / cnt
            outv[c0:c0 + chunk] = np.where(cnt >= 3, mean, np.nan)
    window = max(1, int(round(window_s * track.fps)))
    gap = nan_moving_mean((ma - mh)[:, None], window)[:, 0]
    direction = np.where(gap >= 0, 1.0, -1.0).astype(np.float32)
    undecided = ~np.isfinite(gap)
    if undecided.any() and not undecided.all():
        idx = np.flatnonzero(~undecided)
        near = idx[np.clip(np.searchsorted(idx, np.flatnonzero(undecided)), 0, len(idx) - 1)]
        direction[undecided] = direction[near]
    return direction


def compute_team_features(track: Tracking, team: np.ndarray | None = None,
                          direction_window_s: float = 60.0, chunk: int = 5000) -> TeamFeatures:
    """TeamFeatures for every frame of `track` (the ball is never read)."""
    n, fps = track.n, float(track.fps)
    length, width = float(track.pitch[0]), float(track.pitch[1])
    pitch = np.asarray(track.pitch, np.float32)
    team = np.asarray(track.team if team is None else team, dtype="<U1")
    is_h, is_a = team == "H", team == "A"
    direction = attack_direction_chunked(track, is_h, is_a, direction_window_s, chunk)

    H = np.full((n, len(BASE)), np.nan, np.float32)
    A = np.full((n, len(BASE)), np.nan, np.float32)
    sym = np.full((n, len(SYM)), np.nan, np.float32)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            h0, h1 = max(0, c0 - 1), min(n, c1 + 1)            # one frame of halo for velocities
            sub = track.xy[h0:h1].astype(np.float32) * pitch
            vel = central_velocity(sub, fps)
            sl = slice(c0 - h0, c0 - h0 + (c1 - c0))
            ph, vh, kh = _pack(sub[sl][:, is_h], vel[sl][:, is_h], MAX_PER_TEAM)
            pa, va, ka = _pack(sub[sl][:, is_a], vel[sl][:, is_a], MAX_PER_TEAM)
            d = direction[c0:c1]
            H[c0:c1] = _team_block(ph, vh, kh, pa, ka, length, width, d)
            A[c0:c1] = _team_block(pa, va, ka, ph, kh, length, width, -d)
            sp = np.minimum(np.hypot(np.concatenate([vh, va], 1)[..., 0],
                                     np.concatenate([vh, va], 1)[..., 1]), MAX_SPEED_M_S)
            sym[c0:c1, 0] = np.nanmean(sp, axis=1)
            ch, ca = np.nanmean(ph, axis=1), np.nanmean(pa, axis=1)
            sym[c0:c1, 1] = np.hypot(*(ch - ca).T)
            sym[c0:c1, 2] = kh.sum(1) + ka.sum(1)
            sym[c0:c1, 3] = np.nanmean(np.where(np.isfinite(sp), (sp < STILL_M_S).astype(np.float32), np.nan), 1)
            sym[c0:c1, 4] = np.nanstd(sp, axis=1)
            pall = np.concatenate([ph, pa], 1)
            sym[c0:c1, 5] = (np.abs(pall[..., 1] - width / 2) > width / 2 - NEAR_TOUCH_M).sum(1)
            sym[c0:c1, 6] = (np.abs(pall[..., 0] - length / 2) > length / 2 - NEAR_GOAL_M).sum(1)
    sm = nan_moving_mean(sym[:, :1], 5)[:, 0]
    k = max(1, int(round(fps)))
    sym[k:, 7] = sm[k:] - sm[:-k]
    H = np.concatenate([H, _deltas(H, fps)], axis=1)
    A = np.concatenate([A, _deltas(A, fps)], axis=1)
    return TeamFeatures(H, A, sym, direction, block_names(), fps, tuple(track.pitch),
                        track.ball.astype(np.float32) * pitch)

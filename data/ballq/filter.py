"""The grid Bayes filter: predict, update, forwards; the same backwards; multiply for the smoothed
posterior. Runs in chunks with overlap so the full match fits in memory.

world (dict) is the in-memory form io.py produces:
    xy        (n, S, 2) player positions, NaN where absent
    ball_obs  (n, 2) detector ball, NaN where missing
    team      (S,) 'H' | 'A' | '?'
    fps, pitch
optional: speed (S per frame), hom_err (n,), dead (n,) bool, white_team, kicks (set), restarts [(r, x, y)]
"""
import numpy as np
from .grid import Grid
from .motion import predict, rolling_kernel_sigma
from .observe import detector_likelihood, holder_likelihood, holder_candidates, restart_prior, trust, gate_tau
from . import features as F


def _frame_context(world, cfg, r, grid):
    """everything the motion and observation models need for frame r"""
    xy, ball = world["xy"], world["ball_obs"]
    pitch, fps = world["pitch"], world["fps"]
    players = [(k, float(xy[r, k, 0]), float(xy[r, k, 1]), 0.0) for k in range(xy.shape[1]) if np.isfinite(xy[r, k, 0])]
    obs = ball[r] if np.all(np.isfinite(ball[r])) else None
    return players, obs


def _velocity_cells(world, grid, r, cap_ms):
    """ball velocity in cells per frame from the last two trusted observations, capped"""
    b = world["ball_obs"]
    if r >= 1 and np.all(np.isfinite(b[r])) and np.all(np.isfinite(b[r - 1])) and not world["air"][r]:
        dx, dy = b[r] - b[r - 1]
        sp = float(np.hypot(dx * world["pitch"][0], dy * world["pitch"][1]) * world["fps"])
        if sp > cap_ms:
            k = cap_ms / sp; dx *= k; dy *= k
        return grid.to_cells(dx, dy), min(sp, cap_ms)
    return (0.0, 0.0), 0.0


def prepare(world, cfg):
    """fill in the derived per frame arrays the filter needs, once"""
    n = len(world["ball_obs"]); pitch, fps = world["pitch"], world["fps"]
    fcfg, mcfg = cfg["features"], cfg["motion"]
    speed = F.ball_speed_frames(world["ball_obs"], pitch, fps)
    accel = F.ball_accel(speed, fps)
    near_d, near_k = F.nearest_player_m(world["ball_obs"], world["xy"], pitch)
    world.setdefault("speed_frames", speed)
    world["accel"] = accel
    world["near_d"], world["near_k"] = near_d, near_k
    world["air"] = F.air_ball_flags(speed, accel, near_d, mcfg["speed_cap_ms"], fcfg["kick_accel_ms2"], fcfg["near_player_m"])
    world.setdefault("accel_xy", F.ball_accel_xy(world["ball_obs"], pitch, fps))
    world.setdefault("kicks", set(F.kicks_from_accel(speed, fps, fcfg["kick_accel_ms2"], fcfg["kick_min_gap_s"], accel=world["accel_xy"])))
    world.setdefault("dead", np.zeros(n, bool))
    world.setdefault("hom_err", np.full(n, np.nan))
    world["white"] = F.white_blend_flags(near_d, near_k, world["team"], world.get("white_team"), cfg["observe"]["white_blend_radius_m"])
    world.setdefault("restarts", [])
    world.setdefault("tau", np.zeros(n))
    return world


def _likelihood(world, cfg, grid, r, predicted=None):
    """predicted: the forward prediction for frame r, used once to gate the point. The backward pass
    passes nothing and reuses the tau the forward pass settled on, so both passes see the same frame."""
    o = cfg["observe"]
    players, obs = _frame_context(world, cfg, r, grid)
    sigma, tau = trust(o, air=bool(world["air"][r]), white=bool(world["white"][r]), hom_err=world["hom_err"][r], dead=bool(world["dead"][r]))
    if predicted is not None:
        tau = gate_tau(predicted, grid, obs, tau, o["gate_radius_m"], o["gate_ratio_min"], o["gate_tau"])
        world["tau"][r] = tau if obs is not None else 0.0
    elif obs is not None and world["tau"][r] > 0:
        tau = float(world["tau"][r])
    L = detector_likelihood(grid, obs, sigma, tau)
    cands = holder_candidates(obs, players, cfg["features"]["near_player_m"], {}, world["pitch"]) if obs is not None else []
    if cands and not world["dead"][r]:
        L = L * holder_likelihood(grid, cands, o["holder_sigma_m"], o["holder_weight_max"])
    for (rr, x, y) in world["restarts"]:
        if rr == r:
            L = L * restart_prior(grid, (x, y), o["restart_sigma_m"], o["restart_weight"])
    return L


def _step(p, world, cfg, grid, r, direction):
    """predict from frame r-direction into frame r, then weigh by frame r's likelihood"""
    m = cfg["motion"]
    (drow, dcol), sp = _velocity_cells(world, grid, r, m["speed_cap_ms"])
    if direction < 0:
        drow, dcol = -drow, -dcol
    sigma_cells = rolling_kernel_sigma(sp, m["rolling_sigma_m"], m["rolling_sigma_per_ms"]) / grid.cell_m
    # players near the current mass carry the ball
    (mx, my), _ = grid.mean_cov(p)
    xy = world["xy"][r]
    carried = [(float(xy[k, 0]), float(xy[k, 1])) for k in range(xy.shape[0]) if np.isfinite(xy[k, 0])
               and np.hypot((xy[k, 0] - mx) * grid.pitch[0], (xy[k, 1] - my) * grid.pitch[1]) <= m["carried_radius_m"]]
    weights = m["weights_dead"] if world["dead"][r] else m["weights"]
    q = predict(p, grid, (drow, dcol), sigma_cells, carried, weights, carried_sigma_cells=m["carried_sigma_m"] / grid.cell_m)
    q = q * _likelihood(world, cfg, grid, r, predicted=q if direction > 0 else None)
    s = q.sum()
    return q / s if s > 0 else np.full_like(q, 1.0 / q.size)


def forward_backward(grid, world, cfg, frames):
    """smoothed posterior for a range of frames; returns per frame mean, cov, entropy, mass per slot, plus the sparse grid"""
    world = prepare(world, cfg)
    frames = list(frames); n = len(frames)
    flat = np.full(grid.shape, 1.0 / (grid.shape[0] * grid.shape[1]))
    fwd = np.empty((n,) + grid.shape, dtype=np.float32)
    p = flat * _likelihood(world, cfg, grid, frames[0]); p /= p.sum(); fwd[0] = p
    for i in range(1, n):
        p = _step(p, world, cfg, grid, frames[i], +1); fwd[i] = p
    out = {"mean": np.zeros((n, 2)), "cov": np.zeros((n, 2, 2)), "entropy": np.zeros(n), "mass_total": np.zeros(n),
           "mass_within": {}, "post": [None] * n}
    S = world["xy"].shape[1]
    out["mass_within"] = {rad: np.zeros((n, S)) for rad in (1.0, 2.0)}
    # backward: beta messages, multiply with forward
    b = np.ones(grid.shape)
    for i in range(n - 1, -1, -1):
        post = fwd[i] * b; post /= post.sum()
        out["post"][i] = post.astype(np.float32)
        (mx, my), cov = grid.mean_cov(post)
        out["mean"][i] = (mx, my); out["cov"][i] = cov; out["entropy"][i] = grid.entropy_bits(post); out["mass_total"][i] = post.sum()
        xy = world["xy"][frames[i]]
        for k in range(S):
            if np.isfinite(xy[k, 0]):
                for rad in (1.0, 2.0):
                    out["mass_within"][rad][i, k] = grid.mass_within(post, float(xy[k, 0]), float(xy[k, 1]), rad)
        if i > 0:
            # beta for frame i-1: predict backwards from frame i's weighted message
            wb = b * _likelihood(world, cfg, grid, frames[i])
            b = _step(wb / max(wb.sum(), 1e-30), world, cfg, grid, frames[i - 1], -1)
            b = b / b.max()
    return out


def run_chunked(grid, world, cfg, start=0, end=None, progress=None):
    """forward_backward over the whole range in overlapping chunks; returns the same dict with per frame arrays"""
    n_all = len(world["ball_obs"]); end = n_all - 1 if end is None else end
    chunk, ov = cfg["grid"]["chunk_frames"], cfg["grid"]["overlap_frames"]
    res = None
    S = world["xy"].shape[1]
    total = end - start + 1
    # the full posterior of every frame would not fit in memory for a match (0.5 MB a frame), so only
    # the cells above mass_min_out are kept, as (flat cell index, mass) pairs
    thr = cfg["grid"]["mass_min_out"]
    acc = {"mean": np.zeros((total, 2)), "cov": np.zeros((total, 2, 2)), "entropy": np.zeros(total), "mass_total": np.zeros(total),
           "mass_within": {1.0: np.zeros((total, S)), 2.0: np.zeros((total, S))}, "sparse": [None] * total}
    a = start
    while a <= end:
        b = min(end, a + chunk - 1)
        fa, fb = max(start, a - ov), min(end, b + ov)
        out = forward_backward(grid, world, cfg, range(fa, fb + 1))
        for i, f in enumerate(range(fa, fb + 1)):
            if a <= f <= b:
                j = f - start
                acc["mean"][j] = out["mean"][i]; acc["cov"][j] = out["cov"][i]; acc["entropy"][j] = out["entropy"][i]
                acc["mass_total"][j] = out["mass_total"][i]
                post = out["post"][i]; keep = np.flatnonzero(post.ravel() >= thr)
                acc["sparse"][j] = (keep.astype(np.int32), post.ravel()[keep].astype(np.float32))
                for rad in (1.0, 2.0):
                    acc["mass_within"][rad][j] = out["mass_within"][rad][i]
        if progress:
            progress(b - start + 1, total)
        a = b + 1
    return acc

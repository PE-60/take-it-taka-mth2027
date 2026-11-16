"""How we know the filter helps without hand labels (rule 1).

holdout_mask     hide the detector ball for one second in every ten; the filter must fill the gap
holdout_scores   error of the filtered mean against the hidden points, and whether the stated
                 uncertainty is honest (fraction of hidden points inside the 1 and 2 sigma ellipses;
                 a 2D Gaussian puts 39 % and 86 % there)
kalman_baseline  constant velocity Kalman smoother on the same observations, the thing to beat
ablations        rerun with parts switched off and report GT agreement of the possession heuristic
"""
import copy
import numpy as np

from .grid import Grid
from .filter import prepare, run_chunked
from .possession import lh_posterior


def holdout_mask(n, fps, every_s=10.0, hide_s=1.0, offset_s=5.0):
    """boolean (n,) True where the detector ball is hidden from the filter"""
    m = np.zeros(n, bool)
    a = int(offset_s * fps)
    while a < n:
        m[a:min(n, a + int(hide_s * fps))] = True
        a += int(every_s * fps)
    return m


def run_filter(world, cfg, grid=None):
    w = {k: (v.copy() if isinstance(v, np.ndarray) else copy.copy(v)) for k, v in world.items() if k not in ("tau", "air", "white", "accel", "near_d", "near_k", "speed_frames")}
    w = prepare(w, cfg)
    grid = grid or Grid(w["pitch"], cfg["grid"]["cell_m"])
    return w, run_chunked(grid, w, cfg)


def holdout_scores(world, cfg, mask, clean):
    """clean: (n,) bool, frames whose detector point we treat as truth (not air, not white blend, trusted).
    Returns dict of metrics in metres and fractions."""
    hidden = copy.copy(world)
    hidden["ball_obs"] = world["ball_obs"].copy()
    hidden["ball_obs"][mask] = np.nan
    w, res = run_filter(hidden, cfg)
    truth = world["ball_obs"]
    use = mask & clean & np.isfinite(truth[:, 0])
    px, py = world["pitch"]
    d = np.column_stack([(res["mean"][:, 0] - truth[:, 0]) * px, (res["mean"][:, 1] - truth[:, 1]) * py])[use]
    cov = res["cov"][use] * np.array([[px * px, px * py], [px * py, py * py]])
    maha = np.array([float(v @ np.linalg.solve(c + 1e-9 * np.eye(2), v)) for v, c in zip(d, cov)])
    err = np.hypot(d[:, 0], d[:, 1])
    return {"n_hidden": int(use.sum()), "median_err_m": float(np.median(err)) if len(err) else None, "p90_err_m": float(np.percentile(err, 90)) if len(err) else None,
            "inside_1sigma": float(np.mean(maha <= 1.0)) if len(err) else None, "inside_2sigma": float(np.mean(maha <= 4.0)) if len(err) else None,
            "expected_1sigma": 0.393, "expected_2sigma": 0.865}


def kalman_baseline(ball_obs, pitch, fps, q_ms2=8.0, r_m=0.5, mask=None):
    """constant velocity Kalman filter + RTS smoother in metres; returns mean positions (n, 2) in pitch units"""
    n = len(ball_obs); dt = 1.0 / fps
    obs = ball_obs.copy()
    if mask is not None:
        obs[mask] = np.nan
    z = np.column_stack([obs[:, 0] * pitch[0], obs[:, 1] * pitch[1]])
    Fm = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    Q = q_ms2 ** 2 * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2], [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]); R = np.eye(2) * r_m ** 2
    first = np.flatnonzero(np.isfinite(z[:, 0]))
    x = np.array([z[first[0], 0], z[first[0], 1], 0, 0]) if len(first) else np.zeros(4)
    P = np.eye(4) * 25.0
    xs_p, Ps_p, xs_f, Ps_f = [], [], [], []
    for r in range(n):
        x = Fm @ x; P = Fm @ P @ Fm.T + Q
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        if np.all(np.isfinite(z[r])):
            S = Hm @ P @ Hm.T + R; K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (z[r] - Hm @ x); P = (np.eye(4) - K @ Hm) @ P
        xs_f.append(x.copy()); Ps_f.append(P.copy())
    xs = xs_f[-1].copy(); out = np.zeros((n, 2)); out[-1] = xs[:2]
    for r in range(n - 2, -1, -1):
        C = Ps_f[r] @ Fm.T @ np.linalg.inv(Ps_p[r + 1])
        xs = xs_f[r] + C @ (xs - xs_p[r + 1]); out[r] = xs[:2]
    return np.column_stack([out[:, 0] / pitch[0], out[:, 1] / pitch[1]])


def kalman_holdout(world, mask, clean):
    m = kalman_baseline(world["ball_obs"], world["pitch"], world["fps"], mask=mask)
    truth = world["ball_obs"]; use = mask & clean & np.isfinite(truth[:, 0])
    err = np.hypot((m[use, 0] - truth[use, 0]) * world["pitch"][0], (m[use, 1] - truth[use, 1]) * world["pitch"][1])
    return {"median_err_m": float(np.median(err)) if len(err) else None, "p90_err_m": float(np.percentile(err, 90)) if len(err) else None}


ABLATIONS = {
    "no_holder": {"observe": {"holder_weight_max": 0.0}, "motion": {"weights": {"rolling": 0.97, "carried": 0.0, "anywhere": 0.03}}},
    "no_gate": {"observe": {"gate_ratio_min": 0.0}},
    "no_air_flag": {"observe": {"air_ball_tau": None}},    # None: same as base_tau, handled below
    "no_restart_prior": {"observe": {"restart_weight": 0.0}},
}


def _deep_update(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _merge(cfg, over):
    """cfg with the ablation's overrides applied; None for air_ball_tau means 'same as base_tau'"""
    out = copy.deepcopy(cfg)
    _deep_update(out, over)
    if out["observe"].get("air_ball_tau") is None:
        out["observe"]["air_ball_tau"] = out["observe"]["base_tau"]
    return out


def gt_agreement(world, res, cfg):
    gt = world.get("gt") or {}
    pc = cfg["possession"]
    mass_tp = res["mass_within"][1.0] if pc["tp_m"] == 1.0 else res["mass_within"][2.0]
    lh = lh_posterior(world["xy"], res["mean"], mass_tp, world["team"], world["pitch"], world["fps"], pc, world["kicks"])
    hits = [(gt.get(world.get("start", 0) + r) == lh["team"][r]) for r in range(len(lh["team"])) if gt.get(world.get("start", 0) + r) in ("H", "A")]
    return float(np.mean(hits)) if hits else None


def ablations(world, cfg, names=None, progress=None):
    out = {}
    for name in (names or ABLATIONS):
        w, res = run_filter(world, _merge(cfg, ABLATIONS[name]))
        out[name] = {"gt_agreement_lh_posterior": gt_agreement(w, res, cfg), "mean_entropy_bits": float(res["entropy"].mean())}
        if progress:
            progress(name, out[name])
    return out

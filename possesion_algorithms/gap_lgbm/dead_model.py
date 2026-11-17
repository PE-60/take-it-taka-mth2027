"""P(dead ball) per frame from team-neutral features, for gap_lgbm_dead.

Labels come from the dead-ball detector (dead_ball.json), but only from its surest parts:
  dead   frames of a dead interval inside the off_pitch, still_gap or baseline (no broadcast
         view / ball lost with it) terms — not no_counterpart (halftime, before / after the
         broadcast), which is trivially different and never decoded anyway
  alive  frames outside every dead interval
  none   everything within `margin` frames of a dead interval's start or end, where the
         detector's boundary is least sure, and frames without data
The detector's own inputs (broadcast view, ball off the pitch) are not features, so the model is
independent evidence rather than a copy of the detector.

Training windows (intervals.sample_windows) fall anywhere in play and, more often, straddle the
start or end of a dead ball. Inside a window the ball is ignored, exactly as in a real gap; the
ball at the window's edges is used where it is seen.

Features that favour neither team: the symmetric block (mean speed, still players, speed
spread, players near the lines, slowing down, ...), each team-block column as max and min over
the two teams, where the frame is in its window, and the team-neutral edge-ball columns.
No mirroring is needed. CV is grouped over the same time blocks as the team model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possesion_algorithms.gap_lgbm import model as M
from possesion_algorithms.gap_lgbm.features import SYM, TeamFeatures
from possesion_algorithms.gap_lgbm.intervals import Gaps, runs

CTX = ("t_since_s", "t_till_s", "win_len_s", "win_frac")
DEAD_EDGE = ("ball_before_speed", "ball_before_touchline_m", "ball_before_goalline_m",
             "ball_after_speed", "ball_after_touchline_m", "ball_after_goalline_m", "ball_shift_m")
_EDGE_IDX = [[c for c, _ in M.EDGE_COLS].index(c) for c in DEAD_EDGE]


def dead_feature_names(tf: TeamFeatures) -> list[str]:
    return (list(SYM) + [f"max_{c}" for c in tf.block] + [f"min_{c}" for c in tf.block]
            + list(CTX) + list(DEAD_EDGE))


def dead_labels(dead: np.ndarray, confident: np.ndarray, excluded: np.ndarray, valid: np.ndarray,
                margin: int) -> np.ndarray:
    """(n,) 1 sure dead, 0 alive, NaN not used."""
    y = np.full(len(dead), np.nan, np.float32)
    y[valid & ~dead] = 0.0
    y[confident & dead] = 1.0
    for a, b in runs(dead):
        y[max(0, a - margin):a + margin] = np.nan
        y[max(0, b - margin):b + margin] = np.nan
    y[excluded | ~valid] = np.nan
    return y


def edge_known(win: Gaps, ball_seen: np.ndarray):
    n = len(ball_seen)
    kb = (win.a - 1 >= 0) & ball_seen[np.clip(win.a - 1, 0, n - 1)]
    ka = (win.b < n) & ball_seen[np.clip(win.b, 0, n - 1)]
    return kb, ka


def build_dead_X(tf: TeamFeatures, win: Gaps, rows: np.ndarray, gap: np.ndarray,
                 ball_seen: np.ndarray) -> np.ndarray:
    fps = float(tf.fps)
    off = rows - win.a[gap]
    L = win.length[gap].astype(np.float32)
    ctx = np.column_stack([off / fps, (win.b[gap] - rows) / fps, L / fps, (off + 0.5) / L]).astype(np.float32)
    kb, ka = edge_known(win, ball_seen)
    edge = M.edge_features(tf, win, kb, ka)[:, _EDGE_IDX]
    H, A = tf.H[rows], tf.A[rows]
    return np.concatenate([tf.sym[rows], np.fmax(H, A), np.fmin(H, A), ctx, edge[gap]], axis=1)


def predict_dead(booster, tf, win: Gaps, ball_seen: np.ndarray, num_iteration=None) -> np.ndarray:
    rows, gap = win.frames()
    if not len(rows):
        return np.zeros(0)
    return booster.predict(build_dead_X(tf, win, rows, gap, ball_seen), num_iteration=num_iteration)


@dataclass
class DeadCV:
    folds: list
    summary: dict
    oof: np.ndarray          # flat over win.frames(), NaN outside the scored folds
    best_iterations: list
    prior: float             # share of dead among the training rows


def _metrics(p, y) -> dict:
    yb = y > 0.5
    pred = p > 0.5
    tp = int((pred & yb).sum())
    out = {"frames": int(len(y)), "dead_share": float(yb.mean()) if len(y) else float("nan")}
    if len(y):
        out.update(logloss=M.logloss(p, yb), auc=M.auc(p, yb), acc=float((pred == yb).mean()),
                   precision=tp / max(1, int(pred.sum())), recall=tp / max(1, int(yb.sum())))
    return out


def fold_line(m: dict) -> str:
    def f(k):
        v = m.get(k)
        return "  -  " if v is None or v != v else f"{v:.3f}"
    return (f"dead fold {m.get('fold', 'all')!s:>3}  windows {m.get('windows', 0):5d}  frames {m.get('frames', 0):7d}  "
            f"dead {f('dead_share')}  it {m.get('best_iteration', '-')!s:>4}  logloss {f('logloss')}  "
            f"auc {f('auc')}  acc {f('acc')}  precision {f('precision')}  recall {f('recall')}")


def _subsample(win: Gaps, rows, gap, stride, rng):
    offset = rng.integers(0, stride, size=win.n)
    return ((rows - win.a[gap]) + offset[gap]) % stride == 0


def cross_validate_dead(tf, win: Gaps, y_frame: np.ndarray, ball_seen: np.ndarray, block: int,
                        n_folds: int = 5, stride: int = 4, lgbm_params: dict | None = None,
                        seed: int = 0, log=print) -> DeadCV:
    params = {**M.LGBM_DEFAULTS, **(lgbm_params or {})}
    rng = np.random.default_rng(seed + 7)
    rows, gap = win.frames()
    y = y_frame[rows]
    valid = np.isfinite(y)
    sub = _subsample(win, rows, gap, stride, rng) & valid
    blk = win.a // block
    fold_of = blk % n_folds
    fold = fold_of[gap]
    X = build_dead_X(tf, win, rows, gap, ball_seen)
    oof = np.full(len(rows), np.nan)
    folds, its = [], []
    for f in range(n_folds):
        test = fold == f
        train_blocks = np.unique(blk[fold_of != f])
        if not (test & valid).any() or not len(train_blocks):
            continue
        val_blocks = rng.choice(train_blocks, size=max(1, int(round(0.2 * len(train_blocks)))), replace=False)
        is_val = np.isin(blk[gap], val_blocks)
        tr, va = sub & ~test & ~is_val, sub & ~test & is_val
        if not tr.any() or not va.any() or len(np.unique(y[tr])) < 2:
            continue
        booster = M._fit(params, X[tr], y[tr], X[va], y[va], seed=seed + 100 + f)
        it = int(booster.best_iteration or booster.current_iteration())
        its.append(it)
        oof[test] = booster.predict(X[test], num_iteration=it)
        sel = test & valid
        m = _metrics(oof[sel], y[sel])
        m.update(fold=f, best_iteration=it, windows=int(len(np.unique(gap[sel]))))
        folds.append(m)
        log(f"gap_lgbm_dead: dead fold {f} done: {it} trees, auc {m.get('auc', float('nan')):.3f}")
    scored = valid & np.isfinite(oof)
    summary = _metrics(oof[scored], y[scored])
    summary["windows"] = int(len(np.unique(gap[scored]))) if scored.any() else 0
    prior = float(y[sub].mean()) if sub.any() else 0.5
    return DeadCV(folds, summary, oof, its, prior)


def fit_final_dead(tf, win: Gaps, y_frame: np.ndarray, ball_seen: np.ndarray, rounds: int, stride: int = 4,
                   lgbm_params: dict | None = None, seed: int = 0):
    params = {**M.LGBM_DEFAULTS, **(lgbm_params or {})}
    rng = np.random.default_rng(seed + 2000)
    rows, gap = win.frames()
    y = y_frame[rows]
    sub = _subsample(win, rows, gap, stride, rng) & np.isfinite(y)
    X = build_dead_X(tf, win, rows[sub], gap[sub], ball_seen)
    return M._fit(params, X, y[sub], rounds=max(1, int(rounds)), seed=seed + 2000)

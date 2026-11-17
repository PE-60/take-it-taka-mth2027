"""Dataset from team T's point of view, LightGBM with grouped CV, baselines, and the gap HMM.

Every gap frame becomes two rows: T = H ([H block, A block, ...], label "H has it") and T = A
([A block, H block, ...], label "A has it"), with the anchors flipped to T's side. The base
rate is then exactly 0.5 and the model treats both teams alike. At prediction time
P(H) = (p(T = H) + 1 - p(T = A)) / 2.

Besides the per-frame team features, each row gets its gap's context: the anchors, where it is
in the gap, and the ball at the gap's edges (last seen before, first seen after: position,
velocity, distance to the touchline and goal line, in T's attacking frame).

CV: gaps are grouped into time blocks of `block_s` seconds; fold = block % n_folds, so every
fold has blocks from both halves. Pretend gaps never cross a block edge (intervals.py), and
each fold's early stopping uses a random 20 % of the *training* blocks, never the held-out fold.

The HMM has two states (H, A). Emissions are the per-frame probabilities raised to a
temperature tau <= 1 (neighbouring frames are not independent evidence), a team change costs
log(p_switch), and the chain starts from the team before the gap and must end on the team
after it, when they are known. (p_switch, tau) are tuned on the out-of-fold pretend gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from possesion_algorithms.gap_lgbm.features import DIFF_OF, SYM, TeamFeatures
from possesion_algorithms.gap_lgbm.intervals import Gaps

GAP_COLS = ("before", "after", "anchors_agree", "t_since_s", "t_till_s", "gap_len_s", "gap_frac")

# The ball at the edges of the gap, in T's attacking frame: the last seen frame before it (a - 1)
# and the first seen frame after it (b). Velocity over EDGE_VEL_FRAMES frames outside the gap.
# An edge is NaN when its anchor is unknown, so real and pretend gaps hide the same things.
# (name, flips sign when T changes side)
EDGE_COLS = (
    ("ball_before_x", True), ("ball_before_y", True), ("ball_before_vx", True), ("ball_before_vy", True),
    ("ball_before_speed", False), ("ball_before_touchline_m", False), ("ball_before_goalline_m", False),
    ("ball_after_x", True), ("ball_after_y", True), ("ball_after_vx", True), ("ball_after_vy", True),
    ("ball_after_speed", False), ("ball_after_touchline_m", False), ("ball_after_goalline_m", False),
    ("ball_shift_x", True), ("ball_shift_m", False),
)
EDGE_VEL_FRAMES = 5
MAX_BALL_SPEED = 40.0

LGBM_DEFAULTS = {
    "objective": "binary",
    "num_leaves": 15,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "learning_rate": 0.05,
    "verbose": -1,
    # RULES.md: rerunning must reproduce the same file. With these, LightGBM gives the same trees
    # for the same data and seed, whatever the thread count.
    "deterministic": True,
    "force_row_wise": True,
}


def feature_names(tf: TeamFeatures) -> list[str]:
    return ([f"T_{c}" for c in tf.block] + [f"O_{c}" for c in tf.block] + list(SYM)
            + [f"diff_{c}" for c in DIFF_OF] + list(GAP_COLS) + [c for c, _ in EDGE_COLS])


def edge_features(tf: TeamFeatures, gaps: Gaps, known_before: np.ndarray | None = None,
                  known_after: np.ndarray | None = None) -> np.ndarray:
    """(G, len(EDGE_COLS)) ball at the gap edges from H's side (flip the anti columns for A).
    An edge counts only where known_* says so (default: where its team anchor is known)."""
    out = np.full((gaps.n, len(EDGE_COLS)), np.nan, np.float32)
    if tf.ball is None or gaps.n == 0:
        return out
    n, k, fps = len(tf.ball), EDGE_VEL_FRAMES, float(tf.fps)
    length, width = float(tf.pitch[0]), float(tf.pitch[1])

    def at(rows, other, known):
        """Position, velocity (rows minus other, per second, signed so it points forwards in
        time) and line distances at `rows`, in H's frame; NaN where unknown or out of range."""
        ok = known & (rows >= 0) & (rows < n) & (other >= 0) & (other < n)
        r, o = np.clip(rows, 0, n - 1), np.clip(other, 0, n - 1)
        p = np.where(ok[:, None], tf.ball[r], np.nan)
        q = np.where(ok[:, None], tf.ball[o], np.nan)
        d = tf.direction[r][:, None]
        v = (p - q) * (fps / k) * np.sign(r - o)[:, None]
        sp = np.hypot(v[:, 0], v[:, 1])
        v = np.where((sp <= MAX_BALL_SPEED)[:, None], v, np.nan)
        rel = (p - np.array([length / 2, width / 2], np.float32)) * d
        return np.column_stack([rel[:, 0], rel[:, 1], v[:, 0] * d[:, 0], v[:, 1] * d[:, 0],
                                np.where(sp <= MAX_BALL_SPEED, sp, np.nan),
                                width / 2 - np.abs(rel[:, 1]), length / 2 - np.abs(rel[:, 0])])

    kb = gaps.before != 0 if known_before is None else np.asarray(known_before, bool)
    ka = gaps.after != 0 if known_after is None else np.asarray(known_after, bool)
    pre = at(gaps.a - 1, gaps.a - 1 - k, kb)
    post = at(gaps.b, gaps.b + k, ka)
    out[:, 0:7] = pre
    out[:, 7:14] = post
    out[:, 14] = post[:, 0] - pre[:, 0]
    out[:, 15] = np.hypot(post[:, 0] - pre[:, 0], post[:, 1] - pre[:, 1])
    return out


EDGE_ANTI = np.array([anti for _, anti in EDGE_COLS])


def build_X(tf: TeamFeatures, gaps: Gaps, rows: np.ndarray, gap: np.ndarray, t_is_h: bool) -> np.ndarray:
    """Feature rows for frames `rows` of gaps `gap`, from the side of T = H (or T = A)."""
    mine, theirs = (tf.H, tf.A) if t_is_h else (tf.A, tf.H)
    s = 1 if t_is_h else -1
    di = [tf.block.index(c) for c in DIFF_OF]
    off = rows - gaps.a[gap]
    fps = float(tf.fps)
    L = gaps.length[gap].astype(np.float32)
    before = gaps.before[gap].astype(np.float32) * s
    after = gaps.after[gap].astype(np.float32) * s
    g = np.column_stack([before, after, before * after,
                         off / fps, (gaps.b[gap] - rows) / fps, L / fps, (off + 0.5) / L])
    edge = edge_features(tf, gaps)
    if not t_is_h:
        edge[:, EDGE_ANTI] *= -1
    return np.concatenate([mine[rows], theirs[rows], tf.sym[rows],
                           mine[rows][:, di] - theirs[rows][:, di], g.astype(np.float32),
                           edge[gap]], axis=1)


# --------------------------------------------------------------------------- metrics

def logloss(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y) -> float:
    y = np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    ranks = (cum - (counts - 1) / 2.0)[inv]                  # average ranks, 1-based
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def accuracy(p, y) -> float:
    """Accuracy of p > 0.5; a 0.5 counts as half right."""
    if not len(y):
        return float("nan")
    return float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == np.asarray(y, bool))))


def baseline_prob(gaps: Gaps, rows: np.ndarray, gap: np.ndarray, mode: str) -> np.ndarray:
    """P(H) of the simple fills: 'before', 'after' or 'midpoint', falling back to the other anchor."""
    b, a = gaps.before[gap].astype(int), gaps.after[gap].astype(int)
    if mode == "before":
        s = np.where(b != 0, b, a)
    elif mode == "after":
        s = np.where(a != 0, a, b)
    elif mode == "midpoint":
        first = (rows - gaps.a[gap]) < gaps.length[gap] / 2.0
        s = np.where(first, np.where(b != 0, b, a), np.where(a != 0, a, b))
    else:
        raise ValueError(mode)
    return np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))


# --------------------------------------------------------------------------- HMM

def viterbi_gaps(p_h: np.ndarray, gaps: Gaps, p_switch: float, tau: float,
                 bucket: int = 256) -> np.ndarray:
    """Most likely H/A path per gap. p_h is flat over gaps.frames() order; returns True = H."""
    out = np.zeros(len(p_h), bool)
    lens = gaps.length
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    ls, lw = np.log1p(-p_switch), np.log(p_switch)
    p = np.clip(p_h, 1e-6, 1 - 1e-6)
    em = tau * np.stack([np.log(p), np.log1p(-p)], axis=1)          # (N, 2): state 0 = H, 1 = A
    order = np.argsort(lens, kind="stable")
    for k in range(0, len(order), bucket):
        g = order[k:k + bucket]
        L = int(lens[g].max())
        t = np.arange(L)
        active = t[None, :] < lens[g][:, None]                       # (G, L)
        idx = np.where(active, starts[g][:, None] + t[None, :], 0)
        E = np.where(active[..., None], em[idx], 0.0)                # (G, L, 2)
        bstate = np.where(gaps.before[g] > 0, 0, np.where(gaps.before[g] < 0, 1, -1))
        astate = np.where(gaps.after[g] > 0, 0, np.where(gaps.after[g] < 0, 1, -1))
        init = np.full((len(g), 2), np.log(0.5))
        for j in (0, 1):
            init[:, j] = np.where(bstate == j, ls, np.where(bstate == 1 - j, lw, init[:, j]))
        delta = init + E[:, 0]
        back = np.zeros((len(g), L, 2), bool)                        # True = came from the other state
        for i in range(1, L):
            stay = delta + ls
            sw = delta[:, ::-1] + lw
            came_other = sw > stay
            new = np.where(came_other, sw, stay) + E[:, i]
            act = active[:, i]
            delta = np.where(act[:, None], new, delta)
            back[:, i] = came_other & act[:, None]
        end = np.zeros((len(g), 2))
        for j in (0, 1):
            end[:, j] = np.where(astate == j, ls, np.where(astate == 1 - j, lw, 0.0))
        state = np.argmax(delta + end, axis=1)
        path = np.zeros((len(g), L), np.int8)
        for i in range(L - 1, -1, -1):
            path[:, i] = state
            flip = back[np.arange(len(g)), i, state]
            state = np.where(flip, 1 - state, state)
        out_idx = idx[active]
        out[out_idx] = path[active] == 0
    return out


def tune_hmm(p_h, y_h, valid, gaps: Gaps, switch_grid, tau_grid) -> dict:
    """(p_switch, tau) with the best decoded accuracy on frames with a teacher label; ties go to
    the smaller p_switch (more sceptical of team changes)."""
    rows = []
    for ps in sorted(switch_grid):
        for tau in sorted(tau_grid):
            dec = viterbi_gaps(p_h, gaps, ps, tau)
            acc = float(np.mean(dec[valid] == y_h[valid])) if valid.any() else float("nan")
            rows.append({"p_switch": float(ps), "tau": float(tau), "accuracy": acc})
    best = max(rows, key=lambda r: (r["accuracy"], -r["p_switch"]))
    return {"best": best, "grid": rows}


# --------------------------------------------------------------------------- CV

@dataclass
class CVResult:
    folds: list
    summary: dict
    hmm: dict
    best_iterations: list
    oof_p_h: np.ndarray
    importance: list = field(default_factory=list)


def _lgb():
    try:
        import lightgbm
    except ImportError as e:   # pragma: no cover
        raise ImportError("gap_lgbm needs lightgbm:  pip install lightgbm") from e
    return lightgbm


def _fit(params, X, y, Xv=None, yv=None, rounds=2000, early=100, seed=0):
    lgb = _lgb()
    p = {**params, "seed": seed}
    dtrain = lgb.Dataset(X, y, free_raw_data=False)
    if Xv is None:
        return lgb.train(p, dtrain, num_boost_round=rounds)
    dval = lgb.Dataset(Xv, yv, reference=dtrain)
    return lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(early, verbose=False)])


def predict_h(booster, XH, XA, num_iteration=None) -> np.ndarray:
    return 0.5 * (booster.predict(XH, num_iteration=num_iteration)
                  + 1.0 - booster.predict(XA, num_iteration=num_iteration))


def _fold_metrics(p, y, gaps, rows, gap, mask):
    """Model vs baselines on the frames in mask (y = H has it)."""
    out = {"frames": int(mask.sum()), "gaps": int(len(np.unique(gap[mask])))}
    if not mask.any():
        return out
    pm, ym = p[mask], y[mask]
    out.update(logloss=logloss(pm, ym), auc=auc(pm, ym), acc_model=accuracy(pm, ym))
    for mode in ("before", "after", "midpoint"):
        out[f"acc_{mode}"] = accuracy(baseline_prob(gaps, rows[mask], gap[mask], mode), ym)
    differ = (gaps.before[gap] != gaps.after[gap]) & (gaps.before[gap] != 0) & (gaps.after[gap] != 0)
    dm = differ[mask]
    out["frames_differ"] = int(dm.sum())
    if dm.any():
        out["acc_model_differ"] = accuracy(pm[dm], ym[dm])
        out["acc_midpoint_differ"] = accuracy(
            baseline_prob(gaps, rows[mask][dm], gap[mask][dm], "midpoint"), ym[dm])
    return out


def cross_validate(tf: TeamFeatures, gaps: Gaps, teacher_sign: np.ndarray, block: int, n_folds: int = 5,
                   stride: int = 4, lgbm_params: dict | None = None, seed: int = 0,
                   switch_grid=(0.001, 0.003, 0.01, 0.03, 0.1), tau_grid=(0.2, 0.35, 0.5, 0.75, 1.0),
                   log=print) -> CVResult:
    """Grouped CV over time blocks on the pretend gaps, then HMM tuning on the out-of-fold
    probabilities. Trains on every `stride`-th frame, scores on every frame."""
    params = {**LGBM_DEFAULTS, **(lgbm_params or {})}
    rng = np.random.default_rng(seed)
    rows, gap = gaps.frames()
    sign = teacher_sign[rows]
    valid = sign != 0
    y = sign > 0
    offset = rng.integers(0, stride, size=gaps.n)
    sub = (((rows - gaps.a[gap]) + offset[gap]) % stride == 0) & valid
    blk = gaps.a // block
    fold_of_gap = blk % n_folds
    fold = fold_of_gap[gap]
    XH = build_X(tf, gaps, rows, gap, True)
    XA = build_X(tf, gaps, rows, gap, False)

    oof = np.full(len(rows), 0.5)
    folds, best_its = [], []
    for f in range(n_folds):
        test = fold == f
        train_blocks = np.unique(blk[fold_of_gap != f])
        if not test.any() or not len(train_blocks):
            continue
        val_blocks = rng.choice(train_blocks, size=max(1, int(round(0.2 * len(train_blocks)))), replace=False)
        is_val = np.isin(blk[gap], val_blocks)
        tr = sub & ~test & ~is_val
        va = sub & ~test & is_val
        Xtr = np.vstack([XH[tr], XA[tr]])
        ytr = np.concatenate([y[tr], ~y[tr]]).astype(float)
        Xva = np.vstack([XH[va], XA[va]])
        yva = np.concatenate([y[va], ~y[va]]).astype(float)
        booster = _fit(params, Xtr, ytr, Xva, yva, seed=seed + f)
        it = int(booster.best_iteration or booster.current_iteration())
        best_its.append(it)
        oof[test] = predict_h(booster, XH[test], XA[test], num_iteration=it)
        m = _fold_metrics(oof, y, gaps, rows, gap, test & valid)
        m.update(fold=f, best_iteration=it, train_rows=int(len(ytr)))
        folds.append(m)
        log(f"gap_lgbm: fold {f} done: {it} trees, auc {m.get('auc', float('nan')):.3f}")

    allm = _fold_metrics(oof, y, gaps, rows, gap, valid & np.isin(fold, [m["fold"] for m in folds]))
    hmm = tune_hmm(oof, y, valid, gaps, switch_grid, tau_grid)
    best = hmm["best"]
    dec = viterbi_gaps(oof, gaps, best["p_switch"], best["tau"])
    for m in folds:
        sel = (fold == m["fold"]) & valid
        m["acc_hmm"] = float(np.mean(dec[sel] == y[sel])) if sel.any() else float("nan")
    allm["acc_hmm"] = float(np.mean(dec[valid] == y[valid])) if valid.any() else float("nan")
    allm["switches_per_gap_hmm"] = _switches(dec, gaps)
    allm["switches_per_gap_teacher"] = _switches(y, gaps)
    allm["switches_per_gap_argmax"] = _switches(oof > 0.5, gaps)
    return CVResult(folds, allm, hmm, best_its, oof)


def _switches(h: np.ndarray, gaps: Gaps) -> float:
    if gaps.n == 0:
        return float("nan")
    rows, gap = gaps.frames()
    change = np.zeros(len(h), bool)
    change[1:] = (h[1:] != h[:-1]) & (gap[1:] == gap[:-1])
    return float(change.sum() / gaps.n)


def _fold_line(m: dict) -> str:
    def f(k):
        v = m.get(k)
        return "   -  " if v is None or v != v else f"{v:.3f}"
    return (f"fold {m.get('fold', 'all')!s:>3}  gaps {m.get('gaps', 0):5d}  frames {m.get('frames', 0):7d}  "
            f"it {m.get('best_iteration', '-')!s:>4}  logloss {f('logloss')}  auc {f('auc')}  "
            f"acc model {f('acc_model')}  hmm {f('acc_hmm')}  | before {f('acc_before')}  "
            f"after {f('acc_after')}  mid {f('acc_midpoint')}  | before!=after: model {f('acc_model_differ')}  "
            f"mid {f('acc_midpoint_differ')}  ({m.get('frames_differ', 0)} fr)")


def report_lines(cv: CVResult) -> list[str]:
    lines = [_fold_line(m) for m in cv.folds]
    lines.append(_fold_line({**cv.summary, "fold": "all"}))
    b = cv.hmm["best"]
    s = cv.summary
    lines.append(f"hmm: p_switch {b['p_switch']}  tau {b['tau']}  decoded acc {b['accuracy']:.3f}  "
                 f"team changes per gap: hmm {s['switches_per_gap_hmm']:.2f}  "
                 f"argmax {s['switches_per_gap_argmax']:.2f}  teacher {s['switches_per_gap_teacher']:.2f}")
    return lines


def fit_final(tf: TeamFeatures, gaps: Gaps, teacher_sign: np.ndarray, rounds: int, stride: int = 4,
              lgbm_params: dict | None = None, seed: int = 0):
    """One model on every pretend gap, `rounds` trees (the median best iteration of the CV)."""
    params = {**LGBM_DEFAULTS, **(lgbm_params or {})}
    rng = np.random.default_rng(seed + 1000)
    rows, gap = gaps.frames()
    sign = teacher_sign[rows]
    offset = rng.integers(0, stride, size=gaps.n)
    sub = (((rows - gaps.a[gap]) + offset[gap]) % stride == 0) & (sign != 0)
    r, g, y = rows[sub], gap[sub], sign[sub] > 0
    X = np.vstack([build_X(tf, gaps, r, g, True), build_X(tf, gaps, r, g, False)])
    yy = np.concatenate([y, ~y]).astype(float)
    return _fit(params, X, yy, rounds=max(1, int(rounds)), seed=seed)

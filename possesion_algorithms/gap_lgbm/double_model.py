"""The ball-aware model of gap_lgbm_double: P(H) on frames where the ball is seen.

Features: the team-shape features of lgbm_shape (both team blocks, the symmetric block, the
differences) plus ball_features (nearest players now and in 1 s, players near the ball, ball
position and velocity in the team's attacking frame, kicks). Rows are mirrored (T = H, T = A).

Labels: the teacher (lh_alternative_V2) on alive frames where it sees the ball and has a team.

Out of fold only: the frames are split into time blocks (fold = block % n_folds) and every
alive frame with the ball seen is predicted by the model that did not train on its block, so the
model cannot memorise the teacher's individual mistakes. These out-of-fold probabilities are what
the final HMM uses. `purge` frames at each block edge are left out of training (the 1 s
look-ahead and deltas would otherwise reach into the held-out block).

The HMM's p_switch and tau for these frames are tuned by decoding every alive interval from the
out-of-fold probabilities (ball-missing frames neutral) and scoring against the teacher; no
setting can match the teacher's random mistakes, so this rewards generalisation, not copying.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possesion_algorithms.gap_lgbm import model as M
from possesion_algorithms.gap_lgbm.ball_features import BallFeatures, ball_feature_names, build_ball_X
from possesion_algorithms.gap_lgbm.features import TeamFeatures
from possesion_algorithms.gap_lgbm.intervals import Gaps, runs
from possesion_algorithms.gap_lgbm.shape_model import build_shape_X, shape_feature_names

TAG = "gap_lgbm_double"


def double_feature_names(tf: TeamFeatures) -> list[str]:
    return shape_feature_names(tf) + ball_feature_names()


def build_double_X(tf: TeamFeatures, bf: BallFeatures, rows: np.ndarray, t_is_h: bool) -> np.ndarray:
    return np.concatenate([build_shape_X(tf, rows, t_is_h), build_ball_X(bf, rows, t_is_h)], axis=1)


def predict_double(booster, tf, bf, rows, num_iteration=None) -> np.ndarray:
    return M.predict_h(booster, build_double_X(tf, bf, rows, True), build_double_X(tf, bf, rows, False),
                       num_iteration)


@dataclass
class BallModel:
    oof: np.ndarray          # (n,) out-of-fold P(H) on alive, ball-seen frames; NaN elsewhere
    folds: list
    summary: dict
    hmm: dict                # tuning grid and best (p_switch, tau)
    top_features: list


def train_ball_model(tf: TeamFeatures, bf: BallFeatures, sign: np.ndarray, live: np.ndarray,
                     ball_seen: np.ndarray, params: dict, lgbm_params: dict, log=print) -> BallModel:
    n, fps = len(sign), float(tf.fps)
    seed, stride, n_folds = int(params["seed"]), int(params["stride"]), int(params["n_folds"])
    block = max(1, int(round(params["block_s"] * fps)))
    purge = int(round(params["purge_s"] * fps))
    rng = np.random.default_rng(seed + 500)
    lp = {**M.LGBM_DEFAULTS, **lgbm_params}

    target = live & ball_seen                       # frames to predict
    labelled = target & (sign != 0)
    blk = np.arange(n) // block
    fold_of = blk % n_folds
    pos = np.arange(n) % block
    clean = (pos >= purge) & (pos < block - purge)
    offset = int(rng.integers(0, stride))
    train_ok = labelled & clean & ((np.arange(n) + offset) % stride == 0)
    log(f"{TAG}: ball model: {int(target.sum())} alive frames with the ball seen, {int(labelled.sum())} with a "
        f"teacher team; training on every {stride}th")

    oof = np.full(n, np.nan)
    folds, gains = [], []
    for f in range(n_folds):
        train_blocks = np.unique(blk[target & (fold_of != f)])
        test = np.flatnonzero(target & (fold_of == f))
        if not len(test) or not len(train_blocks):
            continue
        val_blocks = rng.choice(train_blocks, size=max(1, int(round(0.2 * len(train_blocks)))), replace=False)
        is_val = np.isin(blk, val_blocks)
        tr = np.flatnonzero(train_ok & (fold_of != f) & ~is_val)
        va = np.flatnonzero(train_ok & (fold_of != f) & is_val)
        if not len(tr) or not len(va):
            continue
        ytr, yva = sign[tr] > 0, sign[va] > 0
        booster = M._fit(lp, np.vstack([build_double_X(tf, bf, tr, True), build_double_X(tf, bf, tr, False)]),
                         np.r_[ytr, ~ytr].astype(float),
                         np.vstack([build_double_X(tf, bf, va, True), build_double_X(tf, bf, va, False)]),
                         np.r_[yva, ~yva].astype(float), seed=seed + 500 + f)
        it = int(booster.best_iteration or booster.current_iteration())
        oof[test] = predict_double(booster, tf, bf, test, it)
        gains.append(booster.feature_importance("gain", iteration=it))
        sel = test[labelled[test]]
        p, y = oof[sel], sign[sel] > 0
        m = {"fold": f, "frames": int(len(sel)), "best_iteration": it, "logloss": M.logloss(p, y),
             "auc": M.auc(p, y), "acc_vs_teacher": M.accuracy(p, y)}
        folds.append(m)
        log(f"{TAG}: ball fold {f}: {it} trees, {len(sel)} frames, logloss {m['logloss']:.3f}  auc {m['auc']:.3f}  "
            f"acc vs teacher {m['acc_vs_teacher']:.3f}")

    sel = labelled & np.isfinite(oof)
    summary = {"frames": int(sel.sum())}
    if sel.any():
        summary.update(logloss=M.logloss(oof[sel], sign[sel] > 0), auc=M.auc(oof[sel], sign[sel] > 0),
                       acc_vs_teacher=M.accuracy(oof[sel], sign[sel] > 0))

    line = (f"{TAG}: ball model all folds: logloss {summary.get('logloss', float('nan')):.3f}  "
            f"auc {summary.get('auc', float('nan')):.3f}  acc vs teacher "
            f"{summary.get('acc_vs_teacher', float('nan')):.3f}")
    if params.get("use_hmm", True):
        alive = runs(live)
        segs = Gaps(np.array([a for a, _ in alive], np.int64), np.array([b for _, b in alive], np.int64),
                    np.zeros(len(alive), np.int8), np.zeros(len(alive), np.int8))
        rows, _ = segs.frames()
        p = np.where(np.isfinite(oof[rows]), oof[rows], 0.5)
        switch_grid = [float(params["p_switch"])] if params["p_switch"] else [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        tau_grid = [float(params["tau_ball"])] if params["tau_ball"] else [0.2, 0.35, 0.5, 0.75, 1.0]
        hmm = M.tune_hmm(p, sign[rows] > 0, labelled[rows], segs, switch_grid, tau_grid)
        summary["acc_hmm_vs_teacher"] = hmm["best"]["accuracy"]
        line += (f"; HMM p_switch {hmm['best']['p_switch']}  tau {hmm['best']['tau']}  decoded acc vs teacher "
                 f"{hmm['best']['accuracy']:.3f}")
    else:                       # no HMM: frames are labelled one by one, nothing to tune
        hmm = {"best": {"p_switch": None, "tau": 1.0, "accuracy": None}, "grid": []}
    log(line)
    names = double_feature_names(tf)
    mean_gain = np.mean(gains, axis=0) if gains else np.zeros(len(names))
    top = [[k, round(float(g), 1)] for k, g in sorted(zip(names, mean_gain), key=lambda t: -t[1])[:20]]
    return BallModel(oof, folds, summary, hmm, top)

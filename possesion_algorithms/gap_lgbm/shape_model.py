"""lgbm_shape: possession from team shape alone, on every alive frame (a test of the model).

Unlike gap_lgbm, nothing is kept from the teacher: every alive frame gets P(H) from a LightGBM
model that sees only the two teams (their per-team blocks, the symmetric block and the
differences; no ball, no gap context, no anchors), and a two-state HMM per alive interval turns
that into spells. Dead-ball frames are D (the runner's dead-ball mode).

The teacher (lh_alternative_V2 on the corrected keepers) still provides the training labels, on
the frames where it sees the ball; its carried-forward labels in ball-missing frames are not
used. Rows are mirrored (T = H and T = A) as in model.py. CV is grouped over time blocks of
frames (fold = block % n_folds), with `purge` frames dropped at each block edge so the 1 s
deltas cannot leak across; the HMM's p_switch and tau are tuned by decoding whole alive
intervals from the out-of-fold probabilities and scoring them against the teacher.
"""

from __future__ import annotations

import time

import numpy as np

from possesion_algorithms.gap_lgbm import model as M
from possesion_algorithms.gap_lgbm.features import DIFF_OF, SYM, TeamFeatures
from possesion_algorithms.gap_lgbm.intervals import Gaps, runs, team_sign

TAG = "lgbm_shape"


def shape_feature_names(tf: TeamFeatures) -> list[str]:
    return ([f"T_{c}" for c in tf.block] + [f"O_{c}" for c in tf.block] + list(SYM)
            + [f"diff_{c}" for c in DIFF_OF])


def build_shape_X(tf: TeamFeatures, rows: np.ndarray, t_is_h: bool) -> np.ndarray:
    mine, theirs = (tf.H, tf.A) if t_is_h else (tf.A, tf.H)
    di = [tf.block.index(c) for c in DIFF_OF]
    return np.concatenate([mine[rows], theirs[rows], tf.sym[rows], mine[rows][:, di] - theirs[rows][:, di]], axis=1)


def predict_shape(booster, tf, rows, num_iteration=None) -> np.ndarray:
    return M.predict_h(booster, build_shape_X(tf, rows, True), build_shape_X(tf, rows, False), num_iteration)


def fit_and_label_shape(teacher: np.ndarray, tf: TeamFeatures, live: np.ndarray, ball_seen: np.ndarray,
                        params: dict, log=print) -> tuple[np.ndarray, dict]:
    t_start = time.time()
    n, fps = len(teacher), float(tf.fps)
    sign = team_sign(teacher)
    lgbm_params = {**M.LGBM_DEFAULTS, "num_leaves": int(params["num_leaves"]),
                   "min_data_in_leaf": int(params["min_data_in_leaf"]),
                   "feature_fraction": float(params["feature_fraction"]),
                   "bagging_fraction": float(params["bagging_fraction"]),
                   "learning_rate": float(params["learning_rate"])}
    seed, stride, n_folds = int(params["seed"]), int(params["stride"]), int(params["n_folds"])
    block = max(1, int(round(params["block_s"] * fps)))
    purge = int(round(params["purge_s"] * fps))
    rng = np.random.default_rng(seed + 300)

    rows_all = np.flatnonzero(live)
    labelled = live & ball_seen & (sign != 0)
    blk = np.arange(n) // block
    fold_of_frame = blk % n_folds
    pos = np.arange(n) % block
    clean = (pos >= purge) & (pos < block - purge)
    offset = int(rng.integers(0, stride))
    train_ok = labelled & clean & ((np.arange(n) + offset) % stride == 0)
    log(f"{TAG}: {int(live.sum())} alive frames, {int(labelled.sum())} with a teacher label and the ball "
        f"seen; training on every {stride}th of them")

    oof = np.full(n, np.nan)
    folds, its = [], []
    for f in range(n_folds):
        test_blocks = np.unique(blk[live & (fold_of_frame == f)])
        train_blocks = np.unique(blk[live & (fold_of_frame != f)])
        if not len(test_blocks) or not len(train_blocks):
            continue
        val_blocks = rng.choice(train_blocks, size=max(1, int(round(0.2 * len(train_blocks)))), replace=False)
        is_val = np.isin(blk, val_blocks)
        tr = np.flatnonzero(train_ok & (fold_of_frame != f) & ~is_val)
        va = np.flatnonzero(train_ok & (fold_of_frame != f) & is_val)
        if not len(tr) or not len(va):
            continue
        y_tr, y_va = sign[tr] > 0, sign[va] > 0
        Xtr = np.vstack([build_shape_X(tf, tr, True), build_shape_X(tf, tr, False)])
        Xva = np.vstack([build_shape_X(tf, va, True), build_shape_X(tf, va, False)])
        booster = M._fit(lgbm_params, Xtr, np.r_[y_tr, ~y_tr].astype(float), Xva,
                         np.r_[y_va, ~y_va].astype(float), seed=seed + 300 + f)
        it = int(booster.best_iteration or booster.current_iteration())
        its.append(it)
        test = np.flatnonzero(live & (fold_of_frame == f))
        oof[test] = predict_shape(booster, tf, test, it)
        sel = test[labelled[test]]
        p, y = oof[sel], sign[sel] > 0
        m = {"fold": f, "frames": int(len(sel)), "best_iteration": it, "logloss": M.logloss(p, y),
             "auc": M.auc(p, y), "acc": M.accuracy(p, y)}
        folds.append(m)
        log(f"{TAG}: fold {f}: {it} trees, {len(sel)} frames, logloss {m['logloss']:.3f}  auc {m['auc']:.3f}  "
            f"acc vs teacher {m['acc']:.3f}")

    # HMM tuning: decode every alive interval from the out-of-fold probabilities
    alive_runs = runs(live)
    segs = Gaps(np.array([a for a, _ in alive_runs], np.int64), np.array([b for _, b in alive_runs], np.int64),
                np.zeros(len(alive_runs), np.int8), np.zeros(len(alive_runs), np.int8))
    rows, _ = segs.frames()
    p_oof = np.where(np.isfinite(oof[rows]), oof[rows], 0.5)
    y_rows = sign[rows] > 0
    valid = labelled[rows]
    switch_grid = [float(params["p_switch"])] if params["p_switch"] else [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03]
    tau_grid = [float(params["tau"])] if params["tau"] else [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    hmm = M.tune_hmm(p_oof, y_rows, valid, segs, switch_grid, tau_grid)
    best = hmm["best"]
    sel = valid & np.isfinite(oof[rows])
    summary = {"frames": int(sel.sum()), "logloss": M.logloss(p_oof[sel], y_rows[sel]),
               "auc": M.auc(p_oof[sel], y_rows[sel]), "acc": M.accuracy(p_oof[sel], y_rows[sel]),
               "acc_hmm": best["accuracy"]}
    log(f"{TAG}: all folds: logloss {summary['logloss']:.3f}  auc {summary['auc']:.3f}  acc vs teacher "
        f"{summary['acc']:.3f}; HMM p_switch {best['p_switch']}  tau {best['tau']}  decoded acc vs teacher "
        f"{best['accuracy']:.3f}")

    rounds = int(np.median(its)) if its else 200
    tr = np.flatnonzero(train_ok)
    y = sign[tr] > 0
    booster = M._fit(lgbm_params, np.vstack([build_shape_X(tf, tr, True), build_shape_X(tf, tr, False)]),
                     np.r_[y, ~y].astype(float), rounds=rounds, seed=seed + 399)
    labels = np.full(n, "", dtype="<U1")
    p_final = np.full(n, np.nan)                    # P(home) per alive frame from the final model
    if len(rows_all):
        p = predict_shape(booster, tf, rows)
        p_final[rows] = p
        dec = M.viterbi_gaps(p, segs, best["p_switch"], best["tau"])
        labels[rows] = np.where(dec, "H", "A")
    agree = float(np.mean(labels[rows][valid] == teacher[rows][valid])) if valid.any() else float("nan")
    gain = booster.feature_importance("gain")
    top = sorted(zip(shape_feature_names(tf), gain.tolist()), key=lambda t: -t[1])[:20]
    report = {"cv_folds": folds, "cv_all": summary, "hmm_grid": hmm["grid"],
              "hmm_used": {"p_switch": best["p_switch"], "tau": best["tau"]}, "final_rounds": rounds,
              "agreement_with_teacher_in_sample": agree, "alive_intervals": segs.n,
              "top_features_gain": [[k, round(g, 1)] for k, g in top],
              "seconds": round(time.time() - t_start, 1)}
    # per-frame probabilities for the caller to write out; not part of the JSON report
    report["_proba"] = {"p_home": p_final, "p_home_oof": oof}
    log(f"{TAG}: final model {rounds} trees; agrees with the teacher on {100 * agree:.1f}% of its labelled frames "
        f"(in sample); {report['seconds']} s")
    return labels, report

"""gap_lgbm_double: every frame between fixed dead balls decoded by one three-state HMM.

    1. dead balls from the classical detector (dead_ball.json)
    2. the teacher (lh_alternative_V2, corrected keepers) on each alive interval
    3. where the ball is missing (and at the start of play): the team model and the dead model of
       gap_lgbm_dead, trained exactly as there
    4. where the ball is seen: a second, ball-aware LightGBM (double_model.py) trained on the
       teacher's labels, used only through its out-of-fold probabilities

Frames:
  fixed dead   ball seen inside a dead interval, or no_counterpart   -> D (the detector decides)
  free         everything else with data                             -> decoded H / A / D
               ball seen:    ball-aware model, H or A only (dead is never added where the ball
                             is seen)
               ball missing: team model + dead model as in gap_lgbm_dead (detector bonus on
                             frames it calls dead)
The teacher's labels are no longer fixed: where the ball-aware model is unsure and the HMM's
switching cost or the surrounding frames point the other way, the teacher is overruled.

With use_hmm=false (the default) there is no HMM at the end: every free frame takes the state with
the highest emission (H if P(H) > 0.5 where the ball is seen; H, A or dead from the team and dead
models where it is missing). There is then no switching cost, no minimum dead-ball length and no
restart rule, and the output can flicker between H and A where P(H) is near 0.5.

HMM settings (use_hmm=true): p_switch and the ball frames' tau are tuned on the ball model's out-of-fold
decoding; the ball-missing frames' tau comes from the team model; p_stop, p_restart and q from
the dead model's windows (pipeline_dead.train_dead_parts).
"""

from __future__ import annotations

import time

import numpy as np

from possesion_algorithms.base import Tracking
from possesion_algorithms.gap_lgbm import dead_model as DM
from possesion_algorithms.gap_lgbm import hmm3
from possesion_algorithms.gap_lgbm.ball_features import compute_ball_features
from possesion_algorithms.gap_lgbm.double_model import train_ball_model
from possesion_algorithms.gap_lgbm.intervals import Gaps, runs, team_sign
from possesion_algorithms.gap_lgbm.pipeline import lgbm_params_from, load_or_compute, train_team_model
from possesion_algorithms.gap_lgbm.pipeline_dead import (anchors, change_line, change_summary, dead_report,
                                                         detector_frames, team_model_report, team_prob,
                                                         train_dead_parts)

TAG = "gap_lgbm_double"


def fit_and_label_double(track: Tracking, masks: dict, params: dict, log=print) -> tuple[np.ndarray, dict]:
    t_start = time.time()
    n = track.n
    fr = detector_frames(track, masks)
    valid, ball_seen, dead, no_cp, live = fr.valid, fr.ball_seen, fr.dead, fr.no_cp, fr.live

    teacher, tf = load_or_compute(track, live, params, bool(params["cache"]), log)
    sign = team_sign(teacher)

    # --- 3. models for ball-missing frames (as gap_lgbm_dead)
    tm = train_team_model(track, live, teacher, tf, params, log, tag=TAG)
    free_missing = valid & ~no_cp & (~ball_seen | (live & (sign == 0)))
    dp = train_dead_parts(track, fr, sign, tf, tm, free_missing, params, log, tag=TAG)

    # --- 4. ball-aware model, out of fold, for ball-seen frames
    t0 = time.time()
    bf = compute_ball_features(track, tf.direction)
    bm = train_ball_model(tf, bf, sign, live, ball_seen, params, lgbm_params_from(params), log)
    p_switch = bm.hmm["best"]["p_switch"]
    tau_ball = bm.hmm["best"]["tau"]
    log(f"{TAG}: ball model in {time.time() - t0:.1f} s")

    # --- emissions per frame
    e_all = np.zeros((n, 3))
    team_code = np.zeros(n, np.int8)                       # anchors for the team / dead models
    seen_team = live & ball_seen & (sign != 0)
    team_code[seen_team] = sign[seen_team]
    team_code[(dead & ball_seen) | no_cp] = 2
    miss = valid & ~no_cp & ~ball_seen
    mr = runs(miss)
    if mr:
        ma = np.array([a for a, _ in mr], np.int64)
        mb = np.array([b for _, b in mr], np.int64)
        gb, ga = anchors(team_code, ma, mb)
        gaps = Gaps(ma, mb, gb, ga)
        rows, _ = gaps.frames()
        p_h = team_prob(tm, tf, gaps)
        p_d = DM.predict_dead(dp.booster, tf, gaps, ball_seen)
        bonus = np.where(dead[rows], float(params["detector_logodds"]), 0.0)
        e_all[rows] = hmm3.emissions(p_h, p_d, dp.cv.prior, tm.tau, dead_bonus=bonus)
    seen_rows = np.flatnonzero(live & ball_seen)
    p_ball = np.where(np.isfinite(bm.oof[seen_rows]), bm.oof[seen_rows], 0.5)
    pb = np.clip(p_ball, 1e-6, 1 - 1e-6)
    e_all[seen_rows, hmm3.H] = tau_ball * np.log(2 * pb)
    e_all[seen_rows, hmm3.A] = tau_ball * np.log(2 * (1 - pb))
    e_all[seen_rows, hmm3.D] = hmm3.NEG

    # --- decode every free run between fixed dead frames
    fixed_dead = (dead & ball_seen) | no_cp
    free = valid & ~fixed_dead
    fr_runs = runs(free)
    code = np.where(fixed_dead, 2, 0).astype(np.int8)
    labels = np.full(n, "", dtype="<U1")
    labels[fixed_dead] = "D"
    changes = {}
    if fr_runs:
        fa = np.array([a for a, _ in fr_runs], np.int64)
        fb = np.array([b for _, b in fr_runs], np.int64)
        sb, sa = anchors(code, fa, fb)
        segs = Gaps(fa, fb, sb, sa)
        rows, _ = segs.frames()
        restart = hmm3.restart_logprob(dp.last_team[np.clip(fa - 1, 0, n - 1)], dp.q)
        t0 = time.time()
        if params["use_hmm"]:
            dec = hmm3.viterbi3(e_all[rows], segs, restart, p_switch, dp.best["p_stop"], dp.best["p_restart"],
                                dp.k)
        else:
            # frame by frame: the state with the highest emission (P(H) > 0.5 on ball-seen frames;
            # on ball-missing frames dead wins where the dead model, with the detector bonus, beats alive)
            dec = np.argmax(e_all[rows], axis=1).astype(np.int8)
        labels[rows] = np.array(["H", "A", "D"])[dec]
        changes = change_summary(dec, rows, dead, live, sign)
        seen_t = live[rows] & ball_seen[rows] & (sign[rows] != 0)
        changes["teacher_overruled_seen_frames"] = int(np.sum(
            seen_t & (np.where(dec == hmm3.H, 1, -1) != sign[rows])))
        changes["teacher_overruled_missing_frames"] = changes["team_changed_frames"] - \
            changes["teacher_overruled_seen_frames"]
        changes["decoder"] = "hmm" if params["use_hmm"] else "frame by frame (no HMM)"
        log(f"{TAG}: decoded {segs.n} runs ({len(rows)} frames) in {time.time() - t0:.1f} s: " + change_line(changes)
            + f"; of the team changes, {changes['teacher_overruled_seen_frames']} where the ball is seen")

    dead_model, hmm_rep = dead_report(dp, tf, params, p_switch, tm.tau)
    hmm_rep["used"]["tau_ball"] = tau_ball
    report = {
        "changes": changes,
        "ball_model": {"cv_folds": bm.folds, "cv_all": bm.summary, "hmm_grid": bm.hmm["grid"],
                       "top_features_gain": bm.top_features},
        "team_model": team_model_report(tm, tf),
        "dead_model": dead_model,
        "hmm3": hmm_rep,
        "seconds": round(time.time() - t_start, 1),
    }
    log(f"{TAG}: done in {report['seconds']} s")
    return labels, report

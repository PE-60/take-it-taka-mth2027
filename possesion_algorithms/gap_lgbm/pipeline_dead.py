"""gap_lgbm_dead: team model + dead-vs-alive model + three-state HMM over everything the ball
leaves unclear.

Every frame of the tracking is one of
  fixed team   alive (detector), ball seen, the teacher has a team      -> the teacher's label
  fixed dead   ball seen inside a dead interval, or no_counterpart     -> D (the detector decides)
  free         ball missing (alive or dead), or ball seen in alive play before the teacher's
               first holder (start of play)                            -> decoded: H, A or D
  no data      origin < 0                                              -> '' (X)
Dead is allowed only on free frames where the ball is missing, so the model can extend, shorten
or add dead balls only where nobody sees the ball.

The team model is the one gap_lgbm trains. The dead model is dead_model.py. The HMM (hmm3.py)
decodes each run of free frames between its fixed neighbours. Its p_switch and tau come from the
team model's CV; p_stop and p_restart are tuned on windows around dead balls (out-of-fold dead
probabilities, no detector bonus); q (the restart rule) is estimated from the teacher at
restarts. On the real segments, frames the detector calls dead get `detector_logodds` added to
the dead emission, so the model needs real evidence to overrule it.

`detector_frames`, `train_dead_parts` and `dead_report` are shared with gap_lgbm_double.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from possesion_algorithms.base import Tracking
from possesion_algorithms.gap_lgbm import dead_model as DM
from possesion_algorithms.gap_lgbm import hmm3
from possesion_algorithms.gap_lgbm import model as M
from possesion_algorithms.gap_lgbm.intervals import Gaps, runs, sample_windows, team_sign
from possesion_algorithms.gap_lgbm.pipeline import (lgbm_params_from, load_or_compute, top_features,
                                                    train_team_model)

TAG = "gap_lgbm_dead"
CONFIDENT_TERMS = ("off_pitch", "still_gap", "baseline")


def load_dead_masks(path: str, start: int, n: int) -> dict:
    """{'dead', 'no_counterpart', 'off_pitch', 'still_gap', 'baseline'} masks over rows of a track
    starting at tactical frame `start`, from dead_ball.json."""
    from dead_ball_investigation.dead_ball import intervals_to_mask
    if not os.path.isfile(path):
        raise IOError(f"no dead-ball intervals at {path}\n"
                      "  make them:  python dead_ball_investigation/run.py --from-exports")
    with open(path) as fh:
        doc = json.load(fh)
    tl = (doc.get("timelines") or {}).get("tactical")
    if not tl:
        raise ValueError(f"{path} has no tactical timeline")
    terms = tl.get("terms") or {}
    out = {"dead": intervals_to_mask(tl["dead_ball"], start, n)}
    for k in ("no_counterpart",) + CONFIDENT_TERMS:
        if k not in terms:
            raise ValueError(f"{path}: tactical timeline has no term {k!r} (rerun dead_ball_investigation/run.py)")
        out[k] = intervals_to_mask(terms[k], start, n)
    return out


def forward_fill_team(sign: np.ndarray, where: np.ndarray) -> np.ndarray:
    """(n,) last nonzero sign at or before each row, among rows in `where`; 0 before the first."""
    s = np.where(where, sign, 0)
    idx = np.where(s != 0, np.arange(len(s)), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, s[np.maximum(idx, 0)], 0).astype(np.int8)


def estimate_restart_q(sign: np.ndarray, live: np.ndarray) -> tuple[float, int]:
    """Share of stoppages after which the teacher's first team differs from its last team before."""
    lr = runs(live)
    flips = []
    for (a0, b0), (a1, b1) in zip(lr, lr[1:]):
        before = sign[a0:b0][sign[a0:b0] != 0]
        after = sign[a1:b1][sign[a1:b1] != 0]
        if len(before) and len(after):
            flips.append(after[0] != before[-1])
    if len(flips) < 5:
        return 0.5, len(flips)
    return float(np.clip(np.mean(flips), 0.05, 0.95)), len(flips)


def anchors(code: np.ndarray, a: np.ndarray, b: np.ndarray):
    n = len(code)
    before = np.where(a > 0, code[np.clip(a - 1, 0, n - 1)], 0).astype(np.int8)
    after = np.where(b < n, code[np.clip(b, 0, n - 1)], 0).astype(np.int8)
    return before, after


def team_only(g: Gaps) -> Gaps:
    """The same gaps with dead anchors turned into 'team unknown', for the team model."""
    tb = np.where(np.abs(g.before) == 1, g.before, 0).astype(np.int8)
    ta = np.where(np.abs(g.after) == 1, g.after, 0).astype(np.int8)
    return Gaps(g.a, g.b, tb, ta)


def team_prob(tm, tf, g: Gaps) -> np.ndarray:
    tg = team_only(g)
    rows, gap = tg.frames()
    if not len(rows):
        return np.zeros(0)
    return M.predict_h(tm.booster, M.build_X(tf, tg, rows, gap, True), M.build_X(tf, tg, rows, gap, False))


@dataclass
class Frames:
    """The detector's view of one tracking, shared by gap_lgbm_dead and gap_lgbm_double."""
    valid: np.ndarray        # origin >= 0
    ball_seen: np.ndarray
    dead: np.ndarray
    no_cp: np.ndarray
    confident: np.ndarray    # sure dead (training label)
    live: np.ndarray         # valid & not dead


def detector_frames(track: Tracking, masks: dict) -> Frames:
    valid = track.origin >= 0
    dead = masks["dead"] & valid
    no_cp = masks["no_counterpart"] & valid
    confident = np.zeros(track.n, bool)
    for k in CONFIDENT_TERMS:
        confident |= masks[k]
    confident &= dead & ~no_cp
    return Frames(valid, np.isfinite(track.ball[:, 0]), dead, no_cp, confident, valid & ~dead)


@dataclass
class DeadParts:
    """The dead-vs-alive model, the restart rule and the tuned stop / restart costs."""
    booster: object
    cv: DM.DeadCV
    rounds: int
    win: Gaps
    dead_runs: list
    q: float
    q_est: float
    q_n: int
    k: int                   # minimum dead-ball length in frames
    last_team: np.ndarray    # (n,) teacher's last team at or before each row, in alive play
    tuned: dict
    no_dead_state: dict

    @property
    def best(self) -> dict:
        return self.tuned["best"]


def train_dead_parts(track: Tracking, fr: Frames, sign: np.ndarray, tf, tm, free: np.ndarray, params: dict,
                     log=print, tag: str = TAG) -> DeadParts:
    """Dead model (windows, CV, final fit), restart rule, and p_stop / p_restart tuned on windows."""
    n, fps = track.n, float(track.fps)
    dead, live, valid, no_cp, ball_seen = fr.dead, fr.live, fr.valid, fr.no_cp, fr.ball_seen
    block = max(1, int(round(params["block_s"] * fps)))
    margin = int(round(params["edge_margin_s"] * fps))
    y_dead = DM.dead_labels(dead, fr.confident, no_cp, valid, margin)
    lengths = np.array([b - a for a, b in runs(free)], np.int64)
    lengths = lengths[(lengths > 1) & (lengths <= params["max_window_s"] * fps)]
    dead_runs = [r for r in runs(dead) if not no_cp[r[0]:r[1]].any()]
    bounds = np.array([x for a, b in dead_runs for x in (a, b)], np.int64)
    rng = np.random.default_rng(int(params["seed"]) + 11)
    win = sample_windows(valid & ~no_cp, lengths, rng, coverage=params["dead_coverage"], boundaries=bounds,
                         per_boundary=int(params["boundary_windows"]), block=block)
    log(f"{tag}: dead model: {len(dead_runs)} dead balls to learn from, {win.n} windows "
        f"({int(win.length.sum())} frames); sure-dead frames {int(np.sum(y_dead == 1))}, "
        f"alive frames {int(np.sum(y_dead == 0))}")
    if win.n == 0:
        raise ValueError(f"{tag}: no training windows for the dead model")
    lgbm_params = lgbm_params_from(params)
    t0 = time.time()
    dcv = DM.cross_validate_dead(tf, win, y_dead, ball_seen, block, n_folds=int(params["n_folds"]),
                                 stride=int(params["stride"]), lgbm_params=lgbm_params,
                                 seed=int(params["seed"]), log=log)
    for m in dcv.folds:
        log(DM.fold_line(m))
    log(DM.fold_line({**dcv.summary, "fold": "all"}))
    log(f"{tag}: dead model CV in {time.time() - t0:.1f} s")
    d_rounds = int(np.median(dcv.best_iterations)) if dcv.best_iterations else 200
    booster = DM.fit_final_dead(tf, win, y_dead, ball_seen, d_rounds, stride=int(params["stride"]),
                                lgbm_params=lgbm_params, seed=int(params["seed"]))

    q_est, q_n = estimate_restart_q(sign, live)
    q = float(params["q"]) or q_est
    last_team = forward_fill_team(sign, live)
    k = max(2, int(round(params["min_dead_s"] * fps)))
    truth_code = np.full(n, -1, np.int8)
    truth_code[y_dead == 1] = hmm3.D
    alive_team = (y_dead == 0) & (sign != 0)
    truth_code[alive_team & (sign > 0)] = hmm3.H
    truth_code[alive_team & (sign < 0)] = hmm3.A
    win_code = np.zeros(n, np.int8)
    win_code[live & (sign != 0)] = sign[live & (sign != 0)]
    win_code[dead] = 2
    wb, wa = anchors(win_code, win.a, win.b)
    wseg = Gaps(win.a, win.b, wb, wa)
    wrows, _ = wseg.frames()
    scored = np.isfinite(dcv.oof)
    e_w = hmm3.emissions(team_prob(tm, tf, wseg), np.where(scored, dcv.oof, dcv.prior), dcv.prior, tm.tau)
    truth_w = np.where(scored, truth_code[wrows], -1)
    restart_w = hmm3.restart_logprob(last_team[np.clip(win.a - 1, 0, n - 1)], q)
    stop_grid = [float(params["p_stop"])] if params["p_stop"] else [1e-4, 3e-4, 1e-3, 3e-3]
    restart_grid = [float(params["p_restart"])] if params["p_restart"] else [1e-3, 3e-3, 1e-2, 3e-2]
    t0 = time.time()
    tuned = hmm3.tune(e_w, wseg, restart_w, truth_w, tm.p_switch, k, stop_grid, restart_grid)
    best = tuned["best"]
    team_only_dec = hmm3.viterbi3(np.where(np.arange(3) == hmm3.D, hmm3.NEG, e_w), wseg, restart_w,
                                  tm.p_switch, best["p_stop"], best["p_restart"], k)
    base = hmm3.score(team_only_dec, truth_w)
    log(f"{tag}: restart rule q {q:.2f} (estimated {q_est:.2f} from {q_n} restarts); "
        f"HMM tuned in {time.time() - t0:.1f} s: p_stop {best['p_stop']}  p_restart {best['p_restart']}  "
        f"accuracy {best['accuracy']:.3f}  dead precision {best['dead_precision']:.3f}  "
        f"dead recall {best['dead_recall']:.3f}  team accuracy {best['team_accuracy']:.3f}  "
        f"(H/A only, no dead state: accuracy {base['accuracy']:.3f})")
    return DeadParts(booster, dcv, d_rounds, win, dead_runs, q, q_est, q_n, k, last_team, tuned, base)


def dead_report(dp: DeadParts, tf, params: dict, p_switch: float, tau: float) -> tuple[dict, dict]:
    dead_model = {"cv_folds": dp.cv.folds, "cv_all": dp.cv.summary, "prior": dp.cv.prior,
                  "final_rounds": dp.rounds, "windows": {"n": dp.win.n, "frames": int(dp.win.length.sum())},
                  "dead_balls": len(dp.dead_runs),
                  "top_features_gain": top_features(dp.booster, DM.dead_feature_names(tf))}
    hmm = {"used": {"p_switch": p_switch, "tau": tau, "p_stop": dp.best["p_stop"],
                    "p_restart": dp.best["p_restart"], "q": dp.q, "min_dead_frames": dp.k,
                    "detector_logodds": float(params["detector_logodds"])},
           "q_estimated": dp.q_est, "q_restarts": dp.q_n, "grid": dp.tuned["grid"],
           "no_dead_state": dp.no_dead_state}
    return dead_model, hmm


def team_model_report(tm, tf) -> dict:
    return {"cv_folds": tm.cv.folds, "cv_all": tm.cv.summary, "hmm_grid": tm.cv.hmm["grid"],
            "p_switch": tm.p_switch, "tau": tm.tau, "final_rounds": tm.rounds,
            "pretend_gaps": {"n": tm.pretend.n, "frames": int(tm.pretend.length.sum())},
            "top_features_gain": top_features(tm.booster, M.feature_names(tf))}


def fit_and_label_dead(track: Tracking, masks: dict, params: dict, log=print) -> tuple[np.ndarray, dict]:
    t_start = time.time()
    n = track.n
    fr = detector_frames(track, masks)
    valid, ball_seen, dead, no_cp, live = fr.valid, fr.ball_seen, fr.dead, fr.no_cp, fr.live

    teacher, tf = load_or_compute(track, live, params, bool(params["cache"]), log)
    sign = team_sign(teacher)
    tm = train_team_model(track, live, teacher, tf, params, log, tag=TAG)

    # --- frame codes: fixed team (+1/-1), fixed dead (2), free (0), no data (0, not free)
    code = np.zeros(n, np.int8)
    fixed_team = live & ball_seen & (sign != 0)
    code[fixed_team] = sign[fixed_team]
    code[(dead & ball_seen) | no_cp] = 2
    free = valid & ~no_cp & (~ball_seen | (live & (sign == 0)))
    free_runs = runs(free)

    dp = train_dead_parts(track, fr, sign, tf, tm, free, params, log)

    # --- decode the real free runs
    fa = np.array([a for a, _ in free_runs], np.int64)
    fb = np.array([b for _, b in free_runs], np.int64)
    sb, sa = anchors(code, fa, fb)
    segs = Gaps(fa, fb, sb, sa)
    labels = np.full(n, "", dtype="<U1")
    labels[fixed_team] = np.where(sign[fixed_team] > 0, "H", "A")
    labels[(dead & ball_seen) | no_cp] = "D"
    report_changes = {}
    if segs.n:
        rows, _ = segs.frames()
        p_h = team_prob(tm, tf, segs)
        p_d = DM.predict_dead(dp.booster, tf, segs, ball_seen)
        bonus = np.where(dead[rows], float(params["detector_logodds"]), 0.0)
        e = hmm3.emissions(p_h, p_d, dp.cv.prior, tm.tau, allow_dead=~ball_seen[rows], dead_bonus=bonus)
        restart = hmm3.restart_logprob(dp.last_team[np.clip(fa - 1, 0, n - 1)], dp.q)
        dec = hmm3.viterbi3(e, segs, restart, tm.p_switch, dp.best["p_stop"], dp.best["p_restart"], dp.k)
        labels[rows] = np.array(["H", "A", "D"])[dec]
        report_changes = change_summary(dec, rows, dead, live, sign)
        log(f"{TAG}: {segs.n} free runs ({len(rows)} frames): " + change_line(report_changes))

    dead_model, hmm_rep = dead_report(dp, tf, params, tm.p_switch, tm.tau)
    report = {
        "changes": {"free_runs": segs.n, **report_changes},
        "team_model": team_model_report(tm, tf),
        "dead_model": dead_model,
        "hmm3": hmm_rep,
        "seconds": round(time.time() - t_start, 1),
    }
    log(f"{TAG}: done in {report['seconds']} s")
    return labels, report


def change_summary(dec, rows, dead, live, sign) -> dict:
    """What the decoding changed on `rows` compared with the detector and the teacher."""
    was_dead = dead[rows]
    had_team = live[rows] & (sign[rows] != 0)
    start_of_play = live[rows] & (sign[rows] == 0)
    return {
        "free_frames": int(len(rows)),
        "dead_added_frames": int(np.sum((dec == hmm3.D) & ~was_dead)),
        "dead_removed_frames": int(np.sum((dec != hmm3.D) & was_dead)),
        "team_changed_frames": int(np.sum(had_team & (dec != hmm3.D)
                                          & (np.where(dec == hmm3.H, 1, -1) != sign[rows]))),
        "start_of_play_filled_frames": int(np.sum(start_of_play & (dec != hmm3.D))),
    }


def change_line(c: dict) -> str:
    return (f"dead added on {c['dead_added_frames']} frames, removed on {c['dead_removed_frames']}; team changed "
            f"vs teacher on {c['team_changed_frames']}; start of play filled on {c['start_of_play_filled_frames']}")

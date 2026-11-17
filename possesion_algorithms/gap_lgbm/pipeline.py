"""Teacher labels, pretend gaps, CV, final model, decoded target gaps — one call.

The teacher is lh_alternative_V2's ``label_lh_alt``, run on each alive run on its own (as the
runner does with dead_ball on), on whatever tracking it is given — GapLGBM hands it the tracking
with the corrected goalkeepers already swapped in, as lh_alt_gk_dead uses. Its labels are kept on
frames where the ball is seen; inside target gaps they are replaced by the HMM-decoded model
output. Only gaps the teacher enters with a team are predicted; gaps at the start of play
(before its first holder) keep the teacher's empty label.

Teacher labels and team features are cached in gap_lgbm/cache/ (git-ignored), keyed by a hash
of the tracking, the alive mask, the teacher parameters and FEATURE_VERSION, so a rerun that
only changes model or HMM settings skips the slow part.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from possesion_algorithms.algorithms.lh_alternative_V2 import label_lh_alt
from possesion_algorithms.base import Tracking
from possesion_algorithms.gap_lgbm import model as M
from possesion_algorithms.gap_lgbm.features import TeamFeatures, block_names, compute_team_features
from possesion_algorithms.gap_lgbm.intervals import (Gaps, runs, sample_pretend_gaps, target_gaps, team_sign,
                                                      visible_runs)

FEATURE_VERSION = 2
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TEACHER_KEYS = ("t_p", "t_a", "horizon_s", "kick_win", "min_hold_s")


def teacher_labels(track: Tracking, live: np.ndarray, params: dict) -> np.ndarray:
    out = np.full(track.n, "", dtype="<U1")
    tp = {k: params[k] for k in TEACHER_KEYS}
    for a, b in runs(live):
        out[a:b] = label_lh_alt(track.ball[a:b], track.xy[a:b], track.team, track.kick[a:b],
                                track.pitch, track.fps, tp)
    return out


def _cache_key(track: Tracking, live: np.ndarray, params: dict) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"v": FEATURE_VERSION, "start": track.start, "n": track.n, "S": track.n_slots,
                         "pitch": list(track.pitch), "fps": track.fps,
                         "teacher": {k: params[k] for k in TEACHER_KEYS}}, sort_keys=True).encode())
    h.update(track.team.tobytes())
    h.update(np.packbits(live).tobytes())
    h.update(np.packbits(np.isfinite(track.ball[:, 0])).tobytes())
    h.update(np.packbits(track.kick).tobytes())
    h.update(np.ascontiguousarray(track.xy[::97]).tobytes())
    return h.hexdigest()[:16]


def load_or_compute(track: Tracking, live: np.ndarray, params: dict, use_cache: bool, log=print):
    key = _cache_key(track, live, params)
    path = os.path.join(CACHE_DIR, f"gap_lgbm_{key}.npz")
    if use_cache and os.path.isfile(path):
        z = np.load(path)
        log(f"gap_lgbm: teacher labels and features from cache ({os.path.relpath(path)})")
        tf = TeamFeatures(z["H"], z["A"], z["sym"], z["direction"], block_names(), float(track.fps),
                          tuple(track.pitch), track.ball.astype(np.float32) * np.asarray(track.pitch, np.float32))
        return z["teacher"].astype("<U1"), tf
    t0 = time.time()
    teacher = teacher_labels(track, live, params)
    log(f"gap_lgbm: teacher (lh_alternative_V2) in {time.time() - t0:.1f} s")
    t0 = time.time()
    tf = compute_team_features(track)
    log(f"gap_lgbm: team features in {time.time() - t0:.1f} s")
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, teacher=teacher, H=tf.H, A=tf.A, sym=tf.sym, direction=tf.direction)
    return teacher, tf


def switch_rate(teacher_sign: np.ndarray, live: np.ndarray, ball_seen: np.ndarray) -> float:
    """Team changes per frame of the teacher on consecutive alive, ball-seen frames."""
    ok = live & ball_seen & (teacher_sign != 0)
    pair = ok[1:] & ok[:-1]
    if not pair.any():
        return float("nan")
    return float((teacher_sign[1:] != teacher_sign[:-1])[pair].mean())


def lgbm_params_from(params: dict) -> dict:
    return {"num_leaves": int(params["num_leaves"]), "min_data_in_leaf": int(params["min_data_in_leaf"]),
            "feature_fraction": float(params["feature_fraction"]),
            "bagging_fraction": float(params["bagging_fraction"]),
            "learning_rate": float(params["learning_rate"])}


@dataclass
class TeamModel:
    """The H vs A model and everything it was trained on."""
    booster: object
    cv: M.CVResult
    targets: Gaps           # gaps the teacher enters with a team
    skipped: tuple          # (gaps, frames) the teacher enters without a team
    pretend: Gaps
    rate: float             # teacher team changes per visible frame
    p_switch: float
    tau: float
    rounds: int


def train_team_model(track: Tracking, live: np.ndarray, teacher: np.ndarray, tf: TeamFeatures, params: dict,
                     log=print, tag: str = "gap_lgbm") -> TeamModel:
    """Pretend gaps, grouped CV, 2-state HMM tuning and the final H vs A model."""
    fps = float(track.fps)
    ball_seen = np.isfinite(track.ball[:, 0])
    sign = team_sign(teacher)
    all_gaps = target_gaps(live, ball_seen, teacher)
    targets = target_gaps(live, ball_seen, teacher, require_team=True)
    skipped = (all_gaps.n - targets.n, int(all_gaps.length.sum() - targets.length.sum()))
    vis = visible_runs(live, ball_seen)
    block = max(1, int(round(params["block_s"] * fps)))
    lengths = targets.length
    lengths = lengths[lengths <= params["max_gap_s"] * fps]
    rng = np.random.default_rng(int(params["seed"]))
    pretend = sample_pretend_gaps(
        vis, lengths, teacher, track.kick, rng, coverage=params["coverage"],
        context=int(round(params["context_s"] * fps)), kick_bias=params["kick_bias"],
        p_no_before=float(np.mean(targets.before == 0)) if targets.n else 0.0,
        p_no_after=float(np.mean(targets.after == 0)) if targets.n else 0.0, block=block)
    rate = switch_rate(sign, live, ball_seen)
    log(f"{tag}: team model: {targets.n} gaps the teacher enters with a team "
        f"({int(targets.length.sum())} frames), {skipped[0]} without ({skipped[1]} frames); "
        f"{pretend.n} pretend gaps ({int(pretend.length.sum())} frames); "
        f"teacher team changes per visible frame {rate:.4f}")
    if pretend.n == 0:
        raise ValueError(f"{tag}: no pretend gaps could be placed (no visible runs long enough)")
    lgbm_params = lgbm_params_from(params)
    switch_grid = [0.001, 0.003, 0.01, 0.03, 0.1]
    if rate == rate and 1e-5 < rate < 0.5:
        switch_grid.append(round(rate, 5))
    t0 = time.time()
    cv = M.cross_validate(tf, pretend, sign, block, n_folds=int(params["n_folds"]), stride=int(params["stride"]),
                          lgbm_params=lgbm_params, seed=int(params["seed"]), switch_grid=switch_grid, log=log)
    for line in M.report_lines(cv):
        log(line)
    log(f"{tag}: team model CV in {time.time() - t0:.1f} s")
    rounds = int(np.median(cv.best_iterations)) if cv.best_iterations else 200
    booster = M.fit_final(tf, pretend, sign, rounds, stride=int(params["stride"]), lgbm_params=lgbm_params,
                          seed=int(params["seed"]))
    p_switch = float(params["p_switch"]) or cv.hmm["best"]["p_switch"]
    tau = float(params["tau"]) or cv.hmm["best"]["tau"]
    return TeamModel(booster, cv, targets, skipped, pretend, rate, p_switch, tau, rounds)


def top_features(booster, names, k: int = 20) -> list:
    gain = booster.feature_importance("gain")
    return [[n, round(g, 1)] for n, g in sorted(zip(names, gain.tolist()), key=lambda t: -t[1])[:k]]


def fit_and_label(track: Tracking, live: np.ndarray, params: dict, log=print) -> tuple[np.ndarray, dict]:
    """Labels (n,) H/A/'' for every row of `track` and a report of how they were made."""
    t_start = time.time()
    fps = float(track.fps)
    live = np.asarray(live, bool) & (track.origin >= 0)
    teacher, tf = load_or_compute(track, live, params, bool(params["cache"]), log)
    tm = train_team_model(track, live, teacher, tf, params, log)
    targets, skipped, cv = tm.targets, tm.skipped, tm.cv
    log(f"gap_lgbm: {skipped[0]} gaps ({skipped[1]} frames) skipped: the teacher has no team "
        f"before them (start of play); they keep the teacher's label")

    labels = np.where(live, teacher, "").astype("<U1")
    if targets.n:
        rows, gap = targets.frames()
        p = M.predict_h(tm.booster, M.build_X(tf, targets, rows, gap, True), M.build_X(tf, targets, rows, gap, False))
        dec = M.viterbi_gaps(p, targets, tm.p_switch, tm.tau)
        labels[rows] = np.where(dec, "H", "A")
        changed = float(np.mean(labels[rows] != teacher[rows]))
    else:
        changed = 0.0

    report = {
        "target_gaps": {"n": targets.n, "frames": int(targets.length.sum()),
                        "median_s": float(np.median(targets.length) / fps) if targets.n else None,
                        "before_unknown": float(np.mean(targets.before == 0)) if targets.n else None,
                        "after_unknown": float(np.mean(targets.after == 0)) if targets.n else None,
                        "frames_changed_vs_teacher": changed},
        "skipped_gaps": {"n": skipped[0], "frames": skipped[1]},
        "pretend_gaps": {"n": tm.pretend.n, "frames": int(tm.pretend.length.sum())},
        "teacher_switch_rate": tm.rate,
        "cv_folds": cv.folds, "cv_all": cv.summary, "hmm_grid": cv.hmm["grid"],
        "hmm_used": {"p_switch": tm.p_switch, "tau": tm.tau}, "final_rounds": tm.rounds,
        "top_features_gain": top_features(tm.booster, M.feature_names(tf)),
        "seconds": round(time.time() - t_start, 1),
    }
    log(f"gap_lgbm: final model {tm.rounds} trees, HMM p_switch {tm.p_switch} tau {tm.tau}; "
        f"{100 * changed:.1f}% of gap frames differ from the teacher's carry-forward; "
        f"{report['seconds']} s in total")
    return labels, report

#!/usr/bin/env python3
"""Inspect geom_hmm_dead on the match (or synthetic) — find why it may lose to LH.

Does not tune on labels. Uses the public 5-minute GT by default to measure whether the
possession axis and anti features separate H from A, and compares ablations:

    geom_hmm_dead          the full model
    near_gap_only          sticky 1D HMM on near_gap alone (local ball cue)
    poss_axis_flipped      same as geom but with -poss (catches H/A sign bugs)
    lh_dead                Link & Hoernig on the same dead-ball intervals (if runnable)

Examples:

    # Real match from viewer exports + dead_ball.json + public GT
    python possesion_algorithms/inspect_geom_hmm.py --from-exports

    # Synthetic smoke test (no proprietary data)
    python possesion_algorithms/inspect_geom_hmm.py --synthetic

    # Write JSON + per-frame CSV for plotting
    python possesion_algorithms/inspect_geom_hmm.py --from-exports \\
        --out-json evaluation/results/geom_hmm_inspect.json \\
        --out-csv  evaluation/results/geom_hmm_inspect_series.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possesion_algorithms.algorithms.geometry_hmm import (GeometryHMMDeadBall, fit_geometry)  # noqa: E402
from possesion_algorithms.algorithms.link_hoernig import LinkHoernigDeadBall  # noqa: E402
from possesion_algorithms.base import TACTICAL  # noqa: E402
from possesion_algorithms.inputs import MatchInputs  # noqa: E402
from possesion_algorithms.paths import DEAD_BALL_JSON, DEFAULT_SYNC, GT_DIR, RESULTS_DIR  # noqa: E402
from possesion_algorithms.runner import label_match  # noqa: E402
from possesion_algorithms.unsupervised.diagnose import (  # noqa: E402
    build_report, format_report, place_on_timeline, write_report, write_series_csv,
)
from possesion_algorithms.unsupervised.synthetic import make_synthetic_match  # noqa: E402


def load_gt(path: str | None, n_frames: int) -> np.ndarray:
    import glob
    if path is None:
        hits = sorted(glob.glob(os.path.join(GT_DIR, "*_possession_gt_public.csv")))
        if not hits:
            raise SystemExit(f"no public GT in {GT_DIR}")
        path = hits[0]
    sys.path.insert(0, os.path.join(os.path.dirname(GT_DIR)))
    import possession_metrics as pm
    labels = pm.read_label_csv(path)["label"].to_numpy()
    labels = np.array([str(v).strip().upper() for v in labels], dtype="<U1")
    if len(labels) != n_frames:
        raise SystemExit(f"GT length {len(labels)} != n_frames {n_frames}")
    print(f"GT: {os.path.relpath(path)}  graded={(labels != 'X').sum()}")
    return labels


def public_window(gt: np.ndarray) -> slice:
    graded = np.flatnonzero(gt != "X")
    if not len(graded):
        return slice(0, len(gt))
    return slice(int(graded[0]), int(graded[-1]) + 1)


def run_synthetic(args) -> int:
    m = make_synthetic_match(n_frames=args.synthetic_frames, seed=args.seed,
                             miss_ball_frac=0.35, ghost_episodes=4)
    live = m.labels != "D"
    fit = fit_geometry(m.track, live, {**GeometryHMMDeadBall.params, "with_dead": False,
                                       "n_iter": args.n_iter})
    pred = fit.labels.copy()
    pred[~live] = "D"
    pred[pred == ""] = "X"
    # LH baseline on the same oracle dead balls
    from possesion_algorithms.algorithms.link_hoernig import LinkHoernig
    lh = LinkHoernig().label(m.track)
    lh_full = lh.copy()
    last = "X"
    for i, lab in enumerate(lh_full):
        if lab in ("H", "A"):
            last = lab
        elif lab == "":
            lh_full[i] = last if last in ("H", "A") else "X"
    lh_full[~live] = "D"

    gt = m.labels
    report = build_report(fit, gt, pred, live_full=live,
                          baselines={"lh_dead": lh_full},
                          window=slice(0, len(gt)), fps=float(m.track.fps))
    print(format_report(report))
    if args.out_json:
        write_report(report, args.out_json)
        print(f"wrote {args.out_json}")
    if args.out_csv:
        poss = fit.poss
        near = fit.features.column("near_gap")
        write_series_csv(args.out_csv, gt, pred, poss, near, live,
                         fit.anomaly.ball_mask, slice(0, len(gt)))
        print(f"wrote {args.out_csv}")
    return 0


def run_match(args) -> int:
    inputs = MatchInputs(from_exports=args.from_exports, sync_path=args.sync,
                         dead_ball_path=args.dead_ball_json)
    gt = load_gt(args.gt, inputs.n_frames)
    window = public_window(gt)

    algo = GeometryHMMDeadBall(n_iter=args.n_iter)
    pred, summary = label_match(algo, inputs)
    fit = algo.fit
    if fit is None:
        raise SystemExit("geom_hmm_dead produced no fit")
    print(f"geom_hmm_dead: {summary['counts']}  diag={fit.diag}")

    # Live mask on full timeline from dead_ball.json
    dead = inputs.dead_ball()
    live_full = ~dead

    baselines = {}
    try:
        lh_pred, lh_sum = label_match(LinkHoernigDeadBall(), inputs)
        baselines["lh_dead"] = lh_pred
        print(f"lh_dead: {lh_sum['counts']}")
    except (IOError, ValueError) as e:
        print(f"lh_dead skipped: {e}")

    report = build_report(fit, gt, pred, live_full=live_full, baselines=baselines,
                          window=window, fps=float(inputs.tracking(TACTICAL).fps))
    print(format_report(report))

    if args.out_json:
        write_report(report, args.out_json)
        print(f"wrote {args.out_json}")
    if args.out_csv:
        n = inputs.n_frames
        poss = place_on_timeline(fit.track_start, n, fit.poss)
        near = place_on_timeline(fit.track_start, n, fit.features.column("near_gap"))
        live = live_full
        anom = place_on_timeline(fit.track_start, n, fit.anomaly.ball_mask.astype(np.float32)) > 0.5
        write_series_csv(args.out_csv, gt, pred, poss, near, live, anom, window)
        print(f"wrote {args.out_csv}")

    # Also drop the submission CSV next to results if asked
    if args.write_submission:
        from possesion_algorithms.runner import write_csv, validate, score
        path = os.path.join(args.out_dir, "geom_hmm_dead.csv")
        write_csv(path, pred)
        validate(path, inputs.n_frames)
        s = score(path)
        print(f"wrote {path}" + (f"  public spellF1={s['spell_f1_mean']:.4f}" if s else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="run on artificial tracking (no match data needed)")
    ap.add_argument("--synthetic-frames", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--from-exports", action="store_true",
                    help="read match_viewer/export instead of raw ATD")
    ap.add_argument("--sync", default=DEFAULT_SYNC)
    ap.add_argument("--dead-ball-json", default=DEAD_BALL_JSON)
    ap.add_argument("--gt", default=None, help="frame,label CSV (default: public GT)")
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--out-json", default=None,
                    help="write the full report JSON")
    ap.add_argument("--out-csv", default=None,
                    help="write per-frame series for the scored window")
    ap.add_argument("--write-submission", action="store_true",
                    help="also write evaluation/results/geom_hmm_dead.csv")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    if args.synthetic:
        return run_synthetic(args)
    return run_match(args)


if __name__ == "__main__":
    sys.exit(main())

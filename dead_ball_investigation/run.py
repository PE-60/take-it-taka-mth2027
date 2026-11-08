"""Compute dead-ball intervals for both recordings and write match_viewer/export/dead_ball.json.

    python dead_ball_investigation/run.py                 # from the raw data files (parses both ATDs)
    python dead_ball_investigation/run.py --from-exports  # from match_viewer exports (fast)
    python dead_ball_investigation/run.py --no-counterpart-missing --out other.json

Every file in the package also runs as a file from any directory; `python -m dead_ball_investigation`
works from the repo root.

match_viewer.html picks up export/dead_ball.json on load: a "dead ball (sync)" row group in the
single viewer and in the dual viewer (match_viewer.py --dual --serve).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.dead_ball import FORMULA, SyncedMatch, annotate_dead_ball, to_document  # noqa: E402
from dead_ball_investigation.paths import EXPORT_DIR, REPO, require_repo  # noqa: E402
from dead_ball_investigation.signals import (load_from_export, load_source, load_tags,  # noqa: E402
                                             load_tags_from_export)
from dead_ball_investigation.sync import DEFAULT_SYNC, SyncMap  # noqa: E402

DEFAULT_OUT = os.path.join(EXPORT_DIR, "dead_ball.json")
PUBLIC_GT = os.path.join(REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def gt_check(dead, start, gt_path=PUBLIC_GT):
    """Derived dead ball vs GT label D on the labelled tactical frames, or None without a GT file."""
    if not os.path.isfile(gt_path):
        return None
    import pandas as pd
    gt = pd.read_csv(gt_path)
    gt.columns = [c.strip().lower() for c in gt.columns]
    f = gt["frame"].to_numpy(dtype=np.int64)
    lab = gt["label"].astype(str).str.strip().str.upper().to_numpy()
    keep = (lab != "X") & (f >= start) & (f < start + len(dead))
    if not keep.any():
        return None
    pred, truth = dead[f[keep] - start], lab[keep] == "D"
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    return {"gt": os.path.relpath(gt_path, REPO), "frames": int(keep.sum()), "gt_dead": int(truth.sum()),
            "pred_dead": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else None,
            "accuracy": round(float((pred == truth).mean()), 4)}


def graded_blocks():
    try:
        from games import GAME, half_video_blocks
        return [[a, b] for _h, a, b in half_video_blocks(GAME)]
    except Exception:  # noqa: BLE001
        return None


def build_match(args) -> tuple[SyncedMatch, dict]:
    sync = SyncMap.from_json(args.sync)
    if args.from_exports:
        tac, br = load_from_export(args.tactical_export), load_from_export(args.broadcast_export)
        tags = load_tags_from_export(args.tactical_export)
        inputs = {"from": "exports", "tactical": args.tactical_export, "broadcast": args.broadcast_export}
    else:
        tac, br = load_source("tactical"), load_source("broadcast")
        try:
            tags = load_tags()
        except SystemExit as e:
            print(f"smart tagging not loaded: {e}")
            tags = []
        inputs = {"from": "raw data"}
    for name, s in (("tactical", tac), ("broadcast", br)):
        if s.source != name:
            sys.exit(f"--{name}-export holds a {s.source} export")
    return SyncedMatch(tac, br, sync, tags), inputs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync", default=DEFAULT_SYNC, help="sync file from synchronize_videos.py (default match_viewer/export/sync.json)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSON (default match_viewer/export/dead_ball.json)")
    ap.add_argument("--from-exports", action="store_true",
                    help="read the signals from match_viewer exports instead of parsing the raw files")
    ap.add_argument("--tactical-export", default="export/tactical_full", help="with --from-exports (relative to match_viewer/)")
    ap.add_argument("--broadcast-export", default="export/broadcast_full", help="with --from-exports (relative to match_viewer/)")
    ap.add_argument("--no-counterpart-missing", action="store_true",
                    help="treat the other recording's signals as missing (True) where a frame has no counterpart; "
                         "default: not missing, so an unmapped frame is never dead on that evidence alone")
    ap.add_argument("--max-gap", type=int, default=0,
                    help="a ball-lost interval joins the dead ball when it overlaps a no-view interval or comes "
                         "within this many frames of one (default 0: it has to share a frame)")
    args = ap.parse_args(argv)
    require_repo()

    match, inputs = build_match(args)
    print(f"\nsync: {len(match.sync.anchors)} anchor(s) from {args.sync}")
    for a in match.sync.anchors:
        print(f"  {a['label']:<32s} T {a['tactical']:>6d}  B {a['broadcast']:>6d}  offset {a['broadcast'] - a['tactical']:+d}")
    print(f"formula: {FORMULA}" + (f"  (max gap {args.max_gap} frames)" if args.max_gap else ""))

    result = annotate_dead_ball(match, args.no_counterpart_missing, args.max_gap)
    doc = to_document(match, result, os.path.relpath(os.path.abspath(args.sync), REPO),
                      args.no_counterpart_missing, graded_blocks(), inputs, args.max_gap)
    t = result["tactical"]
    check = gt_check(t.dead, t.start)
    if check:
        doc["timelines"]["tactical"]["public_gt"] = check

    for side, r in result.items():
        s = doc["timelines"][side]["stats"]
        print(f"\n{side}: frames {r.start}..{r.start + len(r.dead) - 1}, counterpart on {s['covered_frames']} "
              f"({100 * s['covered_frames'] / max(1, s['frames']):.1f}%)")
        for k in ("dead", "no_view", "ball_lost", "ball_lost_kept"):
            print(f"  {k:<14s} {s[k]['frames']:>7d} frames ({100 * s[k]['share']:5.1f}%) in {s[k]['intervals']} intervals")
        if "graded" in s:
            g = s["graded"]
            print(f"  graded frames: {100 * g['dead_share']:.1f}% dead ({g['dead_frames']} / {g['frames']})")
    if check:
        print(f"\npublic GT ({check['frames']} labelled frames, {check['gt_dead']} D): "
              f"precision {check['precision']}  recall {check['recall']}  IoU {check['iou']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh)
    print(f"\nwrote {args.out}")
    return 0


def cli() -> int:
    try:
        return main()
    except (IOError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())

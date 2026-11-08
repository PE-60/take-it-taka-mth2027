"""Offline runner: process a tactical (or other) MP4 frame range → referee_signals.json."""

from __future__ import annotations

import argparse
import os
import sys

import cv2

from .classify import TemporalSmoother
from .detect import DetectorConfig, detect_frame, detections_to_schema
from .io import save_signals
from .schema import empty_document
from .seeds import default_seeds_path, load_seeds

VIEWER_EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "match_viewer", "export")


def process_video(
    video: str,
    seeds_path: str | None,
    out_path: str,
    start: int = 0,
    end: int | None = None,
    source: str = "tactical",
    smooth: int | None = None,
    progress_every: int = 100,
) -> dict:
    seeds = load_seeds(seeds_path)
    cfg = DetectorConfig(seeds=seeds)
    win = smooth if smooth is not None else int((seeds.get("classifier") or {}).get("smooth_window", 9))
    smoother = TemporalSmoother(win)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or seeds.get("fps") or 25)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or seeds.get("width") or 1920)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or seeds.get("height") or 1080)
    if end is None:
        end = max(0, n - 1) if n else start
    end = max(start, end)

    doc = empty_document(source=source, fps=fps, width=W, height=H, seeds_path=seeds_path or default_seeds_path())
    detections = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(start):
            if not cap.grab():
                raise SystemExit("video shorter than --start")

    for f in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            print(f"stream ended at frame {f}", file=sys.stderr)
            break
        raw = detect_frame(frame, cfg, frame_idx=f)
        for r in raw:
            sig, conf = smoother.update(r["id"], r.get("signal", "none"), float(r.get("confidence") or 0))
            r["signal"] = sig
            r["confidence"] = conf
        detections.extend(detections_to_schema(raw))
        if progress_every and (f - start) % progress_every == 0:
            print(f"frame {f}/{end}  detections so far {len(detections)}", flush=True)

    cap.release()
    doc["detections"] = detections
    save_signals(doc, out_path)
    print(f"wrote {len(detections)} detections → {out_path}")
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Detect AR flags on a match video (Stein-style color/contour).")
    ap.add_argument("--video", required=True, help="path to MP4 (tactical preferred)")
    ap.add_argument("--seeds", default=None, help="JSON color seeds (default: package tactical defaults)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: match_viewer/export/referee_signals_<source>.json)")
    ap.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--smooth", type=int, default=None, help="temporal majority window (default from seeds)")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args(argv)

    out = args.out or os.path.join(VIEWER_EXPORT, f"referee_signals_{args.source}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    process_video(
        video=args.video,
        seeds_path=args.seeds,
        out_path=out,
        start=args.start,
        end=args.end,
        source=args.source,
        smooth=args.smooth,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()

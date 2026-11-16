"""Click ROIs on a video frame to write calibrated HSV seeds.

Usage:
  python -m referee_signals.calibrate_colors --video data/tactical/BAR-ATH_tactical.mp4 --frame 1000 --out my_seeds.json

Keys while the window is open:
  s  — sample shirt ROI (drag)
  f  — sample flag ROI
  k  — sample shorts ROI
  g  — sample grass ROI
  w  — write seeds and quit
  n / p — next / previous frame (±25)
  q  — quit without writing
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import cv2
import numpy as np

from .colors import hsv_stats_from_roi
from .seeds import default_seeds_path, load_seeds, save_seeds

MODE_KEYS = {"s": "kit_shirt_hsv", "f": "flag_hsv", "k": "kit_shorts_hsv", "g": "grass_hsv"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Calibrate referee_signals HSV seeds from video ROIs.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="base seeds JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    seeds = load_seeds(args.seeds or default_seeds_path())
    seeds = copy.deepcopy(seeds)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    f = max(0, min(args.frame, max(0, n - 1)))
    mode = "kit_shirt_hsv"
    dragging = False
    pt0 = pt1 = None

    def read(fi: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        return fr if ok else None

    frame = read(f)
    if frame is None:
        sys.exit("could not read frame")

    win = "calibrate_colors"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, pt0, pt1, frame, seeds
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            pt0 = (x, y)
            pt1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            pt1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            pt1 = (x, y)
            if pt0 and pt1:
                x0, x1 = sorted([pt0[0], pt1[0]])
                y0, y1 = sorted([pt0[1], pt1[1]])
                if x1 - x0 > 2 and y1 - y0 > 2:
                    lo, hi = hsv_stats_from_roi(frame, x0, y0, x1, y1)
                    seeds[mode] = [{"name": "calibrated", "lower": lo.tolist(), "upper": hi.tolist()}]
                    print(f"{mode}: lower={lo.tolist()} upper={hi.tolist()}")

    cv2.setMouseCallback(win, on_mouse)

    print("mode=shirt (s). Keys: s/f/k/g modes, n/p frames, w write, q quit")
    while True:
        vis = frame.copy()
        cv2.putText(vis, f"frame {f}  mode={mode}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if pt0 and pt1:
            cv2.rectangle(vis, pt0, pt1, (0, 255, 255), 1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("w"):
            save_seeds(seeds, args.out)
            print(f"wrote {args.out}")
            break
        if key in (ord("s"), ord("f"), ord("k"), ord("g")):
            mode = MODE_KEYS[chr(key)]
            print(f"mode={mode}")
        if key == ord("n"):
            f = min(f + 25, max(0, n - 1))
            frame = read(f) or frame
            pt0 = pt1 = None
        if key == ord("p"):
            f = max(0, f - 25)
            frame = read(f) or frame
            pt0 = pt1 = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Render annotated PNGs for kick contacts (QA without the full match_viewer).

    python -m ball_detection.run_preview --max-kicks 10
    # -> ball_detection/export/preview/kick_<frame>.png
"""

from __future__ import annotations

import argparse
import os
import sys

from ball_detection.attribute import TouchAttribution
from ball_detection.geometry import project_points
from ball_detection.inputs import data_available, load_homography, load_tracking, open_video, read_frame
from ball_detection.overlay import draw_candidate_pixels, draw_track_on_frame
from ball_detection.paths import EXPORT_DIR, ensure_export_dir, require_repo
from ball_detection.results import load_attributions, load_detections

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preview PNG overlays for ball_detection kicks.")
    ap.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    ap.add_argument("--detections", default=None)
    ap.add_argument("--attributions", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-kicks", type=int, default=20)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_repo()
    if cv2 is None:
        print("OpenCV (cv2) required for previews", file=sys.stderr)
        return 2
    if not data_available(args.source):
        print(
            f"match data for source={args.source!r} is not available.\n"
            "Overlay drawing is covered by: python -m ball_detection.tests.test_stage_c",
            file=sys.stderr,
        )
        return 2

    det_path = args.detections or os.path.join(EXPORT_DIR, "kick_ball_tracks.json")
    attr_path = args.attributions or os.path.join(EXPORT_DIR, "touch_attributions.json")
    if not os.path.isfile(det_path):
        print(f"missing {det_path}", file=sys.stderr)
        return 2
    detections = load_detections(det_path)
    attrs_by_kick: dict[int, TouchAttribution] = {}
    if os.path.isfile(attr_path):
        for a in load_attributions(attr_path).attributions:
            attrs_by_kick[int(a.kick_frame)] = a

    out_dir = args.out_dir or os.path.join(EXPORT_DIR, "preview")
    ensure_export_dir(out_dir)

    tracks = detections.tracks[: args.max_kicks]
    if not tracks:
        print("no tracks")
        return 0

    frames = sorted({(t.contact.frame if t.contact else t.kick_frame) for t in tracks})
    first, last = min(frames), max(frames)
    track = load_tracking(args.source, first=first, last=last)
    H_all, _e, valid = load_homography(args.source)

    cap = open_video(args.source)
    n_written = 0
    try:
        for tr in tracks:
            f = tr.contact.frame if tr.contact is not None else tr.kick_frame
            frame = read_frame(cap, f)
            if frame is None:
                continue
            attr = attrs_by_kick.get(tr.kick_frame)
            vis = draw_track_on_frame(frame, tr, attr, video_frame=f)
            # Candidate dots from attribution, projected.
            if attr is not None and attr.candidates and valid[f]:
                row = f - track.start
                if 0 <= row < track.n:
                    uv_list = []
                    id_to_i = {str(s): i for i, s in enumerate(track.slot_ids)}
                    for c in attr.candidates[:5]:
                        i = id_to_i.get(str(c.slot_id))
                        if i is None:
                            continue
                        ply = track.xy[row, i]
                        pix = project_points(H_all[f], ply.reshape(1, 2))[0]
                        if np_finite(pix):
                            uv_list.append((float(pix[0]), float(pix[1]), c.team, c.score))
                    draw_candidate_pixels(vis, uv_list)
            path = os.path.join(out_dir, f"kick_{f:06d}.png")
            cv2.imwrite(path, vis)
            n_written += 1
    finally:
        cap.release()

    print(f"wrote {n_written} previews under {out_dir}")
    return 0


def np_finite(pix) -> bool:
    import numpy as np
    return bool(np.isfinite(pix).all())


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI: Stage C touch / team attribution from Stage B tracks.

    python -m ball_detection.run_attribute
    python -m ball_detection.run_attribute \\
        --detections ball_detection/export/kick_ball_tracks.json \\
        --banks ball_detection/export/appearance_banks.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from ball_detection.appearance import load_banks
from ball_detection.attribute import AttributeParams, attribute_touch
from ball_detection.gk import resolve_goalkeeper_slot_ids
from ball_detection.inputs import data_available, load_homography, load_tracking, open_video, read_frame
from ball_detection.paths import EXPORT_DIR, ensure_export_dir, require_repo
from ball_detection.results import (
    AttributionExport,
    load_detections,
    save_attributions,
)
from ball_detection.viewer_sidecar import build_sidecar, save_sidecar


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Attribute kick contacts to outfield teams (Stage C).")
    ap.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    ap.add_argument("--detections", default=None,
                    help="kick_ball_tracks.json (default: ball_detection/export/kick_ball_tracks.json)")
    ap.add_argument("--banks", default=None)
    ap.add_argument("--out", default=None,
                    help="default: ball_detection/export/touch_attributions.json")
    ap.add_argument("--sidecar", default=None,
                    help="match_viewer sidecar (default: ball_detection/export/ball_detection.json "
                         "and copy under match_viewer/export/)")
    ap.add_argument("--gk-csv", default=None)
    ap.add_argument("--max-dist-m", type=float, default=3.0)
    ap.add_argument("--no-video", action="store_true",
                    help="distance-only attribution (skip kit hist from frames)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_repo()

    det_path = args.detections or os.path.join(EXPORT_DIR, "kick_ball_tracks.json")
    if not os.path.isfile(det_path):
        print(f"missing detections JSON: {det_path}\n"
              "  run Stage B first: python -m ball_detection.run_detect", file=sys.stderr)
        if not data_available(args.source):
            print("Stage C logic is covered by: python -m ball_detection.tests.test_stage_c",
                  file=sys.stderr)
            return 2
        return 2

    if not data_available(args.source):
        print(
            f"match data for source={args.source!r} is not available.\n"
            "Stage C logic is covered by: python -m ball_detection.tests.test_stage_c",
            file=sys.stderr,
        )
        return 2

    detections = load_detections(det_path)
    banks_path = args.banks or os.path.join(EXPORT_DIR, "appearance_banks.json")
    banks = load_banks(banks_path) if os.path.isfile(banks_path) else None
    if banks is None:
        print(f"warning: no banks at {banks_path}; distance-only attribution", file=sys.stderr)

    gk_csv = args.gk_csv
    if gk_csv is None:
        cand = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_correction", "goalkeeper_trackers", "goalkeeper_tracks.csv",
        )
        gk_csv = cand if os.path.isfile(cand) else None
    gk_ids = resolve_goalkeeper_slot_ids(csv_path=gk_csv)

    # Load the span covering all kick windows.
    frames_needed = []
    for t in detections.tracks:
        frames_needed.append(t.kick_frame)
        if t.contact is not None:
            frames_needed.append(t.contact.frame)
        frames_needed.extend(d.frame for d in t.detections)
    if not frames_needed:
        print("no tracks in detections export")
        return 0
    first, last = min(frames_needed), max(frames_needed)
    print(f"loading ATD frames {first}..{last}...")
    track = load_tracking(args.source, first=first, last=last)
    H_all, _reproj, valid = load_homography(args.source)
    params = AttributeParams(max_dist_m=args.max_dist_m)

    cap = None if args.no_video else open_video(args.source)
    attrs = []
    try:
        for i, tr in enumerate(detections.tracks):
            contact_f = tr.contact.frame if tr.contact is not None else tr.kick_frame
            row = contact_f - track.start
            if row < 0 or row >= track.n:
                attrs.append(attribute_touch(
                    tr, np.zeros((0, 2)), np.array([], dtype="<U1"), [],
                    np.eye(3), track.pitch_m, gk_ids=gk_ids, params=params))
                continue
            H = H_all[contact_f].copy()
            if not valid[contact_f]:
                H[:] = np.nan
            frame = None
            if cap is not None and not args.no_video:
                frame = read_frame(cap, contact_f)
            attrs.append(attribute_touch(
                tr, track.xy[row], track.team, track.slot_ids, H, track.pitch_m,
                frame_bgr=frame, banks=banks, gk_ids=gk_ids, params=params,
            ))
            if (i + 1) % 25 == 0:
                print(f"  … {i + 1}/{len(detections.tracks)}")
    finally:
        if cap is not None:
            cap.release()

    export = AttributionExport(
        source=args.source,
        banks_path=banks_path if banks is not None else "",
        detections_path=det_path,
        attributions=attrs,
    )
    out = args.out or os.path.join(ensure_export_dir(EXPORT_DIR), "touch_attributions.json")
    ensure_export_dir(os.path.dirname(out) or ".")
    save_attributions(export, out)

    labeled = sum(1 for a in attrs if a.team in ("H", "A"))
    print(f"wrote {out}  ({labeled}/{len(attrs)} with team)")

    sidecar = build_sidecar(detections, export)
    side_path = args.sidecar or os.path.join(EXPORT_DIR, "ball_detection.json")
    save_sidecar(sidecar, side_path)
    # Convenience copy for the HTML viewer (same pattern as referee_signals).
    viewer_copy = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "match_viewer", "export", "ball_detection.json",
    )
    try:
        ensure_export_dir(os.path.dirname(viewer_copy))
        save_sidecar(sidecar, viewer_copy)
        print(f"wrote sidecar {side_path} and {viewer_copy}")
    except OSError as e:
        print(f"wrote sidecar {side_path} (viewer copy failed: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

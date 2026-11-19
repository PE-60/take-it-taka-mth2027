"""CLI: harvest ball + outfield-kit appearance banks from ATD + homography + video.

Works without gated data only in the sense that it exits with a clear message; unit tests cover
the math on synthetic frames.

    python -m ball_detection.build_banks
    python -m ball_detection.build_banks --max-frames 500 --stride 5
"""

from __future__ import annotations

import argparse
import os
import sys

from ball_detection.appearance import AppearanceBanks, save_banks
from ball_detection.banks import BallCropParams, KitCropParams, accumulate_frame
from ball_detection.gk import resolve_goalkeeper_slot_ids
from ball_detection.harvest import (
    BallHarvestParams,
    KitHarvestParams,
    ball_harvest_mask,
    kit_harvest_mask,
    subsample_indices,
)
from ball_detection.inputs import (
    data_available,
    load_homography_slice,
    load_tracking,
    open_video,
    read_frame,
)
from ball_detection.paths import EXPORT_DIR, ensure_export_dir, require_repo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build ball/kit appearance banks (Stage A).")
    ap.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    ap.add_argument("--out", default=None, help="output JSON (default: ball_detection/export/appearance_banks.json)")
    ap.add_argument("--gk-csv", default=None,
                    help="optional goalkeeper_tracks.csv to union extra source ids")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--stride", type=int, default=10, help="keep every N-th harvestable frame")
    ap.add_argument("--max-frames", type=int, default=2000, help="cap on frames read from video")
    ap.add_argument("--ball-clearance-m", type=float, default=2.5,
                    help="min metres from outfield players when sampling the ball (avoid shoes)")
    ap.add_argument("--kit-ball-clearance-m", type=float, default=1.0)
    ap.add_argument("--max-reproj-error", type=float, default=1.0)
    ap.add_argument("--ball-radius-px", type=float, default=6.0)
    ap.add_argument("--min-bright-frac", type=float, default=0.35,
                    help="reject ball crops that are not white/light enough")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_repo()

    if not data_available(args.source):
        print(
            f"match data for source={args.source!r} is not available under data/ "
            "(or MTAH_DATA_DIR).\n"
            "Stage A logic is covered by: python -m ball_detection.tests.test_stage_a\n"
            "When data is present, re-run this command to write appearance banks.",
            file=sys.stderr,
        )
        return 2

    gk_csv = args.gk_csv
    if gk_csv is None:
        cand = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_correction", "goalkeeper_trackers", "goalkeeper_tracks.csv",
        )
        gk_csv = cand if os.path.isfile(cand) else None
    gk_ids = resolve_goalkeeper_slot_ids(csv_path=gk_csv)
    print(f"excluding goalkeeper slots: {sorted(gk_ids)}")

    print(f"loading ATD ({args.source})...")
    track = load_tracking(args.source, first=args.first, last=args.last)
    print(f"loading homography slice {track.start}..{track.start + track.n - 1}...")
    H, reproj, valid = load_homography_slice(
        args.source, first=track.start, last=track.start + track.n - 1)

    ball_params = BallHarvestParams(
        clearance_m=args.ball_clearance_m,
        max_reproj_error=args.max_reproj_error,
    )
    kit_params = KitHarvestParams(
        ball_clearance_m=args.kit_ball_clearance_m,
        max_reproj_error=args.max_reproj_error,
    )
    ball_mask = ball_harvest_mask(
        track.ball, track.xy, track.pitch_m,
        slot_ids=track.slot_ids, gk_ids=gk_ids,
        valid_H=valid, reproj_error=reproj, fps=track.fps, params=ball_params,
    )
    kit_mask = kit_harvest_mask(
        track.xy, track.team, track.ball, track.pitch_m,
        slot_ids=track.slot_ids, gk_ids=gk_ids,
        valid_H=valid, reproj_error=reproj, params=kit_params,
    )
    # Visit frames that are useful for either bank.
    useful = ball_mask | kit_mask.any(axis=1)
    frame_rows = subsample_indices(useful, stride=max(1, args.stride), limit=args.max_frames)
    print(f"harvest candidates: ball_frames={int(ball_mask.sum())} "
          f"kit_player_frames={int(kit_mask.sum())} reading={len(frame_rows)}")

    banks = AppearanceBanks(
        source=args.source,
        gk_excluded=sorted(gk_ids),
    )
    banks.ball.meta.update({"clearance_m": args.ball_clearance_m, "radius_px": args.ball_radius_px})
    crop_ball = BallCropParams(radius_px=args.ball_radius_px,
                               min_bright_frac=args.min_bright_frac)
    crop_kit = KitCropParams()

    cap = open_video(args.source)
    totals = {"ball": 0, "kit_H": 0, "kit_A": 0}
    try:
        for i, row in enumerate(frame_rows):
            video_frame = track.start + int(row)
            frame = read_frame(cap, video_frame)
            if frame is None:
                continue
            if not valid[row]:
                continue
            counts = accumulate_frame(
                banks, frame, H[row],
                ball_xy=track.ball[row],
                take_ball=bool(ball_mask[row]),
                players_xy=track.xy[row],
                team=track.team,
                kit_slot_mask=kit_mask[row],
                ball_params=crop_ball,
                kit_params=crop_kit,
            )
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            if (i + 1) % 100 == 0:
                print(f"  … {i + 1}/{len(frame_rows)} frames  samples={totals}")
    finally:
        cap.release()

    out = args.out or os.path.join(ensure_export_dir(EXPORT_DIR), "appearance_banks.json")
    ensure_export_dir(os.path.dirname(out) or ".")
    save_banks(banks, out)
    print(f"wrote {out}")
    print(f"samples: ball={banks.ball.n_samples} "
          + " ".join(f"kit_{t}={banks.kits[t].n_samples}" for t in sorted(banks.kits)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

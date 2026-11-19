"""CLI: Stage B kick-window ball detection (memory-budgeted parallel windows).

    python -m ball_detection.run_detect
    python -m ball_detection.run_detect --first 11185 --last 18684 \\
        --max-memory-mb 4096 --workers 4 --kick-stride 2
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ball_detection.appearance import AppearanceBanks, load_banks
from ball_detection.detect import DetectParams, iter_kick_windows, track_kick_window
from ball_detection.gk import resolve_goalkeeper_slot_ids
from ball_detection.inputs import (
    data_available,
    load_homography_slice,
    load_tracking,
    open_video,
    read_frame,
)
from ball_detection.kicks import KICK_METHODS, KickWindow, detect_kicks
from ball_detection.parallel import choose_workers, estimate_window_mb, subsample_windows
from ball_detection.paths import EXPORT_DIR, ensure_export_dir, require_repo
from ball_detection.results import DetectExport, KickBallTrack, save_detections
from ball_detection.roi import RoiParams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Kick-window ball re-detection (Stage B).")
    ap.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    ap.add_argument("--banks", default=None,
                    help="appearance_banks.json from Stage A "
                         "(default: ball_detection/export/appearance_banks.json)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: ball_detection/export/kick_ball_tracks.json)")
    ap.add_argument("--gk-csv", default=None)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--pre", type=int, default=8, help="frames before kick")
    ap.add_argument("--post", type=int, default=12, help="frames after kick")
    ap.add_argument("--max-kicks", type=int, default=None,
                    help="cap how many kicks to process after stride (default: all)")
    ap.add_argument("--kick-stride", type=int, default=1,
                    help="process every N-th detected kick (default 1 = all)")
    ap.add_argument("--kick-method", default="vector_peak", choices=KICK_METHODS,
                    help="kick detector: vector_peak (‖Δv‖ threshold) or extrema "
                         "(local min/max of a_x,a_y with prominence/width)")
    ap.add_argument("--kick-threshold", type=float, default=4.0,
                    help="vector_peak: ATD ‖Δv‖·fps peak threshold in m/s² (higher → fewer kicks)")
    ap.add_argument("--kick-prominence", type=float, default=3.0,
                    help="extrema: min vertical separation of a local min/max in m/s²")
    ap.add_argument("--kick-min-width-s", type=float, default=0.0,
                    help="extrema: min half-prominence width in seconds (reject spikes)")
    ap.add_argument("--kick-max-width-s", type=float, default=0.40,
                    help="extrema: max half-prominence width in seconds "
                         "(reject slow drifts; use a negative value to disable)")
    ap.add_argument("--kick-smooth-s", type=float, default=0.08,
                    help="extrema: moving-average window in seconds before finding extrema")
    ap.add_argument("--kick-refractory-s", type=float, default=0.3,
                    help="min seconds between accepted kicks (both methods)")
    ap.add_argument("--min-score", type=float, default=0.32)
    ap.add_argument("--pad-px", type=int, default=36,
                    help="expand the ATD-prior search ROI by this many pixels "
                         "(raise when the ball is airborne / far from the ground projection)")
    ap.add_argument("--locked-roi-pad-px", type=int, default=28,
                    help="search box around the locked tracker prediction once a blob is found "
                         "(raise for high balls that drift off the ground projection)")
    ap.add_argument("--player-radius-m", type=float, default=5.0,
                    help="include outfield player feet within this pitch distance of the prior "
                         "when building the ROI (larger = bigger search area near clusters)")
    ap.add_argument("--max-link-dist-px", type=float, default=28.0,
                    help="hard max jump between consecutive detections (no teleports)")
    ap.add_argument("--foot-exclude-px", type=float, default=14.0,
                    help="suppress blobs within this many pixels of projected feet")
    ap.add_argument("--min-bright-frac", type=float, default=0.30,
                    help="reject candidates that are not white/light enough (shoes)")
    ap.add_argument("--max-memory-mb", type=float, default=4096.0,
                    help="RAM budget for concurrent kick windows (default 4096 MiB)")
    ap.add_argument("--workers", type=int, default=None,
                    help="requested parallel windows (capped by --max-memory-mb; "
                         "default = as many as the budget allows)")
    return ap.parse_args(argv)


def _video_size(source: str) -> tuple[int, int]:
    import cv2
    cap = open_video(source)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    finally:
        cap.release()
    return max(w, 1), max(h, 1)


def _read_window_frames(cap, track_start: int, win_start: int, win_end: int):
    frames = []
    for idx in range(win_start, win_end):
        frames.append(read_frame(cap, track_start + idx))
    return frames


def _process_one_window(
        source: str,
        win: KickWindow,
        track_start: int,
        H: np.ndarray,
        ball: np.ndarray,
        xy: np.ndarray,
        banks: AppearanceBanks | None,
        pitch_m: tuple[float, float],
        fps: float,
        slot_ids,
        gk_ids,
        params: DetectParams,
) -> KickBallTrack:
    """One kick window in its own thread: own VideoCapture (safe concurrent seeks)."""
    cap = open_video(source)
    try:
        frames = _read_window_frames(cap, track_start, win.start, win.end)
    finally:
        cap.release()
    if any(fr is None for fr in frames):
        return KickBallTrack(
            kick_frame=track_start + win.kick_index,
            window_start=track_start + win.start,
            window_end=track_start + win.end,
        )
    try:
        return track_kick_window(
            frames, win, H, ball, xy, banks, pitch_m, fps,
            frame_offset=track_start, slot_ids=slot_ids, gk_ids=gk_ids, params=params,
        )
    finally:
        del frames


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_repo()

    if not data_available(args.source):
        print(
            f"match data for source={args.source!r} is not available under data/ "
            "(or MTAH_DATA_DIR).\n"
            "Stage B logic is covered by: python -m ball_detection.tests.test_stage_b\n"
            "When data + Stage A banks are present, re-run this command.",
            file=sys.stderr,
        )
        return 2

    if args.kick_stride < 1:
        print("--kick-stride must be >= 1", file=sys.stderr)
        return 2
    if args.max_memory_mb <= 0:
        print("--max-memory-mb must be > 0", file=sys.stderr)
        return 2

    banks_path = args.banks or os.path.join(EXPORT_DIR, "appearance_banks.json")
    banks = None
    if os.path.isfile(banks_path):
        banks = load_banks(banks_path)
        print(f"loaded banks {banks_path} "
              f"(ball_samples={banks.ball.n_samples})")
    else:
        print(f"warning: no banks at {banks_path}; using bright-pixel fallback",
              file=sys.stderr)

    gk_csv = args.gk_csv
    if gk_csv is None:
        cand = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_correction", "goalkeeper_trackers", "goalkeeper_tracks.csv",
        )
        gk_csv = cand if os.path.isfile(cand) else None
    gk_ids = resolve_goalkeeper_slot_ids(csv_path=gk_csv)

    print(f"loading ATD ({args.source}) frames "
          f"{args.first if args.first is not None else 0}.."
          f"{args.last if args.last is not None else 'end'}...")
    track = load_tracking(args.source, first=args.first, last=args.last)
    print(f"loading homography slice {track.start}..{track.start + track.n - 1}...")
    H, _reproj, valid = load_homography_slice(
        args.source, first=track.start, last=track.start + track.n - 1)
    H = np.asarray(H, dtype=np.float64).copy()
    H[~np.asarray(valid, dtype=bool)] = np.nan

    max_width = None if args.kick_max_width_s < 0 else args.kick_max_width_s
    kick_mask, _accel = detect_kicks(
        track.ball, track.pitch_m, track.fps,
        threshold=args.kick_threshold,
        refractory_s=args.kick_refractory_s,
        method=args.kick_method,
        prominence=args.kick_prominence,
        min_width_s=args.kick_min_width_s,
        max_width_s=max_width,
        smooth_s=args.kick_smooth_s,
    )
    all_windows = iter_kick_windows(kick_mask, args.pre, args.post, max_kicks=None)
    windows = subsample_windows(all_windows, args.kick_stride, args.max_kicks)
    n_raw = int(kick_mask.sum())
    print(f"kicks detected ({args.kick_method}): {n_raw}  after stride={args.kick_stride}"
          f"{'' if args.max_kicks is None else f' max={args.max_kicks}'}: "
          f"processing {len(windows)}")

    width, height = _video_size(args.source)
    frames_per_window = int(args.pre) + int(args.post) + 1
    window_mb = estimate_window_mb(width, height, frames_per_window)
    n_workers = choose_workers(args.max_memory_mb, window_mb, workers=args.workers)
    print(f"parallelism: ~{window_mb:.0f} MiB/window @ {width}x{height}×{frames_per_window}f  "
          f"budget={args.max_memory_mb:.0f} MiB → workers={n_workers}")

    params = DetectParams(
        min_score=args.min_score,
        max_link_dist_px=args.max_link_dist_px,
        max_pred_dist_px=max(args.max_link_dist_px * 1.3, args.max_link_dist_px + 4),
        foot_exclude_px=args.foot_exclude_px,
        min_bright_frac=args.min_bright_frac,
        locked_roi_pad_px=args.locked_roi_pad_px,
        roi=RoiParams(
            pad_px=args.pad_px,
            player_radius_m=args.player_radius_m,
        ),
    )

    tracks: list[KickBallTrack | None] = [None] * len(windows)
    if not windows:
        pass
    elif n_workers == 1:
        cap = open_video(args.source)
        try:
            for i, win in enumerate(windows):
                frames = _read_window_frames(cap, track.start, win.start, win.end)
                if any(fr is None for fr in frames):
                    tracks[i] = KickBallTrack(
                        kick_frame=track.start + win.kick_index,
                        window_start=track.start + win.start,
                        window_end=track.start + win.end,
                    )
                else:
                    tracks[i] = track_kick_window(
                        frames, win, H, track.ball, track.xy, banks, track.pitch_m,
                        track.fps, frame_offset=track.start, slot_ids=track.slot_ids,
                        gk_ids=gk_ids, params=params,
                    )
                del frames
                if (i + 1) % 10 == 0:
                    done = [t for t in tracks[: i + 1] if t is not None]
                    with_c = sum(1 for t in done if t.contact is not None)
                    print(f"  … {i + 1}/{len(windows)} kicks  contacts={with_c}")
                    gc.collect()
        finally:
            cap.release()
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_window,
                    args.source, win, track.start, H, track.ball, track.xy, banks,
                    track.pitch_m, track.fps, track.slot_ids, gk_ids, params,
                ): i
                for i, win in enumerate(windows)
            }
            finished = 0
            for fut in as_completed(futures):
                i = futures[fut]
                tracks[i] = fut.result()
                finished += 1
                if finished % 10 == 0 or finished == len(windows):
                    done = [t for t in tracks if t is not None]
                    with_c = sum(1 for t in done if t.contact is not None)
                    print(f"  … {finished}/{len(windows)} kicks  contacts={with_c}")
                    gc.collect()

    final_tracks = [t for t in tracks if t is not None]
    export = DetectExport(
        source=args.source,
        banks_path=banks_path if banks is not None else "",
        tracks=final_tracks,
    )
    out = args.out or os.path.join(ensure_export_dir(EXPORT_DIR), "kick_ball_tracks.json")
    ensure_export_dir(os.path.dirname(out) or ".")
    save_detections(export, out)

    with_contact = sum(1 for t in final_tracks if t.contact is not None)
    print(f"wrote {out}")
    print(f"tracks={len(final_tracks)} with_contact={with_contact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

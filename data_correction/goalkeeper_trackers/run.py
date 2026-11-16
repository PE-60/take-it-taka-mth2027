"""Derive corrected home/away goalkeeper tracks and write goalkeeper_tracks.csv.

    python data_correction/goalkeeper_trackers/run.py
    python data_correction/goalkeeper_trackers/run.py --trackers 24,46
    python -m data_correction.goalkeeper_trackers

Reads the tactical ATD + homography (+ video size for camera visibility), assigns each visible
keeper observation to the team defending that pitch half, and disambiguates same-half overlaps
with raw ATD speed. The source tracker slots are left unchanged. Output lands next to this
package (default: data_correction/goalkeeper_trackers/goalkeeper_tracks.csv), the same place
dead_ball_investigation/dead_ball.json lives relative to its suite.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_correction.goalkeeper_trackers.derive import (DEFAULT_TRACKERS,  # noqa: E402
                                                         derive_goalkeeper_tracks, status_counts,
                                                         write_csv)
from data_correction.goalkeeper_trackers.load import load_keeper_inputs  # noqa: E402
from data_correction.goalkeeper_trackers.paths import GOALKEEPER_CSV, REPO, require_repo  # noqa: E402


def parse_trackers(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected exactly two tracker ids, e.g. 24,46")
    return parts[0], parts[1]


def parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1920x1080")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1920x1080") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("video size must be positive")
    return width, height


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=GOALKEEPER_CSV,
                    help="output CSV (default data_correction/goalkeeper_trackers/goalkeeper_tracks.csv)")
    ap.add_argument("--trackers", type=parse_trackers, default=DEFAULT_TRACKERS,
                    help="comma-separated source ATD tracker ids (default 24,46)")
    ap.add_argument("--video-size", type=parse_size, default=None,
                    help="override WxH instead of reading the tactical video header")
    ap.add_argument("--game", default=None, help="games.py key (default GAME)")
    args = ap.parse_args(argv)
    require_repo()

    inputs = load_keeper_inputs(
        tracker_ids=tuple(args.trackers),
        game=args.game,
        video_wh=args.video_size,
    )
    derived = derive_goalkeeper_tracks(
        inputs.positions,
        inputs.speeds,
        inputs.visible,
        inputs.home_defending_half,
        inputs.away_defending_half,
        tracker_ids=inputs.tracker_ids,
        team_names=inputs.team_names,
        fps=inputs.fps,
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_rows = write_csv(derived, out_path)

    print(f"\ngame: {inputs.game}  source: {inputs.source}  frames: {inputs.n_frames}  "
          f"trackers: {','.join(inputs.tracker_ids)}")
    print(f"teams: home={inputs.team_names['home']}  away={inputs.team_names['away']}")
    for side, counts in status_counts(derived).items():
        labelled = int(np.isfinite(derived.position[side]).all(axis=1).sum())
        print(f"  {side:<5s} labelled {labelled:>7d} frames  statuses={counts}")
    print(f"\nwrote {os.path.relpath(out_path, REPO)} ({n_rows} rows)")
    return 0


def cli() -> int:
    try:
        return main()
    except (OSError, IOError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())

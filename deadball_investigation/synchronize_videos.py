"""Find kick-off sync frames for the tactical and broadcast recordings.

The two videos are not aligned. This script picks shared real-world moments — the first-half and
second-half kick-offs — on each timeline, so a later tool can map frames between them.

    python synchronize_videos.py                       # prints and writes export/sync.json
    python synchronize_videos.py --out other.json
    python synchronize_videos.py --no-write

export/sync.json is what `match_viewer.py --dual` reads. Each anchor is a pair of video frames
showing the same moment; from an anchor up to the next one, broadcast = tactical + offset.

Second-half kick-off
--------------------
Tactical: the first kick-off tag whose Half mentions 2 (or, failing that, the first one more than
50 minutes after the first kick-off), then the first ball appearance from there, as above.
Broadcast: the longest stretch after the first kick-off where fewer than 5 players are tracked
(halftime, at least 2 minutes), then the first frame after it with ball speed > 10 m/s.

Tactical kick-off frame
-----------------------
Take the earliest Smart Tagging instance whose Type is a kick-off, then from that tag's start
walk the tactical ATD until the ball first appears (non-NaN position). Smart Tagging lives on
the tactical video only; start/end are seconds from frame 0 (see DATA.md / starter/load_smart_tagging.py).

Broadcast kick-off frame
------------------------
The ATD ball *speed channel* is typically empty (match_viewer prints a note and falls back). Ball
speed is therefore derived the same way as match_viewer.bspeed_f: consecutive normalised (x, y)
positions, scaled by the pitch in metres, times fps. On the broadcast ATD (one record per
broadcast video frame), return the first frame where that speed exceeds 10 m/s.

Both videos must report the same fps in their container metadata; otherwise the script stops.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import cv2

sys.path.append("..")

from games import DATA_DIR, GAME, GAMES, require

# Params
FIELD_SEP = ":"
MISSING = "N"  # first character of "NaN" in the raw ATD
BALL_SPEED_THRESHOLD_MS = 10.0
KICKOFF_TYPE_NEEDLE = "kick off"  # matches "Kick Off (Start)" and "Kick Off (After Goal)"
SECOND_HALF_MIN_GAP_MIN = 50      # fallback: 2nd-half kick-off is at least this long after the 1st
HALFTIME_MIN_GAP_S = 120          # broadcast halftime: at least this long with < HALFTIME_MAX_PLAYERS tracked
HALFTIME_MAX_PLAYERS = 5
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "export", "sync.json")


def video_fps(path: str) -> float:
    """Container fps from OpenCV, or raise if the file cannot be opened / has no rate."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("could not open video: {}".format(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("no usable fps in video metadata: {}".format(path))
    return fps


def assert_same_fps(tactical_video: str, broadcast_video: str, tol: float = 1e-3) -> float:
    """Require both containers to advertise the same frame rate; return that rate."""
    fps_t = video_fps(tactical_video)
    fps_b = video_fps(broadcast_video)
    if abs(fps_t - fps_b) > tol:
        raise ValueError(
            "frame rates differ: tactical={:.6g} fps, broadcast={:.6g} fps".format(fps_t, fps_b))
    return fps_t


def load_kickoff_tags(smart_tagging_path: str, fps: float):
    """Smart Tagging instances whose Type mentions kick-off, earliest first.

    Codes/values are matched case-insensitively (DATA.md). Frames are round(seconds * fps).
    """
    out = []
    for instance in ET.parse(smart_tagging_path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        type_text = (tags.get("Type") or "").strip().lower()
        if KICKOFF_TYPE_NEEDLE not in type_text:
            continue
        try:
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
        except (TypeError, ValueError):
            continue
        out.append({
            "code": (instance.findtext("code") or "").strip(),
            "type": tags.get("Type"),
            "half": tags.get("Half"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
        })
    out.sort(key=lambda e: (e["start_frame"], e["end_frame"]))
    return out


def parse_ball_xy(ball_field: str):
    """Normalised (x, y) from an ATD ball group, or (None, None) when the ball is missing.

    DATA.md / kloppy expose ball_x, ball_y only; the optional third token is not relied on for
    speed (match_viewer often finds it empty and derives speed from positions instead).
    """
    ball_field = ball_field.strip()
    if not ball_field or ball_field[0] == MISSING:
        return None, None
    parts = ball_field.split(",")
    if len(parts) < 2:
        return None, None
    try:
        x = float(parts[0])
        y = float(parts[1])
    except ValueError:
        return None, None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None, None
    return x, y


def first_ball_appearance(atd_raw_path: str, from_frame: int) -> int:
    """First video frame >= from_frame where the tactical ATD has a ball position."""
    with open(atd_raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                video_frame = int(parts[0]) - 1  # ATD frame_id is 1-based
            except ValueError:
                continue
            if video_frame < from_frame:
                continue
            x, y = parse_ball_xy(parts[2])
            if x is not None:
                return video_frame
    raise RuntimeError(
        "no ball appearance in tactical ATD at or after frame {}".format(from_frame))


def frame_ball_speed_ms(prev_xy, xy, pitch, fps: float) -> float:
    """Ball speed in m/s between two consecutive frames — same formula as match_viewer.frame_speed.

    Coordinates are normalised 0–1; pitch is (length_m, width_m). Speed at frame i is the
    displacement from frame i-1, times fps.
    """
    length_m, width_m = pitch
    dx = (xy[0] - prev_xy[0]) * length_m
    dy = (xy[1] - prev_xy[1]) * width_m
    return math.hypot(dx, dy) * fps


def first_ball_speed_above(atd_raw_path: str, threshold_ms: float, fps: float, pitch,
                           from_frame: int = 0) -> int:
    """First broadcast-video frame >= from_frame whose frame-derived ball speed exceeds threshold_ms.

    Streams the broadcast ATD (aligned 1:1 with the broadcast video). Speed is computed from
    successive ball positions, not from the ATD speed channel.
    """
    prev_frame = None
    prev_xy = None
    with open(atd_raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                video_frame = int(parts[0]) - 1
            except ValueError:
                continue
            if video_frame < from_frame:
                continue
            xy = parse_ball_xy(parts[2])
            if xy[0] is None:
                # Missing ball breaks the consecutive pair, same as NaN in match_viewer's array.
                prev_frame, prev_xy = None, None
                continue
            if prev_xy is not None and video_frame == prev_frame + 1:
                speed = frame_ball_speed_ms(prev_xy, xy, pitch, fps)
                if speed > threshold_ms:
                    return video_frame
            prev_frame, prev_xy = video_frame, xy
    raise RuntimeError(
        "no broadcast ATD frame with frame-derived ball speed > {} m/s".format(threshold_ms))


def tracked_players(players_field: str) -> int:
    """Number of players with a finite (x, y) in an ATD players group ("x,y,...;x,y,...;...")."""
    return sum(1 for p in players_field.split(";") if parse_ball_xy(p)[0] is not None)


def halftime_gap(atd_raw_path: str, from_frame: int, fps: float):
    """(first, last) video frame of the longest stretch >= from_frame with few tracked players,
    or None if no stretch is at least HALFTIME_MIN_GAP_S long."""
    best, run_start, prev = None, None, None
    with open(atd_raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                video_frame = int(parts[0]) - 1
            except ValueError:
                continue
            if video_frame < from_frame:
                continue
            empty = tracked_players(parts[1]) < HALFTIME_MAX_PLAYERS
            if empty and run_start is None:
                run_start = video_frame
            elif not empty and run_start is not None:
                if best is None or prev - run_start > best[1] - best[0]:
                    best = (run_start, prev)
                run_start = None
            prev = video_frame
    if run_start is not None and (best is None or prev - run_start > best[1] - best[0]):
        best = (run_start, prev)
    if best is None or (best[1] - best[0] + 1) < HALFTIME_MIN_GAP_S * fps:
        return None
    return best


def second_half_kickoff_tag(kickoffs, first_tag, fps: float):
    """The kick-off tag that starts the second half (Half mentions 2, else the first one long after)."""
    for tag in kickoffs:
        if "2" in (tag.get("half") or "") and tag["start_frame"] > first_tag["start_frame"]:
            return tag
    min_gap = SECOND_HALF_MIN_GAP_MIN * 60 * fps
    for tag in kickoffs:
        if tag["start_frame"] - first_tag["start_frame"] > min_gap:
            return tag
    return None


def compute_sync(tactical_video, tactical_atd, smart_tagging, broadcast_video, broadcast_atd, pitch,
                 log=print):
    """Kick-off anchors on both timelines -> the dict written to sync.json."""
    fps = assert_same_fps(tactical_video, broadcast_video)
    log("video fps: {:.6g} (tactical and broadcast metadata agree)".format(fps))

    kickoffs = load_kickoff_tags(smart_tagging, fps)
    if not kickoffs:
        raise RuntimeError("no kick-off Smart Tagging instances found in {}".format(smart_tagging))
    first_tag = kickoffs[0]
    log("first kick-off smart tag: Type={!r} Half={!r} frames {}..{} ({:.3f}s .. {:.3f}s)".format(
        first_tag["type"], first_tag["half"], first_tag["start_frame"], first_tag["end_frame"],
        first_tag["start_s"], first_tag["end_s"]))

    t1 = first_ball_appearance(tactical_atd, first_tag["start_frame"])
    b1 = first_ball_speed_above(broadcast_atd, BALL_SPEED_THRESHOLD_MS, fps, pitch)
    anchors = [{"label": "1st-half kick-off", "tactical": t1, "broadcast": b1}]
    log("")
    log("tactical kick-off:  {}".format(frame_timestamp(t1, fps)))
    log("broadcast kick-off: {}".format(frame_timestamp(b1, fps)))
    log("offset (broadcast - tactical): {:+d} frames".format(b1 - t1))

    tag2 = second_half_kickoff_tag(kickoffs, first_tag, fps)
    gap = halftime_gap(broadcast_atd, b1, fps)
    if tag2 is None:
        log("warning: no second-half kick-off tag found, only the first-half anchor is written")
    elif gap is None:
        log("warning: no halftime gap found in the broadcast ATD, only the first-half anchor is written")
    else:
        t2 = first_ball_appearance(tactical_atd, tag2["start_frame"])
        b2 = first_ball_speed_above(broadcast_atd, BALL_SPEED_THRESHOLD_MS, fps, pitch, from_frame=gap[1] + 1)
        anchors.append({"label": "2nd-half kick-off", "tactical": t2, "broadcast": b2})
        log("")
        log("broadcast halftime gap: frames {}..{} ({:.1f} min)".format(gap[0], gap[1], (gap[1] - gap[0] + 1) / fps / 60))
        log("tactical 2nd-half kick-off:  {}".format(frame_timestamp(t2, fps)))
        log("broadcast 2nd-half kick-off: {}".format(frame_timestamp(b2, fps)))
        log("offset (broadcast - tactical): {:+d} frames ({:+d} vs 1st half)".format(b2 - t2, (b2 - t2) - (b1 - t1)))
    return {
        "fps": fps,
        "tactical_video": os.path.basename(tactical_video),
        "broadcast_video": os.path.basename(broadcast_video),
        "anchors": anchors,
        "note": "from each anchor up to the next, broadcast_frame = tactical_frame + (broadcast - tactical)",
    }


def frame_timestamp(frame: int, fps: float) -> str:
    """Human-readable clock for a 0-based video frame: H:MM:SS.mmm (plus the frame index)."""
    total_s = frame / fps
    hours = int(total_s // 3600)
    minutes = int((total_s % 3600) // 60)
    seconds = total_s % 60
    return "frame {:>6}  @  {}:{:02d}:{:06.3f}".format(frame, hours, minutes, seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write the anchors (default export/sync.json)")
    ap.add_argument("--no-write", action="store_true", help="only print, do not write a JSON file")
    args = ap.parse_args()

    fps_game = GAMES[GAME]["fps"]
    pitch = GAMES[GAME]["pitch"]
    tactical_video, tactical_atd, smart_tagging = require(
        ["video", "atd_raw", "smart_tagging"], source="tactical")
    broadcast_video, broadcast_atd = require(
        ["video", "atd_raw"], source="broadcast")

    print("data dir: {}".format(DATA_DIR))
    print("game:     {}".format(GAME))

    sync = compute_sync(tactical_video, tactical_atd, smart_tagging, broadcast_video, broadcast_atd, pitch)
    if abs(sync["fps"] - fps_game) > 1e-3:
        print("warning: container fps ({:.6g}) differs from games.py fps ({})".format(
            sync["fps"], fps_game))
    if not args.no_write:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(sync, fh, indent=2)
        print()
        print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (IOError, RuntimeError, ValueError, KeyError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        sys.exit(1)

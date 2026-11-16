"""Find kick-off sync frames for the tactical and broadcast recordings.

The two videos are not aligned. This script picks shared real-world moments — the first-half and
second-half kick-offs — on each timeline, so a later tool can map frames between them.

    python match_viewer/synchronize_videos.py                    # prints and writes match_viewer/export/sync.json
    python match_viewer/synchronize_videos.py --out other.json
    python match_viewer/synchronize_videos.py --no-write
    python match_viewer/synchronize_videos.py --first-half-xcorr # 1st half by cross-correlation too (see below)

match_viewer/export/sync.json is what `match_viewer.py --dual` reads. Each anchor is a pair of video frames
showing the same moment; from an anchor up to the next one, broadcast = tactical + offset.

Second half: ball-speed cross-correlation
-----------------------------------------
Tactical: the first kick-off tag whose Half mentions 2 (or, failing that, the first one more than
50 minutes after the first kick-off). From that tag's start, take the first whole stretch where the
ball is never missing (at least SIGNATURE_MIN_S long; shorter blips are skipped) and use its
frame-derived ball speed as a signature.
Broadcast: frame-derived ball speed in the window from 45 minutes after the first-half kick-off
alignment to 45 minutes before the end of the video. The signature is slid over that window with
a normalised cross-correlation that ignores frames where either side has no ball; the best lag
gives the broadcast frame of the signature start, and so the second-half offset.

First half by cross-correlation (optional, --first-half-xcorr)
-------------------------------------------------------------
Same technique from the first kick-off tag's start, with the signature start searched in the
first 45 minutes of the broadcast. It replaces the speed-threshold kick-off below (which is still
computed, printed for comparison, and used as the fallback if the correlation finds nothing).
The second-half search window then starts 45 min after the earlier of the two kick-off estimates.

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
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # games.py lives at the repo root
from games import DATA_DIR, GAME, GAMES, require  # noqa: E402

# Params
FIELD_SEP = ":"
MISSING = "N"  # first character of "NaN" in the raw ATD
BALL_SPEED_THRESHOLD_MS = 10.0
KICKOFF_TYPE_NEEDLE = "kick off"  # matches "Kick Off (Start)" and "Kick Off (After Goal)"
SECOND_HALF_MIN_GAP_MIN = 50      # fallback: 2nd-half kick-off is at least this long after the 1st
SIGNATURE_MIN_S = 4.0             # shortest ball-present stretch accepted as the tactical signature
SEARCH_MARGIN_MIN = 45            # broadcast window: this long after the 1st kick-off .. this long before the end
SMOOTH_FRAMES = 5                 # NaN-aware moving average on both speed curves (1 = off)
MAX_PLAUSIBLE_SPEED_MS = 40.0     # faster "speeds" are tracking teleports; treated as missing so one spike cannot dominate
MIN_OVERLAP = 0.25                # a lag counts only if this share of the signature overlaps broadcast ball data
MIN_OVERLAP_S = 3.0               # ... and at least this much overlap in absolute terms (broadcast ball is ~50% visible)
PEAK_EXCLUSION_S = 2.0            # runner-up peak must be at least this far from the best one
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


def video_frame_count(path: str) -> int:
    """Container frame count from OpenCV, or raise if it is not available."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("could not open video: {}".format(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if n <= 0:
        raise ValueError("no usable frame count in video metadata: {}".format(path))
    return n


def load_ball_track(atd_raw_path: str):
    """Ball (x, y) per 0-based video frame as two float arrays, NaN where the ball is missing."""
    frames, xs, ys = [], [], []
    with open(atd_raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                video_frame = int(parts[0]) - 1
            except ValueError:
                continue
            x, y = parse_ball_xy(parts[2])
            if x is not None:
                frames.append(video_frame)
                xs.append(x)
                ys.append(y)
    if not frames:
        raise RuntimeError("no ball positions in {}".format(atd_raw_path))
    n = max(frames) + 1
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bx[frames] = xs
    by[frames] = ys
    return bx, by


def ball_speed_curve(bx, by, pitch, fps: float):
    """Frame-derived ball speed (m/s) per frame, same formula as frame_ball_speed_ms.
    NaN on the first frame, wherever this frame or the previous one has no ball, and above
    MAX_PLAUSIBLE_SPEED_MS."""
    length_m, width_m = pitch
    sp = np.full(len(bx), np.nan)
    sp[1:] = np.hypot(np.diff(bx) * length_m, np.diff(by) * width_m) * fps
    sp[sp > MAX_PLAUSIBLE_SPEED_MS] = np.nan
    return sp


def nan_smooth(v, k: int):
    """Centred moving average of width k that skips NaNs; stays NaN where v is NaN."""
    if k <= 1:
        return v.copy()
    ok = np.isfinite(v)
    ker = np.ones(k)
    num = np.convolve(np.where(ok, v, 0.0), ker, "same")
    den = np.convolve(ok.astype(float), ker, "same")
    out = np.full(len(v), np.nan)
    out[ok] = num[ok] / den[ok]
    return out


def ball_present_runs(bx, from_frame: int):
    """(first, last) frames of consecutive ball-present stretches starting at or after from_frame."""
    present = np.isfinite(bx[from_frame:])
    edges = np.diff(np.concatenate([[0], present.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [(int(a) + from_frame, int(b) + from_frame) for a, b in zip(starts, ends)]


def _xcorr_valid(a, t):
    """sum_i t[i] * a[k + i] for every k with the template fully inside a (FFT)."""
    n = len(a) + len(t) - 1
    size = 1 << (n - 1).bit_length()
    full = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(t[::-1], size), size)[:n]
    return full[len(t) - 1:len(a)]


def masked_ncc(signal, template, min_overlap: float = MIN_OVERLAP, min_samples: int = 0):
    """Normalised cross-correlation of template against signal at every 'valid' lag, using only
    samples where both are finite. Returns (scores, overlap counts); lags where fewer than
    min_overlap of the template's finite samples (or fewer than min_samples) overlap are -inf."""
    ms, mt = np.isfinite(signal).astype(float), np.isfinite(template).astype(float)
    s, t = np.where(ms > 0, signal, 0.0), np.where(mt > 0, template, 0.0)
    n = _xcorr_valid(ms, mt)
    sum_s, sum_ss = _xcorr_valid(s, mt), _xcorr_valid(s * s, mt)
    sum_t, sum_tt = _xcorr_valid(ms, t), _xcorr_valid(ms, t * t)
    sum_st = _xcorr_valid(s, t)
    n_safe = np.maximum(np.rint(n), 1.0)
    cov = sum_st - sum_s * sum_t / n_safe
    var = (sum_ss - sum_s ** 2 / n_safe) * (sum_tt - sum_t ** 2 / n_safe)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = cov / np.sqrt(var)
    bad = (np.rint(n) < max(min_overlap * mt.sum(), min_samples)) | ~(var > 1e-9) | ~np.isfinite(score)
    score[bad] = -np.inf
    return score, n


class BallTracks:
    """Tactical and broadcast ball tracks plus smoothed speed curves, loaded once on first use."""

    def __init__(self, tactical_atd, broadcast_atd, pitch, fps):
        self.paths, self.pitch, self.fps, self._cache = {"tactical": tactical_atd, "broadcast": broadcast_atd}, pitch, fps, {}

    def get(self, source):
        """(x, y, smoothed speed) for 'tactical' or 'broadcast'."""
        if source not in self._cache:
            x, y = load_ball_track(self.paths[source])
            self._cache[source] = (x, y, nan_smooth(ball_speed_curve(x, y, self.pitch, self.fps), SMOOTH_FRAMES))
        return self._cache[source]


def xcorr_anchor(tracks, tag_start, lo, hi, label, fps, log=print, signature_s: float = 0.0):
    """Anchor from the tactical ball-speed signature (first ball stretch at or after tag_start)
    matched into the broadcast, with the signature start searched in broadcast frames lo..hi.
    signature_s > 0 extends the signature to that many seconds from the start of that stretch
    (gaps included; they are ignored by the masked correlation).
    Returns the anchor dict, or None (with a warning logged) when it cannot be found."""
    tx, _, t_speed = tracks.get("tactical")
    runs = ball_present_runs(tx, tag_start)
    min_len = int(round(SIGNATURE_MIN_S * fps))
    run = next((r for r in runs if r[1] - r[0] + 1 >= min_len), None)
    if run is None:
        log("warning: no ball-present stretch of {:.0f} s after frame {}".format(SIGNATURE_MIN_S, tag_start))
        return None
    skipped = sum(1 for r in runs if r[0] < run[0])
    sig_start = run[0] + 1                             # speed needs the previous frame, so skip the first
    sig_end = max(run[1], run[0] + int(round(signature_s * fps))) if signature_s > 0 else run[1]
    template = t_speed[sig_start:sig_end + 1]
    log("")
    log("tactical signature: frames {}..{} ({:.1f} s{}{})".format(
        run[0], sig_end, (sig_end - run[0] + 1) / fps,
        " of ball" if sig_end == run[1] else ", {:.0f}% with ball".format(100 * np.isfinite(template).mean()),
        ", skipped {} shorter stretch(es) before it".format(skipped) if skipped else ""))

    _, _, b_speed = tracks.get("broadcast")
    lo = max(0, lo)
    if hi < lo:
        log("warning: broadcast search window is empty ({} .. {})".format(lo, hi))
        return None
    stop = min(len(b_speed), hi + len(template))
    window = b_speed[lo:stop]
    if len(window) < len(template):
        log("warning: broadcast ball data ends before the search window")
        return None
    log("broadcast search window: signature start in {} .. {}".format(
        frame_timestamp(lo, fps), frame_timestamp(min(hi, lo + len(window) - len(template)), fps)))

    ncc, overlap = masked_ncc(window, template, min_samples=int(round(MIN_OVERLAP_S * fps)))
    if not np.isfinite(ncc).any():
        log("warning: no lag in the broadcast window has enough overlapping ball data")
        return None
    # A few seconds of overlap correlate highly by chance, so lags are ranked by the Fisher z
    # significance atanh(r) * sqrt(n - 3), which weighs the correlation against the overlap size.
    n = np.maximum(np.rint(overlap), 0)
    score = np.where(np.isfinite(ncc), np.arctanh(np.clip(ncc, -0.999, 0.999)) * np.sqrt(np.maximum(n - 3, 0)), -np.inf)
    k = int(np.argmax(score))
    best = float(score[k])
    excl = int(round(PEAK_EXCLUSION_S * fps))
    rest = score.copy()
    rest[max(0, k - excl):k + excl + 1] = -np.inf
    runner = float(rest.max()) if np.isfinite(rest).any() else float("nan")
    b_sig = lo + k
    offset = b_sig - sig_start
    log("cross-correlation: best at broadcast {}: correlation {:.3f} over {:.0f}% of the signature "
        "(score {:.3f}), runner-up score {:.3f}".format(
            frame_timestamp(b_sig, fps), float(ncc[k]), 100 * n[k] / np.isfinite(template).sum(), best, runner))
    if float(ncc[k]) < 0.4 or (math.isfinite(runner) and runner > 0.8 * best):
        log("warning: weak or ambiguous match; check it in the dual viewer (anchors ▾ → go)")
    return {"label": label, "tactical": run[0], "broadcast": run[0] + offset,
            "method": "ball-speed cross-correlation", "correlation": round(float(ncc[k]), 4), "score": round(best, 4),
            "runner_up": round(runner, 4) if math.isfinite(runner) else None}


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
                 log=print, signature_s: float = 0.0, first_half_xcorr: bool = False):
    """Kick-off anchors on both timelines -> the dict written to sync.json.
    first_half_xcorr: also align the first half by ball-speed cross-correlation (broadcast window:
    its first SEARCH_MARGIN_MIN minutes) and use that instead of the speed-threshold kick-off."""
    fps = assert_same_fps(tactical_video, broadcast_video)
    log("video fps: {:.6g} (tactical and broadcast metadata agree)".format(fps))

    kickoffs = load_kickoff_tags(smart_tagging, fps)
    if not kickoffs:
        raise RuntimeError("no kick-off Smart Tagging instances found in {}".format(smart_tagging))
    first_tag = kickoffs[0]
    log("first kick-off smart tag: Type={!r} Half={!r} frames {}..{} ({:.3f}s .. {:.3f}s)".format(
        first_tag["type"], first_tag["half"], first_tag["start_frame"], first_tag["end_frame"],
        first_tag["start_s"], first_tag["end_s"]))

    tracks = BallTracks(tactical_atd, broadcast_atd, pitch, fps)
    margin = int(round(SEARCH_MARGIN_MIN * 60 * fps))

    t1 = first_ball_appearance(tactical_atd, first_tag["start_frame"])
    b1 = first_ball_speed_above(broadcast_atd, BALL_SPEED_THRESHOLD_MS, fps, pitch)
    a1 = {"label": "1st-half kick-off", "tactical": t1, "broadcast": b1}
    log("")
    log("tactical kick-off:  {}".format(frame_timestamp(t1, fps)))
    log("broadcast kick-off: {}  (first ball speed > {:.0f} m/s)".format(frame_timestamp(b1, fps), BALL_SPEED_THRESHOLD_MS))
    log("offset (broadcast - tactical): {:+d} frames".format(b1 - t1))

    if first_half_xcorr:
        log("")
        log("first half by ball-speed cross-correlation (broadcast: first {} min):".format(SEARCH_MARGIN_MIN))
        x1 = xcorr_anchor(tracks, first_tag["start_frame"], 0, margin, "1st half (ball-speed xcorr)", fps,
                          log=log, signature_s=signature_s)
        if x1 is None:
            log("falling back to the speed-threshold kick-off for the first half")
        else:
            off_x = x1["broadcast"] - x1["tactical"]
            log("offset (broadcast - tactical): {:+d} frames ({:+d} vs speed threshold)".format(off_x, off_x - (b1 - t1)))
            a1 = x1
    anchors = [a1]
    # broadcast frame of the tactical kick-off; the earlier estimate, so a bad match can't empty the 2nd-half window
    b1_aligned = min(b1, a1["broadcast"] + (t1 - a1["tactical"]))

    tag2 = second_half_kickoff_tag(kickoffs, first_tag, fps)
    if tag2 is None:
        log("warning: no second-half kick-off tag found, only the first-half anchor is written")
    else:
        log("")
        log("second-half kick-off smart tag: Type={!r} Half={!r} frames {}..{}".format(
            tag2["type"], tag2["half"], tag2["start_frame"], tag2["end_frame"]))
        n_b = video_frame_count(broadcast_video)
        a2 = xcorr_anchor(tracks, tag2["start_frame"], b1_aligned + margin, n_b - 1 - margin,
                          "2nd half (ball-speed xcorr)", fps, log=log, signature_s=signature_s)
        if a2 is None:
            log("only the first-half anchor is written")
        else:
            anchors.append(a2)
            t2, b2 = a2["tactical"], a2["broadcast"]
            log("tactical 2nd-half anchor:  {}".format(frame_timestamp(t2, fps)))
            log("broadcast 2nd-half anchor: {}".format(frame_timestamp(b2, fps)))
            log("offset (broadcast - tactical): {:+d} frames ({:+d} vs 1st half)".format(
                b2 - t2, (b2 - t2) - (a1["broadcast"] - a1["tactical"])))
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
    ap.add_argument("--signature-s", type=float, default=0.0,
                    help="2nd half: use this many seconds of tactical ball speed from the first ball stretch "
                         "as the signature (gaps allowed) instead of just that stretch; try 60 if the match is ambiguous")
    ap.add_argument("--first-half-xcorr", action="store_true",
                    help="also align the first half by ball-speed cross-correlation against the first 45 min "
                         "of the broadcast (instead of the first ball speed > 10 m/s)")
    args = ap.parse_args()

    fps_game = GAMES[GAME]["fps"]
    pitch = GAMES[GAME]["pitch"]
    tactical_video, tactical_atd, smart_tagging = require(
        ["video", "atd_raw", "smart_tagging"], source="tactical")
    broadcast_video, broadcast_atd = require(
        ["video", "atd_raw"], source="broadcast")

    print("data dir: {}".format(DATA_DIR))
    print("game:     {}".format(GAME))

    sync = compute_sync(tactical_video, tactical_atd, smart_tagging, broadcast_video, broadcast_atd, pitch,
                        signature_s=args.signature_s, first_half_xcorr=args.first_half_xcorr)
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

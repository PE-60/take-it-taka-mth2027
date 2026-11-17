"""Kick-window ball re-detection: motion + appearance blobs inside a spatial ROI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ball_detection.appearance import (
    AppearanceBanks,
    ball_color_mask_bgr,
    bgr_to_hsv,
    bright_fraction,
    circular_mask,
    grass_mask_hsv,
    hist_intersection,
    hsv_histogram,
)
from ball_detection.geometry import project_points
from ball_detection.kicks import KickWindow
from ball_detection.results import BallDetection, KickBallTrack
from ball_detection.roi import PixelROI, RoiParams, search_roi

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class DetectParams:
    radius_px: float = 6.0
    min_radius_px: float = 2.5
    max_radius_px: float = 12.0
    motion_thresh: int = 28
    min_blob_area: int = 5
    max_blob_area: int = 220
    w_hist: float = 0.40
    w_motion: float = 0.20
    w_prior: float = 0.40
    min_score: float = 0.32
    # Hard continuity: reject any jump larger than this (no soft exceptions).
    max_link_dist_px: float = 28.0
    # Predicted-position gate (last + velocity); slightly looser than max_link.
    max_pred_dist_px: float = 36.0
    min_bright_frac: float = 0.30
    # Suppress blobs near projected player feet (shoes).
    foot_exclude_px: float = 14.0
    # Once locked onto a blob, search in a tighter box around the prediction.
    locked_roi_pad_px: int = 28
    roi: RoiParams = RoiParams()


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for Stage B detection")


def motion_mask(prev_bgr: np.ndarray | None, cur_bgr: np.ndarray, thresh: int) -> np.ndarray:
    """Boolean HxW motion residual inside the ROI crop."""
    _require_cv2()
    gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    if prev_bgr is None or prev_bgr.shape != cur_bgr.shape:
        return np.zeros(gray.shape, dtype=bool)
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev)
    return diff >= int(thresh)


def _bright_mask(bgr: np.ndarray) -> np.ndarray:
    """White / light-grey ball prior (stricter than older S<=80 / V>=160)."""
    return ball_color_mask_bgr(bgr)


def _candidate_from_components(mask: np.ndarray, min_area: int, max_area: int):
    """Yield (cy, cx, area) for connected components in a boolean mask."""
    _require_cv2()
    m = np.asarray(mask, dtype=np.uint8) * 255
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cx, cy = float(centroids[lab, 0]), float(centroids[lab, 1])
        yield cy, cx, area


def foot_exclusion_mask(
        height: int,
        width: int,
        feet_uv_roi: list[tuple[float, float]],
        radius_px: float,
) -> np.ndarray:
    """True where search is allowed (False near projected feet)."""
    ok = np.ones((height, width), dtype=bool)
    if radius_px <= 0 or not feet_uv_roi:
        return ok
    yy, xx = np.ogrid[:height, :width]
    r2 = float(radius_px) ** 2
    for u, v in feet_uv_roi:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ok &= ((xx - u) ** 2 + (yy - v) ** 2) > r2
    return ok


def score_candidate(
        roi_bgr: np.ndarray,
        cy: float,
        cx: float,
        radius_px: float,
        ball_hist: np.ndarray | None,
        motion: np.ndarray,
        prior_uv_roi: tuple[float, float] | None,
        params: DetectParams,
) -> tuple[float, float, float, float]:
    """Return (score, hist_score, motion_score, prior_score). 0 if too shoe-like."""
    h, w = roi_bgr.shape[:2]
    r = float(radius_px)
    if cx < r or cy < r or cx >= w - r or cy >= h - r:
        return 0.0, 0.0, 0.0, 0.0

    mask = circular_mask(h, w, cy, cx, r)
    bright_frac = bright_fraction(roi_bgr, mask=mask)
    if bright_frac < float(params.min_bright_frac):
        return 0.0, 0.0, 0.0, 0.0

    hsv = bgr_to_hsv(roi_bgr)
    usable = mask & ~grass_mask_hsv(hsv) & ball_color_mask_bgr(roi_bgr)
    if int(usable.sum()) < 4:
        usable = mask & ball_color_mask_bgr(roi_bgr)
    if int(usable.sum()) < 4:
        return 0.0, 0.0, 0.0, 0.0

    hist = hsv_histogram(roi_bgr, mask=usable.astype(np.uint8) * 255)
    if ball_hist is not None and float(hist.sum()) > 0:
        hist_score = hist_intersection(hist, ball_hist)
    else:
        hist_score = bright_frac

    mot = motion[mask]
    motion_score = float(mot.mean()) if mot.size else 0.0

    if prior_uv_roi is None:
        prior_score = 0.35
    else:
        pu, pv = prior_uv_roi
        dist = float(np.hypot(cx - pu, cy - pv))
        scale = max(params.max_link_dist_px, 1.0)
        prior_score = float(np.exp(-dist / scale))

    score = (params.w_hist * hist_score
             + params.w_motion * motion_score
             + params.w_prior * prior_score)
    # Mild bonus for looking white.
    score = float(score) * (0.85 + 0.15 * bright_frac)
    return score, hist_score, motion_score, prior_score


def detect_in_roi(
        frame_bgr: np.ndarray,
        prev_bgr: np.ndarray | None,
        roi: PixelROI,
        ball_hist: np.ndarray | None,
        prior_uv: tuple[float, float] | None,
        frame_index: int,
        params: DetectParams | None = None,
        *,
        feet_uv: list[tuple[float, float]] | None = None,
) -> BallDetection | None:
    """Best ball candidate inside ``roi`` for one frame, or None if below threshold."""
    params = params or DetectParams()
    _require_cv2()
    if roi.width < 4 or roi.height < 4:
        return None
    ys, xs = roi.as_slice()
    crop = frame_bgr[ys, xs]
    prev_crop = None if prev_bgr is None else prev_bgr[ys, xs]
    motion = motion_mask(prev_crop, crop, params.motion_thresh)
    bright = _bright_mask(crop)

    feet_roi: list[tuple[float, float]] = []
    if feet_uv:
        for u, v in feet_uv:
            feet_roi.append((u - roi.x0, v - roi.y0))
    allowed = foot_exclusion_mask(crop.shape[0], crop.shape[1], feet_roi, params.foot_exclude_px)

    # Prefer white pixels; allow motion only near the prior (kicks / blur).
    search = bright & allowed
    if prior_uv is not None:
        pu, pv = prior_uv[0] - roi.x0, prior_uv[1] - roi.y0
        near = circular_mask(crop.shape[0], crop.shape[1], pv, pu,
                             radius=max(params.max_link_dist_px, params.radius_px * 3))
        search = search | (motion & near & allowed & bright)
    else:
        search = search & (motion | bright)

    prior_roi = None
    if prior_uv is not None:
        prior_roi = (prior_uv[0] - roi.x0, prior_uv[1] - roi.y0)

    radius = float(params.radius_px)
    best: BallDetection | None = None
    for cy, cx, area in _candidate_from_components(
            search, params.min_blob_area, params.max_blob_area):
        r = float(np.clip(np.sqrt(area / np.pi), params.min_radius_px, params.max_radius_px))
        for trial_r in sorted({r, radius}):
            score, hs, ms, ps = score_candidate(
                crop, cy, cx, trial_r, ball_hist, motion, prior_roi, params)
            if score < params.min_score:
                continue
            det = BallDetection(
                frame=int(frame_index),
                u=float(cx + roi.x0),
                v=float(cy + roi.y0),
                score=score,
                radius_px=trial_r,
                hist_score=hs,
                motion_score=ms,
                prior_score=ps,
                source="blob",
            )
            if best is None or det.score > best.score:
                best = det
    return best


def _prior_uv(H: np.ndarray, anchor_pitch: np.ndarray | None) -> tuple[float, float] | None:
    if anchor_pitch is None:
        return None
    pix = project_points(H, np.asarray(anchor_pitch, dtype=np.float64).reshape(1, 2))[0]
    if not np.isfinite(pix).all():
        return None
    return float(pix[0]), float(pix[1])


def _player_feet_uv(H: np.ndarray, players_xy: np.ndarray) -> list[tuple[float, float]]:
    players_xy = np.asarray(players_xy, dtype=np.float64).reshape(-1, 2)
    out: list[tuple[float, float]] = []
    finite = np.isfinite(players_xy).all(axis=1)
    if not finite.any():
        return out
    pix = project_points(H, players_xy[finite])
    for p in pix:
        if np.isfinite(p).all():
            out.append((float(p[0]), float(p[1])))
    return out


def choose_contact(track: KickBallTrack) -> BallDetection | None:
    """Prefer detection on the kick frame; else highest score in the window."""
    if not track.detections:
        return None
    on_kick = [d for d in track.detections if d.frame == track.kick_frame]
    if on_kick:
        return max(on_kick, key=lambda d: d.score)
    return max(track.detections, key=lambda d: d.score)


def _locked_roi(center_uv: tuple[float, float], pad: int, frame_wh: tuple[int, int]) -> PixelROI:
    u, v = center_uv
    p = max(8, int(pad))
    return PixelROI(int(u) - p, int(v) - p, int(u) + p + 1, int(v) + p + 1).clip(*frame_wh)


def track_kick_window(
        frames_bgr: list[np.ndarray],
        window: KickWindow,
        homographies: np.ndarray,
        ball: np.ndarray,
        players_xy: np.ndarray,
        banks: AppearanceBanks | None,
        pitch_m: tuple[float, float],
        fps: float = 25.0,
        *,
        frame_offset: int = 0,
        slot_ids=None,
        gk_ids=None,
        params: DetectParams | None = None,
) -> KickBallTrack:
    """Run detection over ``frames_bgr`` aligned to ``window.start .. window.end-1``.

    Continuity: once a detection exists, the next frame must lie within ``max_link_dist_px``
    of the velocity-predicted position — teleports are dropped (miss that frame).
    """
    params = params or DetectParams()
    ball_hist = None
    if banks is not None and banks.ball.n_samples > 0:
        ball_hist = banks.ball.mean_hist()
        mean_r = banks.ball.mean_radius_px()
        if mean_r is not None:
            params = DetectParams(
                **{**params.__dict__, "radius_px": float(np.clip(mean_r, params.min_radius_px, params.max_radius_px)),
                   "roi": params.roi},
            )

    h0, w0 = frames_bgr[0].shape[:2]
    track = KickBallTrack(
        kick_frame=frame_offset + window.kick_index,
        window_start=frame_offset + window.start,
        window_end=frame_offset + window.end,
    )
    prev = None
    last_uv: tuple[float, float] | None = None
    last_vel: tuple[float, float] | None = None

    for i, frame in enumerate(frames_bgr):
        idx = window.start + i
        video_frame = frame_offset + idx
        H = homographies[idx]
        if not np.isfinite(H).all():
            prev = frame
            continue
        ply = players_xy[idx] if players_xy.ndim == 3 else players_xy
        feet = _player_feet_uv(H, ply)

        predicted = None
        if last_uv is not None:
            if last_vel is not None:
                predicted = (last_uv[0] + last_vel[0], last_uv[1] + last_vel[1])
            else:
                predicted = last_uv

        if predicted is not None:
            roi = _locked_roi(predicted, params.locked_roi_pad_px, (w0, h0))
            prior_uv = predicted
        else:
            roi, anchor = search_roi(
                H, ball, ply, idx, (w0, h0), pitch_m, fps,
                slot_ids=slot_ids, gk_ids=gk_ids, params=params.roi,
            )
            if roi is None:
                prev = frame
                continue
            prior_uv = _prior_uv(H, anchor)

        det = detect_in_roi(
            frame, prev, roi, ball_hist, prior_uv, video_frame, params, feet_uv=feet)

        if det is not None and last_uv is not None:
            # Hard continuity vs last detection.
            dist_last = float(np.hypot(det.u - last_uv[0], det.v - last_uv[1]))
            if dist_last > params.max_link_dist_px:
                det = None
            elif predicted is not None:
                dist_pred = float(np.hypot(det.u - predicted[0], det.v - predicted[1]))
                if dist_pred > params.max_pred_dist_px:
                    det = None

        if det is not None:
            if last_uv is not None:
                last_vel = (det.u - last_uv[0], det.v - last_uv[1])
            last_uv = (det.u, det.v)
            track.detections.append(det)
        # Missed frame: keep last_uv/vel so we do not re-seed on a shoe across the pitch.
        prev = frame

    track.contact = choose_contact(track)
    return track


def detect_kicks_in_tracking(
        frames_by_index: dict[int, np.ndarray],
        kick_mask: np.ndarray,
        homographies: np.ndarray,
        ball: np.ndarray,
        players_xy: np.ndarray,
        banks: AppearanceBanks | None,
        pitch_m: tuple[float, float],
        fps: float = 25.0,
        *,
        frame_offset: int = 0,
        pre: int = 8,
        post: int = 12,
        slot_ids=None,
        gk_ids=None,
        params: DetectParams | None = None,
        max_kicks: int | None = None,
) -> list[KickBallTrack]:
    """Detect using a preloaded frame dict (tests / small ranges). Prefer streaming in CLI."""
    from ball_detection.kicks import kick_windows

    windows = kick_windows(kick_mask, pre=pre, post=post, n_frames=len(ball))
    if max_kicks is not None:
        windows = windows[: int(max_kicks)]
    out: list[KickBallTrack] = []
    for win in windows:
        frames = []
        missing = False
        for idx in range(win.start, win.end):
            fr = frames_by_index.get(idx)
            if fr is None:
                missing = True
                break
            frames.append(fr)
        if missing or not frames:
            out.append(KickBallTrack(
                kick_frame=frame_offset + win.kick_index,
                window_start=frame_offset + win.start,
                window_end=frame_offset + win.end,
            ))
            continue
        out.append(track_kick_window(
            frames, win, homographies, ball, players_xy, banks, pitch_m, fps,
            frame_offset=frame_offset, slot_ids=slot_ids, gk_ids=gk_ids, params=params,
        ))
    return out


def iter_kick_windows(kick_mask: np.ndarray, pre: int, post: int, max_kicks: int | None = None):
    from ball_detection.kicks import kick_windows
    windows = kick_windows(kick_mask, pre=pre, post=post, n_frames=len(kick_mask))
    if max_kicks is not None:
        windows = windows[: int(max_kicks)]
    return windows

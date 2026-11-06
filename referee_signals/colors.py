"""HSV masking helpers (Stein-style color segmentation without CUDA)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def hsv_ranges_to_mask(hsv: np.ndarray, ranges: Iterable[Dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges or []:
        lo = np.array(r["lower"], dtype=np.uint8)
        hi = np.array(r["upper"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def exclude_mask(src: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(src, cv2.bitwise_not(exclude))


def morph_clean(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def band_mask(h: int, w: int, y0_frac: float, y1_frac: float) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    y0 = int(max(0, min(h, h * y0_frac)))
    y1 = int(max(0, min(h, h * y1_frac)))
    m[y0:y1, :] = 255
    return m


def contour_stats(contour) -> Dict[str, float]:
    x, y, bw, bh = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    m = cv2.moments(contour)
    if m["m00"] > 1e-6:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    else:
        cx, cy = x + bw / 2.0, y + bh / 2.0
    # Orientation from horizontal: 0 = flat, 90 = vertical (raised stick).
    # Prefer axis-aligned bbox aspect when the blob is clearly tall/wide; else PCA.
    if bh >= 1.35 * bw:
        angle = 90.0
    elif bw >= 1.35 * bh:
        angle = 0.0
    else:
        pts = contour.reshape(-1, 2).astype(np.float64)
        angle = 45.0
        if len(pts) >= 2:
            pts = pts - pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts, full_matrices=False)
            vx, vy = vt[0]
            angle = abs(float(np.degrees(np.arctan2(vy, vx))))
            if angle > 90:
                angle = 180.0 - angle
    return {
        "x": float(x), "y": float(y), "w": float(bw), "h": float(bh),
        "area": area, "cx": float(cx), "cy": float(cy), "angle": float(angle),
        "aspect": float(bh / max(1.0, bw)),
    }


def expand_bbox(
    x: float, y: float, w: float, h: float,
    head: float, feet: float, side: float,
    frame_w: int, frame_h: int,
) -> Tuple[int, int, int, int]:
    """Stein-style body expansion from a jersey blob (sh/sf/sa style factors)."""
    cx = x + w / 2.0
    top = y - head * h
    bot = y + h + feet * h
    left = cx - w / 2.0 - side * w
    right = cx + w / 2.0 + side * w
    x0 = int(max(0, left))
    y0 = int(max(0, top))
    x1 = int(min(frame_w - 1, right))
    y1 = int(min(frame_h - 1, bot))
    return x0, y0, x1, y1


def pad_bbox(bbox: Sequence[float], pad_frac: float, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    px, py = pad_frac * bw, pad_frac * bh
    return (
        int(max(0, x0 - px)),
        int(max(0, y0 - py)),
        int(min(frame_w - 1, x1 + px)),
        int(min(frame_h - 1, y1 + py)),
    )


def hsv_stats_from_roi(bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) HSV bounds from an ROI with percentile clipping."""
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return np.array([0, 0, 0]), np.array([180, 255, 255])
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    lo = np.percentile(pixels, 5, axis=0)
    hi = np.percentile(pixels, 95, axis=0)
    # Widen slightly
    lo = np.clip(lo - [3, 15, 15], [0, 0, 0], [180, 255, 255])
    hi = np.clip(hi + [3, 15, 15], [0, 0, 0], [180, 255, 255])
    return lo.astype(int), hi.astype(int)

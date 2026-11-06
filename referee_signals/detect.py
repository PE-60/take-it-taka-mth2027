"""Stein-style official + flag detection on a single BGR frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .colors import (
    band_mask,
    contour_stats,
    exclude_mask,
    expand_bbox,
    hsv_ranges_to_mask,
    morph_clean,
    pad_bbox,
)
from .geometry import attach_geometry
from .classify import classify_geometry


@dataclass
class DetectorConfig:
    seeds: Dict[str, Any]
    prefer_touchline_bands: bool = True

    @property
    def official_cfg(self) -> Dict[str, Any]:
        return self.seeds.get("official") or {}

    @property
    def flag_cfg(self) -> Dict[str, Any]:
        return self.seeds.get("flag") or {}

    @property
    def classifier_cfg(self) -> Dict[str, Any]:
        return self.seeds.get("classifier") or {}


def _find_official_blobs(
    kit_mask: np.ndarray,
    band: np.ndarray,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    masked = cv2.bitwise_and(kit_mask, band)
    masked = morph_clean(masked, open_k=3, close_k=7)
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = float(cfg.get("min_area", 80))
    max_a = float(cfg.get("max_area", 25000))
    min_asp = float(cfg.get("min_aspect", 0.25))
    max_asp = float(cfg.get("max_aspect", 2.8))
    out = []
    for c in contours:
        st = contour_stats(c)
        if st["area"] < min_a or st["area"] > max_a:
            continue
        if not (min_asp <= st["aspect"] <= max_asp):
            continue
        out.append(st)
    out.sort(key=lambda s: -s["area"])
    return out


def _pick_flag_in_roi(
    flag_mask: np.ndarray,
    roi: Tuple[int, int, int, int],
    body_bbox: Tuple[int, int, int, int],
    body_area: float,
    shirt_mask: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Pick flag contour in ROI, preferring blobs outside the jersey core.

    Neon-yellow kits overlap flag yellow; subtract shirt pixels and score by
    distance from torso so the held flag wins over the shirt itself.
    """
    x0, y0, x1, y1 = roi
    sub = flag_mask[y0:y1, x0:x1].copy()
    if sub.size == 0:
        return None
    if shirt_mask is not None:
        # Only suppress kit yellow *inside the body box* — never the raised/side flag,
        # which shares neon yellow with the shirt but sits outside the jersey bbox.
        bx0, by0, bx1, by1 = body_bbox
        body_only = np.zeros_like(shirt_mask)
        body_only[by0:by1, bx0:bx1] = shirt_mask[by0:by1, bx0:bx1]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shirt_core = cv2.erode(body_only[y0:y1, x0:x1], k, iterations=1)
        sub = cv2.bitwise_and(sub, cv2.bitwise_not(shirt_core))
    sub = morph_clean(sub, open_k=2, close_k=3)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = float(cfg.get("min_area", 12))
    min_frac = float(cfg.get("min_area_frac_of_body", 0.01))
    max_frac = float(cfg.get("max_area_frac_of_body", 0.85))
    bx0, by0, bx1, by1 = body_bbox
    torso_cx = (bx0 + bx1) / 2.0
    torso_cy = by0 + 0.40 * max(1.0, by1 - by0)
    bh = max(1.0, by1 - by0)
    best = None
    best_score = -1.0
    for c in contours:
        st = contour_stats(c)
        st["cx"] += x0
        st["cy"] += y0
        st["x"] += x0
        st["y"] += y0
        if st["area"] < min_a:
            continue
        frac = st["area"] / max(1.0, body_area)
        if frac < min_frac or frac > max_frac:
            continue
        # Inside jersey rectangle → heavy penalty (likely shirt, not flag)
        inside = bx0 <= st["cx"] <= bx1 and by0 <= st["cy"] <= by1
        dist = ((st["cx"] - torso_cx) ** 2 + (st["cy"] - torso_cy) ** 2) ** 0.5 / bh
        score = st["area"] * (0.25 if inside else 1.0) * (1.0 + dist)
        if score > best_score:
            best_score = score
            best = st
    return best


def detect_frame(
    bgr: np.ndarray,
    config: DetectorConfig,
    frame_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Detect AR candidates + flags; return list of geometry+signal dicts (unsmoothed)."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    seeds = config.seeds

    shirt = hsv_ranges_to_mask(hsv, seeds.get("kit_shirt_hsv"))
    # shorts help body extent but are optional; OR with shirt for blob finding
    shorts = hsv_ranges_to_mask(hsv, seeds.get("kit_shorts_hsv"))
    kit = cv2.bitwise_or(shirt, shorts)
    grass = hsv_ranges_to_mask(hsv, seeds.get("grass_hsv"))
    kit = exclude_mask(kit, grass)
    flag_m = hsv_ranges_to_mask(hsv, seeds.get("flag_hsv"))
    # Keep flag colors even on grass; do not exclude grass from flag mask

    bands_cfg = seeds.get("touchline_bands") or {
        "top": {"y0_frac": 0.0, "y1_frac": 0.28},
        "bottom": {"y0_frac": 0.72, "y1_frac": 1.0},
    }
    ocfg = config.official_cfg
    fcfg = config.flag_cfg
    results: List[Dict[str, Any]] = []

    for band_name, official_id in (("top", "ar_top"), ("bottom", "ar_bottom")):
        bc = bands_cfg.get(band_name) or {"y0_frac": 0.0, "y1_frac": 1.0}
        band = band_mask(h, w, bc["y0_frac"], bc["y1_frac"])
        blobs = _find_official_blobs(kit, band, ocfg)
        if not blobs:
            continue
        # Prefer largest blob in this band (one AR per touchline when visible)
        st = blobs[0]
        x0, y0, x1, y1 = expand_bbox(
            st["x"], st["y"], st["w"], st["h"],
            float(ocfg.get("expand_head", 0.55)),
            float(ocfg.get("expand_feet", 1.35)),
            float(ocfg.get("expand_side", 0.45)),
            w, h,
        )
        body_area = max(1.0, (x1 - x0) * (y1 - y0))
        roi = pad_bbox((x0, y0, x1, y1), float(ocfg.get("flag_roi_pad", 0.55)), w, h)
        flag = _pick_flag_in_roi(flag_m, roi, (x0, y0, x1, y1), body_area, shirt, fcfg)
        official = {
            "id": official_id,
            "band": band_name,
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "jersey": st,
        }
        packed = attach_geometry(official, flag)
        sig, conf = classify_geometry(packed.get("rel"), config.classifier_cfg)
        packed["signal"] = sig
        packed["confidence"] = conf
        if frame_idx is not None:
            packed["f"] = int(frame_idx)
        results.append(packed)
    return results


def detections_to_schema(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .schema import detection_dict

    out = []
    for r in rows:
        flag = r.get("flag")
        out.append(detection_dict(
            f=int(r.get("f", 0)),
            official_id=r["id"],
            bbox_xyxy=r["bbox"],
            signal=r.get("signal", "none"),
            confidence=float(r.get("confidence") or 0),
            flag_centroid=[flag["cx"], flag["cy"]] if flag else None,
            flag_angle_deg=float(flag["angle"]) if flag else None,
            rel=r.get("rel"),
            role="assistant",
        ))
    return out

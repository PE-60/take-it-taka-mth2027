"""Flag-vs-body relative geometry features."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def relative_features(
    body_bbox: Tuple[float, float, float, float],
    flag_cx: float,
    flag_cy: float,
    flag_angle_deg: float,
    flag_area: float,
) -> Dict[str, float]:
    x0, y0, x1, y1 = body_bbox
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    torso_cx = (x0 + x1) / 2.0
    torso_cy = y0 + 0.40 * bh  # approximate chest / upper torso
    dx = (flag_cx - torso_cx) / bh
    dy = (flag_cy - torso_cy) / bh
    length_ratio = (flag_area ** 0.5) / bh
    raised = 1.0 if dy <= -0.28 else 0.0
    side = 1.0 if dx >= 0 else -1.0  # +1 right of torso in image coords
    return {
        "dx": float(dx),
        "dy": float(dy),
        "angle": float(flag_angle_deg),
        "side": float(side),
        "raised": float(raised),
        "length_ratio": float(length_ratio),
        "body_h": float(bh),
        "body_w": float(bw),
    }


def attach_geometry(official: Dict[str, Any], flag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(official)
    if not flag:
        out["flag"] = None
        out["rel"] = None
        return out
    rel = relative_features(
        (official["bbox"][0], official["bbox"][1], official["bbox"][2], official["bbox"][3]),
        flag["cx"], flag["cy"], flag["angle"], flag["area"],
    )
    out["flag"] = flag
    out["rel"] = rel
    return out

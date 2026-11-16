"""JSON schema helpers for referee_signals.json."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SIGNAL_LABELS = (
    "none",
    "unknown",
    "attention_raised",
    "throw_in_left",
    "throw_in_right",
    "corner",
    "goal_kick",
    "substitution",
)

# Viewer / timeline glyph + BGR color (OpenCV) per signal
SIGNAL_STYLE = {
    "none": {"glyph": "cross", "color": [80, 80, 80], "hex": "#505050"},
    "unknown": {"glyph": "cross", "color": [120, 120, 120], "hex": "#787878"},
    "attention_raised": {"glyph": "tri_up", "color": [0, 220, 255], "hex": "#ffdc00"},
    "throw_in_left": {"glyph": "diamond", "color": [255, 160, 0], "hex": "#00a0ff"},
    "throw_in_right": {"glyph": "diamond", "color": [0, 160, 255], "hex": "#ffa000"},
    "corner": {"glyph": "circle", "color": [80, 80, 255], "hex": "#ff5050"},
    "goal_kick": {"glyph": "square", "color": [80, 200, 80], "hex": "#50c850"},
    "substitution": {"glyph": "star", "color": [220, 100, 220], "hex": "#dc64dc"},
}


def empty_document(
    source: str = "tactical",
    fps: float = 25.0,
    width: int = 1920,
    height: int = 1080,
    seeds_path: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "fps": fps,
        "width": width,
        "height": height,
        "seeds": seeds_path,
        "officials": [
            {"id": "ar_top", "role": "assistant", "band": "top"},
            {"id": "ar_bottom", "role": "assistant", "band": "bottom"},
        ],
        "detections": [],
        "phase2_pose": False,
        "phase3_broadcast": False,
    }


def detection_dict(
    f: int,
    official_id: str,
    bbox_xyxy: List[float],
    signal: str = "none",
    confidence: float = 0.0,
    flag_centroid: Optional[List[float]] = None,
    flag_angle_deg: Optional[float] = None,
    rel: Optional[Dict[str, float]] = None,
    role: str = "assistant",
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "f": int(f),
        "official_id": official_id,
        "official_role": role,
        "bbox_xyxy": [float(x) for x in bbox_xyxy],
        "signal": signal if signal in SIGNAL_LABELS else "unknown",
        "confidence": float(confidence),
    }
    if flag_centroid is not None:
        d["flag_centroid"] = [float(flag_centroid[0]), float(flag_centroid[1])]
    if flag_angle_deg is not None:
        d["flag_angle_deg"] = float(flag_angle_deg)
    if rel is not None:
        d["rel"] = {k: float(v) for k, v in rel.items()}
    if extras:
        d.update(extras)
    return d

"""Phase 2 stub: center-referee hand / arm pose signals (Law 5).

Not used in phase 1. When MediaPipe (or similar) is available, implement
`estimate_pose_in_bbox` and map arm angles to Law 5 labels:

  - advantage          both arms forward
  - direct_free_kick   one arm pointing direction
  - indirect_free_kick one arm raised vertically
  - penalty            arm pointing to penalty spot
  - yellow_card / red_card  card-colored blob near raised hand
  - var_tv             both hands outline a rectangle
  - countdown_5        raised hand counting

Install (optional, not required for phase 1):
  pip install mediapipe
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

LAW5_LABELS = (
    "advantage",
    "direct_free_kick",
    "indirect_free_kick",
    "penalty",
    "yellow_card",
    "red_card",
    "var_tv",
    "countdown_5",
    "none",
    "unknown",
)


class PoseBackendUnavailable(RuntimeError):
    pass


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


def estimate_pose_in_bbox(
    bgr,
    bbox_xyxy: Tuple[float, float, float, float],
) -> Optional[Dict[str, Any]]:
    """Return pose landmarks in full-frame coords, or None if backend missing / no person.

    Phase 2 implementation hook — raises PoseBackendUnavailable if mediapipe is not installed.
    """
    if not mediapipe_available():
        raise PoseBackendUnavailable(
            "mediapipe is not installed; phase 2 pose estimation is unavailable. "
            "Phase 1 flag detection does not need it."
        )
    # Placeholder so the import path is exercised once mediapipe exists.
    # Real landmark extraction belongs here.
    import mediapipe as mp  # noqa: F401
    return {
        "backend": "mediapipe",
        "bbox": list(bbox_xyxy),
        "landmarks": [],
        "note": "stub — wire PoseLandmarker here",
    }


def classify_law5_pose(pose: Optional[Dict[str, Any]]) -> Tuple[str, float]:
    """Map pose features to a Law 5 label. Stub always returns unknown until phase 2 lands."""
    if not pose or not pose.get("landmarks"):
        return "none", 0.0
    return "unknown", 0.1


def detect_center_ref_signals(bgr, official_bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Phase 2 entry: for each center-ref bbox, estimate pose and classify."""
    out = []
    for o in official_bboxes:
        bbox = o.get("bbox_xyxy") or o.get("bbox")
        if not bbox:
            continue
        try:
            pose = estimate_pose_in_bbox(bgr, tuple(bbox))
        except PoseBackendUnavailable:
            return []
        sig, conf = classify_law5_pose(pose)
        out.append({
            "official_id": o.get("official_id", "center"),
            "official_role": "center",
            "bbox_xyxy": list(bbox),
            "signal": sig,
            "confidence": conf,
            "pose": pose,
        })
    return out

"""Kick-local ball detection and kit-aware touch attribution (experimental).

Stage A: learn ball + outfield kit appearance from ATD + homography.
Stage B: re-detect the ball in short windows around kick peaks.
Stage C: attribute the contact to an outfield team via distance + kit match.
"""

from ball_detection.appearance import AppearanceBanks, load_banks, save_banks
from ball_detection.gk import default_goalkeeper_slot_ids, resolve_goalkeeper_slot_ids
from ball_detection.results import (
    AttributionExport,
    DetectExport,
    load_attributions,
    load_detections,
    save_attributions,
    save_detections,
)

__all__ = [
    "AppearanceBanks",
    "AttributionExport",
    "DetectExport",
    "default_goalkeeper_slot_ids",
    "load_attributions",
    "load_banks",
    "load_detections",
    "resolve_goalkeeper_slot_ids",
    "save_attributions",
    "save_banks",
    "save_detections",
]

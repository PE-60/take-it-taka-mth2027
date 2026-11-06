"""Assistant-referee flag detection (Stein-style color/contour) and Law 6 signal heuristics.

Phase 1: kit + flag HSV masks, relative geometry, rule classifier.
Phase 2: optional center-ref hand pose stub (see pose.py).
Phase 3: broadcast adapter hooks (see broadcast.py) — unused until tactical QA fails.
"""

from .schema import SIGNAL_LABELS, empty_document, detection_dict
from .detect import detect_frame, DetectorConfig
from .classify import classify_geometry, TemporalSmoother
from .io import load_signals, save_signals, index_by_frame

__all__ = [
    "SIGNAL_LABELS",
    "empty_document",
    "detection_dict",
    "detect_frame",
    "DetectorConfig",
    "classify_geometry",
    "TemporalSmoother",
    "load_signals",
    "save_signals",
    "index_by_frame",
]

__version__ = "0.1.0"

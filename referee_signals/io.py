"""Load / save referee_signals.json and build frame indexes for the viewer."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def load_signals(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def save_signals(doc: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)


def index_by_frame(doc: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Map absolute video frame -> list of detections on that frame."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for d in doc.get("detections") or []:
        f = int(d["f"])
        out.setdefault(f, []).append(d)
    return out


def detections_in_range(doc: Dict[str, Any], start: int, end: int) -> List[Dict[str, Any]]:
    return [d for d in (doc.get("detections") or []) if start <= int(d["f"]) <= end]


def signal_events_for_meta(doc: Dict[str, Any], start: int, end: int) -> List[Dict[str, Any]]:
    """Compact per-detection records for meta.json (export-relative frames)."""
    from .schema import SIGNAL_STYLE

    events = []
    for d in detections_in_range(doc, start, end):
        sig = d.get("signal") or "none"
        if sig in ("none",):
            continue
        style = SIGNAL_STYLE.get(sig, SIGNAL_STYLE["unknown"])
        events.append({
            "f": int(d["f"]) - start,  # export-relative
            "abs_f": int(d["f"]),
            "official_id": d.get("official_id"),
            "signal": sig,
            "confidence": float(d.get("confidence") or 0),
            "bbox_xyxy": d.get("bbox_xyxy"),
            "flag_centroid": d.get("flag_centroid"),
            "flag_angle_deg": d.get("flag_angle_deg"),
            "rel": d.get("rel"),
            "glyph": style["glyph"],
            "color": style["color"],
            "hex": style["hex"],
        })
    return events


def find_signals_path(here: str, explicit: Optional[str] = None, source: str = "tactical") -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = [
        os.path.join(here, "export", f"referee_signals_{source}.json"),
        os.path.join(here, f"referee_signals_{source}.json"),
        os.path.join(here, "referee_signals.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

"""Phase 3 — optional broadcast-video adapter (hooks only).

Use only if tactical AR flag geometry is not adequate. Broadcast has cuts/replays
and is **not** synchronised to the tactical / submission timeline.

Hooks:
  - load_broadcast_seeds(): separate HSV seeds (lighting / kit may differ)
  - run_on_broadcast(video, seeds, out): same detector, source='broadcast'
  - align_to_tactical(...): intentionally unimplemented — requires a manual or
    feature-based cross-timeline sync you do not have yet

Do not merge broadcast detections into tactical meta.json by default.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def load_broadcast_seeds(path: Optional[str] = None) -> Dict[str, Any]:
    from .seeds import load_seeds, default_seeds_path

    if path and os.path.isfile(path):
        return load_seeds(path)
    # Fall back to tactical defaults with source retagged — calibrate separately.
    seeds = load_seeds(default_seeds_path())
    seeds = dict(seeds)
    seeds["source"] = "broadcast"
    seeds["notes"] = (
        (seeds.get("notes") or "")
        + " | BROADCAST: re-calibrate kit/flag HSV; timeline is independent of tactical."
    )
    return seeds


def output_path_for_broadcast(here: str) -> str:
    return os.path.join(here, "export", "referee_signals_broadcast.json")


def align_to_tactical(
    broadcast_doc: Dict[str, Any],
    tactical_doc: Dict[str, Any],
    offset_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Map broadcast detections onto tactical frames.

    Not implemented: broadcast and tactical clocks diverge (cuts, replays).
    Pass an explicit constant offset only for rough experiments.
    """
    if offset_frames is None:
        raise NotImplementedError(
            "Cross-timeline sync is phase 3. Provide offset_frames for a naive shift, "
            "or keep broadcast detections on their own timeline."
        )
    out = dict(tactical_doc)
    dets = []
    for d in broadcast_doc.get("detections") or []:
        nd = dict(d)
        nd["f"] = int(d["f"]) + int(offset_frames)
        nd["from_broadcast"] = True
        dets.append(nd)
    out["detections"] = list(out.get("detections") or []) + dets
    out["phase3_broadcast"] = True
    return out


def should_use_broadcast(tactical_qa: Dict[str, Any]) -> bool:
    """Heuristic gate from a manual QA summary dict.

    Expected keys (optional): flag_recall, mean_ar_height_px, usable_fraction
    """
    recall = float(tactical_qa.get("flag_recall") or 1.0)
    height = float(tactical_qa.get("mean_ar_height_px") or 40)
    usable = float(tactical_qa.get("usable_fraction") or 1.0)
    return recall < 0.4 or height < 18 or usable < 0.3

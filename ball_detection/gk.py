"""Goalkeeper ATD slot ids to exclude from kit appearance learning.

Keepers wear different kits that can look similar across teams (or to the ball/lines).
Source of truth: data_correction.goalkeeper_trackers.DEFAULT_TRACKERS ("24", "46").
"""

from __future__ import annotations

import csv
import os
from typing import Iterable

import numpy as np

# Fallback if the goalkeeper_trackers package cannot be imported.
_FALLBACK_TRACKERS = ("24", "46")


def default_goalkeeper_slot_ids() -> frozenset[str]:
    """Raw ATD tracker slot ids that are the two keeper slots for this match."""
    try:
        from data_correction.goalkeeper_trackers import DEFAULT_TRACKERS
        return frozenset(str(t) for t in DEFAULT_TRACKERS)
    except Exception:
        return frozenset(_FALLBACK_TRACKERS)


def goalkeeper_slot_ids_from_csv(path: str) -> frozenset[str]:
    """Unique non-empty source_track_id values from goalkeeper_tracks.csv."""
    ids: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        key = fields.get("source_track_id")
        if key is None:
            raise ValueError(f"{path}: missing source_track_id column")
        for row in reader:
            value = (row.get(key) or "").strip()
            if value and value.lower() not in ("", "nan", "none"):
                ids.add(value)
    return frozenset(ids)


def resolve_goalkeeper_slot_ids(
        csv_path: str | None = None,
        extra: Iterable[str] | None = None,
) -> frozenset[str]:
    """DEFAULT_TRACKERS, optionally unioned with ids seen in the derived CSV and extras."""
    ids = set(default_goalkeeper_slot_ids())
    if csv_path and os.path.isfile(csv_path):
        ids |= set(goalkeeper_slot_ids_from_csv(csv_path))
    if extra:
        ids |= {str(x).strip() for x in extra if str(x).strip()}
    return frozenset(ids)


def outfield_slot_mask(slot_ids, gk_ids: Iterable[str] | None = None) -> np.ndarray:
    """Boolean mask True for outfield (non-GK) slots. ``slot_ids`` length S."""
    gk = frozenset(str(x) for x in (gk_ids if gk_ids is not None else default_goalkeeper_slot_ids()))
    slots = np.asarray([str(s) for s in slot_ids], dtype=object)
    return np.array([s not in gk for s in slots], dtype=bool)

"""Color seed load/save. Defaults match neon-yellow AR kit + yellow/orange flag."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict

_DEFAULTS = os.path.join(os.path.dirname(__file__), "defaults", "seeds_tactical.json")


def default_seeds_path() -> str:
    return _DEFAULTS


def load_seeds(path: str | None = None) -> Dict[str, Any]:
    p = path or _DEFAULTS
    with open(p) as fh:
        return json.load(fh)


def save_seeds(seeds: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(seeds, fh, indent=2)


def merge_hsv_range(seeds: Dict[str, Any], key: str, lower, upper, name: str = "calibrated") -> Dict[str, Any]:
    out = deepcopy(seeds)
    out[key] = [{"name": name, "lower": [int(x) for x in lower], "upper": [int(x) for x in upper]}]
    return out

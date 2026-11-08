"""Tactical <-> broadcast frame mapping from match_viewer/export/sync.json.

Same rule as SyncMap in match_viewer/match_dual.html, so a frame maps to the same counterpart here
and in the dual viewer: anchors are pairs of video frames showing the same moment, and from an
anchor up to the next one `broadcast = tactical + (anchor.broadcast - anchor.tactical)`. Frames
before the first anchor use the first offset. Where a segment would run past the start of the next
anchor on the other timeline, that overlap has no counterpart (content cut on the other side).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.paths import EXPORT_DIR  # noqa: E402

DEFAULT_SYNC = os.path.join(EXPORT_DIR, "sync.json")
SIDES = ("tactical", "broadcast")
NO_COUNTERPART = -1


def _other(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return SIDES[1 - SIDES.index(side)]


class SyncMap:
    """Piecewise-constant frame offset between the two recordings."""

    def __init__(self, anchors, fps: float = 25.0):
        A = sorted(({"tactical": int(a["tactical"]), "broadcast": int(a["broadcast"]),
                     "label": a.get("label", "anchor")} for a in anchors), key=lambda a: a["tactical"])
        b = [a["broadcast"] for a in A]
        if any(y <= x for x, y in zip(b, b[1:])):
            raise ValueError("sync anchors must be increasing on both timelines: {}".format(
                [(a["tactical"], a["broadcast"]) for a in A]))
        self.anchors, self.fps = A, float(fps)
        self._at = {s: np.array([a[s] for a in A], dtype=np.int64) for s in SIDES}

    @classmethod
    def from_json(cls, path: str = DEFAULT_SYNC) -> "SyncMap":
        if not os.path.isfile(path):
            raise IOError(f"no sync file at {path}\n  run:  python match_viewer/synchronize_videos.py")
        with open(path) as fh:
            doc = json.load(fh)
        return cls(doc.get("anchors") or [], fps=doc.get("fps", 25.0))

    def offset_at(self, frames, side: str = "tactical"):
        """Offset (other - side) in force at each frame of `side`."""
        f = np.asarray(frames, dtype=np.int64)
        if not self.anchors:
            return np.zeros_like(f)
        d = _other(side)
        k = np.maximum(np.searchsorted(self._at[side], f, side="right") - 1, 0)
        return self._at[d][k] - self._at[side][k]

    def map(self, frames, side: str = "tactical"):
        """Counterpart on the other timeline of each frame of `side`; NO_COUNTERPART (-1) where none.

        Scalars in, scalar out. Range checks against the other video's length are the caller's job
        (see SyncMap.index).
        """
        scalar = np.ndim(frames) == 0
        f = np.atleast_1d(np.asarray(frames, dtype=np.int64))
        if not self.anchors:
            out = f.copy()
        else:
            d = _other(side)
            k = np.maximum(np.searchsorted(self._at[side], f, side="right") - 1, 0)
            out = f - self._at[side][k] + self._at[d][k]
            nxt = k + 1
            has_next = nxt < len(self.anchors)
            cut = np.zeros(len(f), dtype=bool)
            cut[has_next] = out[has_next] >= self._at[d][nxt[has_next]]
            out[cut] = NO_COUNTERPART
        out[out < 0] = NO_COUNTERPART
        return int(out[0]) if scalar else out

    def index(self, start: int, end: int, side: str, other_start: int, other_end: int):
        """For every frame start..end of `side`, the row in the other source's arrays (which cover
        other_start..other_end), or -1 where there is no counterpart inside that range."""
        g = self.map(np.arange(start, end + 1), side)
        ok = (g >= other_start) & (g <= other_end)
        return np.where(ok, g - other_start, NO_COUNTERPART)

    def to_json(self):
        return {"fps": self.fps, "anchors": [dict(a) for a in self.anchors]}

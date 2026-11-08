"""Dead-ball intervals from the synchronised tactical and broadcast recordings.

    no view   = broadcast homography missing  ∩  broadcast player count ≠ 22
    ball lost = tactical ball missing         ∩  broadcast ball missing
    dead ball = no view ∪ (the ball-lost intervals that overlap a no-view interval)

A ball-lost interval is taken whole when it shares at least one frame with a no-view interval
(or comes within `max_gap` frames of one); ball-lost intervals on their own are not dead ball.

The terms are evaluated frame by frame on either timeline: the other recording's signals are
pulled across through the sync map first. Frames with no counterpart (halftime cut, outside the
other video) take `no_counterpart` for every signal of the missing side, False by default, so a
missing counterpart is never by itself evidence of a dead ball.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.signals import SourceSignals  # noqa: E402
from dead_ball_investigation.sync import SyncMap  # noqa: E402

FORMULA = ("(broadcast homography missing ∩ broadcast player count ≠ 22) "
           "∪ (the (tactical ball missing ∩ broadcast ball missing) intervals overlapping it)")
TERMS = {
    "no_view": "broadcast homography missing ∩ broadcast player count ≠ 22",
    "ball_lost": "tactical ball missing ∩ broadcast ball missing (every interval)",
    "ball_lost_kept": "the ball_lost intervals that overlap a no_view interval (part of the dead ball)",
}


def runs_overlapping(mask, seed, max_gap: int = 0):
    """The runs of `mask` that share a frame with `seed`, or come within max_gap frames of it."""
    mask, seed = np.asarray(mask, bool), np.asarray(seed, bool)
    if max_gap > 0 and seed.any():
        seed = np.convolve(seed, np.ones(2 * max_gap + 1), mode="same") > 0
    ids = np.cumsum(mask & ~np.concatenate([[False], mask[:-1]])) * mask
    keep = np.unique(ids[mask & seed])
    return np.isin(ids, keep[keep > 0])


def dead_ball_terms(b_hom_missing, b_count_off, t_ball_missing, b_ball_missing, max_gap: int = 0):
    """(no_view, ball_lost, ball_lost_kept, dead) masks from four frame-aligned boolean masks."""
    no_view = np.asarray(b_hom_missing, bool) & np.asarray(b_count_off, bool)
    ball_lost = np.asarray(t_ball_missing, bool) & np.asarray(b_ball_missing, bool)
    kept = runs_overlapping(ball_lost, no_view, max_gap)
    return no_view, ball_lost, kept, no_view | kept


def dead_ball_mask(b_hom_missing, b_count_off, t_ball_missing, b_ball_missing, max_gap: int = 0):
    """The dead-ball formula on frame-aligned masks."""
    return dead_ball_terms(b_hom_missing, b_count_off, t_ball_missing, b_ball_missing, max_gap)[3]


def mask_to_intervals(mask, start: int = 0):
    """Runs of True as [first, last] inclusive frame pairs, offset by start."""
    m = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [[int(x) + start, int(y) + start] for x, y in zip(a, b)]


def intervals_to_mask(intervals, start: int, n: int):
    """Inverse of mask_to_intervals over frames start .. start + n - 1."""
    out = np.zeros(n, dtype=bool)
    for a, b in intervals:
        a, b = max(int(a) - start, 0), min(int(b) - start, n - 1)
        if b >= a:
            out[a:b + 1] = True
    return out


@dataclass
class Timeline:
    """Every signal of both recordings, aligned to the frames of one of them."""
    source: str
    start: int
    covered: np.ndarray           # frame has a counterpart on the other recording
    t_ball_missing: np.ndarray
    b_ball_missing: np.ndarray
    b_hom_missing: np.ndarray
    b_count_off: np.ndarray
    other_frame: np.ndarray       # counterpart video frame on the other recording, -1 where none

    @property
    def n(self) -> int:
        return len(self.covered)


@dataclass
class DeadBall:
    """Formula output on one timeline."""
    source: str
    start: int
    dead: np.ndarray
    no_view: np.ndarray
    ball_lost: np.ndarray
    ball_lost_kept: np.ndarray
    covered: np.ndarray

    def intervals(self, which: str = "dead"):
        return mask_to_intervals(getattr(self, which), self.start)

    def stats(self, graded=None):
        out = {"frames": int(len(self.dead)), "covered_frames": int(self.covered.sum())}
        for k in ("dead", *TERMS):
            m = getattr(self, k)
            out[k] = {"frames": int(m.sum()), "share": round(float(m.mean()) if len(m) else 0.0, 4),
                      "intervals": len(mask_to_intervals(m))}
        if graded is not None:
            g = np.zeros(len(self.dead), dtype=bool)
            for a, b in graded:
                g |= intervals_to_mask([[a, b]], self.start, len(self.dead))
            out["graded"] = {"frames": int(g.sum()), "dead_frames": int((self.dead & g).sum()),
                             "dead_share": round(float(self.dead[g].mean()) if g.any() else 0.0, 4)}
        return out


@dataclass
class SyncedMatch:
    """Tactical and broadcast signals plus the sync map that relates them, and the Smart Tags."""
    tactical: SourceSignals
    broadcast: SourceSignals
    sync: SyncMap
    tags: list = field(default_factory=list)

    def source(self, side: str) -> SourceSignals:
        return self.tactical if side == "tactical" else self.broadcast

    def timeline(self, side: str = "tactical", no_counterpart: bool = False) -> Timeline:
        """All signals on the frames of `side`; the other side's come through the sync map."""
        own, oth = self.source(side), self.source("broadcast" if side == "tactical" else "tactical")
        row = self.sync.index(own.start, own.end, side, oth.start, oth.end)
        covered = row >= 0

        def pull(arr):
            out = np.full(own.n, bool(no_counterpart))
            out[covered] = arr[row[covered]]
            return out

        if side == "tactical":
            t_ball = own.ball_missing
            b_ball, b_hom, b_count = pull(oth.ball_missing), pull(oth.hom_missing), pull(oth.count_off)
        else:
            t_ball = pull(oth.ball_missing)
            b_ball, b_hom, b_count = own.ball_missing, own.hom_missing, own.count_off
        return Timeline(side, own.start, covered, t_ball, b_ball, b_hom, b_count,
                        np.where(covered, row + oth.start, -1))

    def synced_tags(self):
        """Smart Tagging instances with their span mapped onto the broadcast (None where cut)."""
        out = []
        for e in self.tags:
            ba, bb = (self.sync.map(e[k], "tactical") for k in ("a", "b"))
            out.append({**e, "broadcast_a": None if ba < 0 else ba, "broadcast_b": None if bb < 0 else bb})
        return out


def annotate(match: SyncedMatch, side: str = "tactical", no_counterpart: bool = False, max_gap: int = 0) -> DeadBall:
    """Dead-ball mask (and its terms) on the frames of one recording."""
    tl = match.timeline(side, no_counterpart)
    no_view, ball_lost, kept, dead = dead_ball_terms(tl.b_hom_missing, tl.b_count_off, tl.t_ball_missing,
                                                     tl.b_ball_missing, max_gap)
    return DeadBall(side, tl.start, dead, no_view, ball_lost, kept, tl.covered)


def annotate_dead_ball(match: SyncedMatch, no_counterpart: bool = False, max_gap: int = 0):
    """Dead-ball intervals on both timelines: {"tactical": DeadBall, "broadcast": DeadBall}."""
    return {side: annotate(match, side, no_counterpart, max_gap) for side in ("tactical", "broadcast")}


def to_document(match: SyncedMatch, result, sync_path: str | None = None, no_counterpart: bool = False,
                graded=None, inputs=None, max_gap: int = 0):
    """The JSON written to match_viewer/export/dead_ball.json (read by match_viewer.html)."""
    timelines = {}
    for side, r in result.items():
        timelines[side] = {
            "start": r.start, "end": r.start + len(r.dead) - 1,
            "dead_ball": r.intervals("dead"),
            "terms": {k: r.intervals(k) for k in TERMS},
            "covered": r.intervals("covered"),
            "stats": r.stats(graded if side == "tactical" else None),
        }
    tags = [{"code": e.get("code"), "type": e.get("type"), "team": e.get("team"),
             "tactical": [e["a"], e["b"]], "broadcast": [e["broadcast_a"], e["broadcast_b"]]}
            for e in match.synced_tags()]
    return {
        "version": 2,
        "formula": FORMULA,
        "terms": TERMS,
        "max_gap": int(max_gap),
        "fps": match.sync.fps,
        "frames": "absolute video frames of each timeline, intervals [first, last] inclusive",
        "no_counterpart": bool(no_counterpart),
        "sync": {"path": sync_path, **match.sync.to_json()},
        "inputs": inputs or {},
        "timelines": timelines,
        "smart_tags": tags,
    }

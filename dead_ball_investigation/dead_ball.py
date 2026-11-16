"""Dead-ball intervals from the synchronised tactical and broadcast recordings.

Baseline:
    no view   = broadcast homography missing  ∩  broadcast player count ≠ 22
    ball lost = tactical ball missing         ∩  broadcast ball missing
    baseline  = no view ∪ (the ball-lost intervals that overlap a no-view interval)

A ball-lost interval is taken whole when it shares at least one frame with a no-view interval
(or comes within `max_gap` frames of one); ball-lost intervals on their own are not dead ball.

Then, in this order:
    1. no counterpart   frames without a counterpart in the other video are dead (on the tactical
                        timeline: before the broadcast starts, halftime, after it ends)
    2. off pitch        the tactical ball is off the pitch (x or y ∉ [-0.002, 1.002] for more than
                        1 s, the BALL OFF PITCH row); where the tactical ball is missing, the
                        broadcast ball off the pitch counts instead
    3. still ball       a gap between two dead intervals is dead when the ball stays within
                        `still_radius_m` of where it is first seen in the gap, for the whole gap

The terms are evaluated frame by frame on either timeline: the other recording's signals are
pulled across through the sync map first. Where a frame has no counterpart, the other side's
signals read as not missing (`unmapped_signals_missing` flips that), so for the baseline a missing
counterpart is never evidence of a dead ball; step 1 is what makes it one.

The full method runs on the tactical timeline, the one the submission is indexed to. Broadcast
frames with a tactical counterpart take its result, so both timelines agree; broadcast frames
without one get the baseline and steps 1 and 2 on their own signals (no tactical ball there, so
step 2 reads the broadcast's).
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.signals import OFF_PITCH_MARGIN, OFF_PITCH_MIN_S, SourceSignals  # noqa: E402
from dead_ball_investigation.sync import SyncMap  # noqa: E402

STILL_RADIUS_M = 1.5
FORMULA = ("baseline = (broadcast homography missing ∩ broadcast player count ≠ 22) "
           "∪ (the (tactical ball missing ∩ broadcast ball missing) intervals overlapping it); "
           "then 1. no counterpart in the other video is dead; "
           "2. the tactical ball off the pitch (the broadcast's where the tactical ball is missing) is dead; "
           "3. a gap between dead intervals in which the ball stays still is dead")
TERMS = {
    "no_view": "broadcast homography missing ∩ broadcast player count ≠ 22",
    "ball_lost": "tactical ball missing ∩ broadcast ball missing (every interval)",
    "ball_lost_kept": "the ball_lost intervals that overlap a no_view interval",
    "baseline": "no_view ∪ ball_lost_kept",
    "no_counterpart": "step 1: frames without a counterpart in the other video",
    "off_pitch": "step 2: tactical ball off the pitch (x or y ∉ [-0.002, 1.002] for > 1 s); "
                 "where the tactical ball is missing, the broadcast ball off the pitch",
    "still_gap": "step 3: gaps between dead intervals in which the ball stays within the still radius",
}


@dataclass
class Params:
    """Knobs of the method; the defaults are the full method (baseline, then steps 1, 2 and 3)."""
    max_gap: int = 0                        # baseline: frames between a ball-lost and a no-view interval
    unmapped_signals_missing: bool = False  # other side's signals where a frame has no counterpart
    counterpart_step: bool = True           # step 1
    off_pitch_step: bool = True             # step 2
    off_pitch_margin: float = OFF_PITCH_MARGIN
    off_pitch_min_s: float = OFF_PITCH_MIN_S
    still_step: bool = True                 # step 3
    still_radius_m: float = STILL_RADIUS_M

    def to_json(self):
        return asdict(self)


def runs_overlapping(mask, seed, max_gap: int = 0):
    """The runs of `mask` that share a frame with `seed`, or come within max_gap frames of it."""
    mask, seed = np.asarray(mask, bool), np.asarray(seed, bool)
    if max_gap > 0 and seed.any():
        seed = np.convolve(seed, np.ones(2 * max_gap + 1), mode="same") > 0
    ids = np.cumsum(mask & ~np.concatenate([[False], mask[:-1]])) * mask
    keep = np.unique(ids[mask & seed])
    return np.isin(ids, keep[keep > 0])


def dead_ball_terms(b_hom_missing, b_count_off, t_ball_missing, b_ball_missing, max_gap: int = 0):
    """(no_view, ball_lost, ball_lost_kept, baseline) masks from four frame-aligned boolean masks."""
    no_view = np.asarray(b_hom_missing, bool) & np.asarray(b_count_off, bool)
    ball_lost = np.asarray(t_ball_missing, bool) & np.asarray(b_ball_missing, bool)
    kept = runs_overlapping(ball_lost, no_view, max_gap)
    return no_view, ball_lost, kept, no_view | kept


def dead_ball_mask(b_hom_missing, b_count_off, t_ball_missing, b_ball_missing, max_gap: int = 0):
    """The baseline formula on frame-aligned masks."""
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


def ball_travel(xy) -> float | None:
    """Furthest the ball gets from where it is first seen, in metres; None if it is never seen."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    seen = xy[np.isfinite(xy).all(axis=1)]
    if not len(seen):
        return None
    return float(np.hypot(*(seen - seen[0]).T).max())


def off_pitch(t_ball_seen, t_ball_off, b_ball_off):
    """Step 2: the tactical ball off the pitch; where the tactical ball is not seen, the broadcast's."""
    return np.where(np.asarray(t_ball_seen, bool), np.asarray(t_ball_off, bool), np.asarray(b_ball_off, bool))


def still_gaps(dead, ball_xy_by_preference, radius_m: float = STILL_RADIUS_M):
    """Step 3: (mask of the gaps to fill, stats). A gap is the frames between two consecutive dead
    intervals; it is filled when the ball travels less than radius_m in it. The ball is read from
    the first position array that sees it at all in the gap (tactical before broadcast); a gap in
    which no array sees the ball is left alone."""
    dead = np.asarray(dead, bool)
    fill = np.zeros(len(dead), dtype=bool)
    stats = {"gaps": 0, "filled": 0, "filled_frames": 0, "moving": 0, "unseen": 0}
    iv = mask_to_intervals(dead)
    for (_, b), (a, _) in zip(iv, iv[1:]):
        stats["gaps"] += 1
        travel = next((t for t in (ball_travel(xy[b + 1:a]) for xy in ball_xy_by_preference) if t is not None), None)
        if travel is None:
            stats["unseen"] += 1
        elif travel < radius_m:
            fill[b + 1:a] = True
            stats["filled"] += 1
            stats["filled_frames"] += a - b - 1
        else:
            stats["moving"] += 1
    return fill, stats


@dataclass
class Timeline:
    """Every signal of both recordings, aligned to the frames of one of them."""
    source: str
    start: int
    covered: np.ndarray           # the counterpart is inside the other source's loaded frames
    unmapped: np.ndarray          # no counterpart in the other video at all (sync cut or off its ends)
    t_ball_missing: np.ndarray
    b_ball_missing: np.ndarray
    b_hom_missing: np.ndarray
    b_count_off: np.ndarray
    t_ball_xy: np.ndarray         # (n, 2) metres, NaN where missing
    b_ball_xy: np.ndarray
    t_ball_seen: np.ndarray       # the tactical ball is tracked (False where the frame has no tactical counterpart)
    t_ball_off: np.ndarray        # BALL OFF PITCH of each source, on its own frames, pulled across
    b_ball_off: np.ndarray
    other_frame: np.ndarray       # counterpart video frame on the other recording, -1 where none

    @property
    def n(self) -> int:
        return len(self.covered)


@dataclass
class DeadBall:
    """Method output on one timeline."""
    source: str
    start: int
    dead: np.ndarray
    baseline: np.ndarray
    no_view: np.ndarray
    ball_lost: np.ndarray
    ball_lost_kept: np.ndarray
    no_counterpart: np.ndarray
    off_pitch: np.ndarray
    still_gap: np.ndarray
    covered: np.ndarray
    gaps: dict = field(default_factory=dict)

    def intervals(self, which: str = "dead"):
        return mask_to_intervals(getattr(self, which), self.start)

    def stats(self, graded=None):
        out = {"frames": int(len(self.dead)), "covered_frames": int(self.covered.sum())}
        for k in ("dead", *TERMS):
            m = getattr(self, k)
            out[k] = {"frames": int(m.sum()), "share": round(float(m.mean()) if len(m) else 0.0, 4),
                      "intervals": len(mask_to_intervals(m))}
        if self.gaps:
            out["still_gaps"] = dict(self.gaps)
        if graded is not None:
            g = np.zeros(len(self.dead), dtype=bool)
            for a, b in graded:
                g |= intervals_to_mask([[a, b]], self.start, len(self.dead))
            out["graded"] = {"frames": int(g.sum()), "dead_frames": int((self.dead & g).sum()),
                             "dead_share": round(float(self.dead[g].mean()) if g.any() else 0.0, 4)}
        return out


@dataclass
class SyncedMatch:
    """Tactical and broadcast signals plus the sync map that relates them, and the Smart Tags.

    video_frames: length of each video, to tell "no counterpart" from "counterpart not loaded";
    defaults to the end of the loaded signals."""
    tactical: SourceSignals
    broadcast: SourceSignals
    sync: SyncMap
    tags: list = field(default_factory=list)
    video_frames: dict | None = None

    def source(self, side: str) -> SourceSignals:
        return self.tactical if side == "tactical" else self.broadcast

    def frames_of(self, side: str) -> int:
        return int((self.video_frames or {}).get(side) or self.source(side).end + 1)

    def timeline(self, side: str = "tactical", params: Params | None = None) -> Timeline:
        """All signals on the frames of `side`; the other side's come through the sync map."""
        p = params or Params()
        other = "broadcast" if side == "tactical" else "tactical"
        own, oth = self.source(side), self.source(other)
        row = self.sync.index(own.start, own.end, side, oth.start, oth.end)
        covered = row >= 0
        g = self.sync.map(np.arange(own.start, own.end + 1), side)
        unmapped = (g < 0) | (g >= self.frames_of(other))

        def pull(arr, fill):
            out = np.full((own.n,) + arr.shape[1:], fill, dtype=arr.dtype)
            out[covered] = arr[row[covered]]
            return out

        miss = bool(p.unmapped_signals_missing)
        own_off, oth_off = (s.ball_off(self.sync.fps, p.off_pitch_margin, p.off_pitch_min_s) for s in (own, oth))
        if side == "tactical":
            t_ball, t_xy, t_seen, t_off = own.ball_missing, own.ball_xy, ~own.ball_missing, own_off
            b_ball, b_hom, b_count = pull(oth.ball_missing, miss), pull(oth.hom_missing, miss), pull(oth.count_off, miss)
            b_xy, b_off = pull(oth.ball_xy, np.nan), pull(oth_off, False)
        else:
            t_ball, t_xy = pull(oth.ball_missing, miss), pull(oth.ball_xy, np.nan)
            t_seen, t_off = pull(~oth.ball_missing, False), pull(oth_off, False)
            b_ball, b_hom, b_count, b_xy, b_off = own.ball_missing, own.hom_missing, own.count_off, own.ball_xy, own_off
        return Timeline(side, own.start, covered, unmapped, t_ball, b_ball, b_hom, b_count, t_xy, b_xy,
                        t_seen, t_off, b_off, np.where(covered, row + oth.start, -1))

    def synced_tags(self):
        """Smart Tagging instances with their span mapped onto the broadcast (None where cut)."""
        out = []
        for e in self.tags:
            ba, bb = (self.sync.map(e[k], "tactical") for k in ("a", "b"))
            out.append({**e, "broadcast_a": None if ba < 0 else ba, "broadcast_b": None if bb < 0 else bb})
        return out


def _terms(tl: Timeline, p: Params):
    no_view, ball_lost, kept, base = dead_ball_terms(tl.b_hom_missing, tl.b_count_off, tl.t_ball_missing,
                                                     tl.b_ball_missing, p.max_gap)
    no_cp = tl.unmapped & bool(p.counterpart_step)
    off = off_pitch(tl.t_ball_seen, tl.t_ball_off, tl.b_ball_off) & bool(p.off_pitch_step)
    return no_view, ball_lost, kept, base, no_cp, off


def annotate_dead_ball(match: SyncedMatch, params: Params | None = None):
    """Dead-ball intervals on both timelines: {"tactical": DeadBall, "broadcast": DeadBall}."""
    p = params or Params()
    t = match.timeline("tactical", p)
    no_view, ball_lost, kept, base, no_cp, off = _terms(t, p)
    dead = base | no_cp | off
    still, gaps = (still_gaps(dead, (t.t_ball_xy, t.b_ball_xy), p.still_radius_m) if p.still_step
                   else (np.zeros(t.n, dtype=bool), {}))
    tac = DeadBall("tactical", t.start, dead | still, base, no_view, ball_lost, kept, no_cp, off, still, t.covered, gaps)

    b = match.timeline("broadcast", p)
    no_view, ball_lost, kept, base, no_cp, off = _terms(b, p)
    dead, still = base | no_cp | off, np.zeros(b.n, dtype=bool)
    rows = b.other_frame[b.covered] - t.start
    dead[b.covered], off[b.covered], still[b.covered] = tac.dead[rows], tac.off_pitch[rows], tac.still_gap[rows]
    br = DeadBall("broadcast", b.start, dead, base, no_view, ball_lost, kept, no_cp, off, still, b.covered)
    return {"tactical": tac, "broadcast": br}


def annotate(match: SyncedMatch, side: str = "tactical", params: Params | None = None) -> DeadBall:
    """Dead-ball mask (and its terms) on the frames of one recording."""
    return annotate_dead_ball(match, params)[side]


def to_document(match: SyncedMatch, result, sync_path: str | None = None, params: Params | None = None,
                graded=None, inputs=None):
    """The JSON written to dead_ball_investigation/dead_ball.json (read by match_viewer.html)."""
    p = params or Params()
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
        "version": 4,
        "formula": FORMULA,
        "terms": TERMS,
        "params": p.to_json(),
        "fps": match.sync.fps,
        "frames": "absolute video frames of each timeline, intervals [first, last] inclusive",
        "sync": {"path": sync_path, **match.sync.to_json()},
        "inputs": inputs or {},
        "timelines": timelines,
        "smart_tags": tags,
    }

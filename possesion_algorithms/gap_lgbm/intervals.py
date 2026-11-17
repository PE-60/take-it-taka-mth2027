"""Target gaps, visible runs and pretend gaps.

All indices are rows of one Tracking (0 .. n - 1) and every interval is half-open [a, b).

A target gap is a run of alive frames (inside one alive interval of the dead-ball mask) where
the ball is missing. Its anchors are the teacher's team on the last visible frame before it and
on the first visible frame after it, when that frame is still in the same alive interval;
otherwise the anchor is unknown (0). Anchors are H-relative: +1 H, -1 A, 0 unknown.

A pretend gap is a stretch of a visible run (alive, ball seen throughout) that we treat as if
the ball were missing. Lengths come from the target-gap length distribution; with probability
`kick_bias` a pretend gap starts just after a kick (real gaps cluster around balls in flight).
It keeps `context` visible frames on both sides, so it always has both anchors; the anchors
are then dropped at the rate the target gaps lack them, so both sets look alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEAM_SIGN = {"H": 1, "A": -1}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(a, b)] half-open runs of True in mask."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def team_sign(labels: np.ndarray) -> np.ndarray:
    """H/A/'' labels as +1 / -1 / 0."""
    labels = np.asarray(labels)
    return np.where(labels == "H", 1, np.where(labels == "A", -1, 0)).astype(np.int8)


@dataclass
class Gaps:
    """A set of gaps on one Tracking."""
    a: np.ndarray           # (G,) first row
    b: np.ndarray           # (G,) one past the last row
    before: np.ndarray      # (G,) int8 H-relative anchor before, 0 unknown
    after: np.ndarray       # (G,) int8 H-relative anchor after, 0 unknown

    @property
    def n(self) -> int:
        return len(self.a)

    @property
    def length(self) -> np.ndarray:
        return self.b - self.a

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        """(rows, gap index) of every frame of every gap, gap by gap."""
        lens = self.length
        gap = np.repeat(np.arange(self.n), lens)
        offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        return self.a[gap] + offs, gap


def target_gaps(live: np.ndarray, ball_seen: np.ndarray, teacher: np.ndarray,
                require_team: bool = False) -> Gaps:
    """Ball-missing runs inside each alive run, with the teacher's team on both sides.

    require_team  keep only gaps the teacher enters with a team (its label on the last seen
                  frame before the gap). Gaps that open an alive run, or come before the
                  teacher's first holder in it, are dropped: the start of play is too unsure.
    """
    sign = team_sign(teacher)
    A, B, before, after = [], [], [], []
    for la, lb in runs(live):
        for ga, gb in runs(~ball_seen[la:lb]):
            a, b = la + ga, la + gb
            if require_team and (a == la or sign[a - 1] == 0):
                continue
            A.append(a)
            B.append(b)
            before.append(sign[a - 1] if a > la else 0)     # a - 1 is a seen frame of this run
            after.append(sign[b] if b < lb else 0)
    return Gaps(np.asarray(A, np.int64), np.asarray(B, np.int64),
                np.asarray(before, np.int8), np.asarray(after, np.int8))


def visible_runs(live: np.ndarray, ball_seen: np.ndarray) -> list[tuple[int, int]]:
    return runs(np.asarray(live, bool) & np.asarray(ball_seen, bool))


def sample_pretend_gaps(visible: list[tuple[int, int]], gap_lengths: np.ndarray, teacher: np.ndarray,
                        kick: np.ndarray, rng: np.random.Generator, coverage: float = 1.0,
                        context: int = 25, kick_bias: float = 0.5, p_no_before: float = 0.0,
                        p_no_after: float = 0.0, block: int | None = None) -> Gaps:
    """Pretend gaps inside the visible runs.

    gap_lengths  lengths (frames) to draw from, e.g. the target gaps' lengths
    coverage     expected pretend-gap frames per visible frame (overlaps allowed)
    context      visible frames kept before and after each pretend gap
    block        when set, a pretend gap may not cross a multiple of `block` (CV block edge),
                 nor start within `context` frames after one
    """
    gap_lengths = np.asarray(gap_lengths, np.int64)
    gap_lengths = gap_lengths[gap_lengths > 0]
    if not len(gap_lengths):
        gap_lengths = np.array([25], np.int64)
    mean_len = float(gap_lengths.mean())
    sign = team_sign(teacher)
    kick_rows = np.flatnonzero(np.asarray(kick, bool))
    A, B = [], []
    for va, vb in visible:
        room = vb - va - 2 * context
        if room < 1:
            continue
        k = rng.poisson(coverage * room / mean_len)
        for _ in range(int(k)):
            fit = gap_lengths[gap_lengths <= room]
            if not len(fit):
                break
            L = int(rng.choice(fit))
            lo, hi = va + context, vb - context - L          # start in [lo, hi]
            start = None
            if kick_bias > 0 and rng.random() < kick_bias:
                ks = kick_rows[(kick_rows + 1 >= lo) & (kick_rows + 1 <= hi)]
                if len(ks):
                    start = int(rng.choice(ks)) + 1 + int(rng.integers(0, 3))
                    start = min(max(start, lo), hi)
            if start is None:
                start = int(rng.integers(lo, hi + 1))
            if block:
                first, last = start // block, (start + L - 1) // block
                if first != last or start - first * block < context:
                    continue
            if sign[start - 1] == 0 or sign[start + L] == 0:
                continue                                      # teacher has no team yet
            A.append(start)
            B.append(start + L)
    a, b = np.asarray(A, np.int64), np.asarray(B, np.int64)
    before = sign[a - 1].astype(np.int8) if len(a) else np.zeros(0, np.int8)
    after = sign[b].astype(np.int8) if len(b) else np.zeros(0, np.int8)
    if len(a):
        before = np.where(rng.random(len(a)) < p_no_before, 0, before).astype(np.int8)
        after = np.where(rng.random(len(a)) < p_no_after, 0, after).astype(np.int8)
    return Gaps(a, b, before, after)


def sample_windows(ok: np.ndarray, lengths: np.ndarray, rng: np.random.Generator, coverage: float = 0.5,
                   boundaries: np.ndarray | None = None, per_boundary: int = 2,
                   block: int | None = None) -> Gaps:
    """Windows for the dead-vs-alive model: stretches of `ok` frames treated as if the ball were
    missing, labelled frame by frame elsewhere. Some fall anywhere (`coverage` window frames per ok
    frame); `per_boundary` more straddle each row in `boundaries` (a dead ball starting or ending),
    so the model sees play stopping and restarting. No window crosses a block edge or leaves its
    run of ok frames. Anchors are 0: the caller fills them in."""
    lengths = np.asarray(lengths, np.int64)
    lengths = lengths[lengths > 1]
    if not len(lengths):
        lengths = np.array([50], np.int64)
    mean_len = float(lengths.mean())
    ok = np.asarray(ok, bool)
    run_id = np.full(len(ok), -1, np.int64)
    ok_runs = runs(ok)
    for i, (a, b) in enumerate(ok_runs):
        run_id[a:b] = i
    A, B = [], []

    def accept(start, L):
        end = start + L
        if start < 1 or end > len(ok) or run_id[start] < 0 or run_id[start] != run_id[end - 1]:
            return
        if block and start // block != (end - 1) // block:
            return
        A.append(start)
        B.append(end)

    for a, b in ok_runs:
        for _ in range(int(rng.poisson(coverage * (b - a) / mean_len))):
            fit = lengths[lengths <= b - a - 1]
            if not len(fit):
                break
            L = int(rng.choice(fit))
            accept(int(rng.integers(a + 1, b - L + 1)) if b - L + 1 > a + 1 else a + 1, L)
    for x in np.asarray(boundaries if boundaries is not None else [], np.int64):
        for _ in range(per_boundary):
            L = int(rng.choice(lengths))
            accept(int(x) - int(rng.integers(1, L)), L)
    a, b = np.asarray(A, np.int64), np.asarray(B, np.int64)
    z = np.zeros(len(a), np.int8)
    return Gaps(a, b, z, z.copy())

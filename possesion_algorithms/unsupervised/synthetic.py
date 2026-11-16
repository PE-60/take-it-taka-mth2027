"""Artificial match tracking with ground-truth possession labels.

The real ATD is proprietary and never read here. This generator produces a short match-like
Tracking plus H/A/D labels, with the same pathologies the model has to survive: missing ball
frames, ghost-ball episodes, fragmented slots, and a goalkeeper filed under the wrong team.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possesion_algorithms.base import Tracking

PITCH = (105.3, 68.0)
FPS = 25.0

# Relative formation anchors in own-half fractions of pitch length / width (x from own goal).
_FORMATION = np.array([
    (0.06, 0.50),                          # GK
    (0.22, 0.15), (0.22, 0.38), (0.22, 0.62), (0.22, 0.85),  # back four
    (0.40, 0.25), (0.40, 0.50), (0.40, 0.75),                # midfield
    (0.58, 0.20), (0.58, 0.50), (0.58, 0.80),                # front three
], dtype=np.float64)


@dataclass
class SyntheticMatch:
    track: Tracking
    labels: np.ndarray          # (n,) H/A/D ground truth
    ghost_ball: np.ndarray      # (n,) True where the ball position is a planted anomaly
    true_team: np.ndarray       # (S,) team of each slot before corruption
    gk_slots: tuple             # slot indices that are goalkeepers


def _anchors(sign: float, push: float, length: float, width: float) -> np.ndarray:
    """11 player targets. sign=+1 attacks towards x=length; push in metres along attack."""
    mid = length / 2.0
    own = 0.0 if sign > 0 else length
    xs = own + sign * (_FORMATION[:, 0] * length + push)
    ys = _FORMATION[:, 1] * width
    xs = np.clip(xs, 2.0, length - 2.0)
    return np.column_stack([xs, ys])


def _spell_lengths(rng: np.random.Generator, n: int, mean_s: float, dead_prob: float,
                   fps: float) -> list[tuple[str, int]]:
    """Sequence of (label, n_frames) covering n frames."""
    spells = []
    left = n
    team = "H"
    while left > 0:
        if spells and rng.random() < dead_prob:
            lab = "D"
            dur = int(rng.integers(int(1.5 * fps), int(8 * fps) + 1))
        else:
            lab = team
            dur = max(int(0.8 * fps), int(rng.lognormal(np.log(mean_s), 0.55) * fps))
            team = "A" if team == "H" else "H"
        dur = min(dur, left)
        spells.append((lab, dur))
        left -= dur
    return spells


def make_synthetic_match(n_frames: int = 4500, fps: float = FPS, pitch: tuple = PITCH,
                         seed: int = 0, miss_ball_frac: float = 0.40,
                         ghost_episodes: int = 4, mislabel_gk: bool = True) -> SyntheticMatch:
    """A few minutes of artificial play with known possession labels."""
    rng = np.random.default_rng(seed)
    length, width = float(pitch[0]), float(pitch[1])
    n = int(n_frames)
    S = 46
    xy = np.full((n, S, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    labels = np.full(n, "D", dtype="<U1")
    ghost = np.zeros(n, bool)
    kick = np.zeros(n, bool)
    true_team = np.full(S, "?", dtype="<U1")
    team = np.full(S, "?", dtype="<U1")

    # Slot layout: H0..H10, A0..A10 live in slots 0..21; remaining slots unused (fragmentation
    # later reuses a few of them).
    h_slots = np.arange(11)
    a_slots = np.arange(11, 22)
    true_team[h_slots] = "H"
    true_team[a_slots] = "A"
    team[:] = true_team
    if mislabel_gk:
        team[0] = "A"   # H goalkeeper filed as away

    spells = _spell_lengths(rng, n, mean_s=12.0, dead_prob=0.28, fps=fps)
    holders = {"H": 9, "A": 20}   # starting front-three holders
    pos_h = _anchors(+1.0, 0.0, length, width)
    pos_a = _anchors(-1.0, 0.0, length, width)
    ball_m = np.array([length / 2, width / 2], dtype=np.float64)
    t = 0
    for lab, dur in spells:
        for k in range(dur):
            if lab == "D":
                push = -4.0
                target_h = _anchors(+1.0, push, length, width)
                target_a = _anchors(-1.0, push, length, width)
                speed_cap = 1.2  # mostly still during stoppages
            else:
                attack = +1.0 if lab == "H" else -1.0
                push = 8.0
                target_h = _anchors(+1.0, push if lab == "H" else -6.0, length, width)
                target_a = _anchors(-1.0, push if lab == "A" else -6.0, length, width)
                holders_team = h_slots if lab == "H" else a_slots
                holder = holders[lab]
                # occasional pass to another teammate
                if k % int(1.6 * fps) == 0 and rng.random() < 0.55:
                    holder = int(rng.choice(holders_team[1:]))  # not the GK
                    holders[lab] = holder
                holder_pos = (pos_h if lab == "H" else pos_a)[holder % 11]
                # ball eases towards the holder; small carry jitter
                ball_m = ball_m + 0.35 * (holder_pos - ball_m)
                ball_m = ball_m + rng.normal(0, 0.15, 2)
                # possessing side's nearest players ease toward the ball
                side_pos = pos_h if lab == "H" else pos_a
                side_target = target_h if lab == "H" else target_a
                near = np.argsort(np.hypot(*(side_pos - ball_m).T))[:3]
                for i in near:
                    side_target[i] = 0.7 * side_target[i] + 0.3 * ball_m
                speed_cap = 7.0

            # move players towards targets with capped step + noise
            for side_pos, side_target, side_slots in (
                    (pos_h, target_h, h_slots),
                    (pos_a, target_a, a_slots)):
                delta = side_target - side_pos
                step = np.clip(delta, -speed_cap / fps, speed_cap / fps)
                side_pos[:] = side_pos + step + rng.normal(0, 0.04, side_pos.shape)
                side_pos[:, 0] = np.clip(side_pos[:, 0], 1.0, length - 1.0)
                side_pos[:, 1] = np.clip(side_pos[:, 1], 1.0, width - 1.0)
                xy[t, side_slots] = (side_pos / np.array([length, width])).astype(np.float32)

            ball[t] = (ball_m / np.array([length, width])).astype(np.float32)
            labels[t] = lab
            t += 1

    # Missing-ball gaps (~miss_ball_frac of frames).
    miss = np.zeros(n, bool)
    covered = 0
    while covered < int(miss_ball_frac * n):
        start = int(rng.integers(0, max(1, n - 10)))
        length_gap = int(rng.integers(5, int(3.5 * fps) + 1))
        end = min(n, start + length_gap)
        miss[start:end] = True
        covered = int(miss.sum())
    ball[miss] = np.nan

    # Ghost-ball episodes: a still ball planted far from every tracked player.
    for _ in range(ghost_episodes):
        start = int(rng.integers(int(2 * fps), max(int(2 * fps) + 1, n - int(3 * fps))))
        end = min(n, start + int(rng.integers(int(1.0 * fps), int(2.5 * fps) + 1)))
        # Pick a pitch point at least 20 m from every player on the start frame.
        ghost_m = None
        for _try in range(40):
            cand = np.array([rng.uniform(5, length - 5), rng.uniform(5, width - 5)])
            players = xy[start]
            present = np.isfinite(players).all(axis=1)
            if not present.any():
                ghost_m = cand
                break
            d = np.hypot(*(players[present] * np.array([length, width]) - cand).T)
            if d.min() >= 20.0:
                ghost_m = cand
                break
        if ghost_m is None:
            ghost_m = np.array([5.0, 5.0])
        ghost_xy = (ghost_m / np.array([length, width])).astype(np.float32)
        for f in range(start, end):
            if labels[f] == "D":
                continue
            ball[f] = ghost_xy
            ghost[f] = True
            miss[f] = False

    # Occasional player dropouts.
    for slot in list(h_slots) + list(a_slots):
        if rng.random() < 0.35:
            start = int(rng.integers(0, max(1, n - 20)))
            end = min(n, start + int(rng.integers(10, int(2 * fps) + 1)))
            xy[start:end, slot] = np.nan

    # Slot fragmentation: copy a midfielder onto a free slot for a stretch, leave original empty.
    free = np.arange(22, min(30, S))
    for i, slot in enumerate(free[:4]):
        src = 5 + (i % 6)
        start = int(rng.integers(0, max(1, n // 2)))
        end = min(n, start + int(rng.integers(int(5 * fps), int(20 * fps) + 1)))
        xy[start:end, slot] = xy[start:end, src]
        xy[start:end, src] = np.nan
        team[slot] = true_team[src]
        true_team[slot] = true_team[src]

    # Kick flag on sharp ball acceleration (approximate).
    metric = ball * np.array(pitch, dtype=np.float32)
    vel = np.full_like(metric, np.nan)
    if n >= 2:
        vel[1:] = (metric[1:] - metric[:-1]) * fps
    acc = np.full(n, np.nan, dtype=np.float32)
    if n >= 3:
        acc[1:-1] = np.hypot(*(vel[2:] - vel[:-2]).T) * (fps / 2.0)
    kick[np.isfinite(acc) & (acc >= 4.0)] = True

    track = Tracking(
        start=0,
        ball=ball,
        xy=xy,
        team=team,
        kick=kick,
        origin=np.zeros(n, dtype=np.int8),
        pitch=tuple(float(x) for x in pitch),
        fps=float(fps),
    )
    return SyntheticMatch(track, labels, ghost, true_team, gk_slots=(0, 11))

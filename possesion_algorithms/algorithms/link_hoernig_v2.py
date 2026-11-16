"""A control-aware, tactical-only successor to the Link & Hoernig heuristic.

The original heuristic can give the ball to an opponent merely because the ball's projected x/y
position passes close to him. V2 requires two pieces of evidence before a player gains possession:

* a discontinuity in the ball's velocity vector occurred near that player; and
* the ball remains near the player after the candidate touch (the gain-retention test).

Using the velocity *vector* detects changes of direction at nearly constant speed, unlike the
original scalar-acceleration kick flag. A player can bootstrap possession at the start of a run
without a detected touch, but only by passing the same gain-retention test. Once a team has been
established, an opposing team can take over only at a retained touch.

Two optional penalties are deliberately disabled by default. ``switch_dv_margin_m_s`` asks for a
larger velocity change on cross-team gains than on same-team touches. ``smooth_switch_penalty``
applies a two-state Viterbi smoother where the parameter is the cost of one team switch measured
in frames of contradictory evidence. They are useful experiments, but cannot be tuned honestly
from synthetic data alone.
"""

from __future__ import annotations

import numpy as np

from possesion_algorithms.base import ORIGIN, TACTICAL, PossessionAlgorithm, Tracking


TEAMS = ("H", "A")


def _window_velocity_change(track: Tracking, window: int) -> np.ndarray:
    """Magnitude of mean outgoing velocity minus mean incoming velocity at every frame.

    Velocities are pitch-metric vectors in m/s. A value is finite only when both windows contain
    ``window`` valid inter-frame velocities, preventing tracking gaps from becoming fake touches.
    """
    n = track.n
    change = np.full(n, np.nan, dtype=np.float32)
    if n < 2 * window + 1:
        return change

    metric_ball = track.ball * np.asarray(track.pitch, dtype=np.float32)
    velocity = np.diff(metric_ball, axis=0) * float(track.fps)
    valid = np.isfinite(velocity).all(axis=1)
    values = np.where(valid[:, None], velocity, 0.0)
    sums = np.vstack((np.zeros((1, 2), dtype=np.float64),
                      np.cumsum(values, axis=0, dtype=np.float64)))
    counts = np.concatenate(([0], np.cumsum(valid.astype(np.int32))))

    frames = np.arange(window, n - window)
    incoming_count = counts[frames] - counts[frames - window]
    outgoing_count = counts[frames + window] - counts[frames]
    complete = (incoming_count == window) & (outgoing_count == window)
    if not complete.any():
        return change

    incoming = (sums[frames] - sums[frames - window]) / float(window)
    outgoing = (sums[frames + window] - sums[frames]) / float(window)
    delta = np.hypot(*(outgoing - incoming).T)
    change[frames[complete]] = delta[complete].astype(np.float32)
    return change


def _event_peaks(values: np.ndarray, threshold: float, refractory: int) -> np.ndarray:
    """Strongest frame in each above-threshold cluster, including nearby clusters."""
    candidates = np.flatnonzero(np.isfinite(values) & (values >= threshold))
    if not len(candidates):
        return np.empty(0, dtype=np.int64)

    events = []
    first = 0
    for i in range(1, len(candidates) + 1):
        if i < len(candidates) and candidates[i] - candidates[i - 1] <= refractory:
            continue
        group = candidates[first:i]
        events.append(int(group[np.argmax(values[group])]))
        first = i
    return np.asarray(events, dtype=np.int64)


def _retains_gain(distances: np.ndarray, player: int, frame: int, horizon: int,
                  radius: float, min_near_fraction: float) -> bool:
    """Whether a candidate stays near the ball throughout a gain look-ahead.

    Requiring the endpoint to be near rejects an ordinary pass-by. Requiring a fraction of the
    whole window to be near also rejects a ball that travels away and happens to cross the player
    again exactly at the endpoint.
    """
    future = min(len(distances) - 1, frame + horizon)
    window = distances[frame:future + 1, player]
    near = np.isfinite(window) & (window <= radius)
    return bool(len(near) and near[-1] and near.mean() >= min_near_fraction)


def _smooth_team_labels(labels: np.ndarray, switch_penalty: int) -> np.ndarray:
    """Viterbi smoothing for H/A labels with a fixed cost for changing team.

    A disagreement with the raw label costs one per frame. Blank/unknown frames have no emission
    preference. With a penalty ``p``, an isolated opposite-team run generally needs at least
    ``2*p`` supporting frames to survive because retaining it requires entering and leaving it.
    """
    if switch_penalty <= 0 or not np.isin(labels, TEAMS).any():
        return labels

    out = labels.copy()
    first = int(np.flatnonzero(np.isin(labels, TEAMS))[0])
    raw = labels[first:]
    n = len(raw)
    costs = np.full((n, 2), np.inf, dtype=np.float64)
    previous = np.zeros((n, 2), dtype=np.int8)

    for state, team in enumerate(TEAMS):
        costs[0, state] = 0.0 if raw[0] in ("", team) else 1.0
    for r in range(1, n):
        for state, team in enumerate(TEAMS):
            stay = costs[r - 1, state]
            change = costs[r - 1, 1 - state] + switch_penalty
            previous[r, state] = state if stay <= change else 1 - state
            emission = 0.0 if raw[r] in ("", team) else 1.0
            costs[r, state] = min(stay, change) + emission

    state = int(np.argmin(costs[-1]))
    states = np.empty(n, dtype=np.int8)
    for r in range(n - 1, -1, -1):
        states[r] = state
        state = int(previous[r, state])
    out[first:] = np.asarray(TEAMS)[states]
    return out


class LinkHoernigV2(PossessionAlgorithm):
    """Velocity-touch plus retained-control possession heuristic for tactical tracking."""

    name = "lh_v2"
    description = ("LH V2: velocity-vector touches plus look-ahead retention on gains; whole video, tactical only")
    dead_ball = False
    source = TACTICAL
    params = {
        "t_p": 1.0,
        "t_a": 2.5,
        "keep_horizon_s": 1.0,
        "gain_horizon_s": 1,
        "gain_min_near_fraction": 0.6,
        "velocity_window_s": 1,
        "touch_dv_m_s": 2.0,
        "touch_window_s": 0.08,
        "touch_refractory_s": 0.20,
        "switch_dv_margin_m_s": 0.0,
        "smooth_switch_penalty": 0,
    }

    def __init__(self, **params):
        super().__init__(**params)
        nonnegative = (
            "t_p", "t_a", "keep_horizon_s", "gain_horizon_s",
            "gain_min_near_fraction", "velocity_window_s", "touch_dv_m_s",
            "touch_window_s", "touch_refractory_s", "switch_dv_margin_m_s",
            "smooth_switch_penalty",
        )
        bad = [name for name in nonnegative if self.params[name] < 0]
        if bad:
            raise ValueError(f"{self.name}: parameters must be non-negative: {bad}")
        if self.params["gain_min_near_fraction"] > 1:
            raise ValueError(f"{self.name}: gain_min_near_fraction must be between 0 and 1")
        if self.params["velocity_window_s"] <= 0:
            raise ValueError(f"{self.name}: velocity_window_s must be positive")

    def label(self, track: Tracking) -> np.ndarray:
        n = track.n
        out = np.full(n, "", dtype="<U1")
        if n == 0 or track.n_slots == 0:
            return out
        if np.any((track.origin >= 0) & (track.origin != ORIGIN[TACTICAL])):
            raise ValueError(f"{self.name}: only tactical tracking is supported")

        fps = float(track.fps)
        distances = track.ball_distances()
        seen = np.isfinite(track.ball).all(axis=1)
        present = np.isfinite(track.xy).all(axis=2)
        eligible = np.isin(track.team, TEAMS)
        if not eligible.any():
            return out
        eligible_distances = distances.copy()
        eligible_distances[:, ~eligible] = np.inf
        nearest = np.argmin(eligible_distances, axis=1)
        nearest_distance = eligible_distances[np.arange(n), nearest]

        velocity_window = max(1, int(round(self.params["velocity_window_s"] * fps)))
        touch_window = int(round(self.params["touch_window_s"] * fps))
        refractory = max(1, int(round(self.params["touch_refractory_s"] * fps)))
        velocity_change = _window_velocity_change(track, velocity_window)
        events = _event_peaks(velocity_change, self.params["touch_dv_m_s"], refractory)

        # Associate a velocity event with the player closest to the ball in a small window around
        # it. The event stays at its physical time; the window only handles finite-difference
        # alignment and tracking jitter.
        touch_player = np.full(n, -1, dtype=np.int32)
        for event in events:
            a, b = max(0, event - touch_window), min(n, event + touch_window + 1)
            local = eligible_distances[a:b]
            flat = int(np.argmin(local))
            row, player = np.unravel_index(flat, local.shape)
            if local[row, player] <= self.params["t_p"]:
                touch_player[event] = int(player)

        keep_horizon = max(1, int(round(self.params["keep_horizon_s"] * fps)))
        gain_horizon = max(1, int(round(self.params["gain_horizon_s"] * fps)))

        def remains_near(player: int, frame: int, horizon: int) -> bool:
            future = min(n - 1, frame + horizon)
            # Nearness at the end of the look-ahead distinguishes a reception from a pass-by.
            return bool(np.isfinite(distances[future, player])
                        and distances[future, player] <= self.params["t_a"])

        holder, last_team = -1, ""
        for frame in range(n):
            if not seen[frame]:
                holder = -1
                out[frame] = last_team
                continue

            if holder >= 0:
                if not present[frame, holder] or not remains_near(holder, frame, keep_horizon):
                    holder = -1

            candidate = int(touch_player[frame])
            if candidate >= 0 and _retains_gain(
                    distances, candidate, frame, gain_horizon, self.params["t_a"],
                    self.params["gain_min_near_fraction"]):
                candidate_team = str(track.team[candidate])
                is_switch = bool(last_team and candidate_team != last_team)
                required_change = (self.params["touch_dv_m_s"]
                                   + (self.params["switch_dv_margin_m_s"] if is_switch else 0.0))
                if velocity_change[frame] >= required_change:
                    holder = candidate

            # At the start of an independently labelled run there may be no observable touch.
            # Stable proximity is sufficient to establish the initial holder, but never to change
            # an already established team.
            if holder < 0 and not last_team:
                candidate = int(nearest[frame])
                if (eligible[candidate] and nearest_distance[frame] <= self.params["t_p"]
                        and _retains_gain(
                            distances, candidate, frame, gain_horizon, self.params["t_a"],
                            self.params["gain_min_near_fraction"])):
                    holder = candidate

            if holder >= 0:
                last_team = str(track.team[holder])
            out[frame] = last_team

        return _smooth_team_labels(out, int(self.params["smooth_switch_penalty"]))

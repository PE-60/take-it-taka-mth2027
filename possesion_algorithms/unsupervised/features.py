"""Per-frame geometry of the ball and both teams, built for an unsupervised model.

The idea: a single tracker (nearest player to the ball) is noisy and absent 41% of the time, but
the *shape* of the game is visible on every frame. The team in possession pushes its lines up,
spreads out and moves the ball and itself towards the opponent's goal; the defending team
compacts, drops and runs at the ball. Encoding these collectively gives a model something to
work with when the ball is missing or mis-tracked.

Features come in two blocks, declared in FEATURES:

* **anti-symmetric** — swapping the H and A labels negates them. They carry possession
  information: who is nearer the ball, who pushes forward, where the ball is relative to each
  side. A model can treat H and A as mirror images of each other through them.
* **symmetric** — unchanged under a swap. They carry alive/dead information: ball speed, who is
  moving at all, whether the ball is tracked or off the pitch, how many players are tracked.

The attacking direction of H is not known a priori and changes at halftime. It is estimated
from the data: over a rolling window, H attacks towards the side where A's centroid is.

Features are NaN when the input they need is missing (ball not tracked, a team with too few
players); the downstream model handles that, this module does not impute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possesion_algorithms.base import Tracking

# name, anti-symmetric?, needs the ball?
FEATURES = (
    ("near_gap", True, True),          # nearest-A distance minus nearest-H distance (m)
    ("near_soft", True, True),         # soft count of H minus A players around the ball
    # Carrier vs presser: near_gap, plus a penalty for closing speed on each team's nearest
    # player. Two defenders sprinting in look like "A has it" on near_soft/near_gap alone;
    # control_gap keeps the still (or co-moving) holder ahead until someone is clearly nearer
    # without charging.
    ("control_gap", True, True),
    # In-flight / pass lookahead (NaN unless the ball travels far enough with future tracking):
    # recv_gap  = control_gap at the future frame (who is on the ball when it arrives)
    # lead_gap  = nearest-A minus nearest-H distance from *current* players to the *future*
    #             ball position (who is running onto the landing spot, not chasing the ball now)
    ("recv_gap", True, True),
    ("lead_gap", True, True),
    ("converge", True, True),          # H minus A approach speed towards the ball (m/s)
    ("ball_motion", True, True),       # ball velocity along H's attacking direction (m/s)
    ("ball_territory", True, True),    # ball position along H's attacking direction (m from centre)
    ("territory", True, False),        # midpoint of both centroids along H's direction (m)
    ("line_diff", True, False),        # H defensive line height minus A's (m)
    ("spread_diff", True, False),      # H spread minus A spread (m)
    ("team_motion", True, False),      # everyone's velocity along H's direction (m/s)
    ("ball_speed", False, True),       # log1p ball speed
    ("ball_accel", False, True),       # log1p ball acceleration magnitude
    ("ball_speed_1s", False, True),    # log1p displacement over one second
    ("near_min", False, True),         # nearest player of either team (m)
    ("converge_all", False, True),     # everyone's approach speed towards the ball (m/s)
    ("ball_off_pitch", False, True),   # 1 when the ball is outside the pitch
    ("ball_missing", False, False),    # 1 when the ball is not tracked (always observed)
    ("players_moving", False, False),  # mean player speed (m/s)
    ("n_present", False, False),       # tracked players / 22
    ("spread_sum", False, False),      # H spread plus A spread (m)
    ("centroid_gap", False, False),    # distance between the two centroids (m)
)
NAMES = tuple(f[0] for f in FEATURES)
ANTI = np.array([f[1] for f in FEATURES])
NEEDS_BALL = np.array([f[2] for f in FEATURES])


@dataclass
class Features:
    values: np.ndarray        # (n, F) float32, NaN where not computable
    names: tuple
    anti: np.ndarray          # (F,) bool
    direction: np.ndarray     # (n,) +1 where H attacks towards x = 1, else -1
    ball_seen: np.ndarray     # (n,) bool, after anomaly masking
    present: np.ndarray       # (n, S) bool, after anomaly masking

    @property
    def n(self) -> int:
        return len(self.values)

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]


def nan_moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving mean along axis 0 that ignores NaN (NaN only where the window is empty)."""
    if window <= 1:
        return x
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    half = window // 2
    pad = [(half, window - 1 - half)] + [(0, 0)] * (x.ndim - 1)
    cs = np.cumsum(np.pad(filled, pad), axis=0)
    cc = np.cumsum(np.pad(valid.astype(np.float64), pad), axis=0)
    cs = np.concatenate([np.zeros_like(cs[:1]), cs])
    cc = np.concatenate([np.zeros_like(cc[:1]), cc])
    n = len(x)
    s = cs[window:window + n] - cs[:n]
    c = cc[window:window + n] - cc[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out.astype(np.float32)


def central_velocity(p: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference velocity along axis 0 (NaN where a neighbour is missing)."""
    v = np.full_like(p, np.nan, dtype=np.float32)
    if len(p) >= 3:
        v[1:-1] = (p[2:] - p[:-2]) * (fps / 2.0)
    if len(p) >= 2:
        v[0] = (p[1] - p[0]) * fps
        v[-1] = (p[-1] - p[-2]) * fps
    return v


def _nanmean_slots(x: np.ndarray, mask: np.ndarray, min_count: int = 1) -> np.ndarray:
    """Mean over slots of x (n, S[, 2]) where mask (n, S) is True and x is finite; NaN below
    min_count contributing slots."""
    finite = np.isfinite(x) if x.ndim == 2 else np.isfinite(x).all(axis=2)
    valid = mask & finite
    count = valid.sum(axis=1)
    sel = valid if x.ndim == 2 else valid[..., None]
    total = np.where(sel, x, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / (count if x.ndim == 2 else count[:, None])
    mean[count < min_count] = np.nan
    return mean.astype(np.float32)


def attack_direction(x: np.ndarray, is_h: np.ndarray, is_a: np.ndarray, present: np.ndarray,
                     window: int, min_players: int = 3) -> np.ndarray:
    """+1 where H attacks towards x = 1, -1 otherwise, per frame.

    The sign of (mean x of A) - (mean x of H) over a centred rolling window: a team's outfield
    block sits on its own half far more often than not, so the opponent's centroid points at
    the goal it attacks. Flips at halftime by itself. Frames the window cannot decide (nobody
    tracked) take the nearest decided value, and +1 when there is none.
    """
    mean_h = _nanmean_slots(x, present & is_h[None, :], min_players)
    mean_a = _nanmean_slots(x, present & is_a[None, :], min_players)
    gap = nan_moving_mean((mean_a - mean_h)[:, None], window)[:, 0]
    direction = np.where(gap >= 0, 1.0, -1.0).astype(np.float32)
    undecided = ~np.isfinite(gap)
    if undecided.any() and not undecided.all():
        idx = np.flatnonzero(~undecided)
        nearest = idx[np.clip(np.searchsorted(idx, np.flatnonzero(undecided)), 0, len(idx) - 1)]
        direction[undecided] = direction[nearest]
    return direction


def compute_features(track: Tracking, team: np.ndarray | None = None,
                     ball_mask: np.ndarray | None = None, player_mask: np.ndarray | None = None,
                     direction_window_s: float = 60.0, smooth_s: float = 0.2,
                     soft_radius_m: float = 3.0, converge_radius_m: float = 15.0,
                     press_weight_s: float = 0.45, still_weight_s: float = 0.35,
                     lookahead_s: float = 1.0, flight_min_m: float = 6.0,
                     flight_speed_m_s: float = 6.0) -> Features:
    """Features for every frame of `track`.

    team         (S,) override of track.team (after goalkeeper relabelling)
    ball_mask    (n,) frames whose ball position is to be ignored (anomalies)
    player_mask  (n, S) slot positions to be ignored (anomalies)
    """
    n, S = track.n, track.n_slots
    fps = float(track.fps)
    pitch = np.asarray(track.pitch, dtype=np.float32)
    length = float(pitch[0])
    team = np.asarray(track.team if team is None else team, dtype="<U1")
    is_h, is_a = team == "H", team == "A"
    is_team = is_h | is_a

    xy = track.xy.astype(np.float32) * pitch
    ball = track.ball.astype(np.float32) * pitch
    present = np.isfinite(xy).all(axis=2) & is_team[None, :]
    if player_mask is not None:
        present &= ~player_mask
    ball_seen = np.isfinite(ball).all(axis=1)
    if ball_mask is not None:
        ball_seen &= ~ball_mask
    xy = np.where(present[..., None], xy, np.nan)
    ball = np.where(ball_seen[:, None], ball, np.nan)

    window = max(1, int(round(direction_window_s * fps)))
    direction = attack_direction(xy[..., 0], is_h, is_a, present, window)
    d_sign = direction[:, None]

    v_ball = central_velocity(ball, fps)                          # (n, 2)
    speed = np.hypot(v_ball[:, 0], v_ball[:, 1])
    a_ball = central_velocity(v_ball, fps)
    accel = np.hypot(a_ball[:, 0], a_ball[:, 1])
    one_s = max(1, int(round(fps)))
    disp_1s = np.full(n, np.nan, dtype=np.float32)
    if n > one_s:
        disp_1s[one_s:] = np.hypot(*(ball[one_s:] - ball[:-one_s]).T)
    v_xy = central_velocity(xy, fps)                              # (n, S, 2)
    p_speed = np.hypot(v_xy[..., 0], v_xy[..., 1])

    diff = ball[:, None, :] - xy                                  # (n, S, 2) player -> ball
    dist = np.hypot(diff[..., 0], diff[..., 1])                   # NaN if either missing
    with np.errstate(invalid="ignore", divide="ignore"):
        approach = (v_xy * diff).sum(axis=2) / dist               # m/s towards the ball
    # On top of the ball, direction to ball is undefined — treat as not pressing.
    approach = np.where(dist < 0.05, 0.0, approach)
    weight = np.exp(-dist / converge_radius_m)
    soft = np.exp(-dist / soft_radius_m)

    def team_min(mask):
        d = np.where(mask, dist, np.inf)
        m = d.min(axis=1)
        return np.where(np.isfinite(m), m, np.nan)

    def team_nearest_approach(mask):
        """Approach speed (m/s toward ball) of that team's nearest player; NaN if none."""
        d = np.where(mask, dist, np.inf)
        idx = np.argmin(d, axis=1)
        app = approach[np.arange(n), idx]
        return np.where(np.isfinite(d[np.arange(n), idx]), app, np.nan)

    def team_weighted(values, mask):
        w = np.where(mask & np.isfinite(values), weight, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(np.where(w > 0, values * w, 0.0), axis=1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)

    ph, pa = present & is_h[None, :], present & is_a[None, :]
    d_h, d_a = team_min(ph), team_min(pa)
    app_h, app_a = team_nearest_approach(ph), team_nearest_approach(pa)
    cen_h, cen_a = _nanmean_slots(xy, ph, 2), _nanmean_slots(xy, pa, 2)

    def spread(mask, cen):
        r2 = ((xy - cen[:, None, :]) ** 2).sum(axis=2)
        return np.sqrt(_nanmean_slots(r2, mask, 2))

    sp_h, sp_a = spread(ph, cen_h), spread(pa, cen_a)

    def defensive_line(mask, sign):
        """Second-deepest player along the team's own attacking direction (skips the keeper)."""
        c = np.where(mask, (xy[..., 0] - length / 2) * sign, np.inf)
        c.sort(axis=1)
        line = c[:, 1] if S > 1 else np.full(n, np.inf)
        return np.where(mask.sum(1) >= 3, line, np.nan)

    line_h = defensive_line(ph, d_sign)
    line_a = defensive_line(pa, -d_sign)

    cols = {}
    cols["near_gap"] = np.clip(d_a - d_h, -20, 20)
    cols["near_soft"] = (np.nansum(np.where(ph, soft, 0.0), 1) - np.nansum(np.where(pa, soft, 0.0), 1))
    cols["near_soft"][~ball_seen] = np.nan
    # control = -distance - press_weight * relu(approach_of_nearest).
    # Gap H−A ≡ (d_a − d_h) + press_weight * (relu(app_a) − relu(app_h)).
    press_h = np.clip(app_h, 0.0, 12.0)
    press_a = np.clip(app_a, 0.0, 12.0)
    control_gap = (d_a - d_h) + float(press_weight_s) * (press_a - press_h)
    cols["control_gap"] = np.clip(control_gap, -25, 25)
    cols["control_gap"][~ball_seen | ~np.isfinite(d_h) | ~np.isfinite(d_a)] = np.nan

    # --- Pass / flight lookahead ---------------------------------------------------------
    # Only while the ball is clearly travelling to a future point we can see. Otherwise NaN
    # so the possession axis falls back to control_gap (nanmean skips these).
    horizon = max(1, int(round(float(lookahead_s) * fps)))
    recv_gap = np.full(n, np.nan, dtype=np.float32)
    lead_gap = np.full(n, np.nan, dtype=np.float32)
    if n > horizon:
        both = ball_seen[:-horizon] & ball_seen[horizon:]
        disp = np.hypot(*(ball[horizon:] - ball[:-horizon]).T)
        fast = np.isfinite(speed[:-horizon]) & (speed[:-horizon] >= float(flight_speed_m_s))
        flying = both & np.isfinite(disp) & (disp >= float(flight_min_m)) & fast
        # Reception: control_gap when the ball arrives (future holders / pressers).
        future_ctrl = cols["control_gap"][horizon:]
        recv_gap[:-horizon] = np.where(flying & np.isfinite(future_ctrl), future_ctrl, np.nan)
        # Lead: who is nearer the *landing spot* right now (intended receiver vs chase pack).
        land = ball[horizon:]
        lead_diff = land[:, None, :] - xy[:-horizon]
        lead_dist = np.hypot(lead_diff[..., 0], lead_diff[..., 1])
        lead_ph = present[:-horizon] & is_h[None, :]
        lead_pa = present[:-horizon] & is_a[None, :]
        lead_d_h = np.where(lead_ph, lead_dist, np.inf).min(axis=1)
        lead_d_a = np.where(lead_pa, lead_dist, np.inf).min(axis=1)
        lead = lead_d_a - lead_d_h
        lead = np.where(np.isfinite(lead_d_h) & np.isfinite(lead_d_a), lead, np.nan)
        # Also credit players already moving toward the landing spot (not toward the ball now).
        v_now = v_xy[:-horizon]
        with np.errstate(invalid="ignore", divide="ignore"):
            lead_app = (v_now * lead_diff).sum(axis=2) / lead_dist
        lead_app = np.where(lead_dist < 0.05, 0.0, lead_app)

        def _nearest_val(mask, dist_m, values):
            d = np.where(mask, dist_m, np.inf)
            idx = np.argmin(d, axis=1)
            rows = np.arange(len(d))
            return np.where(np.isfinite(d[rows, idx]), values[rows, idx], np.nan)

        la_h = _nearest_val(lead_ph, lead_dist, lead_app)
        la_a = _nearest_val(lead_pa, lead_dist, lead_app)
        # Still receiver: a player waiting on the landing spot (low speed) beats a sprinter
        # chasing the ball who happens to be nearer the landing line.
        spd_now = p_speed[:-horizon]
        ls_h = _nearest_val(lead_ph, lead_dist, spd_now)
        ls_a = _nearest_val(lead_pa, lead_dist, spd_now)
        lead_ctrl = (lead
                     + float(press_weight_s) * (np.clip(la_a, 0, 12) - np.clip(la_h, 0, 12))
                     + float(still_weight_s) * (np.clip(ls_a, 0, 12) - np.clip(ls_h, 0, 12)))
        lead_gap[:-horizon] = np.where(flying & np.isfinite(lead_ctrl),
                                       np.clip(lead_ctrl, -25, 25), np.nan)
    cols["recv_gap"] = recv_gap
    cols["lead_gap"] = lead_gap

    cols["converge"] = team_weighted(approach, ph) - team_weighted(approach, pa)
    cols["ball_motion"] = v_ball[:, 0] * direction
    cols["ball_territory"] = (ball[:, 0] - length / 2) * direction
    cols["territory"] = ((cen_h[:, 0] + cen_a[:, 0]) / 2 - length / 2) * direction
    cols["line_diff"] = line_h - line_a
    cols["spread_diff"] = sp_h - sp_a
    cols["team_motion"] = _nanmean_slots(v_xy[..., 0], present, 4) * direction
    cols["ball_speed"] = np.log1p(np.clip(speed, 0, 60))
    cols["ball_accel"] = np.log1p(np.clip(accel, 0, 400))
    cols["ball_speed_1s"] = np.log1p(np.clip(disp_1s, 0, 60))
    cols["near_min"] = np.clip(np.fmin(d_h, d_a), 0, 40)
    cols["converge_all"] = team_weighted(approach, present)
    off = (track.ball < -0.002) | (track.ball > 1.002)
    cols["ball_off_pitch"] = np.where(ball_seen, off.any(axis=1).astype(np.float32), np.nan)
    cols["ball_missing"] = (~ball_seen).astype(np.float32)
    cols["players_moving"] = _nanmean_slots(p_speed, present, 4)
    cols["n_present"] = present.sum(1) / 22.0
    cols["spread_sum"] = sp_h + sp_a
    cols["centroid_gap"] = np.hypot(*(cen_h - cen_a).T)

    values = np.stack([np.asarray(cols[name], dtype=np.float32) for name in NAMES], axis=1)
    smooth = int(round(smooth_s * fps))
    if smooth > 1:
        # Keep flight-gated NaNs sharp — smoothing would bleed pass cues into ground play.
        indicator = np.isin(NAMES, ("ball_off_pitch", "ball_missing", "recv_gap", "lead_gap"))
        values[:, ~indicator] = nan_moving_mean(values[:, ~indicator], smooth)
    return Features(values, NAMES, ANTI.copy(), direction, ball_seen, present)


def mirror(values: np.ndarray, anti: np.ndarray) -> np.ndarray:
    """The same frames with H and A swapped: anti-symmetric features negated."""
    out = np.array(values, copy=True)
    out[:, anti] *= -1
    return out

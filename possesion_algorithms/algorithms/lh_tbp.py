"""Paper-faithful Team Ball Possession (TBP) after Link & Hoernig (2017).

PLOS ONE 12(7): e0179953. https://doi.org/10.1371/journal.pone.0179953

This is a fresh TBP port, separate from the pragmatic ``lh`` heuristic and from ``lh_v2``.

Core rules used here (paper §§3.2–3.5, best IBP/IBA config):

* **IBP start (kick-gated):** a player starts possession on a local peak of ball *vector*
  acceleration ‖Δv_xy‖·fps ≥ ``kick_accel_m_s2``. If the nearest eligible player is within
  ``t_k``, they take it. If nobody is in that radius, possession is inferred from kick direction:
  the intended recipient is the eligible player best aligned with the outgoing ball velocity
  (preferring the current team — a pass — and, when set, the team's attack half). ``detect_ball_kicks``
  is shared with the match-viewer KICK timeline row.
* **Duel (``r_dueling``):** when two or more eligible players lie within ``r_dueling`` of the ball,
  do not give the ball to the nearest. On a kick, resolve possession from the outgoing trajectory:
  the recipient (preferring among the duelers) owns the team label for the rest of the duel.
* **Bootstrap:** before any team is established, allow a naive nearest-within-``t_p`` start so the
  timeline is not blank until the first kick.
* **IBA / holder clear (ball prediction):** while a holder exists, predict the ball one horizon
  ahead at constant velocity; if that prediction is farther than ``t_a`` from the holder's current
  position, clear the individual holder. Team possession stays sticky until an opposing IBP start.
* **TBP labels:** sticky last holder team through voids and missing ball; ``''`` before any team.

Optional jump masking and a short moving-average smoother are available via params (off by default)
so we can A/B against unsmoothed tracking without committing to a full RTS smoother yet.
"""

from __future__ import annotations

import numpy as np

from possesion_algorithms.base import TACTICAL, PossessionAlgorithm, Tracking

ELIGIBLE = ("H", "A")


def _metric_xy(frac: np.ndarray, pitch: tuple) -> np.ndarray:
    return frac * np.asarray(pitch, dtype=np.float32)


def _ball_velocity_m_s(ball_m: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame ball velocity in m/s; row 0 is NaN; gaps yield NaN."""
    n = len(ball_m)
    vel = np.full((n, 2), np.nan, dtype=np.float32)
    if n < 2:
        return vel
    delta = np.diff(ball_m, axis=0) * float(fps)
    both = np.isfinite(ball_m[:-1]).all(axis=1) & np.isfinite(ball_m[1:]).all(axis=1)
    vel[1:] = np.where(both[:, None], delta, np.nan)
    return vel


def _ball_accel_m_s2(vel_m_s: np.ndarray, fps: float) -> np.ndarray:
    """Magnitude of xy velocity change per second: ‖v[t] − v[t−1]‖ · fps (m/s²).

    Unlike d(|v|)/dt, a pure direction change at constant speed still produces a large value.
    """
    n = len(vel_m_s)
    acc = np.full(n, np.nan, dtype=np.float32)
    if n < 2:
        return acc
    delta = np.diff(vel_m_s, axis=0) * float(fps)
    both = np.isfinite(vel_m_s[:-1]).all(axis=1) & np.isfinite(vel_m_s[1:]).all(axis=1)
    mag = np.hypot(delta[:, 0], delta[:, 1])
    acc[1:] = np.where(both, mag, np.nan)
    return acc


def _kick_peaks(accel_m_s2: np.ndarray, threshold: float, refractory: int) -> np.ndarray:
    """Local maxima of vector acceleration at/above threshold, thinned to one per refractory frames."""
    n = len(accel_m_s2)
    kick = np.zeros(n, bool)
    a = np.where(np.isfinite(accel_m_s2), accel_m_s2, -np.inf)
    for i in range(1, n - 1):
        if a[i] >= threshold and a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            kick[i] = True
    if refractory > 0:
        last = -10**9
        for i in np.flatnonzero(kick):
            if i - last < refractory:
                kick[i] = False
            else:
                last = int(i)
    return kick


def detect_ball_kicks(ball_xy, pitch, fps: float, threshold: float = 4.0,
                      refractory_s: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Kick mask + vector accel used by ``lh_tbp`` and the match-viewer KICK row.

    ``ball_xy`` is (n, 2+) pitch fractions (only x,y are used). Acceleration is ‖Δv_xy‖·fps
    in m/s². Returns ``(kick, accel)`` each shaped (n,).
    """
    ball_m = _metric_xy(np.asarray(ball_xy[..., :2], dtype=np.float32), pitch)
    usable = np.isfinite(ball_m).all(axis=1)
    vel = _ball_velocity_m_s(ball_m, fps)
    vel = np.where(usable[:, None], vel, np.nan)
    accel = _ball_accel_m_s2(vel, fps)
    refractory = int(round(float(refractory_s) * float(fps)))
    kick = _kick_peaks(accel, float(threshold), refractory)
    return kick, accel


def _eligible_within(dist: np.ndarray, team: np.ndarray, present: np.ndarray,
                     radius_m: float) -> list[int]:
    """Eligible H/A slot indices with distance ≤ radius_m."""
    out = []
    for k in range(len(team)):
        if not present[k] or team[k] not in ELIGIBLE:
            continue
        d = float(dist[k])
        if np.isfinite(d) and d <= radius_m:
            out.append(k)
    return out


def _recipient_along_kick(ball_m: np.ndarray, players_m: np.ndarray, present: np.ndarray,
                          team: np.ndarray, kick_dir: np.ndarray, last_team: str,
                          min_cos: float, max_dist_m: float, same_team_bonus: float,
                          attack_sign_h: float, candidates: list[int] | None = None,
                          min_dist_m: float = 0.05) -> int:
    """Eligible player the kick is aimed at, or -1 if none is aligned well enough.

    Scores by cosine of (ball→player, kick_dir), with a bonus for ``last_team`` (pass) and for
    lying toward that team's attack half (H attacks toward +x when ``attack_sign_h`` > 0).
    If ``candidates`` is set, only those slots are considered (duel resolution).
    """
    speed = float(np.hypot(kick_dir[0], kick_dir[1]))
    if speed < 1e-6:
        return -1
    direction = kick_dir / speed
    best_k, best_score = -1, -np.inf
    slots = candidates if candidates is not None else range(len(team))
    for k in slots:
        if not present[k] or team[k] not in ELIGIBLE:
            continue
        dvec = players_m[k] - ball_m
        dist = float(np.hypot(dvec[0], dvec[1]))
        if not np.isfinite(dist) or dist < min_dist_m or dist > max_dist_m:
            continue
        cos = float(np.dot(dvec, direction) / dist)
        if cos < min_cos:
            continue
        score = cos
        if last_team and team[k] == last_team:
            score += same_team_bonus
        if last_team in ELIGIBLE and attack_sign_h != 0.0:
            # Attack along pitch length: H uses +attack_sign_h on x, A the opposite.
            sign = attack_sign_h if last_team == "H" else -attack_sign_h
            toward_attack = float(dvec[0]) * sign
            if toward_attack > 0:
                score += 0.05 * min(toward_attack / max_dist_m, 1.0)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _resolve_kick_recipient(ball_m, players_m, present, team, kick_dir, last_team,
                            duelers: list[int], min_cos, max_dist_m, same_team_bonus,
                            attack_sign_h) -> int:
    """Recipient after a kick: prefer a dueler along the trajectory, else any aligned player."""
    if not np.isfinite(kick_dir).all():
        return -1
    if len(duelers) >= 2:
        # Among duelers, allow very small ball→player distance (both are on the ball).
        recv = _recipient_along_kick(
            ball_m, players_m, present, team, kick_dir, last_team,
            min_cos=min(min_cos, 0.0),  # any forward hemisphere among duelers
            max_dist_m=max_dist_m, same_team_bonus=same_team_bonus,
            attack_sign_h=attack_sign_h, candidates=duelers, min_dist_m=0.0,
        )
        if recv >= 0:
            return recv
        # Ball leaving the duel toward someone else.
        return _recipient_along_kick(
            ball_m, players_m, present, team, kick_dir, last_team,
            min_cos, max_dist_m, same_team_bonus, attack_sign_h, candidates=None,
        )
    return _recipient_along_kick(
        ball_m, players_m, present, team, kick_dir, last_team,
        min_cos, max_dist_m, same_team_bonus, attack_sign_h, candidates=None,
    )


def _moving_average(ball_m: np.ndarray, window: int) -> np.ndarray:
    """Causal-ish centered moving average over finite samples; NaNs preserved where no support."""
    if window <= 1:
        return ball_m.copy()
    n = len(ball_m)
    out = ball_m.copy()
    half = window // 2
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        chunk = ball_m[a:b]
        ok = np.isfinite(chunk).all(axis=1)
        if ok.any():
            out[i] = chunk[ok].mean(axis=0)
        else:
            out[i] = np.nan
    return out.astype(np.float32)


def _jump_mask(ball_m: np.ndarray, max_step_m: float) -> np.ndarray:
    """True where the ball is usable: finite and not a teleport vs the previous usable frame."""
    n = len(ball_m)
    usable = np.isfinite(ball_m).all(axis=1)
    if n == 0 or max_step_m <= 0:
        return usable
    last = -1
    for i in range(n):
        if not usable[i]:
            continue
        if last >= 0 and np.hypot(*(ball_m[i] - ball_m[last])) > max_step_m:
            usable[i] = False
            continue
        last = i
    return usable


def _drop_short_segments(usable: np.ndarray, min_frames: int) -> np.ndarray:
    """Clear usable runs shorter than min_frames (paper drops sections < 1 s)."""
    if min_frames <= 1 or not usable.any():
        return usable
    out = usable.copy()
    n = len(usable)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        if j - i < min_frames:
            out[i:j] = False
        i = j
    return out


def _nearest_eligible(dist: np.ndarray, team: np.ndarray, present: np.ndarray) -> tuple[int, float]:
    """Nearest H/A slot that is present; (-1, inf) if none."""
    best_k, best_d = -1, np.inf
    for k in range(len(team)):
        if not present[k] or team[k] not in ELIGIBLE:
            continue
        d = float(dist[k])
        if d < best_d:
            best_d, best_k = d, k
    return best_k, best_d


class LinkHoernigTBP(PossessionAlgorithm):
    name = "lh_tbp"
    description = (
        "Link & Hoernig TBP (paper): kick-gated IBP starts, ball-prediction holder clear, "
        "sticky team; whole video, tactical"
    )
    dead_ball = False
    source = TACTICAL
    params = {
        "t_p": 1.0,
        "t_k": 1.0,
        "t_a": 2.5,
        "horizon_s": 1.0,
        "start_mode": "kick",          # "kick" (paper-best) or "naive"
        "kick_accel_m_s2": 4.0,        # ‖Δv_xy‖·fps peak threshold for a kick
        "kick_refractory_s": 0.3,      # min gap between accepted kick peaks
        "recipient_min_cos": 0.75,     # min alignment ball→player vs kick dir (out-of-radius)
        "recipient_max_m": 40.0,       # ignore candidates farther than this
        "recipient_same_team_bonus": 0.15,
        "attack_sign_h": 1.0,          # H attacks +x; 0 disables the attack-half bonus
        "r_dueling": 2.0,              # ≥2 eligible players inside → duel; resolve via kick trajectory
        "smooth": False,
        "smooth_window": 5,
        "jump_max_step_m": 0.0,        # 0 = disabled; e.g. 8.0 m/frame to mask teleports
        "min_segment_s": 0.0,          # 0 = disabled; paper uses ~1.0
    }

    def label(self, track: Tracking) -> np.ndarray:
        n = track.n
        out = np.full(n, "", dtype="<U1")
        if n == 0 or track.n_slots == 0:
            return out

        t_p = float(self.params["t_p"])
        t_k = float(self.params["t_k"])
        t_a = float(self.params["t_a"])
        horizon_s = float(self.params["horizon_s"])
        kick_thresh = float(self.params["kick_accel_m_s2"])
        kick_refractory_s = float(self.params["kick_refractory_s"])
        recipient_min_cos = float(self.params["recipient_min_cos"])
        recipient_max_m = float(self.params["recipient_max_m"])
        same_team_bonus = float(self.params["recipient_same_team_bonus"])
        attack_sign_h = float(self.params["attack_sign_h"])
        r_dueling = float(self.params["r_dueling"])
        start_mode = str(self.params["start_mode"])
        if start_mode not in ("kick", "naive"):
            raise ValueError(f"{self.name}: start_mode must be 'kick' or 'naive', got {start_mode!r}")

        pitch = track.pitch
        ball_m = _metric_xy(track.ball, pitch)
        xy_m = _metric_xy(track.xy, pitch)

        usable = np.isfinite(ball_m).all(axis=1)
        jump_max = float(self.params["jump_max_step_m"])
        if jump_max > 0:
            usable &= _jump_mask(ball_m, jump_max)
        min_seg = int(round(float(self.params["min_segment_s"]) * track.fps))
        if min_seg > 1:
            usable = _drop_short_segments(usable, min_seg)

        if bool(self.params["smooth"]):
            window = max(1, int(self.params["smooth_window"]))
            # Smooth only on usable samples; leave unusable as NaN for the loop.
            smoothed = _moving_average(np.where(usable[:, None], ball_m, np.nan), window)
            ball_m = np.where(usable[:, None], smoothed, np.nan)

        ball_frac = np.where(usable[:, None], ball_m / np.asarray(pitch, dtype=np.float32), np.nan)
        kicks, _accel = detect_ball_kicks(ball_frac, pitch, track.fps, kick_thresh, kick_refractory_s)
        vel = _ball_velocity_m_s(ball_m, track.fps)
        vel = np.where(usable[:, None], vel, np.nan)

        present = np.isfinite(xy_m).all(axis=2)
        team = track.team

        # Distances from (possibly smoothed) ball to players.
        dist = np.full((n, track.n_slots), np.inf, dtype=np.float32)
        for r in range(n):
            if not usable[r]:
                continue
            delta = xy_m[r] - ball_m[r]
            d = np.hypot(delta[:, 0], delta[:, 1])
            dist[r] = np.where(present[r], d, np.inf)

        holder, last_team = -1, ""
        for r in range(n):
            if not usable[r]:
                holder = -1
                out[r] = last_team
                continue

            # Clear holder if slot vanished or ball prediction puts the ball out of reach.
            if holder >= 0:
                if not present[r, holder]:
                    holder = -1
                else:
                    v = vel[r]
                    if np.isfinite(v).all():
                        b_hat = ball_m[r] + v * horizon_s
                    else:
                        b_hat = ball_m[r]
                    if np.hypot(*(b_hat - xy_m[r, holder])) > t_a:
                        holder = -1

            k, dk = _nearest_eligible(dist[r], team, present[r])
            duelers = _eligible_within(dist[r], team, present[r], r_dueling)
            in_duel = len(duelers) >= 2

            if start_mode == "naive":
                # Naive physical: gain when no holder and nearest within t_p.
                if holder < 0 and k >= 0 and dk <= t_p:
                    holder = k
            else:
                # Kick-gated IBP starts. Bootstrap with naive until a team exists.
                if last_team == "":
                    if holder < 0 and k >= 0 and dk <= t_p:
                        holder = k
                elif kicks[r]:
                    if in_duel:
                        # Two (or more) on the ball: trajectory decides the recipient's team.
                        recv = _resolve_kick_recipient(
                            ball_m[r], xy_m[r], present[r], team, vel[r], last_team, duelers,
                            recipient_min_cos, recipient_max_m, same_team_bonus, attack_sign_h,
                        )
                        if recv >= 0:
                            holder = recv
                    elif k >= 0 and dk <= t_k:
                        # Touch / reception in the interaction radius.
                        holder = k
                    else:
                        # Nobody on the ball: aim along kick direction at a recipient.
                        recv = _resolve_kick_recipient(
                            ball_m[r], xy_m[r], present[r], team, vel[r], last_team, [],
                            recipient_min_cos, recipient_max_m, same_team_bonus, attack_sign_h,
                        )
                        if recv >= 0:
                            holder = recv
                elif in_duel:
                    # Hold the duel-resolved team; do not flip to whichever dueler is nearest.
                    pass
                elif holder < 0 and k >= 0 and dk <= t_p and str(team[k]) == last_team:
                    # Same-team re-gain without a detected kick (dribble / soft touch).
                    holder = k

            if holder >= 0:
                last_team = str(team[holder])
            out[r] = last_team
        return out


class LinkHoernigTBPDeadBall(LinkHoernigTBP):
    name = "lh_tbp_dead"
    description = (
        "lh_tbp on each alive interval of dead_ball.json on its own, D on dead balls; tactical"
    )
    dead_ball = True

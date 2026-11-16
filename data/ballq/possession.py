"""Individual ball possession after Link and Hoernig (2017), PLoS ONE 12(7): e0179953.

Their rules, as the paper states them:
    start  the ball is inside the player's sphere of influence (T_P) and he is the nearest player;
           with kick detection a ball action is required as well: a local maximum of the ball's xy
           acceleration of at least 4 m/s2 while the ball is near him (T_K). That filter is what keeps a
           ball flying over a player's head, or rolling past him, from starting a possession.
    end    the ball position predicted one second ahead (constant velocity) is outside T_A of the
           player for one second, or another player starts a possession, or the game is interrupted.
Here the start takes either a kick within tp or control: the ball moves with the player (relative
speed below ctrl_rel_speed_ms) and is still predicted within ta a second ahead. That covers the
trap and the dribble, which have no 4 m/s2 spike. A ball that only passes by fails both.
The posterior form replaces each yes/no with the posterior mass inside tp.
The team stays with the last holder while nobody holds: a loose ball is still the last team's.
"""
import numpy as np


def _team_sticky(n, holder, team_of):
    team = np.array([""] * n, dtype=object); last = ""
    for r in range(n):
        if holder[r] >= 0:
            last = team_of[holder[r]]
        team[r] = last
    return team


def _kick_near(kicks, r, w):
    return any(k in kicks for k in range(r - w, r + w + 1))


def _vel(track, r, pitch, fps):
    """velocity in m/s at frame r from the frame before (the frame after on the first frame); None when missing"""
    n = len(track)
    for a, b in ((r - 1, r), (r, r + 1)):
        if 0 <= a and b < n and np.all(np.isfinite(track[a])) and np.all(np.isfinite(track[b])):
            return (track[b] - track[a]) * np.array(pitch) * fps
    return None


def lh_point(xy, ball, team_of, pitch, fps, pc, kicks):
    n, S = xy.shape[:2]
    holder = np.full(n, -1, int); cur = -1; h = int(pc["horizon_s"] * fps)
    w = int(round(pc.get("kick_window_s", 0.2) * fps)); end_hold = int(round(pc.get("end_hold_s", 1.0) * fps)); far = 0
    vrel_max = pc.get("ctrl_rel_speed_ms", 2.0)
    pm = np.array(pitch)
    for r in range(n):
        if not np.all(np.isfinite(ball[r])):
            cur = -1; far = 0; continue
        d = np.hypot((xy[r, :, 0] - ball[r, 0]) * pitch[0], (xy[r, :, 1] - ball[r, 1]) * pitch[1])
        d = np.where(np.isfinite(d), d, np.inf)
        best = int(np.argmin(d)); bd = d[best]

        vb = _vel(ball, r, pitch, fps)

        def ahead(k):
            """distance in metres between the ball predicted a second ahead and player k, now"""
            if vb is None:
                return np.inf
            pred = ball[r] * pm + vb * pc["horizon_s"]
            return float(np.hypot(pred[0] - xy[r, k, 0] * pitch[0], pred[1] - xy[r, k, 1] * pitch[1]))

        def gains(k):
            if d[k] > pc["tp_m"]:
                return False
            if _kick_near(kicks, r, w):
                return True
            vp = _vel(xy[:, k], r, pitch, fps)
            if vp is None or vb is None:
                return False
            return float(np.hypot(*(vb - vp))) <= vrel_max and ahead(k) <= pc["ta_m"]

        if cur >= 0 and np.isfinite(xy[r, cur, 0]):
            far = far + 1 if ahead(cur) > pc["ta_m"] else 0
            taken = best != cur and gains(best)
            # a kick that sends the ball out of reach ends the action at once; otherwise the hold lasts end_hold frames
            if far >= (1 if _kick_near(kicks, r, w) else end_hold) or taken:
                cur = -1; far = 0
        else:
            cur = -1; far = 0
        if cur < 0 and gains(best):
            cur = best
        holder[r] = cur
    return {"holder": holder, "team": _team_sticky(n, holder, team_of)}


def lh_posterior(xy, mean, mass_tp, team_of, pitch, fps, pc, kicks):
    """mean: (n, 2) posterior mean; mass_tp: (n, S) posterior mass within tp of each slot.
    Same rules with the mass inside tp in place of the distance test."""
    n, S = xy.shape[:2]
    holder = np.full(n, -1, int); cur = -1; h = int(pc["horizon_s"] * fps)
    w = int(round(pc.get("kick_window_s", 0.2) * fps)); end_hold = int(round(pc.get("end_hold_s", 1.0) * fps)); far = 0
    vrel_max = pc.get("ctrl_rel_speed_ms", 2.0)
    pm = np.array(pitch)
    for r in range(n):
        m = np.where(np.isfinite(xy[r, :, 0]), mass_tp[r], -1.0)
        best = int(np.argmax(m)); bm = m[best]

        vb = _vel(mean, r, pitch, fps)

        def ahead(k):
            if vb is None:
                return np.inf
            pred = mean[r] * pm + vb * pc["horizon_s"]
            return float(np.hypot(pred[0] - xy[r, k, 0] * pitch[0], pred[1] - xy[r, k, 1] * pitch[1]))

        def gains(k):
            if m[k] < pc["mass_gain"]:
                return False
            if _kick_near(kicks, r, w):
                return True
            vp = _vel(xy[:, k], r, pitch, fps)
            if vp is None or vb is None:
                return False
            return float(np.hypot(*(vb - vp))) <= vrel_max and ahead(k) <= pc["ta_m"]

        if cur >= 0 and np.isfinite(xy[r, cur, 0]):
            far = far + 1 if (ahead(cur) > pc["ta_m"] or m[cur] < pc["mass_hold"]) else 0
            taken = best != cur and gains(best)
            if far >= (1 if _kick_near(kicks, r, w) else end_hold) or taken:
                cur = -1; far = 0
        else:
            cur = -1; far = 0
        if cur < 0 and gains(best):
            cur = best
        holder[r] = cur
    return {"holder": holder, "team": _team_sticky(n, holder, team_of)}

"""The control model: where the ball is and who has it, from tracking alone.

Three published ideas, one rule each:

  Maksai, Wang and Fua (CVPR 2016)   the ball is held, free or absent, and its state can only change
                                     through interaction with a player
  Vidal-Codina et al. (2022, FIFA)   the ball is "at" a player inside a possession zone; a run of such
                                     frames is a touch only if the ball's velocity changes across it,
                                     otherwise the ball merely passed by and the run is dropped
  Link and Hoernig (2017)            the velocity change is a ball action (kick); possession stays
                                     with the player until someone else acts on the ball

So: smooth the ball, find the runs of frames where it sits in one player's zone, keep the runs that
are touches, and everything else follows. Between touches the ball is free and nothing changes,
however many players it passes and whatever the homography does to a ball in the air. When the ball
is not seen, the state it had is kept. After a dead ball the taker does not count: possession goes
to the first other player who touches the ball, from the restart on.

Functions, in the order they run:
  smooth_ball      drop teleports, Savitzky-Golay on the seen stretches
  zone_runs        runs of frames with the ball in one player's possession zone
  score_runs       velocity change and duration of each run -> confidence that it is a touch
  build_states     per frame: dead / held / free, the holder, and the possession team
  flight_chords    between two touches, a track that bows off the straight line is a ball in the air
  ball_track       per frame: position, spread and where the position came from
  run_control      all of the above, returning the three tables
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

TOUCH_COLUMNS = ["frame_start", "frame_end", "slot", "team", "duration_s", "dv_ms", "duel", "p", "kind"]
POSSESSION_COLUMNS = ["frame", "state", "holder_slot", "team", "p_team", "p_home", "p_away", "p_dead", "event", "restart_zone", "label"]
TRACK_COLUMNS = ["frame", "x", "y", "sd_m", "source", "raw_x", "raw_y"]


def _segments(mask):
    """[(start, end_exclusive)] of the True stretches of a boolean array"""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def smooth_ball(ball, pitch, fps, speed_cap_ms, window=7, order=2):
    """ball: (n, 2) pitch units, NaN when not seen. Returns metres (n, 2), NaN where unseen or dropped.
    A point the ball could not have reached at the speed cap is dropped."""
    m = ball.astype(float) * np.array(pitch)
    seen = np.isfinite(m[:, 0])
    # a point is kept only if the ball could have reached it from the last kept point at the speed cap.
    # This is a teleport gate (the detector locking onto another white blob), the validation gate every
    # tracker uses; it does not and cannot catch a ball in the air, whose projected track is smooth.
    last = -1
    for r in (np.flatnonzero(seen) if speed_cap_ms else []):
        if last >= 0 and np.hypot(*(m[r] - m[last])) > speed_cap_ms * (r - last) / fps:
            m[r] = np.nan; seen[r] = False
        else:
            last = r
    for a, b in _segments(seen):
        if b - a >= window:
            m[a:b] = savgol_filter(m[a:b], window, order, axis=0)
    return m


def velocity(ball_m, fps):
    """m/s per frame by central difference, NaN where a neighbour is missing"""
    v = np.full_like(ball_m, np.nan)
    v[1:-1] = (ball_m[2:] - ball_m[:-2]) * fps / 2.0
    return v


def zone_runs(ball_m, xy_m, team, dead, ccfg):
    """Runs of frames where the ball is inside one player's possession zone.
    Returns a list of dicts {f0, f1 (inclusive), k, duel}; runs of the same player separated by a
    short gap with nobody else in between are one run."""
    n = len(ball_m)
    d = np.hypot(xy_m[:, :, 0] - ball_m[:, None, 0], xy_m[:, :, 1] - ball_m[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    k = d.argmin(axis=1); dk = d[np.arange(n), k]
    at = (dk <= ccfg["r_pz_m"]) & ~dead
    opp = team[None, :] != team[k][:, None]
    duel = ((d <= ccfg["r_dz_m"]) & opp).any(axis=1)
    runs = []
    for a, b in _segments(at):
        s = a
        for i in range(a + 1, b + 1):
            if i == b or k[i] != k[s]:
                runs.append({"f0": int(s), "f1": int(i - 1), "k": int(k[s]), "duel": bool(duel[s:i].mean() > 0.5)}); s = i
    merged = []
    for r in runs:
        p = merged[-1] if merged else None
        if p and p["k"] == r["k"] and r["f0"] - p["f1"] - 1 <= ccfg["merge_gap_frames"] and not dead[p["f1"]:r["f0"]].any():
            p["f1"] = r["f1"]; p["duel"] = p["duel"] or r["duel"]
        else:
            merged.append(dict(r))
    return merged


def _mean_velocity(vel, frames):
    v = vel[frames]; v = v[np.isfinite(v[:, 0])]
    return v.mean(axis=0) if len(v) else None


def score_runs(runs, vel, dead, fps, ccfg):
    """Adds dv_ms, duration_s and p (confidence that the player acted on the ball) to each run.
    The velocity before and after is taken from the nearest seen frames within vel_search_s, so a ball
    that goes into a player unseen and comes out moving differently still shows its change."""
    n = len(vel); w = ccfg["vel_window_frames"]; look = int(ccfg["vel_search_s"] * fps)
    for r in runs:
        before = [f for f in range(r["f0"] - 1, max(-1, r["f0"] - 1 - look), -1) if not dead[f] and np.isfinite(vel[f, 0])][:w]
        after = [f for f in range(r["f1"] + 1, min(n, r["f1"] + 1 + look)) if not dead[f] and np.isfinite(vel[f, 0])][:w]
        vin = _mean_velocity(vel, before) if before else None
        vout = _mean_velocity(vel, after) if after else None
        # a ball that was not moving before (restart, start of data) or after (trapped) has velocity zero there
        if vin is None:
            vin = np.zeros(2)
        if vout is None:
            vout = np.zeros(2)
        r["dv_ms"] = float(np.hypot(*(vout - vin)))
        r["duration_s"] = (r["f1"] - r["f0"] + 1) / fps
        p_kin = 1 - 0.5 ** ((r["dv_ms"] / ccfg["dv_touch_ms"]) ** 2)        # 0.5 at the threshold
        p_ctl = 1 - 0.5 ** (r["duration_s"] / ccfg["ctrl_touch_s"])          # 0.5 at the threshold
        r["p"] = float((1 - (1 - p_kin) * (1 - p_ctl)) * (ccfg["duel_factor"] if r["duel"] else 1.0))
    return runs


def restart_zone(x, y, pitch):
    """where on the pitch the ball is when play resumes; a description, not an input to anything"""
    if not (np.isfinite(x) and np.isfinite(y)):
        return "unseen"
    L, W = pitch; ex, ey = min(x, L - x), min(y, W - y)
    if ex < 3 and ey < 3:
        return "corner"
    if ey < 2:
        return "touchline"
    if ex < 7 and abs(y - W / 2) < 11:
        return "goal_area"
    if abs(x - L / 2) < 2 and abs(y - W / 2) < 2:
        return "centre"
    return "other"


def build_states(n, touches, seen, dead, team, ball_m, pitch, fps, ccfg):
    """Per frame state (dead / held / free), holder, owner of the possession and its confidence.
    touches: the accepted runs in time order; each gets kind = taker / gain / keep.
    After a dead ball the run already at the ball when play resumes is the taker and does not count;
    the first touch by anyone else owns the ball from the restart frame on."""
    state = np.where(dead, "dead", "free").astype(object)
    holder = np.full(n, -1, int); owner = np.full(n, -1, int); conf = np.zeros(n)
    event = np.full(n, "", dtype=object); zone = np.full(n, "", dtype=object)
    segs = _segments(dead)
    dead_starts = np.array([a for a, _ in segs], dtype=int)
    resume = [b for _, b in segs if b < n]                                  # first live frame after each dead ball
    for b in resume:
        event[b] = "restart"; zone[b] = restart_zone(ball_m[b, 0], ball_m[b, 1], pitch)
    win = int(ccfg["taker_window_s"] * fps)
    cur, cur_conf, pending, taker, taker_decided, ri = -1, 0.0, None, -1, True, 0
    for t in touches:
        while ri < len(resume) and resume[ri] <= t["f0"]:                   # a dead ball ends every ownership
            cur, cur_conf, pending, taker, taker_decided = -1, 0.0, resume[ri], -1, False
            ri += 1
        holder[t["f0"]:t["f1"] + 1] = t["k"]; state[t["f0"]:t["f1"] + 1] = "held"
        if pending is not None and not taker_decided:
            taker_decided = True
            if t["f0"] - pending <= win:
                taker = t["k"]
        if pending is not None and t["k"] == taker:
            t["kind"] = "taker"; continue
        same_team = cur >= 0 and team[t["k"]] == team[cur]
        t["kind"] = "keep" if same_team else "gain"
        cur_conf = 1 - (1 - cur_conf) * (1 - t["p"]) if same_team else t["p"]
        cur = t["k"]
        a = pending if pending is not None else t["f0"]                      # the receiver owns the ball from the restart on
        pending = None
        j = np.searchsorted(dead_starts, a)
        end = int(dead_starts[j]) if j < len(dead_starts) else n            # up to the next dead ball
        owner[a:end] = cur; conf[a:end] = cur_conf
        if t["kind"] == "gain":
            event[t["f0"]] = "gain"
    holder[dead] = -1; state[dead] = "dead"
    # while the ball is not seen the last state is kept: a ball that vanished at a player's feet is still his
    last_state, last_holder = "free", -1
    for r in range(n):
        if dead[r]:
            last_state, last_holder = "free", -1
        elif seen[r]:
            last_state, last_holder = state[r], holder[r]
        elif last_state == "held":
            state[r] = "held"; holder[r] = last_holder
    return state, holder, owner, conf, event, zone


def flight_chords(ball_m, seen, touches, dead, fps, ccfg):
    """A ball in the air is projected onto the ground along the camera ray, so its track bows away
    from the camera and comes back at landing, smoothly, with no jump to detect. But a ball in flight
    moves in a vertical plane at constant horizontal speed, so on the ground it really travels the
    straight line from where it was struck to where it is next touched, at an even pace. Between two
    touches, when the seen track bows more than flight_dev_m from that line, the line is used instead.
    Returns (mask of frames replaced, positions in metres)."""
    n = len(ball_m); mask = np.zeros(n, bool); pos = np.full((n, 2), np.nan)
    for a, b in zip(touches[:-1], touches[1:]):
        f0, f1 = a["f1"], b["f0"]
        if f1 - f0 < 3 or dead[f0:f1 + 1].any() or not (seen[f0] and seen[f1]):
            continue
        p0, p1 = ball_m[f0], ball_m[f1]; chord = p1 - p0; L = float(np.hypot(*chord))
        if L < ccfg["flight_min_m"]:
            continue
        mid = np.arange(f0 + 1, f1); ok = seen[mid]
        if not ok.any():
            continue
        rel = ball_m[mid[ok]] - p0
        dev = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / L        # perpendicular distance to the line
        if dev.max() > ccfg["flight_dev_m"]:
            t = (mid - f0) / (f1 - f0)
            pos[mid] = p0 + chord * t[:, None]; mask[mid] = True
    return mask, pos


def ball_track(ball_m, raw, seen, state, holder, xy_m, hom_err, pitch, fps, ccfg, start=0, flight=None):
    """Where the ball is, per frame, with a spread in metres and the source of the estimate:
    seen (the detector, smoothed), flight (the straight line between two touches, when the seen track
    bows off it), held (the holder's feet while the ball is hidden), interp (a straight line between two
    sightings of a free ball), unknown (hidden too long to say)."""
    n = len(ball_m); pos = np.full((n, 2), np.nan); sd = np.full(n, np.nan); src = np.full(n, "unknown", dtype=object)
    he = np.nan_to_num(hom_err, nan=0.0, posinf=5.0)
    pos[seen] = ball_m[seen]; sd[seen] = np.maximum(0.3, he[seen]); src[seen] = "seen"
    if flight is not None:
        fm, fpos = flight
        pos[fm] = fpos[fm]; sd[fm] = ccfg["flight_sd_m"]; src[fm] = "flight"
    hid_held = ~seen & (state == "held") & (holder >= 0)
    idx = np.flatnonzero(hid_held)
    pos[idx] = xy_m[idx, holder[idx]]; sd[idx] = 1.0; src[idx] = "held"
    # a holder who is himself untracked on a frame gives no position: that frame is not located
    bad = ~np.isfinite(pos).all(axis=1)
    pos[bad] = np.nan; sd[bad] = np.nan; src[bad] = "unknown"
    ok = ~bad
    max_gap = int(ccfg["interp_max_s"] * fps)
    for a, b in _segments(~ok & (state != "dead")):
        if a == 0 or b >= n or b - a > max_gap or not ok[a - 1] or not ok[b]:
            continue
        t = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
        pos[a:b] = pos[a - 1] + (pos[b] - pos[a - 1]) * t[:, None]
        gap_s = np.minimum(np.arange(a, b) - (a - 1), b - np.arange(a, b)) / fps
        sd[a:b] = np.minimum(0.5 + 5.0 * gap_s, 15.0); src[a:b] = "interp"
    return pd.DataFrame({"frame": start + np.arange(n), "x": pos[:, 0] / pitch[0], "y": pos[:, 1] / pitch[1], "sd_m": sd, "source": src,
                         "raw_x": raw[:, 0], "raw_y": raw[:, 1]}, columns=TRACK_COLUMNS)


def run_control(world, dead, ccfg):
    """world: xy (n, S, 2), ball_obs (n, 2), team (S,), slot_ids, pitch, fps, hom_err, start. dead: bool (n,).
    Returns (touches, possession_frames, ball_track) as DataFrames."""
    n = len(world["ball_obs"]); pitch = world["pitch"]; fps = world["fps"]; start = world.get("start", 0)
    team = np.asarray(world["team"]); slots = list(world["slot_ids"])
    ball_m = smooth_ball(world["ball_obs"], pitch, fps, ccfg["speed_cap_ms"], ccfg["smooth_window"])
    seen = np.isfinite(ball_m[:, 0])
    xy_m = world["xy"].astype(float) * np.array(pitch)
    vel = velocity(ball_m, fps)
    runs = score_runs(zone_runs(ball_m, xy_m, team, dead, ccfg), vel, dead, fps, ccfg)
    touches = [r for r in runs if r["p"] >= ccfg["p_touch_min"]]
    state, holder, owner, conf, event, zone = build_states(n, touches, seen, dead, team, ball_m, pitch, fps, ccfg)
    own_team = np.array([team[o] if o >= 0 else "" for o in owner], dtype=object)
    label = np.where(dead, "D", np.where(own_team == "", "X", own_team)).astype(object)
    p_dead = dead.astype(float)
    p_home = np.where(own_team == "H", conf, np.where(own_team == "A", 1 - conf, 0.0)) * (1 - p_dead)
    p_away = np.where(own_team == "A", conf, np.where(own_team == "H", 1 - conf, 0.0)) * (1 - p_dead)
    poss = pd.DataFrame({"frame": start + np.arange(n), "state": state, "holder_slot": [slots[h] if h >= 0 else "" for h in holder], "team": own_team,
                         "p_team": conf, "p_home": p_home, "p_away": p_away, "p_dead": p_dead, "event": event, "restart_zone": zone, "label": label}, columns=POSSESSION_COLUMNS)
    tt = pd.DataFrame([{"frame_start": start + r["f0"], "frame_end": start + r["f1"], "slot": slots[r["k"]], "team": team[r["k"]], "duration_s": r["duration_s"],
                        "dv_ms": r["dv_ms"], "duel": r["duel"], "p": r["p"], "kind": r.get("kind", "passed_by")} for r in runs], columns=TOUCH_COLUMNS)
    flight = flight_chords(ball_m, seen, touches, dead, fps, ccfg)
    track = ball_track(ball_m, world["ball_obs"], seen, state, holder, xy_m, world.get("hom_err", np.zeros(n)), pitch, fps, ccfg, start, flight)
    return tt, poss, track


def validate_outputs(touches, poss, track):
    """the contract, checked before anything is written: columns, one row per frame, ranges"""
    for name, df, cols in (("touches", touches, TOUCH_COLUMNS), ("possession_frames", poss, POSSESSION_COLUMNS), ("ball_track", track, TRACK_COLUMNS)):
        if list(df.columns) != cols:
            raise ValueError(f"{name}: columns {list(df.columns)} != {cols}")
    for name, df in (("possession_frames", poss), ("ball_track", track)):
        f = df["frame"].to_numpy()
        if len(f) and not np.array_equal(f, np.arange(f[0], f[0] + len(f))):
            raise ValueError(f"{name}: frames are not contiguous, one row per frame")
    if len(poss) != len(track):
        raise ValueError("possession_frames and ball_track differ in length")
    for c in ("p_team", "p_home", "p_away", "p_dead"):
        v = poss[c].to_numpy(float)
        if np.isnan(v).any() or v.min() < 0 or v.max() > 1:
            raise ValueError(f"possession_frames.{c} outside 0..1")
    if not set(poss["label"]) <= {"H", "A", "D", "X"} or not set(poss["state"]) <= {"dead", "held", "free"}:
        raise ValueError("possession_frames: unknown label or state")
    if not set(track["source"]) <= {"seen", "flight", "held", "interp", "unknown"}:
        raise ValueError("ball_track: unknown source")
    if track.loc[track.source != "unknown", ["x", "y", "sd_m"]].isna().any().any():
        raise ValueError("ball_track: a located frame has no position or spread")


def manifest(touches, poss, track, ccfg, inputs, config_hash, version):
    """what was made, from what, with which settings: enough to trust or reproduce the tables"""
    src = track.source.value_counts(normalize=True).round(4).to_dict()
    return {"model": "control", "ballq_version": version, "config_hash": config_hash, "config": dict(ccfg), "inputs": inputs,
            "frames": [int(poss.frame.iloc[0]), int(poss.frame.iloc[-1])], "rows": {"touches": len(touches), "possession_frames": len(poss), "ball_track": len(track)},
            "schema": {"touches": TOUCH_COLUMNS, "possession_frames": POSSESSION_COLUMNS, "ball_track": TRACK_COLUMNS},
            "summary": {"labels": poss.label.value_counts().to_dict(), "states": poss.state.value_counts().to_dict(), "touch_kinds": touches.kind.value_counts().to_dict(),
                        "ball_source_share": src, "gains": int((poss.event == "gain").sum()), "restarts": int((poss.event == "restart").sum()),
                        "restart_zones": poss.loc[poss.event == "restart", "restart_zone"].value_counts().to_dict()}}

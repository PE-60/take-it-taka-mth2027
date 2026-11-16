"""The control model on small scripted scenes with known answers."""
import numpy as np
import pytest
from ballq.control import run_control, smooth_ball, zone_runs, restart_zone, POSSESSION_COLUMNS, TRACK_COLUMNS, TOUCH_COLUMNS

PITCH, FPS = (105.3, 68.0), 25


def scene(n=200):
    """home 0 at x=20 m, away 3 just off the pass line at x=42 m, home 1 at x=65 m, away 4 far away; all on y=34 m"""
    xy = np.full((n, 6, 2), np.nan)
    xy[:, 0] = (20 / PITCH[0], 0.5); xy[:, 1] = (65 / PITCH[0], 0.5); xy[:, 3] = (42 / PITCH[0], 0.5 + 0.6 / PITCH[1]); xy[:, 4] = (90 / PITCH[0], 0.2)
    return {"xy": xy, "ball_obs": np.full((n, 2), np.nan), "team": np.array(list("HHHAAA")), "slot_ids": ["1", "2", "3", "21", "22", "23"],
            "pitch": PITCH, "fps": FPS, "hom_err": np.zeros(n), "start": 1000}


def pass_ball(w, kick=20, speed=12.0, x0=20.5, stop_x=64.6):
    n = len(w["ball_obs"])
    for r in range(n):
        x = x0 if r < kick else min(x0 + speed * (r - kick) / FPS, stop_x)
        w["ball_obs"][r] = (x / PITCH[0], 0.5)
    return w


def test_a_pass_through_an_opponents_zone_is_not_a_touch(cfg):
    w = pass_ball(scene()); dead = np.zeros(200, bool)
    touches, poss, track = run_control(w, dead, cfg["control"])
    by_slot = {t.slot: t for t in touches.itertuples()}
    assert by_slot["21"].kind == "passed_by" and by_slot["21"].p < 0.5, "the ball crossed his zone at constant velocity"
    assert by_slot["1"].kind == "gain" and by_slot["2"].kind == "keep"
    assert set(poss.label) == {"H"}, "possession never leaves the home team"
    assert list(poss.columns) == POSSESSION_COLUMNS and list(track.columns) == TRACK_COLUMNS and list(touches.columns) == TOUCH_COLUMNS
    assert poss.frame.iloc[0] == 1000


def test_an_interception_is_a_gain_at_the_touch(cfg):
    w = pass_ball(scene(), stop_x=42.0)                  # the ball stops at the away player
    touches, poss, _ = run_control(w, np.zeros(200, bool), cfg["control"])
    gain = poss[poss.event == "gain"]
    assert list(gain.label) == ["H", "A"]
    arrive = 1000 + 20 + int((42.0 - 20.5 - 1.0) / 12.0 * FPS)
    assert abs(int(gain.frame.iloc[1]) - arrive) <= 3, "the change sits where the ball reaches him, not a second later"
    assert poss.label.iloc[-1] == "A"


def test_a_ball_hidden_at_a_players_feet_stays_his_and_the_track_follows_him(cfg):
    w = pass_ball(scene(), kick=80)
    w["ball_obs"][30:70] = np.nan                        # the detector loses it in the shirt
    touches, poss, track = run_control(w, np.zeros(200, bool), cfg["control"])
    assert set(poss.state.iloc[30:70]) == {"held"} and set(poss.holder_slot.iloc[30:70]) == {"1"}
    assert set(track.source.iloc[30:70]) == {"held"} and track.sd_m.iloc[40] == 1.0
    assert abs(track.x.iloc[40] * PITCH[0] - 20.0) < 0.01
    assert set(poss.label) == {"H"}


def test_an_air_ball_jump_is_dropped_and_bridged(cfg):
    w = pass_ball(scene())
    w["ball_obs"][40:46, 0] += 25 / PITCH[0]             # the projection throws the ball 25 m up the pitch for six frames
    _, poss, track = run_control(w, np.zeros(200, bool), cfg["control"])
    assert set(track.source.iloc[40:46]) == {"interp"}
    true_x = 20.5 + 12.0 * (np.arange(40, 46) - 20) / FPS
    assert np.abs(track.x.iloc[40:46].to_numpy() * PITCH[0] - true_x).max() < 1.5
    assert set(poss.label) == {"H"}


def test_a_ball_in_the_air_is_put_back_on_the_straight_line(cfg):
    """a 44 m lofted pass: the projection bows up to 6 m sideways and comes back at landing, smoothly"""
    w = pass_ball(scene(), speed=22.0)
    fl = np.flatnonzero((w["ball_obs"][:, 0] * PITCH[0] > 20.6) & (w["ball_obs"][:, 0] * PITCH[0] < 64.5))
    u = (fl - fl[0]) / (fl[-1] - fl[0])
    w["ball_obs"][fl, 1] += 6.0 * 4 * u * (1 - u) / PITCH[1]          # height times the camera's slant
    touches, poss, track = run_control(w, np.zeros(200, bool), cfg["control"])
    mid = fl[5:-5]
    assert set(track.source.iloc[mid]) == {"flight"}
    assert np.abs(track.y.iloc[mid].to_numpy() - 0.5).max() * PITCH[1] < 0.5, "back on the line between kick and reception"
    assert np.abs(track.raw_y.iloc[mid].to_numpy() - 0.5).max() * PITCH[1] > 5, "the raw point is kept for comparison"
    assert set(poss.label) == {"H"} and list(touches.kind) == ["gain", "keep"]


def test_after_a_dead_ball_the_taker_does_not_count_and_the_receiver_owns_from_the_restart(cfg):
    """away throws in (slot 21 holds the ball at the restart), home receives 1.4 s later"""
    n = 200; w = scene(n)
    w["xy"][:, 3] = (42 / PITCH[0], 0.01)                # the thrower on the touchline
    w["xy"][:, 1] = (52 / PITCH[0], 0.2)                 # the home receiver
    dead = np.zeros(n, bool); dead[20:80] = True
    for r in range(n):
        if r < 20:
            w["ball_obs"][r] = (20.5 / PITCH[0], 0.5)    # home had it before the stoppage
        elif r < 90:
            w["ball_obs"][r] = (42 / PITCH[0], 0.012)    # in the thrower's hands, through the dead ball and 0.4 s after
        else:
            t = min((r - 90) / 35.0, 1.0)                # thrown to the receiver
            w["ball_obs"][r] = ((42 + 9.6 * t) / PITCH[0], 0.012 + (0.2 - 0.012 - 0.4 / PITCH[1]) * t)
    touches, poss, _ = run_control(w, dead, cfg["control"])
    assert set(poss.label.iloc[20:80]) == {"D"}
    assert poss.event.iloc[80] == "restart" and poss.restart_zone.iloc[80] == "touchline"
    taker = touches[touches.kind == "taker"]
    assert list(taker.slot) == ["21"]
    assert set(poss.label.iloc[80:]) == {"H"}, "the receiver's team from the restart frame on, the thrower's never"
    assert set(poss.label.iloc[:20]) == {"H"}


def test_probabilities_are_consistent(cfg):
    w = pass_ball(scene()); dead = np.zeros(200, bool); dead[150:] = True
    _, poss, track = run_control(w, dead, cfg["control"])
    s = poss.p_home + poss.p_away + poss.p_dead
    live = poss.label != "X"
    assert np.allclose(s[live], 1.0) and poss[["p_home", "p_away", "p_dead", "p_team"]].to_numpy().min() >= 0
    assert (poss.p_dead[150:] == 1).all()
    assert track.sd_m.dropna().min() >= 0.3


def test_restart_zones():
    assert restart_zone(1, 1, PITCH) == "corner" and restart_zone(40, 0.5, PITCH) == "touchline"
    assert restart_zone(3, 34, PITCH) == "goal_area" and restart_zone(52.65, 34, PITCH) == "centre" and restart_zone(30, 20, PITCH) == "other"


def test_outputs_pass_the_contract_and_the_manifest_describes_them(cfg):
    from ballq.control import validate_outputs, manifest
    w = pass_ball(scene()); dead = np.zeros(200, bool); dead[150:] = True
    touches, poss, track = run_control(w, dead, cfg["control"])
    validate_outputs(touches, poss, track)
    m = manifest(touches, poss, track, cfg["control"], {"atd_txt": {"file": "x"}}, "abc", "0")
    assert m["rows"]["possession_frames"] == 200 and m["schema"]["ball_track"] == TRACK_COLUMNS and m["summary"]["restarts"] == 0
    bad = poss.copy(); bad.loc[3, "p_home"] = 1.5
    with pytest.raises(ValueError):
        validate_outputs(touches, bad, track)

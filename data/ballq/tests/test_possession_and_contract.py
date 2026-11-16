import numpy as np
from ballq.possession import lh_point, lh_posterior
from ballq.contract import FRAME_COLUMNS, PLAYER_COLUMNS, validate_frames


def test_point_and_posterior_agree_when_the_posterior_is_tight(world, cfg):
    n = 50
    xy = world["xy"][:n]; ball = world["ball_true"][:n]
    # a tight posterior: mass 1.0 within tp of the true carrier
    mass = np.zeros((n, 6)); mass[:, 0] = 1.0
    mean = ball.copy()
    pt = lh_point(xy, ball, world["team"], world["pitch"], world["fps"], cfg["possession"], kicks=set())
    po = lh_posterior(xy, mean, mass, world["team"], world["pitch"], world["fps"], cfg["possession"], kicks=set())
    assert list(pt["team"]) == list(po["team"]) == ["H"] * n


def test_posterior_version_keeps_the_carrier_when_the_point_is_lost(world, cfg):
    n = 80
    xy = world["xy"][:n]; obs = world["ball_obs"][:n]
    mass = np.zeros((n, 6)); mass[:, 0] = 0.9
    mean = world["ball_true"][:n]
    pt = lh_point(xy, obs, world["team"], world["pitch"], world["fps"], cfg["possession"], kicks=set())
    po = lh_posterior(xy, mean, mass, world["team"], world["pitch"], world["fps"], cfg["possession"], kicks=set())
    assert (pt["holder"][60:70] == -1).all(), "point version has no holder while the ball is lost"
    assert (po["holder"][60:70] == 0).all(), "posterior version keeps the dribbler"
    assert (pt["team"][60:70] == "H").all(), "team is sticky in both"


def test_posterior_hand_over_after_the_pass(world, cfg):
    n = 200
    xy = world["xy"][:n]; mean = world["ball_true"][:n]
    mass = np.zeros((n, 6)); mass[:100, 0] = 0.9; mass[135:, 3] = 0.9
    po = lh_posterior(xy, mean, mass, world["team"], world["pitch"], world["fps"], cfg["possession"], kicks={100})
    assert po["team"][50] == "H" and po["team"][180] == "A"
    assert po["holder"][115] == -1 and po["team"][115] == "H", "in flight: no holder, team sticky"


def test_contract_columns_and_validation():
    import pandas as pd
    df = pd.DataFrame({c: [0.5] for c in FRAME_COLUMNS}); df["frame"] = [11185]; df["holder_slot"] = [-1]; df["source_flags"] = [0]
    validate_frames(df)
    bad = df.copy(); bad["p_air"] = [1.5]
    try:
        validate_frames(bad); assert False
    except ValueError:
        pass
    assert "p_within_1m" in PLAYER_COLUMNS


def test_a_pass_rolling_past_a_defender_does_not_flip_possession(cfg):
    """Home #0 passes along y = 0.5 at 12 m/s. Away #3 stands 0.6 m off the line halfway: the ball goes
    through his sphere of influence without a ball action. Link and Hoernig with kick detection keeps
    home; proximity alone would flip to away for a frame and back."""
    fps, pitch, n = 25, (105.3, 68.0), 120
    xy = np.full((n, 6, 2), np.nan); ball = np.zeros((n, 2))
    xy[:, 0] = (0.20, 0.50); xy[:, 3] = (0.40 + 0.3 / pitch[0], 0.50 + 0.6 / pitch[1]); xy[:, 5] = (0.62, 0.50)
    for r in range(n):
        t = r / fps
        if r < 20:
            ball[r] = (0.20 + 0.5 / pitch[0], 0.50)                      # at the passer's feet
        else:
            ball[r] = (0.20 + 0.5 / pitch[0] + 12.0 * (t - 20 / fps) / pitch[0], 0.50)   # 12 m/s along the line
    team = np.array(["H", "H", "H", "A", "A", "A"])
    kicks = {20}
    res = lh_point(xy, ball, team, pitch, fps, cfg["possession"], kicks)
    passes_defender = [r for r in range(n) if abs(ball[r, 0] - xy[r, 3, 0]) * pitch[0] < 0.8]
    assert passes_defender, "the ball does pass within 1 m of the defender"
    assert all(res["team"][r] == "H" for r in range(20, 100)), "a rolling pass must not change the team"
    assert all(res["holder"][r] != 3 for r in range(n)), "the defender never gains"
    # the receiver at 0.62 traps the ball: stop it there from the frame it arrives
    arrive = next(r for r in range(n) if ball[r, 0] >= 0.62)
    ball[arrive:] = (0.62 + 0.4 / pitch[0], 0.50)
    res2 = lh_point(xy, ball, team, pitch, fps, cfg["possession"], kicks)
    assert res2["team"][-1] == "A" and res2["holder"][-1] == 5, "the trap starts the receiver's possession"

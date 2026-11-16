"""Shared synthetic data for the tests.

A small world: 25 fps, 300 frames, 6 player slots (3 home, 3 away). Home #0 dribbles the ball
for 4 s, kicks it at frame 100, it flies 30 m to away #3 by frame 130, who then carries it.
The detector loses the ball for frames 60..69 and reports a wild air-ball excursion on 110..120.
"""
import numpy as np
import pytest

FPS = 25
N = 300
S = 6
PITCH = (105.3, 68.0)


@pytest.fixture
def world():
    xy = np.full((N, S, 2), np.nan, dtype=np.float32)
    ball_true = np.zeros((N, 2), dtype=np.float32)
    base = [(0.30, 0.50), (0.20, 0.30), (0.25, 0.70), (0.60, 0.50), (0.70, 0.30), (0.75, 0.70)]
    for r in range(N):
        if r < 100:
            bx = 0.30 + 0.003 * r / PITCH[0] * 25 / 25; by = 0.50      # slow dribble
        elif r <= 130:
            bx = 0.30 + 0.30 * (r - 100) / 30; by = 0.50                 # flight
        else:
            bx = 0.60 + 0.002 * (r - 130) / PITCH[0]; by = 0.50
        ball_true[r] = (bx, by)
        for k, (x, y) in enumerate(base):
            if k == 0 and r < 100:
                x = bx - 0.004
            if k == 3 and r > 130:
                x = bx + 0.004
            xy[r, k] = (x, y)
    ball_obs = ball_true.copy()
    ball_obs[60:70] = np.nan                                              # lost
    ball_obs[110:121, 0] += 0.25                                          # air ball excursion: 26 m jump
    team = np.array(["H", "H", "H", "A", "A", "A"])
    return dict(xy=xy, ball_true=ball_true, ball_obs=ball_obs, team=team, fps=FPS, pitch=PITCH)


@pytest.fixture
def cfg():
    import yaml, pathlib
    return yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config.yaml"))

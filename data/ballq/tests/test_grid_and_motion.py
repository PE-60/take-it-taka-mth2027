import numpy as np
from ballq.grid import Grid
from ballq.motion import rolling_kernel_sigma, predict


def test_grid_geometry_and_round_trip():
    g = Grid((105.3, 68.0), 0.25)
    assert g.shape == (272, 422)                      # rows = y, cols = x
    i, j = g.cell(0.5, 0.5)
    x, y = g.centre(i, j)
    assert abs(x - 0.5) < 0.25 / 105.3 and abs(y - 0.5) < 0.25 / 68.0
    assert g.cell(-0.1, 1.2) == (g.shape[0] - 1, 0)   # clamped


def test_distance_grid_is_in_metres():
    g = Grid((105.3, 68.0), 0.5)
    d = g.dist_m(0.5, 0.5)
    assert d.shape == g.shape
    assert d.min() < 0.4
    i, j = g.cell(0.5, 0.5)
    assert abs(d[i, j + 20] - 10.0) < 0.6                   # 20 cells of 0.5 m along x


def test_predict_keeps_mass_and_moves_with_velocity():
    g = Grid((105.3, 68.0), 0.5)
    p = np.zeros(g.shape); i, j = g.cell(0.5, 0.5); p[i, j] = 1.0
    vel_cells = (0.0, 4.0)                                 # 4 cells per frame along x = 50 m/s capped later
    q = predict(p, g, vel_cells=(0.0, 4.0), sigma_cells=1.0, carried_centres=[], weights={"rolling": 1, "carried": 0, "anywhere": 0})
    assert abs(q.sum() - 1) < 1e-9
    ii, jj = np.unravel_index(q.argmax(), q.shape)
    assert jj == j + 4 and ii == i


def test_predict_carried_puts_mass_on_players():
    g = Grid((105.3, 68.0), 0.5)
    p = np.zeros(g.shape); p[g.cell(0.5, 0.5)] = 1.0
    q = predict(p, g, vel_cells=(0, 0), sigma_cells=1.0, carried_centres=[(0.6, 0.51)], weights={"rolling": 0, "carried": 1, "anywhere": 0})
    ii, jj = np.unravel_index(q.argmax(), q.shape)
    assert (ii, jj) == g.cell(0.6, 0.51)


def test_rolling_sigma_grows_with_speed():
    assert rolling_kernel_sigma(0.0, 0.6, 0.03) == 0.6
    assert rolling_kernel_sigma(20.0, 0.6, 0.03) == 1.2

import numpy as np
from ballq.grid import Grid
from ballq.observe import detector_likelihood, holder_likelihood, trust, gate_tau
from ballq.filter import forward_backward
from ballq.features import ball_speed_frames, kicks_from_accel


def test_trust_lowers_tau_for_each_flag(cfg):
    o = cfg["observe"]
    s, tau = trust(o, air=False, white=False, hom_err=0.3, dead=False)
    assert (s, tau) == (o["base_sigma_m"], o["base_tau"])
    assert trust(o, air=True, white=False, hom_err=0.3, dead=False)[1] == o["air_ball_tau"]
    assert trust(o, air=False, white=True, hom_err=0.3, dead=False)[1] == o["white_blend_tau"]
    s2, tau2 = trust(o, air=False, white=False, hom_err=2.0, dead=False)
    assert s2 == s * 2.0 and tau2 == o["homography_bad_tau"]
    assert trust(o, air=False, white=False, hom_err=0.3, dead=True)[0] == s * o["dead_factor"]


def test_gate_lowers_tau_when_the_point_is_unreachable(cfg):
    o = cfg["observe"]; g = Grid((105.3, 68.0), 0.5)
    pred = np.zeros(g.shape); pred[g.cell(0.3, 0.5)] = 1.0
    args = (o["gate_radius_m"], o["gate_ratio_min"], o["gate_tau"])
    assert gate_tau(pred, g, (0.3, 0.5), 0.95, *args) == 0.95
    assert gate_tau(pred, g, (0.6, 0.5), 0.95, *args) == o["gate_tau"]
    # a filter that has lost the ball (flat prediction) must accept the point, or it never recovers
    flat = np.full(g.shape, 1.0 / pred.size)
    assert gate_tau(flat, g, (0.6, 0.5), 0.95, *args) == 0.95


def test_detector_likelihood_peaks_at_observation_and_is_flat_when_missing():
    g = Grid((105.3, 68.0), 0.5)
    L = detector_likelihood(g, (0.4, 0.6), sigma_m=0.5, tau=0.9)
    assert abs(L.min() - 0.1) < 1e-6, 'the flat part keeps every cell alive'
    assert np.unravel_index(L.argmax(), L.shape) == g.cell(0.4, 0.6)
    F = detector_likelihood(g, None, sigma_m=0.5)
    assert np.allclose(F, F.flat[0])


def test_holder_likelihood_is_a_mixture_at_the_players():
    g = Grid((105.3, 68.0), 0.5)
    L = holder_likelihood(g, [((0.3, 0.5), 0.7), ((0.6, 0.5), 0.3)], sigma_m=0.7, weight=0.6)
    a = L[g.cell(0.3, 0.5)]; b = L[g.cell(0.6, 0.5)]; far = L[g.cell(0.9, 0.9)]
    assert a > b > far


def test_forward_backward_recovers_a_lost_ball(world, cfg):
    """Ball hidden 60..69 while the dribbler carries it: the smoothed mean should stay near the truth."""
    g = Grid(world["pitch"], 0.5)
    out = forward_backward(g, world, cfg, frames=range(40, 90))
    err = np.hypot((out["mean"][:, 0] - world["ball_true"][40:90, 0]) * 105.3, (out["mean"][:, 1] - world["ball_true"][40:90, 1]) * 68)
    assert np.nanmedian(err) < 1.5
    assert err[20:30].max() < 3.0                               # the lost stretch
    assert np.allclose(out["mass_total"], 1.0, atol=1e-6)


def test_forward_backward_discounts_the_air_ball(world, cfg):
    """The 26 m excursion at 110..120 gets low trust; the mean should stay on the flight line."""
    g = Grid(world["pitch"], 0.5)
    out = forward_backward(g, world, cfg, frames=range(95, 135))
    excursion = world["ball_obs"][110:121, 0]
    mean_x = out["mean"][15:26, 0]
    truth_x = world["ball_true"][110:121, 0]
    assert np.all(np.abs(mean_x - truth_x) < np.abs(excursion - truth_x) * 0.5)


def test_kicks_from_acceleration(world, cfg):
    sp = ball_speed_frames(world["ball_true"], world["pitch"], world["fps"])
    k = kicks_from_accel(sp, world["fps"], cfg["features"]["kick_accel_ms2"], cfg["features"]["kick_min_gap_s"])
    assert any(98 <= r <= 102 for r in k), k


def test_restart_prior_nudges_but_does_not_override_a_trusted_point(cfg):
    """Tag timing is loose: when the detector clearly sees the ball 19 m from the spot, the ball stays there."""
    from ballq.observe import restart_prior
    o = cfg["observe"]; g = Grid((105.3, 68.0), 0.5)
    L = detector_likelihood(g, (0.68, 0.5), sigma_m=0.5, tau=0.98) * restart_prior(g, (0.5, 0.5), o["restart_sigma_m"], o["restart_weight"])
    peak = g.centre(*np.unravel_index(L.argmax(), L.shape))
    assert np.hypot((peak[0] - 0.68) * 105.3, (peak[1] - 0.5) * 68) < 1.0
    # with no detector point at all the spot is where the mass goes
    F = detector_likelihood(g, None, sigma_m=0.5) * restart_prior(g, (0.5, 0.5), o["restart_sigma_m"], o["restart_weight"])
    peak = g.centre(*np.unravel_index(F.argmax(), F.shape))
    assert np.hypot((peak[0] - 0.5) * 105.3, (peak[1] - 0.5) * 68) < 1.0

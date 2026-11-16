"""How much to believe what we see. Likelihoods over the grid; the filter multiplies them in."""
import numpy as np


def trust(o, air, white, hom_err, dead):
    """How much to believe the detector on one frame: (sigma_m, tau).
    sigma is the width of the Gaussian; tau is the weight of that Gaussian against a flat
    "the detector is wrong" component. A wide Gaussian alone cannot save a frame whose point is
    far off, because the truth still gets zero likelihood; the flat part is what does that."""
    s, tau = o["base_sigma_m"], o["base_tau"]
    if dead:
        s *= o["dead_factor"]; tau = max(tau, o["dead_tau"])
    if air:
        tau = min(tau, o["air_ball_tau"])
    if white:
        tau = min(tau, o["white_blend_tau"])
    if hom_err is not None and np.isfinite(hom_err) and hom_err > o["homography_bad_m"]:
        s *= hom_err / o["homography_bad_m"]
        tau = min(tau, o["homography_bad_tau"])
    return s, tau


def detector_likelihood(grid, obs_xy, sigma_m, tau=1.0):
    """(1 - tau) flat + tau Gaussian around the detector's point; flat when there is no observation"""
    if obs_xy is None or not np.all(np.isfinite(obs_xy)):
        return np.ones(grid.shape)
    L = np.full(grid.shape, 1 - tau)
    si, sj, bump = grid.gaussian_bump(float(obs_xy[0]), float(obs_xy[1]), sigma_m)
    L[si, sj] += tau * bump
    return L


def gate_tau(predicted, grid, obs_xy, tau, radius_m, ratio_min, tau_gated):
    """If the prediction puts almost no mass near the observation compared with the mass it has around
    its own mean, the observation is suspect: a 26 m jump in one frame is not something the ball does.
    The comparison is relative, so a filter that has lost the ball (flat prediction) accepts the next
    point and recovers; an absolute threshold could reject every point for ever.
    Returns the (possibly) lowered tau."""
    if obs_xy is None or not np.all(np.isfinite(obs_xy)):
        return tau
    (mx, my), _ = grid.mean_cov(predicted)
    own = grid.mass_within(predicted, mx, my, radius_m)
    near = grid.mass_within(predicted, float(obs_xy[0]), float(obs_xy[1]), radius_m)
    return min(tau, tau_gated) if near < ratio_min * own else tau


def holder_likelihood(grid, candidates, sigma_m, weight):
    """Players as a second observation: a mixture of Gaussians at the candidate holders' feet,
    blended with a flat likelihood so it can only pull, never veto.
    candidates: [((x, y), prob)] with probs summing to at most 1"""
    if not candidates or weight <= 0:
        return np.ones(grid.shape)
    mix = np.zeros(grid.shape)
    for (x, y), pr in candidates:
        si, sj, bump = grid.gaussian_bump(x, y, sigma_m)
        mix[si, sj] += pr * bump
    if mix.max() > 0:
        mix /= mix.max()
    return (1 - weight) + weight * mix


def restart_prior(grid, spot_xy, sigma_m, weight=0.7):
    """A nudge towards the restart spot, not a command: a trusted detector point elsewhere still wins.
    weight 1 would force the ball onto the spot even when the tag timing is off by a second."""
    L = np.full(grid.shape, 1 - weight)
    si, sj, bump = grid.gaussian_bump(float(spot_xy[0]), float(spot_xy[1]), sigma_m)
    L[si, sj] += weight * bump
    return L


def holder_candidates(ball_xy, players, near_m, closing, pitch):
    """probabilities that each nearby player is the holder, from distance and closing speed.
    players: [(k, x, y, speed)] ; closing: {k: metres per second towards the ball}"""
    out = []
    if ball_xy is None or not np.all(np.isfinite(ball_xy)):
        return out
    scores = []
    for k, x, y, sp in players:
        d = float(np.hypot((x - ball_xy[0]) * pitch[0], (y - ball_xy[1]) * pitch[1]))
        if d > near_m:
            continue
        score = np.exp(-d / 1.0) * (1.0 + 0.2 * max(0.0, closing.get(k, 0.0)))
        scores.append((k, (x, y), score))
    tot = sum(s for _, _, s in scores)
    if tot <= 0:
        return out
    return [((x, y), s / tot * min(1.0, tot)) for k, (x, y), s in scores]

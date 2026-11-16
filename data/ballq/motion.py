"""How the ball gets from one frame to the next. A mixture of three stories:
    rolling   constant velocity plus noise that grows with speed
    carried   the ball sits at a nearby player's feet
    anywhere  a uniform leak over the pitch, the small chance that both stories are wrong
Rolling is a shift plus one separable Gaussian blur; carried is a few local Gaussian bumps; anywhere is
a constant. Nothing here scales with the number of players beyond the bumps."""
import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift


def rolling_kernel_sigma(speed_ms, sigma0_m, per_ms):
    return sigma0_m + per_ms * speed_ms


def _blur(p, sigma_cells):
    return gaussian_filter(p, sigma=max(sigma_cells, 1e-3), mode="constant", cval=0.0)


def predict(p, grid, vel_cells, sigma_cells, carried_centres, weights, carried_sigma_cells=2.0):
    """One step of the motion model.
    p               distribution over the grid (sums to 1)
    vel_cells       (drow, dcol) displacement per frame, in cells
    sigma_cells     rolling noise in cells
    carried_centres [(x, y)] pitch units of the players close to the ball mass
    weights         {rolling, carried, anywhere}; renormalised here, carried is dropped when there is nobody
    """
    w = dict(weights)
    if not carried_centres:
        w["carried"] = 0.0
    tot = sum(w.values()) or 1.0
    w = {k: v / tot for k, v in w.items()}
    out = np.zeros_like(p)
    if w["rolling"]:
        moved = nd_shift(p, shift=vel_cells, order=1, mode="constant", cval=0.0)
        out += w["rolling"] * _blur(moved, sigma_cells)
    if w["carried"]:
        q = np.zeros_like(p)
        for (x, y) in carried_centres:
            si, sj, bump = grid.gaussian_bump(x, y, carried_sigma_cells * grid.cell_m)
            q[si, sj] += bump / bump.sum()
        out += w["carried"] * q / len(carried_centres)
    if w["anywhere"]:
        out += w["anywhere"] / p.size
    s = out.sum()
    return out / s if s > 0 else np.full_like(p, 1.0 / p.size)

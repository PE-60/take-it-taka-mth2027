"""The pitch as a grid of cells. Positions come in as 0..1 pitch units (as the ATD gives them),
cells are square in metres. rows run along y, columns along x."""
import numpy as np


class Grid:
    def __init__(self, pitch_m, cell_m):
        self.pitch = (float(pitch_m[0]), float(pitch_m[1]))
        self.cell_m = float(cell_m)
        self.ncols = int(np.ceil(self.pitch[0] / self.cell_m))
        self.nrows = int(np.ceil(self.pitch[1] / self.cell_m))
        self.shape = (self.nrows, self.ncols)
        # cell centres in pitch units, once
        self.xs = (np.arange(self.ncols) + 0.5) * self.cell_m / self.pitch[0]
        self.ys = (np.arange(self.nrows) + 0.5) * self.cell_m / self.pitch[1]
        self.X, self.Y = np.meshgrid(self.xs, self.ys)

    def cell(self, x, y):
        """pitch units -> (row, col), clamped to the grid"""
        j = int(np.clip(np.floor(x * self.pitch[0] / self.cell_m), 0, self.ncols - 1))
        i = int(np.clip(np.floor(y * self.pitch[1] / self.cell_m), 0, self.nrows - 1))
        return i, j

    def centre(self, i, j):
        """pitch units of cell (i, j); i and j may be arrays"""
        if np.ndim(i):
            return self.xs[np.asarray(j)], self.ys[np.asarray(i)]
        return float(self.xs[j]), float(self.ys[i])

    def to_cells(self, dx_units, dy_units):
        """a displacement in pitch units -> cells (float)"""
        return dy_units * self.pitch[1] / self.cell_m, dx_units * self.pitch[0] / self.cell_m

    def dist_m(self, x, y):
        """distance in metres from every cell centre to a pitch point (the whole grid: slow, for tests)"""
        return np.hypot((self.X - x) * self.pitch[0], (self.Y - y) * self.pitch[1])

    def window(self, x, y, radius_m):
        """row and column slices of the cells within radius_m of a pitch point, plus their distances in metres.
        Everything local goes through here so the cost does not scale with the pitch."""
        i, j = self.cell(x, y)
        h = int(np.ceil(radius_m / self.cell_m)) + 1
        si = slice(max(0, i - h), min(self.nrows, i + h + 1)); sj = slice(max(0, j - h), min(self.ncols, j + h + 1))
        d = np.hypot((self.X[si, sj] - x) * self.pitch[0], (self.Y[si, sj] - y) * self.pitch[1])
        return si, sj, d

    def gaussian_bump(self, x, y, sigma_m, span=4.0):
        """(si, sj, values) of a unit-peak Gaussian around a pitch point, cut at span sigmas"""
        si, sj, d = self.window(x, y, span * sigma_m)
        return si, sj, np.exp(-0.5 * (d / sigma_m) ** 2)

    def mean_cov(self, p):
        """mean (pitch units) and 2x2 covariance (metres^2) of a distribution over the grid"""
        mx = float((p * self.X).sum()); my = float((p * self.Y).sum())
        dx = (self.X - mx) * self.pitch[0]; dy = (self.Y - my) * self.pitch[1]
        cxx = float((p * dx * dx).sum()); cyy = float((p * dy * dy).sum()); cxy = float((p * dx * dy).sum())
        return (mx, my), np.array([[cxx, cxy], [cxy, cyy]])

    def entropy_bits(self, p):
        q = p[p > 0]
        return float(-(q * np.log2(q)).sum())

    def mass_within(self, p, x, y, radius_m):
        si, sj, d = self.window(x, y, radius_m)
        return float(p[si, sj][d <= radius_m].sum())

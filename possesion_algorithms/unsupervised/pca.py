"""Robust scaling and per-block PCA that tolerates missing features.

Features are split into anti-symmetric (possession) and symmetric (alive/dead) blocks. PCA is
fit on each block separately after median/MAD scaling. Frames with gaps are projected by a small
ridge least-squares solve over the observed dimensions of that block, so a missing ball does not
zero out the team-geometry signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockPCA:
    median: np.ndarray      # (F,)
    mad: np.ndarray         # (F,)  (1.4826 * median abs deviation, floored)
    components: np.ndarray  # (k, F) principal axes
    eigenvalues: np.ndarray # (k,)
    anti_mask: np.ndarray   # (F,) which of the *input* columns are anti-symmetric
    names: tuple


@dataclass
class PCAModel:
    anti: BlockPCA
    sym: BlockPCA

    def transform(self, values: np.ndarray, anti: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (n, F) features -> (n, k_anti), (n, k_sym). Missing dims handled per row."""
        z_anti = _project_block(values[:, anti], self.anti)
        z_sym = _project_block(values[:, ~anti], self.sym)
        return z_anti, z_sym

    def scale(self, values: np.ndarray, anti: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Robust-scale only (no PCA), winsorized — a stabler possession axis than PC1."""
        z_a = _winsorize((values[:, anti] - self.anti.median) / self.anti.mad)
        z_s = _winsorize((values[:, ~anti] - self.sym.median) / self.sym.mad)
        return z_a.astype(np.float32), z_s.astype(np.float32)


def _mad_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column median and robust scale (MAD). Scale floored so constant columns stay finite."""
    med = np.nanmedian(x, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=0)
    mad = np.where(np.isfinite(mad) & (mad > 1e-6), mad, 1.0).astype(np.float64)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float64)
    return med, mad


def _winsorize(z: np.ndarray, limit: float = 3.0) -> np.ndarray:
    """Clip robust-scaled values so a few wild frames cannot own a principal axis."""
    return np.clip(z, -limit, limit)


def _fit_block(x: np.ndarray, n_components: int, var_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on rows that are fully observed. Returns median, mad, components, eigenvalues."""
    med, mad = _mad_scale(x)
    full = np.isfinite(x).all(axis=1)
    if full.sum() < max(10, x.shape[1] + 1):
        # Too few complete rows: fall back to mean-imputed covariance.
        z = (np.where(np.isfinite(x), x, med) - med) / mad
    else:
        z = (x[full] - med) / mad
    z = _winsorize(z)
    if len(z) < 2:
        k = min(n_components, x.shape[1])
        return med.astype(np.float32), mad.astype(np.float32), np.eye(k, x.shape[1], dtype=np.float32), np.ones(k, np.float32)

    # SVD of centred (already scaled) data; eigenvalues = singular^2 / (n-1)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    eig = (s ** 2) / max(1, len(z) - 1)
    total = eig.sum()
    if total <= 0:
        k = min(n_components, len(eig))
    else:
        cum = np.cumsum(eig) / total
        k = int(np.searchsorted(cum, var_frac) + 1)
        k = max(1, min(k, n_components, len(eig)))
    return (med.astype(np.float32), mad.astype(np.float32),
            vt[:k].astype(np.float32), eig[:k].astype(np.float32))


def _project_block(x: np.ndarray, block: BlockPCA, ridge: float = 1e-3) -> np.ndarray:
    """Project with per-missing-pattern ridge LS on observed loadings."""
    n, F = x.shape
    k = len(block.components)
    out = np.zeros((n, k), dtype=np.float32)
    if F == 0 or k == 0:
        return out
    z = _winsorize((x - block.median) / block.mad)
    obs = np.isfinite(z)
    # Process complete rows fast.
    complete = obs.all(axis=1)
    if complete.any():
        out[complete] = z[complete] @ block.components.T
    remaining = np.flatnonzero(~complete & obs.any(axis=1))
    if not len(remaining):
        return out
    # Group incomplete rows by their observation mask.
    groups: dict[bytes, list[int]] = {}
    for i in remaining:
        groups.setdefault(obs[i].tobytes(), []).append(int(i))
    C = block.components  # (k, F)
    for idxs in groups.values():
        mask = obs[idxs[0]]
        if not mask.any():
            continue
        C_o = C[:, mask]          # (k, f_obs)
        # scores = argmin ||C_o.T s - z_o||^2 + ridge ||s||^2
        # -> (C_o C_o.T + ridge I) s = C_o z_o
        A = C_o @ C_o.T + ridge * np.eye(k, dtype=np.float64)
        Z = z[idxs][:, mask].astype(np.float64)
        rhs = C_o.astype(np.float64) @ Z.T
        try:
            s = np.linalg.solve(A, rhs).T
        except np.linalg.LinAlgError:
            s = np.zeros((len(idxs), k), dtype=np.float64)
        out[idxs] = s.astype(np.float32)
    return out


def _orient_anti(components: np.ndarray, values_anti: np.ndarray, median: np.ndarray,
                 mad: np.ndarray, anchor: np.ndarray | None) -> np.ndarray:
    """Flip each anti component so high scores go with positive `anchor` (e.g. near_gap).

    PCA axes have an arbitrary sign; without this, H and A swap at random between matches.
    Uses the difference of mean scores on {anchor>0} vs {anchor<0}, which is stabler than a
    correlation when the leading PC is dominated by a different anti feature.
    """
    out = np.array(components, copy=True)
    z = (values_anti - median) / mad
    full = np.isfinite(z).all(axis=1)
    if anchor is not None:
        full &= np.isfinite(anchor)
    if full.sum() < 10:
        for k in range(len(out)):
            if out[k, 0] < 0:
                out[k] *= -1
        return out
    scores = z[full] @ out.T
    if anchor is None:
        for k in range(len(out)):
            if out[k, 0] < 0:
                out[k] *= -1
        return out
    a = anchor[full]
    pos, neg = a > 0, a < 0
    for k in range(len(out)):
        if pos.sum() >= 5 and neg.sum() >= 5:
            # Medians: a few mirrored outliers must not flip the axis (means do).
            if np.median(scores[pos, k]) < np.median(scores[neg, k]):
                out[k] *= -1
        elif out[k, 0] < 0:
            out[k] *= -1
    return out


def fit_pca(values: np.ndarray, anti: np.ndarray, names: tuple,
            n_anti: int = 3, n_sym: int = 3, var_frac: float = 0.90,
            anti_anchor: np.ndarray | None = None) -> PCAModel:
    """Fit block-wise PCA on (n, F) features. `anti` marks anti-symmetric columns.

    anti_anchor  optional (n,) series (typically near_gap) used to fix the sign of anti PCs
    """
    anti = np.asarray(anti, dtype=bool)
    med_a, mad_a, comp_a, eig_a = _fit_block(values[:, anti], n_anti, var_frac)
    med_s, mad_s, comp_s, eig_s = _fit_block(values[:, ~anti], n_sym, var_frac)
    comp_a = _orient_anti(comp_a, values[:, anti], med_a, mad_a, anti_anchor)
    anti_names = tuple(n for n, a in zip(names, anti) if a)
    sym_names = tuple(n for n, a in zip(names, anti) if not a)
    return PCAModel(
        anti=BlockPCA(med_a, mad_a, comp_a, eig_a, np.ones(anti.sum(), bool), anti_names),
        sym=BlockPCA(med_s, mad_s, comp_s, eig_s, np.zeros((~anti).sum(), bool), sym_names),
    )

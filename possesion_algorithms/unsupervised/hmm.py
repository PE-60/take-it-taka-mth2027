"""Sticky Gaussian HMM with mirrored H/A emissions, fit by EM, decoded by Viterbi.

States are always ordered H, A[, D]. The H and A emission means are tied: anti-symmetric
coordinates flip sign, symmetric ones are shared. That halves the parameters and forces the
model to treat the two teams as mirror images — which is exactly how the features were built.

Missing coordinates (ball out of view, anomaly-masked) are marginalised: they contribute nothing
to any state's emission log-likelihood. Stickiness comes from a Dirichlet-style pseudo-count on
the diagonal of the transition matrix, so short flicker costs more than a real turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_NAMES_3 = ("H", "A", "D")
STATE_NAMES_2 = ("H", "A")


@dataclass
class MirroredHMM:
    means: np.ndarray           # (K, D)
    vars: np.ndarray            # (K, D) diagonal variances
    trans: np.ndarray           # (K, K)
    start: np.ndarray           # (K,)
    anti_dims: np.ndarray       # (D,) bool — which latent dims flip under H<->A
    n_states: int
    names: tuple

    def emission_loglik(self, x: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """(n, K) log p(x_t | state), summing only over observed dims."""
        n, D = x.shape
        K = self.n_states
        ll = np.zeros((n, K), dtype=np.float64)
        for k in range(K):
            diff = x - self.means[k]
            var = np.maximum(self.vars[k], 1e-4)
            term = -0.5 * (np.log(2 * np.pi * var) + diff * diff / var)
            term = np.where(observed, term, 0.0)
            ll[:, k] = term.sum(axis=1)
        return ll

    def decode(self, x: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """Viterbi state sequence (n,) int."""
        ll = self.emission_loglik(x, observed)
        return _viterbi(ll, self.trans, self.start)

    def labels(self, states: np.ndarray) -> np.ndarray:
        names = np.asarray(self.names, dtype="<U1")
        return names[states]


def _viterbi(ll: np.ndarray, trans: np.ndarray, start: np.ndarray) -> np.ndarray:
    n, K = ll.shape
    log_t = np.log(np.maximum(trans, 1e-300))
    log_s = np.log(np.maximum(start, 1e-300))
    delta = np.empty((n, K), dtype=np.float64)
    psi = np.empty((n, K), dtype=np.int8)
    delta[0] = log_s + ll[0]
    psi[0] = 0
    for t in range(1, n):
        scores = delta[t - 1][:, None] + log_t  # (K, K): from i to j
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(K)] + ll[t]
    states = np.empty(n, dtype=np.int8)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _forward_backward(ll: np.ndarray, trans: np.ndarray, start: np.ndarray):
    """Scaled forward-backward. Returns gamma (n,K), xi_sum (K,K), loglik."""
    n, K = ll.shape
    emit = np.exp(ll - ll.max(axis=1, keepdims=True))  # stable relative emissions
    alpha = np.empty((n, K), dtype=np.float64)
    scale = np.empty(n, dtype=np.float64)
    alpha[0] = start * emit[0]
    scale[0] = alpha[0].sum() + 1e-300
    alpha[0] /= scale[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ trans) * emit[t]
        scale[t] = alpha[t].sum() + 1e-300
        alpha[t] /= scale[t]
    beta = np.empty((n, K), dtype=np.float64)
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (trans * emit[t + 1] * beta[t + 1]).sum(axis=1)
        beta[t] /= scale[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
    xi_sum = np.zeros((K, K), dtype=np.float64)
    for t in range(n - 1):
        num = alpha[t][:, None] * trans * emit[t + 1] * beta[t + 1]
        xi_sum += num / (num.sum() + 1e-300)
    loglik = np.log(scale).sum() + ll.max(axis=1).sum()  # undo the emit shift roughly
    return gamma, xi_sum, float(np.log(scale + 1e-300).sum())


def _mirror_mean(mean_h: np.ndarray, anti_dims: np.ndarray) -> np.ndarray:
    out = mean_h.copy()
    out[anti_dims] *= -1
    return out


def _init_means(x: np.ndarray, observed: np.ndarray, anti_dims: np.ndarray,
                n_states: int, alive_cue: np.ndarray | None) -> np.ndarray:
    """Seed H/A from the sign of the first anti dim; D from the lowest alive cue."""
    D = x.shape[1]
    means = np.zeros((n_states, D), dtype=np.float64)
    # Impute missing with 0 for init only (features are scaled around 0).
    z = np.where(observed, x, 0.0)
    anti_idx = np.flatnonzero(anti_dims)
    score = z[:, anti_idx[0]] if len(anti_idx) else z[:, 0]
    h_mask = score >= 0
    a_mask = ~h_mask
    if h_mask.any():
        means[0] = z[h_mask].mean(axis=0)
    if a_mask.any():
        means[1] = _mirror_mean(means[0], anti_dims)
    else:
        means[1] = _mirror_mean(means[0], anti_dims)
    # Tie them from the start.
    pooled = 0.5 * (means[0] + _mirror_mean(means[1], anti_dims))
    means[0] = pooled
    means[1] = _mirror_mean(pooled, anti_dims)
    if n_states == 3:
        if alive_cue is not None and np.isfinite(alive_cue).any():
            thr = np.nanpercentile(alive_cue, 25)
            dead = np.isfinite(alive_cue) & (alive_cue <= thr)
        else:
            dead = np.zeros(len(z), bool)
            dead[: max(1, len(z) // 5)] = True
        if dead.any():
            means[2] = z[dead].mean(axis=0)
        else:
            means[2] = 0.0
        # Dead state's anti dims should be ~0.
        means[2, anti_dims] = 0.0
    return means


def fit_hmm(x: np.ndarray, observed: np.ndarray, anti_dims: np.ndarray,
            n_states: int = 3, n_iter: int = 25, sticky: float = 250.0,
            alive_cue: np.ndarray | None = None, fit_mask: np.ndarray | None = None,
            stride: int = 1) -> MirroredHMM:
    """Fit a mirrored Gaussian HMM by Baum-Welch on the rows where fit_mask is True.

    sticky   pseudo-count added to each self-transition (in frame units of the fit sequence)
    stride   keep every stride-th frame for EM (decode still runs at full rate elsewhere)
    """
    anti_dims = np.asarray(anti_dims, dtype=bool)
    n, D = x.shape
    if fit_mask is None:
        fit_mask = np.ones(n, bool)
    idx = np.flatnonzero(fit_mask)
    if stride > 1:
        idx = idx[::stride]
    xs = x[idx].astype(np.float64)
    obs = observed[idx]
    cue = alive_cue[idx] if alive_cue is not None else None

    means = _init_means(xs, obs, anti_dims, n_states, cue)
    vars_ = np.ones((n_states, D), dtype=np.float64)
    # Init vars from data spread.
    z = np.where(obs, xs, np.nan)
    col_var = np.nanvar(z, axis=0)
    col_var = np.where(np.isfinite(col_var) & (col_var > 1e-3), col_var, 1.0)
    vars_[:] = col_var
    trans = np.full((n_states, n_states), 0.05 / max(1, n_states - 1))
    np.fill_diagonal(trans, 0.95)
    start = np.full(n_states, 1.0 / n_states)

    model = MirroredHMM(means, vars_, trans, start, anti_dims, n_states,
                        STATE_NAMES_3 if n_states == 3 else STATE_NAMES_2)

    for _ in range(n_iter):
        ll = model.emission_loglik(xs, obs)
        gamma, xi_sum, _ = _forward_backward(ll, model.trans, model.start)

        # M-step: transitions with sticky prior.
        trans = xi_sum + 1e-3
        np.fill_diagonal(trans, np.diag(trans) + sticky)
        trans /= trans.sum(axis=1, keepdims=True)
        start = gamma[0] / gamma[0].sum()

        # M-step: mirrored means for H/A. Pool gamma_H with gamma_A on the mirrored data.
        g_h, g_a = gamma[:, 0], gamma[:, 1]
        # Effective observations for H: frames under H + mirrored frames under A.
        w_h = g_h[:, None] * obs
        w_a = g_a[:, None] * obs
        xs_a_mirror = xs.copy()
        xs_a_mirror[:, anti_dims] *= -1
        num = (w_h * xs).sum(axis=0) + (w_a * xs_a_mirror).sum(axis=0)
        den = w_h.sum(axis=0) + w_a.sum(axis=0) + 1e-300
        mean_h = num / den
        mean_a = _mirror_mean(mean_h, anti_dims)
        means[0], means[1] = mean_h, mean_a

        # Variances: same pooling. Shared across H/A (mirror leaves variance unchanged).
        diff_h = xs - mean_h
        diff_a = xs_a_mirror - mean_h  # A observations mirrored into H's frame
        var_num = (w_h * diff_h * diff_h).sum(axis=0) + (w_a * diff_a * diff_a).sum(axis=0)
        vars_[0] = np.maximum(var_num / den, 1e-3)
        vars_[1] = vars_[0]

        if n_states == 3:
            w_d = gamma[:, 2][:, None] * obs
            den_d = w_d.sum(axis=0) + 1e-300
            means[2] = (w_d * xs).sum(axis=0) / den_d
            means[2, anti_dims] = 0.0  # dead ball has no team preference
            diff_d = xs - means[2]
            vars_[2] = np.maximum((w_d * diff_d * diff_d).sum(axis=0) / den_d, 1e-3)

        model.means, model.vars, model.trans, model.start = means, vars_, trans, start

    # After fit, ensure state 0 (H) sits on the positive side of the leading anti dim.
    anti_idx = np.flatnonzero(anti_dims)
    if len(anti_idx) and means[0, anti_idx[0]] < means[1, anti_idx[0]]:
        means[[0, 1]] = means[[1, 0]]
        vars_[[0, 1]] = vars_[[1, 0]]
        trans[[0, 1]] = trans[[1, 0]]
        trans[:, [0, 1]] = trans[:, [1, 0]]
        start[[0, 1]] = start[[1, 0]]
        model.means, model.vars, model.trans, model.start = means, vars_, trans, start

    return model


def rescale_sticky_trans(trans: np.ndarray, stride: int) -> np.ndarray:
    """Convert a transition matrix fit at 1/stride rate back to full frame rate.

    If at stride rate P(stay)=p, then at full rate P(stay)=p^(1/stride).
    """
    if stride <= 1:
        return trans
    K = trans.shape[0]
    out = np.empty_like(trans)
    for i in range(K):
        stay = float(trans[i, i]) ** (1.0 / stride)
        stay = min(max(stay, 1e-6), 1.0 - 1e-6)
        leave = 1.0 - stay
        others = trans[i].copy()
        others[i] = 0.0
        s = others.sum()
        if s <= 0:
            out[i] = (1.0 - stay) / max(1, K - 1)
            out[i, i] = stay
        else:
            out[i] = others * (leave / s)
            out[i, i] = stay
    return out


def merge_short_spells(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Absorb runs shorter than min_len into their left neighbour (or right at the head)."""
    if min_len <= 1 or len(labels) == 0:
        return labels
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if (j - i) < min_len:
            fill = out[i - 1] if i > 0 else (out[j] if j < n else out[i])
            out[i:j] = fill
        i = j
    return out

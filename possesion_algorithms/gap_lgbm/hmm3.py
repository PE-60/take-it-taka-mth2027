"""Three-state HMM (H, A, dead) over the free frames of a segment, for gap_lgbm_dead.

A segment is a run of free frames between fixed ones; its anchors are the fixed states just
before and after it (+1 H, -1 A, 2 dead, 0 none). Because the chain is first-order, decoding each
free run between its fixed neighbours is the same as decoding the whole stretch around a
stoppage with those frames fixed.

States: H, A and a chain D1 .. Dk of dead sub-states. Entering the dead ball goes to D1, each
Di moves to Di+1, and only Dk may stay or restart, so a dead ball lasts at least k frames
(min_dead_s). A fixed dead anchor counts as Dk (already long enough).

    H -> A      p_switch            H -> D1     p_stop        H -> H   the rest
    Dk -> H/A   p_restart * restart prior of the segment     Dk -> Dk the rest

The restart prior gives the restart to the team that did not have the ball before the stoppage
with probability q (the restart rule; estimated from the teacher), and 0.5 / 0.5 when unknown.

Emissions are log values per frame and state, built by the caller (`emissions`): the two models'
probabilities divided by their training class rates, raised to tau, -inf where a state is not
allowed (dead only where the ball is missing).
"""

from __future__ import annotations

import numpy as np

from possesion_algorithms.gap_lgbm.intervals import Gaps

NEG = -1e18
H, A, D = 0, 1, 2


def emissions(p_h, p_dead, prior_dead: float, tau: float, allow_dead=None, dead_bonus=None):
    """(N, 3) log emissions for H, A, dead."""
    ph = np.clip(np.asarray(p_h, float), 1e-6, 1 - 1e-6)
    pd = np.clip(np.asarray(p_dead, float), 1e-6, 1 - 1e-6)
    prior = min(max(float(prior_dead), 1e-3), 1 - 1e-3)
    la = np.log((1 - pd) / (1 - prior))
    ld = np.log(pd / prior)
    e = np.stack([la + np.log(2 * ph), la + np.log(2 * (1 - ph)), ld], axis=1) * tau
    if dead_bonus is not None:
        e[:, D] += dead_bonus
    if allow_dead is not None:
        e[~np.asarray(allow_dead, bool), D] = NEG
    return e


def restart_logprob(last_team: np.ndarray, q: float) -> np.ndarray:
    """(G, 2) log P(restart by H), log P(restart by A) given the team that had the ball last."""
    q = min(max(float(q), 1e-3), 1 - 1e-3)
    lt = np.asarray(last_team)
    to_h = np.where(lt > 0, 1 - q, np.where(lt < 0, q, 0.5))
    return np.log(np.stack([to_h, 1 - to_h], axis=1))


def viterbi3(e: np.ndarray, segs: Gaps, restart: np.ndarray, p_switch: float, p_stop: float,
             p_restart: float, k: int, bucket: int = 128) -> np.ndarray:
    """Most likely state (0 H, 1 A, 2 dead) per frame; e is flat over segs.frames()."""
    k = max(2, int(k))
    out = np.zeros(len(e), np.int8)
    if segs.n == 0:
        return out
    lHH, lHA, lHD = np.log(max(1e-12, 1 - p_switch - p_stop)), np.log(p_switch), np.log(p_stop)
    lDD, lDR = np.log(1 - p_restart), np.log(p_restart)
    S = 2 + k
    DK = S - 1
    lens = segs.length
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    order = np.argsort(lens, kind="stable")
    for c in range(0, len(order), bucket):
        g = order[c:c + bucket]
        G, L = len(g), int(lens[g].max())
        t = np.arange(L)
        active = t[None, :] < lens[g][:, None]
        idx = np.where(active, starts[g][:, None] + t[None, :], 0)
        E = e[idx]                                                    # (G, L, 3)
        rH, rA = restart[g, 0], restart[g, 1]
        bef, aft = segs.before[g], segs.after[g]

        init = np.full((G, S), NEG)
        isH, isA, isD, none = bef == 1, bef == -1, bef == 2, bef == 0
        init[isH, H], init[isH, A], init[isH, 2] = lHH, lHA, lHD
        init[isA, A], init[isA, H], init[isA, 2] = lHH, lHA, lHD
        init[isD, H], init[isD, A], init[isD, DK] = lDR + rH[isD], lDR + rA[isD], lDD
        init[none, H] = init[none, A] = init[none, DK] = np.log(1 / 3)
        em = np.concatenate([E[:, 0, :2], np.repeat(E[:, 0, 2:3], k, axis=1)], axis=1)
        delta = init + em
        bpH = np.zeros((G, L), np.int8)
        bpA = np.zeros((G, L), np.int8)
        bpD1 = np.zeros((G, L), np.int8)
        bpDK = np.zeros((G, L), np.int8)
        for i in range(1, L):
            dH, dA, dK = delta[:, H], delta[:, A], delta[:, DK]
            cH = np.stack([dH + lHH, dA + lHA, dK + lDR + rH], axis=1)
            cA = np.stack([dA + lHH, dH + lHA, dK + lDR + rA], axis=1)
            bh, ba = np.argmax(cH, axis=1), np.argmax(cA, axis=1)
            new = np.empty_like(delta)
            new[:, H] = cH[np.arange(G), bh] + E[:, i, H]
            new[:, A] = cA[np.arange(G), ba] + E[:, i, A]
            from_a = dA > dH
            new[:, 2] = np.where(from_a, dA, dH) + lHD + E[:, i, D]
            if k > 2:
                new[:, 3:DK] = delta[:, 2:DK - 1] + E[:, i, D][:, None]
            stay = dK + lDD > delta[:, DK - 1]
            new[:, DK] = np.where(stay, dK + lDD, delta[:, DK - 1]) + E[:, i, D]
            act = active[:, i]
            delta = np.where(act[:, None], new, delta)
            bpH[:, i], bpA[:, i], bpD1[:, i], bpDK[:, i] = bh, ba, from_a, stay

        end = np.zeros((G, S))
        aH, aA, aD = aft == 1, aft == -1, aft == 2
        for mask, own, other, r in ((aH, H, A, rH), (aA, A, H, rA)):
            end[mask, own], end[mask, other] = lHH, lHA
            end[mask, 2:DK] = NEG
            end[mask, DK] = lDR + r[mask]
        end[aD, H] = end[aD, A] = lHD
        state = np.argmax(delta + end, axis=1)
        path = np.zeros((G, L), np.int8)
        last = lens[g] - 1
        rows = np.arange(G)
        for i in range(L - 1, -1, -1):
            on = i <= last
            path[on, i] = state[on]
            if i == 0:
                break
            s = state
            prev = s.copy()
            prev = np.where(s == H, np.choose(bpH[:, i], [H, A, DK]), prev)
            prev = np.where(s == A, np.choose(bpA[:, i], [A, H, DK]), prev)
            prev = np.where(s == 2, np.where(bpD1[:, i] == 1, A, H), prev)
            mid = (s > 2) & (s < DK)
            prev = np.where(mid, s - 1, prev)
            prev = np.where(s == DK, np.where(bpDK[:, i] == 1, DK, DK - 1), prev)
            state = np.where(on, prev, state)
        coll = np.where(path >= 2, D, path).astype(np.int8)
        out[idx[active]] = coll[active]
        del rows
    return out


def score(dec: np.ndarray, truth: np.ndarray) -> dict:
    """Accuracy over frames with a truth (0 H, 1 A, 2 dead; -1 ignored), and dead precision/recall."""
    ok = truth >= 0
    if not ok.any():
        return {"accuracy": float("nan")}
    d, t = dec[ok], truth[ok]
    tp = int(((d == D) & (t == D)).sum())
    alive = t != D
    return {"accuracy": float((d == t).mean()),
            "dead_precision": tp / max(1, int((d == D).sum())),
            "dead_recall": tp / max(1, int((t == D).sum())),
            "team_accuracy": float((d[alive] == t[alive]).mean()) if alive.any() else float("nan")}


def tune(e, segs, restart, truth, p_switch, k, stop_grid, restart_grid) -> dict:
    rows = []
    for ps in sorted(stop_grid):
        for pr in sorted(restart_grid):
            dec = viterbi3(e, segs, restart, p_switch, ps, pr, k)
            rows.append({"p_stop": float(ps), "p_restart": float(pr), **score(dec, truth)})
    best = max(rows, key=lambda r: (r["accuracy"], -r["p_stop"], -r["p_restart"]))
    return {"best": best, "grid": rows}

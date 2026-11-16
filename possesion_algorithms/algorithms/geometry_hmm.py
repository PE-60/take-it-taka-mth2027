"""Unsupervised possession from collective tracking geometry.

Pipeline (all numpy, no labels):

1. **Anomaly detection** — mask ghost balls nobody is playing, teleporting slots, and repair
   goalkeepers filed under the wrong team.
2. **Geometry features** — anti-symmetric team contrasts (who is nearer, who pushes up) plus
   symmetric alive/dead cues (ball speed, motion, how many are tracked).
3. **Dimensionality reduction** — robust scale; the possession axis defaults to
   ``control_gap`` plus in-flight ``recv_gap`` / ``lead_gap`` (future ball trajectory when
   tracked). Pass cues are NaN on the ground, so nanmean falls back to control alone.
   Sym PCA is only used when ``with_dead`` is on.
4. **Sticky mirrored HMM** — H and A are mirror images on the possession axis; D is optional.
   Fit once on the whole match in `prepare`, decode each slice in `label`.

Prefer ``geom_hmm_dead`` when ``dead_ball.json`` is available: the model then only separates H
from A on live intervals, which is what spellF1 rewards (see EVALUATION.md).

``fit_geometry`` returns every intermediate for ``inspect_geom_hmm``; the algorithm classes are a
thin wrapper around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from possesion_algorithms.base import TACTICAL, PossessionAlgorithm, Tracking
from possesion_algorithms.unsupervised.anomaly import AnomalyReport, detect_anomalies
from possesion_algorithms.unsupervised.features import Features, compute_features
from possesion_algorithms.unsupervised.hmm import (MirroredHMM, fit_hmm, merge_short_spells,
                                                    rescale_sticky_trans)
from possesion_algorithms.unsupervised.pca import PCAModel, fit_pca


# Ground: control_gap (with a dead-zone so weak presses do not flip the sticky HMM).
# Flight: lead_gap only — recv_gap is weaker on the public window and dilutes the landing cue.
DEFAULT_POSS_FEATURES = ("control_gap", "lead_gap")
FLIGHT_POSS_FEATURES = ("lead_gap",)

DEFAULT_PARAMS = {
    "n_anti": 3,
    "n_sym": 3,
    "var_frac": 0.90,
    "n_iter": 20,
    "sticky_s": 15.0,          # longer holds: spellF1 dies on mid-possession flips
    "fit_stride": 5,
    # 0 = keep sub-second spells (public GT has possessions under 1 s). Stickiness handles flicker.
    "min_spell_s": 0.0,
    "repair_gk": True,
    "with_dead": True,
    # MAD-scaled |ground axis| below this → 0 (contested / unclear → trust stickiness).
    "ground_deadzone": 0.40,
    # After a dead ball, the restarting team is the opposite of who had it just before.
    "restart_prior": True,
    "restart_peek_s": 0.6,
    # Comma-separated names, or a tuple/list when set in code.
    "poss_features": ",".join(DEFAULT_POSS_FEATURES),
}


def _parse_poss_features(value, names: tuple) -> tuple[str, ...]:
    """Accept a comma-separated string, or a sequence of feature names."""
    if isinstance(value, str):
        parts = tuple(p.strip() for p in value.split(",") if p.strip())
    else:
        parts = tuple(value)
    known = set(names)
    missing = [p for p in parts if p not in known]
    if missing:
        raise ValueError(f"unknown poss_features {missing}; choose from {list(names)}")
    return parts or DEFAULT_POSS_FEATURES


def _live_runs(live: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs where live is True."""
    m = np.concatenate([[False], np.asarray(live, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def apply_restart_opposition(labels: np.ndarray, live: np.ndarray, fps: float,
                             peek_s: float = 0.6) -> np.ndarray:
    """Throw-in / corner / goal-kick prior: restarter after a stoppage ≠ team that had it before.

    For each dead gap with live play on both sides, read who restarts in the first ``peek_s``
    of the following live run, and set the trailing spell of the previous live run to the
    opposite team. Kick-offs after a goal also match this (conceding side restarts).
    """
    out = labels.copy()
    opposite = {"H": "A", "A": "H"}
    peek = max(1, int(round(float(peek_s) * float(fps))))
    runs = _live_runs(live)
    for prev, nxt in zip(runs, runs[1:]):
        a0, b0 = prev
        a1, b1 = nxt
        if a1 <= b0:  # contiguous live — no dead gap
            continue
        head = out[a1:min(b1, a1 + peek)]
        teams, counts = np.unique(head[np.isin(head, ("H", "A"))], return_counts=True)
        if not len(teams):
            continue
        restart = str(teams[int(np.argmax(counts))])
        want = opposite[restart]
        # Trailing spell of the previous live run.
        t = b0 - 1
        last = out[t]
        if last not in ("H", "A"):
            continue
        while t > a0 and out[t - 1] == last:
            t -= 1
        out[t:b0] = want
    return out


@dataclass
class GeometryFit:
    """Everything ``prepare`` computes — kept for inspection, not just the final labels."""
    track_start: int
    labels: np.ndarray              # (n,) on the tracking stretch
    live: np.ndarray
    features: Features
    anomaly: AnomalyReport
    pca: PCAModel
    poss: np.ndarray                # (n,) possession axis (anti nan-mean)
    z_sym: np.ndarray               # (n, k_sym)
    x: np.ndarray                   # (n, 1+k_sym) HMM observations
    observed: np.ndarray
    anti_dims: np.ndarray
    alive_cue: np.ndarray
    model: MirroredHMM
    params: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.labels)


def fit_geometry(track: Tracking, live: np.ndarray | None = None,
                 params: dict | None = None) -> GeometryFit:
    """Run the full unsupervised fit on ``track`` and return intermediates + labels."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n = track.n
    live = np.ones(n, bool) if live is None else np.asarray(live, dtype=bool)
    if n == 0 or track.n_slots == 0:
        empty = np.full(n, "", dtype="<U1")
        # Minimal placeholders so callers can still attribute-access safely.
        report = detect_anomalies(track, repair_gk=False) if n else AnomalyReport(
            np.zeros(0, bool), np.zeros((0, 0), bool), np.array([], dtype="<U1"), 0, 0, [])
        feats = compute_features(track) if n else None
        return GeometryFit(
            track_start=track.start, labels=empty, live=live, features=feats, anomaly=report,
            pca=None, poss=np.zeros(n), z_sym=np.zeros((n, 0)), x=np.zeros((n, 0)),
            observed=np.zeros((n, 0), bool), anti_dims=np.zeros(0, bool),
            alive_cue=np.zeros(n), model=None, params=p, diag={},
        )

    fps = float(track.fps)
    report = detect_anomalies(track, repair_gk=bool(p["repair_gk"]))
    feats = compute_features(track, team=report.team, ball_mask=report.ball_mask,
                             player_mask=report.player_mask)
    anchor = feats.column("near_gap")
    if not np.isfinite(anchor).any():
        anchor = feats.column("territory")
    pca = fit_pca(feats.values, feats.anti, feats.names,
                  n_anti=int(p["n_anti"]), n_sym=int(p["n_sym"]),
                  var_frac=float(p["var_frac"]), anti_anchor=anchor)
    z_anti_raw, _ = pca.scale(feats.values, feats.anti)
    _, z_sym = pca.transform(feats.values, feats.anti)

    poss_names = _parse_poss_features(p["poss_features"], feats.names)
    # Index into the anti block of z_anti_raw (same order as feats.anti columns).
    anti_idx = np.flatnonzero(feats.anti)
    anti_name_to_j = {feats.names[i]: j for j, i in enumerate(anti_idx)}

    def _mean_cols(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        cols = [anti_name_to_j[nm] for nm in names if nm in anti_name_to_j]
        if not cols:
            return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=bool)
        block = z_anti_raw[:, cols]
        finite = np.isfinite(block)
        count = finite.sum(axis=1).astype(np.float64)
        out = np.zeros(n, dtype=np.float64)
        np.divide(np.where(finite, block, 0.0).sum(axis=1), count, out=out, where=count > 0)
        return out, count > 0

    # Prefer trajectory cues while they are defined (ball in flight with a visible future).
    flight_names = tuple(nm for nm in poss_names if nm in FLIGHT_POSS_FEATURES)
    ground_names = tuple(nm for nm in poss_names if nm not in FLIGHT_POSS_FEATURES) or poss_names
    flight_poss, in_flight = _mean_cols(flight_names) if flight_names else (
        np.zeros(n, dtype=np.float64), np.zeros(n, dtype=bool))
    ground_poss, _ = _mean_cols(ground_names)
    deadzone = float(p.get("ground_deadzone", 0.0))
    if deadzone > 0:
        ground_poss = np.where(np.abs(ground_poss) < deadzone, 0.0, ground_poss)
    # Ball moving fast but no usable lead yet: do not trust "runner has the ball" — emit
    # neutral and let stickiness / lead_gap (when it appears) decide.
    speed = np.expm1(np.clip(feats.column("ball_speed"), 0, None))  # undo log1p ≈ m/s
    ball_fast = np.isfinite(speed) & (speed >= 6.0) & feats.ball_seen
    poss = np.where(in_flight, flight_poss,
                    np.where(ball_fast, 0.0, ground_poss))

    # 2-state H/A: sym dims are tied under the mirror constraint, so they do not separate
    # teams. Keep them only when a dead state is in play.
    use_dead = bool(p["with_dead"])
    if use_dead:
        x = np.concatenate([poss[:, None], z_sym.astype(np.float64)], axis=1)
        observed = np.ones_like(x, dtype=bool)
        observed[:, 1:] = np.isfinite(z_sym)
    else:
        x = poss[:, None].copy()
        observed = np.ones((n, 1), dtype=bool)
    anti_dims = np.zeros(x.shape[1], bool)
    anti_dims[0] = True
    alive_stack = np.column_stack([
        feats.column("ball_speed"),
        feats.column("players_moving"),
        1.0 - feats.column("ball_missing"),
    ])
    alive_ok = np.isfinite(alive_stack)
    alive_cue = np.where(alive_ok.any(axis=1),
                         np.nansum(np.where(alive_ok, alive_stack, 0.0), axis=1)
                         / np.maximum(alive_ok.sum(axis=1), 1),
                         np.nan)

    n_states = 3 if use_dead else 2
    stride = max(1, int(p["fit_stride"]))
    sticky = float(p["sticky_s"]) * fps / stride
    model = fit_hmm(
        x, observed, anti_dims,
        n_states=n_states, n_iter=int(p["n_iter"]), sticky=sticky,
        alive_cue=alive_cue, fit_mask=live, stride=stride,
    )
    if stride > 1:
        model.trans = rescale_sticky_trans(model.trans, stride)

    states = model.decode(x, observed)
    labels = model.labels(states).copy()
    labels[~live] = ""
    min_spell_s = float(p["min_spell_s"])
    if min_spell_s > 0:
        min_len = max(1, int(round(min_spell_s * fps)))
        if live.all():
            labels = merge_short_spells(labels, min_len)
        else:
            for a, b in _live_runs(live):
                labels[a:b] = merge_short_spells(labels[a:b], min_len)
    if bool(p.get("restart_prior", False)) and not live.all():
        labels = apply_restart_opposition(labels, live, fps, float(p.get("restart_peek_s", 0.6)))
        labels[~live] = ""

    diag = {
        "n_ball_anomalies": report.n_ball,
        "n_player_anomaly_slots": report.n_player,
        "gk_flips": report.gk_flips,
        "poss_features": list(poss_names),
        "n_anti": 1,
        "n_sym": int(z_sym.shape[1]) if use_dead else 0,
        "n_states": n_states,
        "counts": {k: int((labels == k).sum()) for k in ("H", "A", "D", "")},
        "poss_mean": float(np.nanmean(poss)),
        "poss_std": float(np.nanstd(poss)),
        "in_flight_frac": float(in_flight.mean()),
        "ball_fast_neutral_frac": float((ball_fast & ~in_flight).mean()),
    }
    return GeometryFit(
        track_start=track.start, labels=labels, live=live, features=feats, anomaly=report,
        pca=pca, poss=poss, z_sym=z_sym, x=x, observed=observed, anti_dims=anti_dims,
        alive_cue=alive_cue, model=model, params=p, diag=diag,
    )


def decode_sticky_axis(axis: np.ndarray, live: np.ndarray, fps: float,
                       sticky_s: float = 10.0, min_spell_s: float = 0.0,
                       n_iter: int = 15, fit_stride: int = 5) -> np.ndarray:
    """2-state mirrored HMM on a single 1D axis — used for near_gap / poss ablations."""
    n = len(axis)
    x = np.asarray(axis, dtype=np.float64).reshape(n, 1)
    x = np.where(np.isfinite(x), x, 0.0)
    observed = np.ones((n, 1), dtype=bool)
    anti_dims = np.array([True])
    stride = max(1, int(fit_stride))
    sticky = float(sticky_s) * float(fps) / stride
    model = fit_hmm(x, observed, anti_dims, n_states=2, n_iter=n_iter, sticky=sticky,
                    fit_mask=live, stride=stride)
    if stride > 1:
        model.trans = rescale_sticky_trans(model.trans, stride)
    labels = model.labels(model.decode(x, observed)).copy()
    labels[~live] = ""
    if float(min_spell_s) > 0:
        min_len = max(1, int(round(float(min_spell_s) * fps)))
        for a, b in _live_runs(live):
            labels[a:b] = merge_short_spells(labels[a:b], min_len)
    return labels


class GeometryHMM(PossessionAlgorithm):
    """Geometry features -> anti-mean + sym PCA -> sticky mirrored HMM; whole video, tactical."""

    name = "geom_hmm"
    description = ("Unsupervised geometry HMM: anomaly-mask ATD, ball-local possession axis, "
                   "sticky H/A[/D] Viterbi; whole video, tactical")
    dead_ball = False
    source = TACTICAL
    params = {**DEFAULT_PARAMS}

    def __init__(self, **params):
        super().__init__(**params)
        self._fit: GeometryFit | None = None

    def prepare(self, track: Tracking, live: np.ndarray | None = None) -> None:
        self._fit = fit_geometry(track, live, self.params)

    def label(self, track: Tracking) -> np.ndarray:
        if self._fit is None:
            self.prepare(track, np.ones(track.n, bool))
        assert self._fit is not None
        a = track.start - self._fit.track_start
        b = a + track.n
        if a < 0 or b > self._fit.n:
            self.prepare(track, np.ones(track.n, bool))
            return self._fit.labels.copy()
        return self._fit.labels[a:b].copy()

    @property
    def fit(self) -> GeometryFit | None:
        return self._fit

    @property
    def _diag(self) -> dict | None:
        return None if self._fit is None else self._fit.diag


class GeometryHMMDeadBall(GeometryHMM):
    """Same model; runner supplies alive intervals and writes D on dead balls."""

    name = "geom_hmm_dead"
    description = ("geom_hmm fit/decoded on each alive interval of dead_ball.json; D on dead "
                   "balls; tactical")
    dead_ball = True
    params = {**DEFAULT_PARAMS, "with_dead": False}

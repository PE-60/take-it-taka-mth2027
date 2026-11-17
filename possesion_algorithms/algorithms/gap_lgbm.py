"""LightGBM possession inside ball-missing gaps, trained on pretend gaps, decoded by an HMM.

Visible-ball frames keep lh_alt_gk_dead's labels (the teacher: lh_alternative_V2 on the corrected
goalkeeper tracks, each alive interval on its own). Each run of alive frames where the ball is
missing, and which the teacher enters with a team, gets P(H) per frame from a LightGBM model
that never sees the ball — only both teams' shape, speed and pressure, and the team before /
after the gap — and a two-state HMM with fixed ends turns that into at most a few team changes.
Gaps at the start of play, before the teacher's first holder, keep the teacher's empty label.

The corrected keepers replace the raw keeper slots once, before anything else, so the teacher
and the features both see them. The correction is for the tactical recording, so the source
is tactical; with corrected_gk=false any source works on the raw teams.

The model is trained on pretend gaps cut out of visible play, with grouped CV over time
blocks; the per-fold table is printed and stored in the run's .meta.json under "report".
See possesion_algorithms/gap_lgbm/. p_switch and tau at 0 mean "tuned in CV".
"""

from __future__ import annotations

import numpy as np

from possesion_algorithms.algorithms.lh_alternative_V2 import (LinkHoernigAlt, LinkHoernigAltGkDead,
                                                               load_derived_goalkeepers,
                                                               with_derived_goalkeepers)
from possesion_algorithms.base import TACTICAL, PossessionAlgorithm, Tracking
from possesion_algorithms.paths import GOALKEEPER_CSV


def corrected_goalkeeper_tracking(track: Tracking, csv_path: str, raw_trackers: tuple[str, ...]) -> Tracking:
    """`track` with the raw keeper slots dropped and the corrected home/away keepers appended."""
    derived = load_derived_goalkeepers(csv_path, track.start, track.n)
    xy, team = with_derived_goalkeepers(track, derived, raw_trackers)
    raw = {str(t) for t in raw_trackers}
    ids = [str(s) for s in track.slot_ids if str(s) not in raw] + ["gk_home", "gk_away"]
    return Tracking(track.start, track.ball, xy, team, track.kick, track.origin, track.pitch, track.fps,
                    np.asarray(ids, dtype=object))


class GapLGBM(PossessionAlgorithm):
    name = "gap_lgbm"
    description = ("lh_alt_gk_dead on visible ball; LightGBM (team shape, pressure, anchors) + HMM "
                   "inside ball-missing gaps the teacher enters with a team; corrected keepers; "
                   "dead-ball intervals D; tactical")
    dead_ball = True
    source = TACTICAL
    params = {
        **LinkHoernigAlt.params,        # teacher
        "corrected_gk": True,           # swap in the corrected goalkeepers (tactical only)
        "goalkeeper_csv": "",           # empty -> data_correction/goalkeeper_trackers/goalkeeper_tracks.csv
        "raw_gk_trackers": LinkHoernigAltGkDead.params["raw_gk_trackers"],
        "num_leaves": 15,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "learning_rate": 0.05,
        "n_folds": 5,
        "block_s": 180.0,               # CV group: contiguous time blocks
        "stride": 4,                    # train on every 4th gap frame
        "coverage": 1.0,                # pretend-gap frames per visible frame
        "context_s": 1.0,               # visible ball kept on both sides of a pretend gap
        "kick_bias": 0.5,               # share of pretend gaps that start right after a kick
        "max_gap_s": 30.0,              # longest target-gap length copied to pretend gaps
        "p_switch": 0.0,                # HMM cost of a team change per frame; 0 = tuned in CV
        "tau": 0.0,                     # HMM emission temperature; 0 = tuned in CV
        "seed": 0,
        "cache": True,                  # cache teacher labels + features in gap_lgbm/cache/
    }

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["corrected_gk"] and self.source != TACTICAL:
            raise ValueError(f"{self.name}: the corrected goalkeepers are for the tactical recording; "
                             f"use --source tactical, or --param corrected_gk=false for {self.source}")
        self._labels: np.ndarray | None = None
        self._start: int | None = None
        self.report: dict = {}

    def _csv_path(self) -> str:
        return str(self.params["goalkeeper_csv"] or "").strip() or GOALKEEPER_CSV

    def _raw_trackers(self) -> tuple[str, ...]:
        ids = tuple(p.strip() for p in str(self.params["raw_gk_trackers"]).split(",") if p.strip())
        if len(ids) != 2:
            raise ValueError(f"{self.name}: raw_gk_trackers must be two ids, got {self.params['raw_gk_trackers']!r}")
        return ids

    def prepare(self, track: Tracking, live: np.ndarray | None = None) -> None:
        from possesion_algorithms.gap_lgbm.pipeline import fit_and_label
        live = np.ones(track.n, bool) if live is None else live
        if self.params["corrected_gk"]:
            track = corrected_goalkeeper_tracking(track, self._csv_path(), self._raw_trackers())
        self._labels, self.report = fit_and_label(track, live, self.params)
        self._start = track.start

    def label(self, track: Tracking) -> np.ndarray:
        if self._labels is None:
            self.prepare(track)
        a = track.start - self._start
        return self._labels[a:a + track.n]

    def settings(self) -> dict:
        out = super().settings()
        if self.params["corrected_gk"]:
            out["params"] = {**out["params"], "goalkeeper_csv": self._csv_path()}
        if self.report:
            out["report"] = self.report
        return out


class GapLGBMDead(GapLGBM):
    """gap_lgbm plus a dead-vs-alive model and a three-state (H, A, dead) HMM.

    Decodes every frame the ball leaves unclear — ball missing in play or in a dead interval, and
    the start of play before the teacher's first holder — between the frames that are fixed:
    the teacher's team where the ball is seen in play, D where the ball is seen in a dead
    interval (and on no_counterpart). Dead balls can be added, extended or shortened only where
    the ball is missing; a dead ball lasts at least min_dead_s. Writes D itself, so the runner's
    dead-ball mode is off and dead_ball.json is read here (dead_ball_json). See
    possesion_algorithms/gap_lgbm/pipeline_dead.py. p_stop, p_restart, q at 0 = tuned / estimated.
    """

    name = "gap_lgbm_dead"
    description = ("gap_lgbm + dead-vs-alive LightGBM + 3-state HMM (H, A, dead) over ball-missing "
                   "frames and the start of play; dead balls adjusted only where the ball is missing; tactical")
    dead_ball = False
    params = {
        **GapLGBM.params,
        "dead_ball_json": "",           # empty -> dead_ball_investigation/dead_ball.json
        "edge_margin_s": 1.0,           # detector boundaries: frames this close are not training labels
        "dead_coverage": 0.5,           # random dead-model window frames per frame
        "boundary_windows": 4,          # dead-model windows straddling each dead-ball start / end
        "max_window_s": 60.0,           # longest window length
        "min_dead_s": 1.0,              # a dead ball lasts at least this long
        "detector_logodds": 1.0,        # bonus for dead on frames the detector calls dead
        "p_stop": 0.0,                  # alive -> dead per frame; 0 = tuned
        "p_restart": 0.0,               # dead -> alive per frame (after min_dead_s); 0 = tuned
        "q": 0.0,                       # P(restart by the team without the ball before); 0 = estimated
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.masks: dict | None = None  # set to skip reading dead_ball.json (tests)

    def prepare(self, track: Tracking, live: np.ndarray | None = None) -> None:
        from possesion_algorithms.gap_lgbm.pipeline_dead import fit_and_label_dead, load_dead_masks
        from possesion_algorithms.paths import DEAD_BALL_JSON
        if self.params["corrected_gk"]:
            track = corrected_goalkeeper_tracking(track, self._csv_path(), self._raw_trackers())
        masks = self.masks
        if masks is None:
            path = str(self.params["dead_ball_json"] or "").strip() or DEAD_BALL_JSON
            masks = load_dead_masks(path, track.start, track.n)
        self._labels, self.report = fit_and_label_dead(track, masks, self.params)
        self._start = track.start


class LGBMShape(PossessionAlgorithm):
    """Test: possession from team shape alone on every alive frame; dead-ball intervals D.

    The LightGBM team model sees no ball, no gap context and no anchors; it is trained on the
    teacher's labels (lh_alternative_V2 on the corrected keepers) where the teacher sees the ball,
    and then labels every alive frame itself, through a two-state HMM per alive interval. Nothing
    of the teacher's output is kept. See possesion_algorithms/gap_lgbm/shape_model.py.
    """

    name = "lgbm_shape"
    description = ("test: LightGBM on team shape only (no ball) labels every alive frame, HMM per alive "
                   "interval; trained on lh_alt_gk_dead labels; dead-ball intervals D; tactical")
    dead_ball = True
    source = TACTICAL
    params = {
        **LinkHoernigAlt.params,        # teacher (training labels only)
        "corrected_gk": True,
        "goalkeeper_csv": "",
        "raw_gk_trackers": LinkHoernigAltGkDead.params["raw_gk_trackers"],
        "num_leaves": 15,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "learning_rate": 0.05,
        "n_folds": 5,
        "block_s": 180.0,
        "purge_s": 1.0,                 # frames dropped at each CV block edge
        "stride": 4,
        "p_switch": 0.0,                # 0 = tuned in CV
        "tau": 0.0,                     # 0 = tuned in CV
        "seed": 0,
        "cache": True,
    }

    _csv_path = GapLGBM._csv_path
    _raw_trackers = GapLGBM._raw_trackers

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["corrected_gk"] and self.source != TACTICAL:
            raise ValueError(f"{self.name}: the corrected goalkeepers are for the tactical recording; "
                             f"use --source tactical, or --param corrected_gk=false for {self.source}")
        self._labels: np.ndarray | None = None
        self._start: int | None = None
        self.report: dict = {}

    def prepare(self, track: Tracking, live: np.ndarray | None = None) -> None:
        from possesion_algorithms.gap_lgbm.pipeline import load_or_compute
        from possesion_algorithms.gap_lgbm.shape_model import fit_and_label_shape
        live = np.ones(track.n, bool) if live is None else np.asarray(live, bool)
        if self.params["corrected_gk"]:
            track = corrected_goalkeeper_tracking(track, self._csv_path(), self._raw_trackers())
        live = live & (track.origin >= 0)
        teacher, tf = load_or_compute(track, live, self.params, bool(self.params["cache"]))
        self._labels, self.report = fit_and_label_shape(teacher, tf, live, np.isfinite(track.ball[:, 0]),
                                                        self.params)
        self._start = track.start

    label = GapLGBM.label

    def settings(self) -> dict:
        out = super().settings()
        if self.params["corrected_gk"]:
            out["params"] = {**out["params"], "goalkeeper_csv": self._csv_path()}
        if self.report:
            out["report"] = self.report
        return out


class GapLGBMDouble(GapLGBMDead):
    """gap_lgbm_dead plus a ball-aware LightGBM for the frames where the ball is seen.

    Steps: classical dead balls; lh_alternative_V2 (corrected keepers) on each alive interval;
    the team and dead models of gap_lgbm_dead for ball-missing frames and the start of play; and
    a second LightGBM with ball features (nearest players now and in 1 s, players near the ball,
    ball position / velocity, attacking direction, kicks) trained on the teacher's labels where
    the ball is seen, used only out of fold. Every frame between fixed dead balls is then labelled
    from those models: frame by frame by default, or with use_hmm=true by one three-state HMM
    (switching cost, minimum dead-ball length, restart rule). Either way the teacher can be
    overruled where the ball model disagrees with it. See possesion_algorithms/gap_lgbm/
    pipeline_double.py. tau_ball at 0 = tuned in CV (HMM only).
    """

    name = "gap_lgbm_double"
    description = ("gap_lgbm_dead + ball-aware LightGBM (out of fold, trained on lh_alt_gk_dead) on ball-seen "
                   "frames; every frame between fixed dead balls labelled from the models (HMM optional); tactical")
    params = {
        **GapLGBMDead.params,
        "purge_s": 1.0,                 # ball model: frames dropped at each CV block edge
        "tau_ball": 0.0,                # HMM temperature on ball-seen frames; 0 = tuned
        "use_hmm": False,               # false: label each frame on its own (no HMM at the end)
    }

    def prepare(self, track: Tracking, live: np.ndarray | None = None) -> None:
        from possesion_algorithms.gap_lgbm.pipeline_dead import load_dead_masks
        from possesion_algorithms.gap_lgbm.pipeline_double import fit_and_label_double
        from possesion_algorithms.paths import DEAD_BALL_JSON
        if self.params["corrected_gk"]:
            track = corrected_goalkeeper_tracking(track, self._csv_path(), self._raw_trackers())
        masks = self.masks
        if masks is None:
            path = str(self.params["dead_ball_json"] or "").strip() or DEAD_BALL_JSON
            masks = load_dead_masks(path, track.start, track.n)
        self._labels, self.report = fit_and_label_double(track, masks, self.params)
        self._start = track.start

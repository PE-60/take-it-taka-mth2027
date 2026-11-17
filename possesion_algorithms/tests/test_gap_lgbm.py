"""Synthetic tests for gap_lgbm, no match data required.

    python -m unittest possesion_algorithms.tests.test_gap_lgbm
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from possesion_algorithms.gap_lgbm.features import hull_area  # noqa: E402
from possesion_algorithms.gap_lgbm.intervals import Gaps, sample_pretend_gaps, target_gaps  # noqa: E402
from possesion_algorithms.gap_lgbm.model import auc, baseline_prob, viterbi_gaps  # noqa: E402

try:
    import lightgbm  # noqa: F401
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False


class IntervalsTest(unittest.TestCase):
    def test_target_gaps_and_anchors(self):
        live = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0], bool)
        seen = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1], bool)
        teacher = np.array(list("HHHHAAAHHAAA"))
        g = target_gaps(live, seen, teacher)
        self.assertEqual(g.a.tolist(), [2, 5, 8])
        self.assertEqual(g.b.tolist(), [4, 7, 9])
        self.assertEqual(g.before.tolist(), [1, -1, 0])     # last gap opens its alive run
        self.assertEqual(g.after.tolist(), [-1, 0, -1])     # middle gap runs into a dead ball
        rows, gap = g.frames()
        self.assertEqual(rows.tolist(), [2, 3, 5, 6, 8])
        self.assertEqual(gap.tolist(), [0, 0, 1, 1, 2])

    def test_require_team_skips_start_of_play(self):
        live = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], bool)
        seen = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1], bool)
        teacher = np.array(["", "", "", "H", "H", "H", "H", "A", "", "", "", "A", "A"])
        g = target_gaps(live, seen, teacher, require_team=True)
        # row 1 opens its alive run; the gap at 10 follows a frame without a team
        self.assertEqual(g.a.tolist(), [4, 6])
        self.assertEqual(g.before.tolist(), [1, 1])
        self.assertEqual(target_gaps(live, seen, teacher).n, 4)

    def test_pretend_gaps_keep_context_and_blocks(self):
        rng = np.random.default_rng(0)
        n = 5000
        teacher = np.where(np.arange(n) % 700 < 350, "H", "A")
        g = sample_pretend_gaps([(0, n)], np.array([10, 20, 40]), teacher, np.zeros(n, bool), rng,
                                coverage=1.0, context=25, block=1000)
        self.assertGreater(g.n, 10)
        self.assertTrue((g.a >= 25).all() and (g.b <= n - 25).all())
        self.assertTrue((g.a // 1000 == (g.b - 1) // 1000).all())
        self.assertTrue((g.a % 1000 >= 25).all())
        self.assertTrue(np.isin(g.length, [10, 20, 40]).all())


class FeaturesTest(unittest.TestCase):
    def test_hull_area(self):
        self.assertAlmostEqual(hull_area([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]), 4.0)
        self.assertEqual(hull_area([(0, 0), (1, 1)]), 0.0)


class ModelTest(unittest.TestCase):
    def test_auc(self):
        self.assertAlmostEqual(auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])), 0.75)

    def test_viterbi_fixed_ends_switch_once(self):
        g = Gaps(np.array([0]), np.array([20]), np.array([1], np.int8), np.array([-1], np.int8))
        p = np.r_[np.full(8, 0.9), np.full(12, 0.1)]
        dec = viterbi_gaps(p, g, p_switch=0.01, tau=1.0)
        self.assertEqual(dec.tolist(), [True] * 8 + [False] * 12)

    def test_viterbi_sceptical_of_short_blips(self):
        g = Gaps(np.array([0]), np.array([30]), np.array([1], np.int8), np.array([1], np.int8))
        p = np.full(30, 0.8)
        p[10:13] = 0.3                                        # a 3-frame blip towards A
        self.assertTrue(viterbi_gaps(p, g, p_switch=0.001, tau=0.5).all())

    def test_viterbi_handles_buckets_of_mixed_lengths(self):
        lens = np.array([3, 50, 1, 7])
        a = np.concatenate([[0], np.cumsum(lens)[:-1]]) * 2
        g = Gaps(a, a + lens, np.array([1, -1, 0, 1], np.int8), np.array([1, -1, 0, 0], np.int8))
        # 50 frames of weak H evidence (log-odds 0.08 each, 4 in total) cannot pay for two
        # team changes (2 * log(0.01) = -9.2), so the A/A anchors win
        p = np.concatenate([np.full(3, 0.2), np.full(50, 0.52), [0.7], np.full(7, 0.1)])
        dec = viterbi_gaps(p, g, p_switch=0.01, tau=1.0, bucket=2)
        self.assertTrue(dec[:3].all())                        # anchors H/H beat weak evidence
        self.assertFalse(dec[3:53].any())                     # anchors A/A
        self.assertTrue(dec[53])                              # no anchors: follows p
        self.assertFalse(dec[54:].all())

    def test_edge_features_and_mirroring(self):
        from possesion_algorithms.gap_lgbm.features import SYM, TeamFeatures, block_names
        from possesion_algorithms.gap_lgbm.model import EDGE_COLS, build_X, edge_features, feature_names
        n = 40
        ball = np.full((n, 2), np.nan, np.float32)
        ball[:20] = np.column_stack([np.linspace(40, 59, 20), np.full(20, 30.0)])   # 25 m/s along +x
        ball[30:] = np.column_stack([np.full(10, 80.0), np.full(10, 1.0)])          # still, 1 m from a line
        z = np.zeros((n, len(block_names())), np.float32)
        tf = TeamFeatures(z, z, np.zeros((n, len(SYM)), np.float32), np.ones(n, np.float32), block_names(),
                          25.0, (100.0, 50.0), ball)
        g = Gaps(np.array([20, 20]), np.array([30, 30]), np.array([1, 0], np.int8), np.array([-1, -1], np.int8))
        e = edge_features(tf, g)
        col = {c: i for i, (c, _) in enumerate(EDGE_COLS)}
        self.assertAlmostEqual(float(e[0, col["ball_before_x"]]), 9.0, places=4)       # 59 - 50
        self.assertAlmostEqual(float(e[0, col["ball_before_vx"]]), 25.0, places=3)
        self.assertAlmostEqual(float(e[0, col["ball_after_touchline_m"]]), 1.0, places=4)
        self.assertAlmostEqual(float(e[0, col["ball_after_speed"]]), 0.0, places=4)
        self.assertAlmostEqual(float(e[0, col["ball_shift_x"]]), 21.0, places=4)
        self.assertTrue(np.isnan(e[1, col["ball_before_x"]]))                          # anchor unknown
        rows, gap = g.frames()
        XH, XA = build_X(tf, g, rows, gap, True), build_X(tf, g, rows, gap, False)
        names = feature_names(tf)
        self.assertEqual(XH.shape[1], len(names))
        i = names.index("ball_before_x")
        self.assertAlmostEqual(float(XA[0, i]), -float(XH[0, i]), places=4)
        j = names.index("ball_after_touchline_m")
        self.assertAlmostEqual(float(XA[0, j]), float(XH[0, j]), places=4)

    def test_baselines(self):
        g = Gaps(np.array([0]), np.array([4]), np.array([1], np.int8), np.array([-1], np.int8))
        rows, gap = g.frames()
        self.assertEqual(baseline_prob(g, rows, gap, "midpoint").tolist(), [1, 1, 0, 0])
        self.assertEqual(baseline_prob(g, rows, gap, "after").tolist(), [0, 0, 0, 0])


@unittest.skipUnless(HAVE_LGBM, "lightgbm not installed")
class EndToEndTest(unittest.TestCase):
    def test_synthetic_match(self):
        from possesion_algorithms.algorithms import registry
        from possesion_algorithms.algorithms.gap_lgbm import GapLGBM
        from possesion_algorithms.unsupervised.synthetic import make_synthetic_match
        self.assertIn("gap_lgbm", registry())
        m = make_synthetic_match(n_frames=12000, seed=3)
        live = m.labels != "D"
        algo = GapLGBM(min_data_in_leaf=30, block_s=40.0, n_folds=3, cache=False, corrected_gk=False)
        algo.prepare(m.track, live)
        out = algo.label(m.track)
        self.assertEqual(out.shape, (m.track.n,))
        gap = live & ~np.isfinite(m.track.ball[:, 0])
        self.assertTrue(np.isin(out[gap], ["H", "A", ""]).all())
        self.assertTrue((out[~live] == "").all())
        report = algo.settings()["report"]
        self.assertIn("cv_folds", report)
        # skipped gaps keep the teacher's label: nothing before the first holder of a run
        self.assertGreaterEqual(report["skipped_gaps"]["n"], 1)


class GoalkeeperTest(unittest.TestCase):
    def test_corrected_keepers_replace_raw_slots(self):
        import tempfile
        from possesion_algorithms.algorithms.gap_lgbm import GapLGBM, corrected_goalkeeper_tracking
        from possesion_algorithms.base import TACTICAL_THEN_BROADCAST, Tracking
        n = 4
        xy = np.full((n, 3, 2), 0.5, np.float32)
        trk = Tracking(10, np.full((n, 2), 0.5, np.float32), xy, np.array(["H", "A", "H"]),
                       np.zeros(n, bool), np.zeros(n, np.int8), (100.0, 50.0),
                       slot_ids=np.array(["7", "24", "46"], dtype=object))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gk.csv")
            with open(path, "w") as fh:
                fh.write("frame,team_side,x,y\n11,home,0.05,0.5\n11,away,0.95,0.5\n")
            out = corrected_goalkeeper_tracking(trk, path, ("24", "46"))
        self.assertEqual(out.team.tolist(), ["H", "H", "A"])
        self.assertEqual(out.slot_ids.tolist(), ["7", "gk_home", "gk_away"])
        self.assertAlmostEqual(float(out.xy[1, 1, 0]), 0.05)
        self.assertTrue(np.isnan(out.xy[0, 1, 0]))
        with self.assertRaises(ValueError):
            type("T", (GapLGBM,), {"source": TACTICAL_THEN_BROADCAST})()



class DeadTest(unittest.TestCase):
    def test_viterbi3_minimum_dead_length(self):
        from possesion_algorithms.gap_lgbm import hmm3
        # 30 frames between H and A anchors; the dead model fires for only 5 frames
        e = hmm3.emissions(np.full(30, 0.5), np.r_[np.full(10, 0.1), np.full(5, 0.99), np.full(15, 0.1)], 0.2, 1.0)
        g = Gaps(np.array([0]), np.array([30]), np.array([1], np.int8), np.array([-1], np.int8))
        r = hmm3.restart_logprob(np.array([1]), 0.9)
        dec = hmm3.viterbi3(e, g, r, 0.01, 0.01, 0.01, k=10)
        dead = np.flatnonzero(dec == hmm3.D)
        self.assertTrue(len(dead) == 0 or len(dead) >= 10)
        if len(dead):
            self.assertTrue((np.diff(dead) == 1).all())

    def test_viterbi3_stoppage_then_restart_by_other_team(self):
        from possesion_algorithms.gap_lgbm import hmm3
        p_dead = np.r_[np.full(5, 0.05), np.full(40, 0.95), np.full(5, 0.02)]
        e = hmm3.emissions(np.full(50, 0.5), p_dead, 0.2, 1.0)
        g = Gaps(np.array([0]), np.array([50]), np.array([1], np.int8), np.array([0], np.int8))
        r = hmm3.restart_logprob(np.array([1]), 0.9)          # H had it: A restarts
        dec = hmm3.viterbi3(e, g, r, 0.001, 0.01, 0.05, k=25)
        self.assertTrue((dec[:5] == hmm3.H).all())
        self.assertTrue((dec[5:45] == hmm3.D).all())
        self.assertTrue((dec[45:] == hmm3.A).all())

    def test_viterbi3_dead_only_where_allowed(self):
        from possesion_algorithms.gap_lgbm import hmm3
        allow = np.r_[np.zeros(10, bool), np.ones(10, bool)]
        e = hmm3.emissions(np.full(20, 0.5), np.full(20, 0.99), 0.2, 1.0, allow_dead=allow)
        g = Gaps(np.array([0]), np.array([20]), np.array([1], np.int8), np.array([2], np.int8))
        dec = hmm3.viterbi3(e, g, hmm3.restart_logprob(np.array([1]), 0.5), 0.01, 0.01, 0.01, k=5)
        self.assertTrue((dec[:10] != hmm3.D).all())
        self.assertTrue((dec[10:] == hmm3.D).all())

    def test_dead_labels_margin_and_exclusions(self):
        from possesion_algorithms.gap_lgbm.dead_model import dead_labels
        n = 40
        dead = np.zeros(n, bool)
        dead[10:30] = True
        confident = dead.copy()
        confident[25:30] = False
        excluded = np.zeros(n, bool)
        excluded[0:3] = True
        y = dead_labels(dead, confident, excluded, np.ones(n, bool), margin=2)
        self.assertTrue(np.isnan(y[:3]).all())
        self.assertTrue((y[3:8] == 0).all())
        self.assertTrue(np.isnan(y[8:12]).all() and np.isnan(y[28:32]).all())
        self.assertTrue((y[12:25] == 1).all())
        self.assertTrue(np.isnan(y[25:28]).all())               # dead but not sure
        self.assertTrue((y[32:] == 0).all())

    def test_windows_stay_in_ok_runs_and_blocks(self):
        from possesion_algorithms.gap_lgbm.intervals import sample_windows
        ok = np.ones(3000, bool)
        ok[1000:1100] = False
        w = sample_windows(ok, np.array([20, 50]), np.random.default_rng(0), coverage=1.0,
                           boundaries=np.array([500, 1500]), per_boundary=5, block=700)
        self.assertGreater(w.n, 20)
        for a, b in zip(w.a, w.b):
            self.assertTrue(ok[a:b].all())
            self.assertEqual(a // 700, (b - 1) // 700)
        self.assertTrue(any(a < 500 < b for a, b in zip(w.a, w.b)))

    def test_load_dead_masks(self):
        import json
        import tempfile
        from possesion_algorithms.gap_lgbm.pipeline_dead import load_dead_masks
        doc = {"timelines": {"tactical": {"dead_ball": [[12, 14]], "terms": {
            "no_counterpart": [[10, 10]], "off_pitch": [[12, 13]], "still_gap": [], "baseline": [[14, 20]]}}}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dead.json")
            with open(path, "w") as fh:
                json.dump(doc, fh)
            m = load_dead_masks(path, 10, 8)
        self.assertEqual(np.flatnonzero(m["dead"]).tolist(), [2, 3, 4])
        self.assertEqual(np.flatnonzero(m["no_counterpart"]).tolist(), [0])
        self.assertEqual(np.flatnonzero(m["baseline"]).tolist(), [4, 5, 6, 7])

    @unittest.skipUnless(HAVE_LGBM, "lightgbm not installed")
    def test_end_to_end_synthetic(self):
        from possesion_algorithms.algorithms import registry
        from possesion_algorithms.algorithms.gap_lgbm import GapLGBMDead
        from possesion_algorithms.unsupervised.synthetic import make_synthetic_match
        self.assertIn("gap_lgbm_dead", registry())
        self.assertFalse(GapLGBMDead.dead_ball)
        m = make_synthetic_match(n_frames=12000, seed=3)
        dead = m.labels == "D"
        z = np.zeros(len(dead), bool)
        algo = GapLGBMDead(min_data_in_leaf=30, block_s=40.0, n_folds=3, cache=False, corrected_gk=False)
        algo.masks = {"dead": dead, "no_counterpart": z, "off_pitch": dead, "still_gap": z, "baseline": z}
        algo.prepare(m.track)
        out = algo.label(m.track)
        seen = np.isfinite(m.track.ball[:, 0])
        self.assertTrue((out[dead & seen] == "D").all())       # the detector decides where the ball is seen
        self.assertTrue(np.isin(out, ["H", "A", "D", ""]).all())
        report = algo.settings()["report"]
        for key in ("changes", "team_model", "dead_model", "hmm3"):
            self.assertIn(key, report)



@unittest.skipUnless(HAVE_LGBM, "lightgbm not installed")
class ShapeTest(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        from possesion_algorithms.algorithms import registry
        from possesion_algorithms.algorithms.gap_lgbm import LGBMShape
        from possesion_algorithms.unsupervised.synthetic import make_synthetic_match
        self.assertIn("lgbm_shape", registry())
        m = make_synthetic_match(n_frames=12000, seed=3)
        live = m.labels != "D"
        algo = LGBMShape(min_data_in_leaf=30, block_s=40.0, n_folds=3, cache=False, corrected_gk=False,
                         p_switch=0.003, tau=0.5)
        algo.prepare(m.track, live)
        out = algo.label(m.track)
        self.assertTrue(np.isin(out[live], ["H", "A"]).all())    # every alive frame from the model
        self.assertTrue((out[~live] == "").all())
        self.assertIn("cv_folds", algo.settings()["report"])


class BallFeaturesTest(unittest.TestCase):
    def test_nearest_counts_and_mirroring(self):
        from possesion_algorithms.base import Tracking
        from possesion_algorithms.gap_lgbm.ball_features import (BLOCK, SIGNED, build_ball_X,
                                                                 compute_ball_features)
        n = 60
        xy = np.full((n, 3, 2), np.nan, np.float32)
        xy[:, 0] = (0.50, 0.50)            # H, on the ball
        xy[:, 1] = (0.52, 0.50)            # A, 2 m away
        xy[:, 2] = (0.60, 0.50)            # H, 10 m away
        ball = np.full((n, 2), 0.5, np.float32)
        ball[40:] = (0.53, 0.50)           # from frame 40 the ball is 1 m from A, 3 m from H
        kick = np.zeros(n, bool)
        kick[10] = True
        trk = Tracking(0, ball, xy, np.array(["H", "A", "H"]), kick, np.zeros(n, np.int8), (100.0, 50.0))
        bf = compute_ball_features(trk, np.ones(n, np.float32))
        i = {c: j for j, c in enumerate(BLOCK)}
        s = {c: j for j, c in enumerate(SIGNED)}
        self.assertAlmostEqual(float(bf.H[0, i["near_m"]]), 0.0, places=4)
        self.assertAlmostEqual(float(bf.A[0, i["near_m"]]), 2.0, places=4)
        self.assertEqual(float(bf.A[0, i["n_3m"]]), 1.0)
        self.assertEqual(float(bf.signed[0, s["closest_team"]]), 1.0)        # H closest now
        self.assertEqual(float(bf.signed[20, s["closest_team_1s"]]), -1.0)   # A closest at frame 45
        XH, XA = build_ball_X(bf, np.array([0]), True), build_ball_X(bf, np.array([0]), False)
        k = len(BLOCK)
        self.assertTrue(np.allclose(XH[0, :k], XA[0, k:2 * k]))              # blocks swap
        self.assertEqual(float(XA[0, 2 * k + s["closest_team"]]), -1.0)      # signed columns flip


@unittest.skipUnless(HAVE_LGBM, "lightgbm not installed")
class DoubleTest(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        from possesion_algorithms.algorithms import registry
        from possesion_algorithms.algorithms.gap_lgbm import GapLGBMDouble
        from possesion_algorithms.unsupervised.synthetic import make_synthetic_match
        self.assertIn("gap_lgbm_double", registry())
        m = make_synthetic_match(n_frames=12000, seed=3)
        dead = m.labels == "D"
        z = np.zeros(len(dead), bool)
        seen = np.isfinite(m.track.ball[:, 0])
        for use_hmm in (False, True):
            algo = GapLGBMDouble(min_data_in_leaf=30, block_s=40.0, n_folds=3, cache=False, corrected_gk=False,
                                 p_switch=0.003, tau_ball=0.5, use_hmm=use_hmm)
            algo.masks = {"dead": dead, "no_counterpart": z, "off_pitch": dead, "still_gap": z, "baseline": z}
            algo.prepare(m.track)
            out = algo.label(m.track)
            self.assertTrue((out[dead & seen] == "D").all())       # fixed dead where the ball is seen
            self.assertTrue(np.isin(out[~dead & seen], ["H", "A"]).all())   # never dead where the ball is seen
            report = algo.settings()["report"]
            for key in ("changes", "ball_model", "team_model", "dead_model", "hmm3"):
                self.assertIn(key, report)
            self.assertEqual(report["changes"]["decoder"].startswith("hmm"), use_hmm)


if __name__ == "__main__":
    unittest.main()

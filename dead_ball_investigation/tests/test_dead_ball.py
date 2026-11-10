"""Synthetic tests, no match data required.

    python -m unittest dead_ball_investigation.tests.test_dead_ball
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dead_ball_investigation.dead_ball import (Params, SyncedMatch, annotate, annotate_dead_ball, ball_travel,
                                               dead_ball_mask, dead_ball_terms, intervals_to_mask,
                                               mask_to_intervals, runs_overlapping, to_document)
from dead_ball_investigation.signals import SourceSignals, homography_missing, load_from_export, runs_longer_than
from dead_ball_investigation.sync import SyncMap

DUAL_HTML = os.path.join(os.path.dirname(__file__), "..", "..", "match_viewer", "match_dual.html")
BASELINE = Params(counterpart_step=False, off_pitch_step=False, still_step=False)
PITCH = (100.0, 50.0)


def signals(source, n, start=0, **set_true):
    """All-good signals (ball seen, 22 players, homography valid) with some frames switched off."""
    ball = np.zeros(n, bool)
    hom = np.zeros(n, bool)
    count = np.full(n, 22)
    for f in set_true.get("ball", []):
        ball[f - start] = True
    for f in set_true.get("hom", []):
        hom[f - start] = True
    for f in set_true.get("count", []):
        count[f - start] = 18
    return SourceSignals(source, start, ball, count, hom)


def ball_signals(source, uv):
    """All-good signals with the ball at the pitch fractions `uv` (NaN = not seen)."""
    uv = np.asarray(uv, dtype=np.float64)
    n = len(uv)
    return SourceSignals(source, 0, ~np.isfinite(uv[:, 0]), np.full(n, 22), np.zeros(n, bool), uv * PITCH, PITCH)


class TestSyncMap(unittest.TestCase):
    def setUp(self):
        # 1st half: broadcast = tactical + 50; 2nd half from tactical 100: broadcast = tactical - 20
        self.sync = SyncMap([{"tactical": 10, "broadcast": 60, "label": "1st"},
                             {"tactical": 100, "broadcast": 80, "label": "2nd"}])

    def test_offsets_and_cut(self):
        m = self.sync.map
        self.assertEqual(m(0), 50)                  # before the first anchor: first offset
        self.assertEqual(m(10), 60)
        self.assertEqual(m(29), 79)                 # last tactical frame that still lands before anchor 2's broadcast
        self.assertEqual(m(30), -1)                 # would run past broadcast 80: cut
        self.assertEqual(m(99), -1)
        self.assertEqual(m(100), 80)
        self.assertEqual(m(150), 130)

    def test_inverse(self):
        m = self.sync.map
        self.assertEqual(m(60, "broadcast"), 10)
        self.assertEqual(m(79, "broadcast"), 29)
        self.assertEqual(m(80, "broadcast"), 100)
        self.assertEqual(m(49, "broadcast"), -1)    # would be tactical -1
        for t in (0, 15, 29, 100, 140):
            self.assertEqual(m(m(t), "broadcast"), t)

    def test_index_range(self):
        row = self.sync.index(0, 4, "tactical", 52, 200)
        self.assertEqual(row.tolist(), [-1, -1, 0, 1, 2])   # broadcast 50, 51 are outside 52..200

    def test_no_anchors_and_order(self):
        self.assertEqual(SyncMap([]).map(42), 42)
        with self.assertRaises(ValueError):
            SyncMap([{"tactical": 0, "broadcast": 100}, {"tactical": 10, "broadcast": 50}])

    @unittest.skipUnless(shutil.which("node") and os.path.isfile(DUAL_HTML), "node or match_dual.html not available")
    def test_matches_dual_viewer(self):
        """Same counterparts as SyncMap in match_dual.html, on both timelines."""
        with open(DUAL_HTML, encoding="utf-8") as fh:
            js = re.search(r"class SyncMap \{.*?\n\}\n", fh.read(), re.S).group(0)
        anchors = [{"t": a["tactical"], "b": a["broadcast"]} for a in self.sync.anchors]
        frames = list(range(-5, 200))
        script = ("const other = id => id === 't' ? 'b' : 't';\n" + js +
                  f"const s = new SyncMap({json.dumps(anchors)}); const F = {json.dumps(frames)};\n"
                  "console.log(JSON.stringify([F.map(f => s.map(f, 't')), F.map(f => s.map(f, 'b'))]));")
        out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
        for side, js_vals in zip(("tactical", "broadcast"), out):
            py = self.sync.map(np.array(frames), side).tolist()
            js_norm = [-1 if (v is None or v < 0) else v for v in js_vals]
            self.assertEqual(py, js_norm, side)


class TestFormula(unittest.TestCase):
    def test_truth_table(self):
        # one frame on its own: ball lost only counts together with no view, so dead == no view
        for a, b, c, d in itertools.product((0, 1), repeat=4):   # B hom missing, B count off, T ball, B ball
            no_view, ball_lost, kept, dead = dead_ball_terms([a], [b], [c], [d])
            self.assertEqual(bool(no_view[0]), bool(a and b))
            self.assertEqual(bool(ball_lost[0]), bool(c and d))
            self.assertEqual(bool(kept[0]), bool(a and b and c and d))
            self.assertEqual(bool(dead[0]), bool(a and b), (a, b, c, d))
            self.assertEqual(dead.tolist(), dead_ball_mask([a], [b], [c], [d]).tolist())

    def test_ball_lost_needs_overlap(self):
        nv = intervals_to_mask([[10, 12], [30, 31]], 0, 60)
        bl = intervals_to_mask([[5, 10], [13, 15], [20, 25], [28, 40], [50, 52]], 0, 60)
        on = np.ones(60, bool)
        dead = lambda gap=0: mask_to_intervals(dead_ball_mask(nv, on, bl, on, gap))
        self.assertEqual(dead(), [[5, 12], [28, 40]])           # [13, 15] only touches [10, 12]
        self.assertEqual(dead(1), [[5, 15], [28, 40]])
        self.assertEqual(dead(5), [[5, 15], [20, 25], [28, 40]])  # [20, 25] is 8 frames from 12 but 5 from 30
        self.assertEqual(runs_overlapping(bl, np.zeros(60, bool)).sum(), 0)
        self.assertEqual(mask_to_intervals(runs_overlapping(bl, nv)), [[5, 10], [28, 40]])

    def test_intervals_roundtrip(self):
        m = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
        iv = mask_to_intervals(m, start=100)
        self.assertEqual(iv, [[100, 101], [104, 104], [106, 108]])
        self.assertEqual(intervals_to_mask(iv, 100, len(m)).tolist(), m.tolist())
        self.assertEqual(mask_to_intervals(np.zeros(4, bool)), [])


class TestSyncedMatch(unittest.TestCase):
    def setUp(self):
        sync = SyncMap([{"tactical": 0, "broadcast": 10, "label": "1st"},
                        {"tactical": 60, "broadcast": 50, "label": "2nd"}])
        # tactical frames 0..99, broadcast frames 0..89. T 0..39 <-> B 10..49, T 40..59 unmapped, T 60..99 <-> B 50..89
        tac = signals("tactical", 100, ball=[5, 6, 7, 20, 45, 70])
        br = signals("broadcast", 90,
                     hom=[12, 13, 14, 60, 61, 62, 0, 1],      # T 2..4 and T 70..72; B 0, 1 have no tactical counterpart
                     count=[13, 14, 15, 61, 0, 1],            # overlap with hom: B 13, 14 (T 3, 4), B 61 (T 71), B 0, 1
                     ball=[15, 16, 30, 80])                   # T 5, 6 and T 20; B 80 = T 90 (tactical sees it)
        self.match = SyncedMatch(tac, br, sync, tags=[{"a": 2, "b": 8, "code": "SET PIECES", "type": "Throw-in", "team": "N/A"},
                                                       {"a": 45, "b": 65, "code": "X", "type": None, "team": None}])

    def test_tactical_timeline(self):
        r = annotate(self.match, "tactical", BASELINE)
        self.assertEqual(r.intervals("no_view"), [[3, 4], [71, 71]])
        self.assertEqual(r.intervals("ball_lost"), [[5, 6], [20, 20]])
        self.assertEqual(r.intervals("ball_lost_kept"), [])    # T 5, 6 touch no view at T 4 but do not overlap it
        self.assertEqual(r.intervals("dead"), [[3, 4], [71, 71]])
        self.assertEqual(r.intervals("covered"), [[0, 39], [60, 99]])
        self.assertFalse(r.dead[45])                           # tactical ball missing, no broadcast counterpart
        self.assertEqual(annotate(self.match, "tactical", replace(BASELINE, max_gap=1)).intervals("dead"),
                         [[3, 6], [71, 71]])

    def test_unmapped_signals_policy(self):
        r = annotate(self.match, "tactical", replace(BASELINE, unmapped_signals_missing=True))
        self.assertTrue(r.dead[40:60].all())                   # unmapped frames: B hom ∅ and count off assumed
        self.assertEqual(r.intervals("ball_lost"), [[5, 6], [20, 20], [45, 45]])
        self.assertEqual(r.intervals("ball_lost_kept"), [[45, 45]])  # inside the assumed no view

    def test_step1_no_counterpart_is_dead(self):
        res = annotate_dead_ball(self.match, replace(BASELINE, counterpart_step=True))
        t, b = res["tactical"], res["broadcast"]
        self.assertEqual(t.intervals("no_counterpart"), [[40, 59]])          # the cut between the anchors
        self.assertEqual(t.intervals("dead"), [[3, 4], [40, 59], [71, 71]])
        self.assertEqual(t.intervals("baseline"), [[3, 4], [71, 71]])
        self.assertEqual(b.intervals("no_counterpart"), [[0, 9]])           # B 0..9 come before tactical frame 0
        self.assertEqual(b.intervals("dead"), [[0, 9], [13, 14], [61, 61]])

    def test_broadcast_timeline_agrees(self):
        for params in (BASELINE, replace(BASELINE, max_gap=1), Params()):
            res = annotate_dead_ball(self.match, params)
            t, b = res["tactical"], res["broadcast"]
            for bf in range(90):
                tf = self.match.sync.map(bf, "broadcast")
                if tf >= 0:
                    self.assertEqual(bool(b.dead[bf]), bool(t.dead[tf]), f"{params}: B {bf} / T {tf}")
            self.assertTrue(b.no_view[0] and b.no_view[1])     # broadcast-only term holds without a counterpart
            # the baseline evaluated on the broadcast's own signals agrees with the tactical one
            for bf in range(10, 90):
                self.assertEqual(bool(b.baseline[bf]), bool(t.baseline[self.match.sync.map(bf, "broadcast")]))
        self.assertEqual(annotate(self.match, "broadcast", BASELINE).intervals("dead"), [[0, 1], [13, 14], [61, 61]])

    def test_tags_and_document(self):
        tags = self.match.synced_tags()
        self.assertEqual((tags[0]["broadcast_a"], tags[0]["broadcast_b"]), (12, 18))
        self.assertIsNone(tags[1]["broadcast_a"])              # T 45 is cut from the broadcast
        doc = to_document(self.match, annotate_dead_ball(self.match), "sync.json", graded=[[0, 49]])
        doc = json.loads(json.dumps(doc))
        tl = doc["timelines"]["tactical"]
        self.assertEqual(tl["dead_ball"], [[3, 4], [40, 59], [71, 71]])
        self.assertEqual(tl["terms"]["baseline"], [[3, 4], [71, 71]])
        self.assertEqual(tl["terms"]["ball_lost"], [[5, 6], [20, 20]])
        self.assertEqual(tl["terms"]["ball_lost_kept"], [])
        self.assertEqual(tl["terms"]["no_counterpart"], [[40, 59]])
        self.assertEqual(tl["stats"]["graded"]["dead_frames"], 12)
        self.assertEqual(tl["stats"]["still_gaps"]["unseen"], 2)   # the fixture has no ball positions
        self.assertEqual(doc["params"], Params().to_json())
        self.assertEqual(doc["timelines"]["broadcast"]["end"], 89)
        self.assertEqual(doc["smart_tags"][0]["broadcast"], [12, 18])

    def test_partial_ranges(self):
        """Sources loaded over part of their video only (e.g. a gt-mode export)."""
        tac = signals("tactical", 20, start=10, ball=[12])
        br = signals("broadcast", 90, ball=[22])
        br_hom = signals("broadcast", 90, ball=[22], hom=[21, 22], count=[22])
        frames = {"tactical": 100, "broadcast": 90}
        r = annotate(SyncedMatch(tac, br, self.match.sync, video_frames=frames), "tactical")
        self.assertEqual(r.start, 10)
        self.assertEqual(r.intervals("ball_lost"), [[12, 12]])
        self.assertEqual(r.intervals("dead"), [])               # ball lost without a no view next to it
        r = annotate(SyncedMatch(tac, br_hom, self.match.sync, video_frames=frames), "tactical")
        self.assertEqual(r.intervals("dead"), [[12, 12]])
        # counterpart in the tactical video but not loaded: not a step 1 frame
        b = annotate(SyncedMatch(tac, br, self.match.sync, video_frames=frames), "broadcast")
        self.assertEqual(b.intervals("no_counterpart"), [[0, 9]])
        self.assertEqual(b.intervals("dead"), [[0, 9]])


class TestStillGaps(unittest.TestCase):
    """Step 2 on one timeline: no-view intervals T/B 10-19, 30-39, 50-59, 70-79, 100-109 (sync offset 0)."""

    def setUp(self):
        n, still = 120, (50.0, 30.0)
        t_xy, b_xy = np.tile(still, (n, 1)), np.tile(still, (n, 1))
        t_xy[20:30] += np.c_[np.linspace(0, 0.4, 10), np.linspace(0, -0.4, 10)]   # A: jitters in place
        b_xy[20:30, 0] = np.linspace(40, 60, 10)                                 #    the broadcast ball moves
        t_xy[40:50, 0] = np.linspace(50, 53, 10)                                 # B: rolls 3 m
        t_xy[60:70] = np.nan                                                     # C: only the broadcast sees it
        t_xy[80:100], b_xy[80:100] = np.nan, np.nan                              # D: nobody sees it
        hom = [f for a in (10, 30, 50, 70, 100) for f in range(a, a + 10)]
        tac = SourceSignals("tactical", 0, ~np.isfinite(t_xy[:, 0]), np.full(n, 22), np.zeros(n, bool), t_xy)
        br = signals("broadcast", n, hom=hom, count=hom, ball=list(range(80, 100)))
        br.ball_xy = b_xy
        self.match = SyncedMatch(tac, br, SyncMap([{"tactical": 0, "broadcast": 0}]))

    def test_gaps(self):
        res = annotate_dead_ball(self.match)
        t, b = res["tactical"], res["broadcast"]
        self.assertEqual(t.intervals("still_gap"), [[20, 29], [60, 69]])
        self.assertEqual(t.intervals("dead"), [[10, 39], [50, 79], [100, 109]])
        self.assertEqual(t.gaps, {"gaps": 4, "filled": 2, "filled_frames": 20, "moving": 1, "unseen": 1})
        self.assertEqual(b.intervals("dead"), t.intervals("dead"))
        self.assertEqual(b.intervals("still_gap"), [[20, 29], [60, 69]])
        self.assertFalse(t.dead[:10].any() or t.dead[110:].any())   # no gap before the first or after the last

    def test_radius_and_switches(self):
        self.assertEqual(annotate(self.match, "tactical", Params(still_radius_m=3.5)).intervals("dead"),
                         [[10, 79], [100, 109]])
        self.assertEqual(annotate(self.match, "tactical", Params(still_step=False)).intervals("dead"),
                         [[10, 19], [30, 39], [50, 59], [70, 79], [100, 109]])

    def test_ball_travel(self):
        self.assertIsNone(ball_travel(np.full((4, 2), np.nan)))
        self.assertAlmostEqual(ball_travel([[0, 0], [np.nan, np.nan], [3, 4], [1, 0]]), 5.0)


class TestOffPitch(unittest.TestCase):
    """Step 2 at 25 fps, sync offset 0: off pitch means x or y ∉ [-0.002, 1.002] for more than 25 frames."""

    def test_ball_off_rule(self):
        uv = np.full((120, 2), 0.5)
        uv[0:30, 0] = 1.0019                  # inside the margin
        uv[30:56, 1] = -0.0021                # 26 frames off: kept
        uv[60:85, 0] = 1.05                   # exactly 25 frames (1 s): dropped
        uv[90:120, 0] = 1.05
        uv[105] = np.nan                      # a missing frame splits 90..119 into 15 + 14 frames
        off = ball_signals("tactical", uv).ball_off(25.0)
        self.assertEqual(mask_to_intervals(off), [[30, 55]])
        self.assertEqual(mask_to_intervals(ball_signals("tactical", uv).ball_off(25.0, min_s=0.5)),
                         [[30, 55], [60, 84], [90, 104], [106, 119]])
        self.assertEqual(mask_to_intervals(ball_signals("tactical", uv).ball_off(25.0, margin=0.001)),
                         [[0, 55]])
        self.assertEqual(mask_to_intervals(runs_longer_than([1, 1, 0, 1, 1, 1, 0, 1], 2)), [[3, 5]])

    def setUp(self):
        n = 200
        t, b = np.full((n, 2), 0.5), np.full((n, 2), 0.5)
        t[10:40, 0] = 1.05                    # tactical off 30 frames
        t[50:60, 0] = 1.05                    # tactical off 10 frames only: not off, and the tactical ball is seen
        b[50:90, 0] = 1.05                    # broadcast off 40 frames
        t[60:90] = np.nan                     #   so where the tactical ball is missing, the broadcast's counts
        b[100:140, 1] = -0.1                  # broadcast off, but the tactical ball is seen on the pitch
        t[150:180] = np.nan                   # tactical missing, broadcast on the pitch
        self.t, self.b = ball_signals("tactical", t), ball_signals("broadcast", b)
        self.match = SyncedMatch(self.t, self.b, SyncMap([{"tactical": 0, "broadcast": 0}]))

    def test_tactical_else_broadcast(self):
        res = annotate_dead_ball(self.match, replace(BASELINE, off_pitch_step=True))
        t, b = res["tactical"], res["broadcast"]
        self.assertEqual(t.intervals("off_pitch"), [[10, 39], [60, 89]])
        self.assertEqual(t.intervals("dead"), [[10, 39], [60, 89]])
        self.assertEqual(t.intervals("baseline"), [])
        self.assertEqual(b.intervals("off_pitch"), t.intervals("off_pitch"))
        self.assertEqual(b.intervals("dead"), t.intervals("dead"))
        self.assertEqual(annotate(self.match, "tactical", BASELINE).intervals("off_pitch"), [])

    def test_before_still_step(self):
        """Step 3 fills the gap between two step-2 intervals when the ball stays still in it."""
        t = self.t.ball_xy.copy()
        t[50:60] = t[40]                      # the short tactical run off the pitch becomes the ball sitting still
        tac = SourceSignals("tactical", 0, self.t.ball_missing, self.t.count, self.t.hom_missing, t, PITCH)
        match = SyncedMatch(tac, self.b, self.match.sync)
        r = annotate(match, "tactical", Params())
        self.assertEqual(r.intervals("still_gap"), [[40, 59]])
        self.assertEqual(r.intervals("dead"), [[10, 89]])
        self.assertEqual(annotate(match, "tactical", Params(off_pitch_step=False)).intervals("dead"), [])

    def test_broadcast_frames_without_tactical(self):
        """B 0..39 come before tactical frame 0: the broadcast ball decides there."""
        b = np.full((200, 2), 0.5)
        b[0:30, 0] = -0.5
        match = SyncedMatch(self.t, ball_signals("broadcast", b), SyncMap([{"tactical": 0, "broadcast": 40}]),
                            video_frames={"tactical": 200, "broadcast": 200})
        r = annotate(match, "broadcast", replace(BASELINE, off_pitch_step=True))
        self.assertEqual(r.intervals("off_pitch")[0], [0, 29])
        self.assertEqual(r.intervals("off_pitch")[1:], [[50, 79]])   # = T 10..39


class TestLoaders(unittest.TestCase):
    def test_homography_missing(self):
        H = np.ones((4, 3, 3))
        H[2] = np.nan
        idx = {10: 0, 11: 1, 12: 2, 14: 3}                     # frame 13 has no row
        valid = np.array([1, 0, 1, 1])
        out = homography_missing((idx, H, np.zeros(4), valid), 10, 14)
        self.assertEqual(out.tolist(), [False, True, True, True, False])

    @unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("cv2", "pandas")), "match_viewer needs cv2 and pandas")
    def test_through_match_viewer_loaders(self):
        """Raw ATD + homography files -> match_viewer.load_atd / load_homography -> signals."""
        from dead_ball_investigation.signals import signals_from_viewer_data, viewer
        mv = viewer()
        players = ";".join(f"{0.02 * k:.3f},0.5,1.0" for k in range(22))
        lines = []
        for vf in range(6):                                    # ATD frame_id is video frame + 1
            pl = players if vf != 3 else ";".join(players.split(";")[:20])
            ball = "NaN,NaN,NaN" if vf in (1, 2) else "0.5,0.5,0"
            lines.append(f"{vf + 1}:{pl}:{ball}")
        hom_rows = ["frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"]
        for vf in range(6):
            ok = vf not in (2, 4)
            vals = "1,0,0,0,1,0,0,0,1,0.3,1" if ok else ",,,,,,,,,,0"
            if vf != 5:                                        # frame 5 has no row at all
                hom_rows.append(f"{vf},{vals}")
        with tempfile.TemporaryDirectory() as td:
            atd, hom = os.path.join(td, "x_atd.txt"), os.path.join(td, "x_homography.csv")
            with open(atd, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            with open(hom, "w") as fh:
                fh.write("\n".join(hom_rows) + "\n")
            data, _meta, _home = mv.load_atd(atd, None, 0, 5)
            s = signals_from_viewer_data("tactical", 0, 5, data, mv.load_homography(hom))
        self.assertEqual(s.ball_missing.tolist(), [False, True, True, False, False, False])
        self.assertEqual(s.count.tolist(), [22, 22, 22, 20, 22, 22])
        self.assertEqual(s.hom_missing.tolist(), [False, False, True, False, True, True])

    def test_load_from_export(self):
        n, S = 5, 2
        block = 9 + 6 + 4 * S
        buf = np.zeros((n, block), np.float32)
        buf[1, :9] = np.nan                                    # no homography
        buf[2, 9:11] = np.nan                                  # no ball
        meta = {"source": "broadcast", "start": 100, "n": n, "block": block, "count": [22, 22, 21, 22, 23]}
        with tempfile.TemporaryDirectory() as td:
            buf.tofile(os.path.join(td, "data.bin"))
            with open(os.path.join(td, "meta.json"), "w") as fh:
                json.dump(meta, fh)
            s = load_from_export(td)
        self.assertEqual((s.source, s.start, s.end), ("broadcast", 100, 104))
        self.assertEqual(s.hom_missing.tolist(), [False, True, False, False, False])
        self.assertEqual(s.ball_missing.tolist(), [False, False, True, False, False])
        self.assertEqual(s.count_off.tolist(), [False, False, True, False, True])


if __name__ == "__main__":
    unittest.main()

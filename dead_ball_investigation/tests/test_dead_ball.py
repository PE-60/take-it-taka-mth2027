"""Synthetic tests, no match data required.

    python -m unittest dead_ball_investigation.tests.test_dead_ball
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dead_ball_investigation.dead_ball import (SyncedMatch, annotate, annotate_dead_ball, dead_ball_mask,
                                               dead_ball_terms, intervals_to_mask, mask_to_intervals, to_document)
from dead_ball_investigation.signals import SourceSignals, homography_missing, load_from_export
from dead_ball_investigation.sync import SyncMap

DUAL_HTML = os.path.join(os.path.dirname(__file__), "..", "..", "match_viewer", "match_dual.html")


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
        # columns: B hom missing, B count off, T ball missing, B ball missing
        rows = np.array([[a, b, c, d] for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)], bool)
        dead = dead_ball_mask(*rows.T)
        expect = (rows[:, 0] & rows[:, 1]) | (rows[:, 2] & rows[:, 3])
        self.assertEqual(dead.tolist(), expect.tolist())
        no_view, ball_lost, union = dead_ball_terms(*rows.T)
        self.assertEqual((no_view | ball_lost).tolist(), union.tolist())

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
        r = annotate(self.match, "tactical")
        self.assertEqual(r.intervals("no_view"), [[3, 4], [71, 71]])
        self.assertEqual(r.intervals("ball_lost"), [[5, 6], [20, 20]])
        self.assertEqual(r.intervals("dead"), [[3, 6], [20, 20], [71, 71]])
        self.assertEqual(r.intervals("covered"), [[0, 39], [60, 99]])
        self.assertFalse(r.dead[45])                           # tactical ball missing, no broadcast counterpart

    def test_no_counterpart_policy(self):
        r = annotate(self.match, "tactical", no_counterpart=True)
        self.assertTrue(r.dead[40:60].all())                   # unmapped frames: B hom ∅ and count off assumed
        self.assertEqual(r.intervals("ball_lost"), [[5, 6], [20, 20], [45, 45]])

    def test_broadcast_timeline_agrees(self):
        res = annotate_dead_ball(self.match)
        t, b = res["tactical"], res["broadcast"]
        for bf in range(90):
            tf = self.match.sync.map(bf, "broadcast")
            if tf >= 0:
                self.assertEqual(bool(b.dead[bf]), bool(t.dead[tf]), f"B {bf} / T {tf}")
        self.assertTrue(b.no_view[0] and b.no_view[1])         # broadcast-only term holds without a counterpart
        self.assertEqual(b.intervals("dead"), [[0, 1], [13, 16], [30, 30], [61, 61]])

    def test_tags_and_document(self):
        tags = self.match.synced_tags()
        self.assertEqual((tags[0]["broadcast_a"], tags[0]["broadcast_b"]), (12, 18))
        self.assertIsNone(tags[1]["broadcast_a"])              # T 45 is cut from the broadcast
        doc = to_document(self.match, annotate_dead_ball(self.match), "sync.json", graded=[[0, 49]])
        doc = json.loads(json.dumps(doc))
        tl = doc["timelines"]["tactical"]
        self.assertEqual(tl["dead_ball"], [[3, 6], [20, 20], [71, 71]])
        self.assertEqual(tl["stats"]["graded"]["dead_frames"], 5)
        self.assertEqual(doc["timelines"]["broadcast"]["end"], 89)
        self.assertEqual(doc["smart_tags"][0]["broadcast"], [12, 18])

    def test_partial_ranges(self):
        """Sources loaded over part of their video only (e.g. a gt-mode export)."""
        tac = signals("tactical", 20, start=10, ball=[12])
        br = signals("broadcast", 90, ball=[22])
        r = annotate(SyncedMatch(tac, br, self.match.sync), "tactical")
        self.assertEqual(r.start, 10)
        self.assertEqual(r.intervals("dead"), [[12, 12]])


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

"""Synthetic tests, no match data required.

    python -m unittest discover possesion_algorithms/tests
"""

from __future__ import annotations

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

from dead_ball_investigation.sync import SyncMap  # noqa: E402
from possesion_algorithms import run as cli  # noqa: E402
from possesion_algorithms.algorithms import registry  # noqa: E402
from possesion_algorithms.algorithms.link_hoernig import LinkHoernig  # noqa: E402
from possesion_algorithms.base import (BROADCAST, TACTICAL, TACTICAL_THEN_BROADCAST, PossessionAlgorithm,  # noqa: E402
                                       Tracking)
from possesion_algorithms.inputs import MatchInputs, SourceTrack, load_export  # noqa: E402
from possesion_algorithms.runner import clean_labels, label_match, run, validate  # noqa: E402

PITCH = (100.0, 50.0)


def track(source, n, start=0, S=2, team=("H", "A")):
    """Ball and S players parked at distinct spots, ball on nobody."""
    xy = np.zeros((n, S, 2), np.float32)
    for k in range(S):
        xy[:, k] = (0.1 + 0.8 * k / max(1, S - 1), 0.5)
    ball = np.full((n, 2), 0.5, np.float32)
    return SourceTrack(source, start, ball, xy, np.array(team[:S], dtype="<U1"), np.zeros(n, bool), PITCH)


def on_player(t, frames, k):
    """Put the ball on player k for the given (source-relative) rows."""
    t.ball[frames] = t.xy[frames, k]


class Recorder(PossessionAlgorithm):
    """Labels every frame with the origin of its data (0 -> H, 1 -> A) and remembers what it saw."""
    name = "test_recorder"
    seen: list = []

    def label(self, trk):
        type(self).seen.append((trk.start, trk.n))
        return np.where(trk.origin == 1, "A", "H")


class Blank(PossessionAlgorithm):
    name = "test_blank"

    def label(self, trk):
        return np.full(trk.n, "", dtype="<U1")


class LinkHoernigTest(unittest.TestCase):
    def tracking(self, t):
        return Tracking(t.start, t.ball, t.xy, t.team, t.kick, np.zeros(t.n, np.int8), t.pitch)

    def test_gain_keep_and_sticky_team(self):
        t = track(TACTICAL, 200)
        on_player(t, np.arange(10, 60), 0)        # H has it
        t.ball[60:80] = np.nan                    # tracking lost: stays H
        on_player(t, np.arange(120, 200), 1)      # A wins it
        out = LinkHoernig().label(self.tracking(t))
        self.assertTrue((out[:10] == "").all())
        self.assertTrue((out[10:120] == "H").all())
        self.assertTrue((out[120:] == "A").all())

    def test_kick_reopens_the_gain_test(self):
        t = track(TACTICAL, 100)
        t.xy[:, 1] = t.xy[:, 0] + np.array([0.008, 0], np.float32)   # opponent 0.8 m away
        on_player(t, np.arange(0, 100), 0)
        t.ball[50:] = t.xy[50:, 1]                                    # ball moves to the opponent
        t.kick[50] = True
        out = LinkHoernig().label(self.tracking(t))
        self.assertTrue((out[:50] == "H").all())
        self.assertEqual(out[50], "A")

    def test_variants(self):
        algs = registry()
        for name in ("lh", "lh_dead", "lh_dead_b", "lh_dead_tb"):
            self.assertIn(name, algs)
        self.assertFalse(algs["lh"].dead_ball)
        self.assertTrue(algs["lh_dead"].dead_ball)
        self.assertEqual(algs["lh"].source, TACTICAL)
        self.assertEqual(algs["lh_dead_b"].source, BROADCAST)
        self.assertEqual(algs["lh_dead_tb"].source, TACTICAL_THEN_BROADCAST)
        with self.assertRaises(ValueError):
            LinkHoernig(nope=1)


class InputsTest(unittest.TestCase):
    N = 300

    def setUp(self):
        Recorder.seen = []
        self.t = track(TACTICAL, 200, start=50)            # tactical rows cover frames 50..249
        self.b = track(BROADCAST, 400, start=0, S=3, team=("A", "H", "?"))
        # broadcast = tactical + 20 up to tactical 150, then + 10: tactical 140..149 would land on
        # broadcast frames past the next anchor, so they have no counterpart (a cut in the broadcast)
        self.sync = SyncMap([{"tactical": 0, "broadcast": 20}, {"tactical": 150, "broadcast": 160}])

    def inputs(self, dead=None):
        return MatchInputs(tracks={TACTICAL: self.t, BROADCAST: self.b}, sync=self.sync, n_frames=self.N,
                           dead_ball=dead)

    def test_tactical_on_the_tactical_timeline(self):
        trk = self.inputs().tracking(TACTICAL)
        self.assertEqual((trk.start, trk.n), (50, 200))
        self.assertTrue((trk.origin == 0).all())

    def test_broadcast_synced(self):
        self.b.ball[:, 0] = np.arange(400) / 1000.0         # ball x encodes the broadcast frame
        trk = self.inputs().tracking(BROADCAST)
        f = trk.start + np.arange(trk.n)
        ok = trk.origin == 1
        np.testing.assert_allclose(trk.ball[ok, 0] * 1000, self.sync.map(f[ok], TACTICAL), atol=1e-3)
        self.assertFalse(ok[(f >= 140) & (f < 150)].any())
        self.assertTrue(ok[(f >= 150) & (f < 250)].all())
        self.assertEqual(trk.n_slots, 3)

    def test_tactical_then_broadcast(self):
        self.t.ball[90:100] = np.nan                        # tactical frames 140..149 lose the ball,
        self.t.ball[20:30] = np.nan                         # and so do 70..79, which the broadcast has
        trk = self.inputs().tracking(TACTICAL_THEN_BROADCAST)
        f = trk.start + np.arange(trk.n)
        self.assertEqual(trk.start, 0)                      # broadcast reaches frames before tactical starts
        self.assertEqual(trk.n_slots, 5)
        from_b = set(f[trk.origin == 1].tolist())
        self.assertTrue(set(range(70, 80)) <= from_b)
        self.assertTrue(set(range(0, 50)) <= from_b)
        self.assertFalse(from_b & set(range(140, 150)))     # no counterpart: they stay tactical
        self.assertTrue((trk.origin[(f >= 140) & (f < 150)] == 0).all())
        r = np.flatnonzero(f == 75)[0]
        self.assertTrue(np.isnan(trk.xy[r, :2]).all() and np.isfinite(trk.xy[r, 2:]).all())
        r = np.flatnonzero(f == 100)[0]
        self.assertTrue(np.isfinite(trk.xy[r, :2]).all() and np.isnan(trk.xy[r, 2:]).all())
        self.assertEqual(trk.team.tolist(), ["H", "A", "A", "H", "?"])

    def test_dead_ball_runs_each_alive_interval_on_its_own(self):
        dead = np.zeros(self.N, bool)
        dead[100:120] = dead[0:60] = True
        algo = type("R", (Recorder,), {"dead_ball": True})()
        labels, summary = label_match(algo, self.inputs(dead))
        self.assertEqual(sorted(Recorder.seen), [(60, 40), (120, 130)])
        self.assertTrue((labels[dead] == "D").all())
        self.assertTrue((labels[250:] == "X").all())        # no tactical data there
        self.assertTrue((labels[60:100] == "H").all() and (labels[120:250] == "H").all())
        self.assertEqual(summary["counts"]["D"], 80)

    def test_whole_video_ignores_dead_balls(self):
        dead = np.ones(self.N, bool)
        labels, _ = label_match(Recorder(), self.inputs(dead))
        self.assertEqual(Recorder.seen, [(50, 200)])
        self.assertTrue((labels[50:250] == "H").all())
        self.assertTrue((labels[:50] == "X").all() and (labels[250:] == "X").all())

    def test_broadcast_frames_are_labelled_from_the_broadcast(self):
        self.t.ball[20:30] = np.nan
        labels, summary = label_match(type("R", (Recorder,), {"source": TACTICAL_THEN_BROADCAST})(), self.inputs())
        self.assertTrue((labels[70:80] == "A").all())
        self.assertTrue((labels[80:250] == "H").all())
        self.assertGreater(summary["from_broadcast"], 50)

    def test_clean_labels(self):
        self.assertEqual(clean_labels(["h", "", "?", "D"], 4, "x").tolist(), ["H", "X", "X", "D"])
        with self.assertRaises(ValueError):
            clean_labels(["Q"], 1, "x")
        with self.assertRaises(ValueError):
            clean_labels(["H"], 2, "x")

    def test_missing_dead_ball_json(self):
        inputs = MatchInputs(tracks={TACTICAL: self.t}, n_frames=self.N, dead_ball_path="/nonexistent/dead_ball.json")
        with self.assertRaises(IOError):
            label_match(type("R", (Recorder,), {"dead_ball": True})(), inputs)


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_csv_is_a_valid_submission(self):
        t = track(TACTICAL, 120, start=10)
        on_player(t, np.arange(0, 120), 1)
        dead = np.zeros(150, bool)
        dead[40:50] = True
        inputs = MatchInputs(tracks={TACTICAL: t}, n_frames=150, dead_ball=dead)
        meta = run(type("L", (LinkHoernig,), {"dead_ball": True})(), inputs, self.tmp, "lh_test", do_score=False)
        path = os.path.join(self.tmp, "lh_test.csv")
        validate(path, 150)
        with open(path) as fh:
            rows = fh.read().splitlines()
        self.assertEqual(rows[0], "frame,label")
        self.assertEqual(len(rows), 151)
        self.assertEqual(rows[1], "0,X")
        self.assertEqual(rows[1 + 45], "45,D")
        self.assertEqual(rows[1 + 60], "60,A")
        with open(os.path.join(self.tmp, "lh_test.meta.json")) as fh:
            side = json.load(fh)
        self.assertEqual((side["name"], side["algorithm"], side["dead_ball"], side["source"]),
                         ("lh_test", "lh", True, TACTICAL))
        self.assertEqual(side["counts"], meta["counts"])
        self.assertEqual(side["counts"]["D"], 10)

    def test_blank_answers_become_x(self):
        inputs = MatchInputs(tracks={TACTICAL: track(TACTICAL, 30)}, n_frames=30)
        run(Blank(), inputs, self.tmp, "blank", do_score=False)
        validate(os.path.join(self.tmp, "blank.csv"), 30)

    def test_export_round_trip(self):
        S, n, start = 3, 40, 1000
        data = np.full((n, 15 + 4 * S), np.nan, np.float32)
        data[:, 9], data[:, 10] = np.linspace(0, 1, n), 0.25
        for k in range(S):
            data[:, 15 + 4 * k], data[:, 16 + 4 * k] = 0.1 * k, 0.2 * k
        data[5:8, 19:21] = np.nan
        meta = {"source": "broadcast", "start": start, "n": n, "block": 15 + 4 * S, "n_slots": S, "fps": 25,
                "pitch": [105.3, 68.0], "slots": [{"id": "a", "team": "H"}, {"id": "b", "team": "A"}, {"id": "c", "team": "?"}],
                "events": {"kick": [3, 17]}}
        d = os.path.join(self.tmp, "broadcast_x")
        os.makedirs(d)
        data.tofile(os.path.join(d, "data.bin"))
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        t = load_export(d)
        self.assertEqual((t.source, t.start, t.end), ("broadcast", start, start + n - 1))
        self.assertEqual(t.xy.shape, (n, S, 2))
        self.assertAlmostEqual(float(t.xy[0, 2, 1]), 0.4, places=6)
        self.assertTrue(np.isnan(t.xy[6, 1]).all())
        self.assertEqual(np.flatnonzero(t.kick).tolist(), [3, 17])
        self.assertEqual(t.team.tolist(), ["H", "A", "?"])
        np.testing.assert_allclose(t.ball[:, 0], np.linspace(0, 1, n), atol=1e-6)


VIEWER_HTML = os.path.join(os.path.dirname(__file__), "..", "..", "match_viewer", "match_viewer.html")


@unittest.skipUnless(shutil.which("node"), "node not installed")
class ViewerJsTest(unittest.TestCase):
    """The page's algorithms.js section (and rasterDiff), run in node."""

    def run_js(self, body, payload):
        with open(VIEWER_HTML) as fh:
            html = fh.read()
        section = re.search(r"// ===== algorithms\.js =====(.*?)// ===== annotations\.js =====", html, re.S).group(1)
        helpers = "\n".join(re.search(p, html, re.S).group(0) for p in
                            (r"const bucket = .*?\n", r"const put = .*?\n", r"function rasterDiff\(.*?\n\}\n"))
        src = section + helpers + "\nconst P = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" + body
        out = subprocess.run(["node", "-e", src], input=json.dumps(payload), capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_csv_and_rows(self):
        csv = "frame,label\n0,H\n1,a\n2,D\n3,X\n5,H\n"
        got = self.run_js("process.stdout.write(JSON.stringify({c: [...parseLabelCsv(P.csv)], "
                          "r: [...codesForRows(parseLabelCsv(P.csv), 2, 6)], "
                          "g: agreement(Uint8Array.from([1, 2, 1, 4, 0]), Uint8Array.from([1, 1, 1, 3, 4]))}))",
                          {"csv": csv})
        self.assertEqual(got["c"], [1, 2, 3, 4, 0, 1])
        self.assertEqual(got["r"], [3, 4, 0, 1, 0, 0])
        self.assertEqual(got["g"], {"n": 4, "same": 2, "share": 0.5})

    def test_broadcast_to_tactical_matches_python(self):
        anchors = [{"tactical": 1000, "broadcast": 400}, {"tactical": 5000, "broadcast": 4100},
                   {"tactical": 9000, "broadcast": 8300}, {"tactical": 9500, "broadcast": 8700}]
        frames = list(range(0, 10000, 7))
        got = self.run_js("const m = syncToTactical(P.anchors); process.stdout.write(JSON.stringify(P.frames.map(m)))",
                          {"anchors": anchors, "frames": frames})
        want = SyncMap(anchors).map(np.array(frames), BROADCAST).tolist()
        self.assertEqual([-1 if g is None else g for g in got], want)

    def test_raster_diff(self):
        got = self.run_js("process.stdout.write(JSON.stringify([...rasterDiff(r => P.v[r], 0, 5, 3, [34, 34, 34])]))",
                          {"v": [1, 1, 2, 2, 0, 0]})
        self.assertEqual(got[0:3], [38, 62, 40])          # all agree
        self.assertEqual(got[3:6], [230, 80, 230])        # all disagree
        self.assertEqual(got[6:9], [34, 34, 34])          # nothing to compare


class CliTest(unittest.TestCase):
    def test_auto_name(self):
        cls = registry()["lh"]
        algo = cli.variant(cls, True, TACTICAL_THEN_BROADCAST)(t_p=1.5)
        self.assertEqual(cli.auto_name(cls, algo, {"t_p": 1.5}), "lh__dead_tb__t_p1.5")
        self.assertEqual(cli.auto_name(cls, cli.variant(cls, None, None)(), {}), "lh")
        self.assertIs(cli.variant(cls, False, TACTICAL), cls)

    def test_parse_params(self):
        cls = registry()["lh"]
        self.assertEqual(cli.parse_params(cls, ["t_p=2", "t_a=3.5"]), {"t_p": 2.0, "t_a": 3.5})
        with self.assertRaises(SystemExit):
            cli.parse_params(cls, ["radius=2"])


if __name__ == "__main__":
    unittest.main()

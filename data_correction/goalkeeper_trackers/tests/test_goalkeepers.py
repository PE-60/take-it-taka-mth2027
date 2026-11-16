"""Synthetic tests for derived goalkeeper tracks. No proprietary match data is read.

    python -m unittest data_correction.goalkeeper_trackers.tests.test_goalkeepers
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from data_correction.goalkeeper_trackers.derive import (  # noqa: E402
    CSV_COLUMNS,
    STATUS_BOTH_ACTIVE,
    STATUS_BY_HALF,
    STATUS_NO_LABEL,
    STATUS_NO_SPEED,
    STATUS_NO_VISIBLE,
    STATUS_SPEED,
    STATUS_SPEED_INFERRED,
    defending_half,
    derive_goalkeeper_tracks,
    write_csv,
)
from data_correction.goalkeeper_trackers.load import parse_team_meta, second_half_mask  # noqa: E402
from data_correction.goalkeeper_trackers.visibility import camera_visible, projective_view_valid  # noqa: E402


def identity_homography(n):
    H = np.repeat(np.eye(3, dtype=np.float64)[None, ...], n, axis=0)
    # Map pitch fractions to pixel coords roughly: x' = x * W, y' = y * H via a scaled identity.
    # Use H such that [x,y,1] -> [x*W, y*H, 1].
    W, Hh = 200.0, 100.0
    H[:, 0, 0] = W
    H[:, 1, 1] = Hh
    return H, np.ones(n, bool), (int(W), int(Hh))


class DefendHalfTest(unittest.TestCase):
    def test_flips_after_halftime(self):
        second = np.array([False, False, True, True])
        left_attack = defending_half("left_to_right", second)
        self.assertEqual(left_attack.tolist(), [0, 0, 1, 1])
        right_attack = defending_half("right_to_left", second)
        self.assertEqual(right_attack.tolist(), [1, 1, 0, 0])


class DeriveTest(unittest.TestCase):
    def test_assigns_by_pitch_half_when_separate(self):
        n = 5
        pos = np.zeros((n, 2, 2), np.float32)
        pos[:, 0] = (0.2, 0.5)   # left
        pos[:, 1] = (0.8, 0.5)   # right
        speeds = np.full((n, 2), 1.0, np.float32)
        visible = np.ones((n, 2), bool)
        home = np.zeros(n, np.int8)          # home defends left
        away = np.ones(n, np.int8)           # away defends right
        out = derive_goalkeeper_tracks(pos, speeds, visible, home, away, ("24", "46"),
                                       {"home": "HFC", "away": "AFC"})
        self.assertTrue((out.source_track_id["home"] == "24").all())
        self.assertTrue((out.source_track_id["away"] == "46").all())
        self.assertTrue((out.status["home"] == STATUS_BY_HALF).all())
        self.assertTrue((out.status["away"] == STATUS_BY_HALF).all())
        np.testing.assert_allclose(out.position["home"][:, 0], 0.2)
        np.testing.assert_allclose(out.position["away"][:, 0], 0.8)

    def test_same_half_speed_disambiguates(self):
        n = 4
        pos = np.zeros((n, 2, 2), np.float32)
        pos[:, 0] = (0.2, 0.4)
        pos[:, 1] = (0.25, 0.6)
        speeds = np.full((n, 2), np.nan, np.float32)
        speeds[:, 1] = 2.0                   # only tracker 46 has speed
        visible = np.ones((n, 2), bool)
        home = np.zeros(n, np.int8)
        away = np.ones(n, np.int8)
        out = derive_goalkeeper_tracks(pos, speeds, visible, home, away)
        self.assertTrue((out.source_track_id["home"] == "46").all())
        self.assertTrue((out.status["home"] == STATUS_SPEED).all())
        self.assertTrue((out.status["away"] == STATUS_NO_LABEL).all())

    def test_same_half_infers_across_missing_speed_frames(self):
        n = 5
        pos = np.zeros((n, 2, 2), np.float32)
        pos[:, :] = (0.2, 0.5)
        speeds = np.full((n, 2), np.nan, np.float32)
        speeds[2, 0] = 1.5                   # only tracker 24 has any speed in the overlap
        visible = np.ones((n, 2), bool)
        home = np.zeros(n, np.int8)
        away = np.ones(n, np.int8)
        out = derive_goalkeeper_tracks(pos, speeds, visible, home, away)
        self.assertEqual(out.status["home"][2], STATUS_SPEED)
        self.assertTrue((out.status["home"][[0, 1, 3, 4]] == STATUS_SPEED_INFERRED).all())
        self.assertTrue((out.source_track_id["home"] == "24").all())

    def test_same_half_both_or_neither_stay_unresolved(self):
        n = 2
        pos = np.zeros((n, 2, 2), np.float32)
        pos[:, :] = (0.8, 0.5)
        speeds = np.array([[1.0, 1.0], [np.nan, np.nan]], np.float32)
        visible = np.ones((n, 2), bool)
        home = np.zeros(n, np.int8)
        away = np.ones(n, np.int8)
        out = derive_goalkeeper_tracks(pos, speeds, visible, home, away)
        # Both on right → away's defending half.
        self.assertEqual(out.status["away"][0], STATUS_BOTH_ACTIVE)
        self.assertEqual(out.status["away"][1], STATUS_NO_SPEED)
        self.assertTrue(np.isnan(out.position["away"]).all())
        self.assertTrue((out.status["home"] == STATUS_NO_LABEL).all())

    def test_invisible_stays_blank(self):
        n = 3
        pos = np.full((n, 2, 2), np.nan, np.float32)
        speeds = np.full((n, 2), np.nan, np.float32)
        visible = np.zeros((n, 2), bool)
        home = np.zeros(n, np.int8)
        away = np.ones(n, np.int8)
        out = derive_goalkeeper_tracks(pos, speeds, visible, home, away)
        self.assertTrue((out.status["home"] == STATUS_NO_VISIBLE).all())
        self.assertTrue((out.status["away"] == STATUS_NO_VISIBLE).all())

    def test_csv_long_format(self):
        import csv
        n = 2
        pos = np.zeros((n, 2, 2), np.float32)
        pos[:, 0] = (0.1, 0.5)
        pos[:, 1] = (0.9, 0.5)
        out = derive_goalkeeper_tracks(
            pos, np.ones((n, 2), np.float32), np.ones((n, 2), bool),
            np.zeros(n, np.int8), np.ones(n, np.int8),
            team_names={"home": "H", "away": "A"}, fps=25.0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.csv")
            self.assertEqual(write_csv(out, path), 4)
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(list(rows[0].keys()), list(CSV_COLUMNS))
            self.assertEqual({row["team_side"] for row in rows}, {"home", "away"})


class VisibilityTest(unittest.TestCase):
    def test_projective_rejects_near_singular(self):
        bad = np.zeros((3, 3))
        self.assertFalse(projective_view_valid(bad))

    def test_points_inside_image_are_visible(self):
        n = 3
        H, valid, size = identity_homography(n)
        pos = np.zeros((n, 2, 2), np.float64)
        pos[:, 0] = (0.25, 0.50)     # -> (50, 50) inside 200x100
        pos[:, 1] = (1.50, 0.50)     # off pitch
        pos[1, 0] = (0.95, 0.95)     # -> (190, 95) inside
        pos[2, 0] = np.nan
        vis = camera_visible(pos, H, valid, size)
        self.assertTrue(vis[0, 0] and vis[1, 0])
        self.assertFalse(vis[:, 1].any())
        self.assertFalse(vis[2, 0])


class MetaParseTest(unittest.TestCase):
    def test_parse_team_meta(self):
        xml = """<?xml version="1.0"?>
        <root>
          <Team id="T1"><Name>Home FC</Name>
            <ProviderParameter><Name>attack_direction_first_half</Name>
              <Value>left_to_right</Value></ProviderParameter>
          </Team>
          <Team id="T2"><Name>Away FC</Name>
            <ProviderParameter><Name>attack_direction_first_half</Name>
              <Value>right_to_left</Value></ProviderParameter>
          </Team>
          <Player id="24" teamId="T1"><Name>GK H</Name></Player>
          <Player id="46" teamId="T2"><Name>GK A</Name></Player>
        </root>"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.xml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(xml)
            meta = parse_team_meta(path)
        self.assertEqual(meta["team_names_by_side"]["home"], "Home FC")
        self.assertEqual(meta["attack_by_side"]["away"], "right_to_left")
        self.assertEqual({s["id"] for s in meta["slots"]}, {"24", "46"})

    def test_second_half_mask_uses_games(self):
        mask = second_half_mask(10)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(len(mask), 10)


if __name__ == "__main__":
    unittest.main()

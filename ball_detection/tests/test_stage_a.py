"""Synthetic tests for Stage A (geometry, GK exclusion, harvest, appearance banks).

No proprietary match data is read.

    python -m ball_detection.tests.test_stage_a
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ball_detection.appearance import (  # noqa: E402
    AppearanceBanks,
    hist_intersection,
    hsv_histogram,
    load_banks,
    save_banks,
)
from ball_detection.banks import (  # noqa: E402
    BallCropParams,
    KitCropParams,
    accumulate_frame,
    add_ball_sample,
    add_kit_sample,
)
from ball_detection.geometry import (  # noqa: E402
    load_homography_table,
    pitch_distance_m,
    project,
    project_points,
    write_homography_table,
)
from ball_detection.gk import (  # noqa: E402
    default_goalkeeper_slot_ids,
    goalkeeper_slot_ids_from_csv,
    outfield_slot_mask,
    resolve_goalkeeper_slot_ids,
)
from ball_detection.harvest import (  # noqa: E402
    BallHarvestParams,
    KitHarvestParams,
    ball_harvest_mask,
    kit_harvest_mask,
    subsample_indices,
)


def scaled_identity_H(width=320, height=180):
    """Pitch (0-1) -> pixels via diagonal scale."""
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = float(width)
    H[1, 1] = float(height)
    return H


def paint_scene(width=320, height=180, ball_xy=(0.5, 0.5), home_xy=(0.3, 0.4), away_xy=(0.7, 0.4)):
    """BGR frame with green grass, white ball, blue home torso, red away torso."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 140, 40)  # BGR grass-ish
    H = scaled_identity_H(width, height)

    def _uv(xy):
        u, v = project(H, xy[0], xy[1])
        return int(round(float(np.asarray(u).reshape(())))), int(round(float(np.asarray(v).reshape(()))))

    def dot(xy, color, r):
        uu, vv = _uv(xy)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - uu) ** 2 + (yy - vv) ** 2 <= r * r
        frame[mask] = color

    def torso(xy, color):
        uu, vv = _uv(xy)
        x0, x1 = uu - 12, uu + 12
        y0, y1 = vv - 40, vv - 10
        frame[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = color

    torso(home_xy, (200, 80, 20))   # blue-ish shirt
    torso(away_xy, (40, 40, 200))   # red-ish shirt
    dot(ball_xy, (240, 240, 240), 5)
    return frame, H


class GeometryTest(unittest.TestCase):
    def test_project_roundtrip_corners(self):
        H = scaled_identity_H(200, 100)
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        px = project_points(H, pts)
        np.testing.assert_allclose(px, [[0, 0], [200, 0], [200, 100], [0, 100]], atol=1e-6)

    def test_pitch_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        self.assertAlmostEqual(float(pitch_distance_m(a, b, (105.0, 68.0))), 105.0)

    def test_homography_io(self):
        H = np.stack([scaled_identity_H(10, 10), scaled_identity_H(20, 20)])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "H.csv")
            write_homography_table(path, H, [0.1, 0.2], [True, True])
            matrices, err, valid = load_homography_table(path, n_frames=2)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(matrices[0], H[0])
            np.testing.assert_allclose(err, [0.1, 0.2])


class GkTest(unittest.TestCase):
    def test_default_slots(self):
        ids = default_goalkeeper_slot_ids()
        self.assertEqual(ids, frozenset({"24", "46"}))

    def test_outfield_mask_excludes_keepers(self):
        slots = ["1", "24", "10", "46", "7"]
        mask = outfield_slot_mask(slots)
        self.assertEqual(mask.tolist(), [True, False, True, False, True])

    def test_csv_union(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "gk.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("frame,side,source_track_id\n0,home,24\n0,away,46\n1,home,99\n")
            ids = resolve_goalkeeper_slot_ids(csv_path=path)
            self.assertIn("24", ids)
            self.assertIn("46", ids)
            self.assertIn("99", ids)
            self.assertEqual(goalkeeper_slot_ids_from_csv(path), frozenset({"24", "46", "99"}))


class HarvestTest(unittest.TestCase):
    def setUp(self):
        self.pitch = (105.0, 68.0)
        self.n, self.S = 6, 4
        self.slot_ids = np.array(["1", "24", "10", "46"], dtype=object)
        self.team = np.array(["H", "H", "A", "A"], dtype="<U1")
        self.ball = np.full((self.n, 2), 0.5, dtype=np.float64)
        self.xy = np.zeros((self.n, self.S, 2), dtype=np.float64)
        # Outfield far from ball; keepers near ball (should not block ball harvest).
        self.xy[:, 0] = (0.2, 0.2)   # H outfield ~far
        self.xy[:, 1] = (0.51, 0.5)  # GK near ball
        self.xy[:, 2] = (0.8, 0.8)   # A outfield far
        self.xy[:, 3] = (0.49, 0.5)  # GK near ball
        self.valid = np.ones(self.n, bool)
        self.reproj = np.full(self.n, 0.2)

    def test_ball_clears_when_only_gk_nearby(self):
        mask = ball_harvest_mask(
            self.ball, self.xy, self.pitch,
            slot_ids=self.slot_ids, valid_H=self.valid, reproj_error=self.reproj,
            params=BallHarvestParams(clearance_m=2.0, max_ball_speed_ms=None),
        )
        self.assertTrue(mask.all())

    def test_ball_rejects_nearby_outfield(self):
        self.xy[:, 0] = (0.505, 0.5)  # H outfield on top of ball
        mask = ball_harvest_mask(
            self.ball, self.xy, self.pitch,
            slot_ids=self.slot_ids, valid_H=self.valid, reproj_error=self.reproj,
            params=BallHarvestParams(clearance_m=2.0, max_ball_speed_ms=None),
        )
        self.assertFalse(mask.any())

    def test_kit_excludes_gk_slots(self):
        mask = kit_harvest_mask(
            self.xy, self.team, self.ball, self.pitch,
            slot_ids=self.slot_ids, valid_H=self.valid, reproj_error=self.reproj,
            params=KitHarvestParams(ball_clearance_m=0.0),
        )
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 1].any())
        self.assertTrue(mask[:, 2].all())
        self.assertFalse(mask[:, 3].any())

    def test_subsample(self):
        mask = np.array([True, False, True, True, False, True])
        idx = subsample_indices(mask, stride=2, limit=10)
        self.assertEqual(idx.tolist(), [0, 3])


class AppearanceBanksTest(unittest.TestCase):
    def test_ball_and_kit_samples_separate(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 required")

        frame, H = paint_scene()
        banks = AppearanceBanks(gk_excluded=["24", "46"])
        self.assertTrue(add_ball_sample(banks, frame, np.array([0.5, 0.5]), H,
                                        params=BallCropParams(radius_px=6, exclude_grass=True)))
        self.assertTrue(add_kit_sample(banks, frame, np.array([0.3, 0.4]), "H", H,
                                       params=KitCropParams(exclude_grass=True)))
        self.assertTrue(add_kit_sample(banks, frame, np.array([0.7, 0.4]), "A", H,
                                       params=KitCropParams(exclude_grass=True)))
        self.assertEqual(banks.ball.n_samples, 1)
        self.assertEqual(banks.kits["H"].n_samples, 1)
        self.assertEqual(banks.kits["A"].n_samples, 1)

        # Kits should not look like each other as much as each looks like itself.
        h_hist = banks.kits["H"].mean_hist()
        a_hist = banks.kits["A"].mean_hist()
        self.assertIsNotNone(h_hist)
        self.assertIsNotNone(a_hist)
        self.assertLess(hist_intersection(h_hist, a_hist), 0.85)

    def test_accumulate_respects_slot_mask(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 required")
        frame, H = paint_scene()
        banks = AppearanceBanks()
        players = np.array([[0.3, 0.4], [0.7, 0.4]])
        team = np.array(["H", "A"], dtype="<U1")
        # Only allow away.
        counts = accumulate_frame(
            banks, frame, H,
            ball_xy=np.array([0.5, 0.5]), take_ball=True,
            players_xy=players, team=team, kit_slot_mask=np.array([False, True]),
            ball_params=BallCropParams(radius_px=6),
            kit_params=KitCropParams(),
        )
        self.assertEqual(counts["ball"], 1)
        self.assertEqual(counts.get("kit_A", 0), 1)
        self.assertEqual(counts.get("kit_H", 0), 0)

    def test_json_roundtrip(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 required")
        frame, H = paint_scene()
        banks = AppearanceBanks(source="tactical", gk_excluded=["24", "46"])
        add_ball_sample(banks, frame, np.array([0.5, 0.5]), H, params=BallCropParams(radius_px=6))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "banks.json")
            save_banks(banks, path)
            loaded = load_banks(path)
            self.assertEqual(loaded.ball.n_samples, 1)
            self.assertEqual(loaded.gk_excluded, ["24", "46"])
            self.assertGreater(hist_intersection(loaded.ball.mean_hist(), banks.ball.mean_hist()), 0.99)

    def test_hsv_histogram_normalised(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 required")
        patch = np.zeros((16, 16, 3), dtype=np.uint8)
        patch[:] = (0, 0, 255)  # red in BGR
        hist = hsv_histogram(patch)
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

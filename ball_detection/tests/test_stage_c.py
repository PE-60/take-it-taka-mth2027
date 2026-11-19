"""Synthetic tests for Stage C (attribution) and overlays / sidecar.

    python -m ball_detection.tests.test_stage_c
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

from ball_detection.appearance import AppearanceBanks  # noqa: E402
from ball_detection.attribute import (  # noqa: E402
    AttributeParams,
    attribute_touch,
    score_candidates,
)
from ball_detection.banks import BallCropParams, KitCropParams, add_ball_sample, add_kit_sample  # noqa: E402
from ball_detection.overlay import draw_track_on_frame  # noqa: E402
from ball_detection.results import (  # noqa: E402
    AttributionExport,
    BallDetection,
    DetectExport,
    KickBallTrack,
    load_attributions,
    save_attributions,
)
from ball_detection.viewer_sidecar import build_sidecar  # noqa: E402


def scaled_H(width=320, height=180):
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = float(width)
    H[1, 1] = float(height)
    return H


def paint(width=320, height=180, ball=(0.5, 0.5), home=(0.48, 0.5), away=(0.7, 0.5)):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 140, 40)
    H = scaled_H(width, height)

    def torso(xy, color):
        u, v = int(round(xy[0] * width)), int(round(xy[1] * height))
        frame[max(0, v - 40):max(0, v - 10), max(0, u - 12):u + 12] = color

    def ball_dot(xy):
        u, v = int(round(xy[0] * width)), int(round(xy[1] * height))
        yy, xx = np.ogrid[:height, :width]
        frame[(xx - u) ** 2 + (yy - v) ** 2 <= 25] = (235, 235, 235)

    torso(home, (200, 80, 20))
    torso(away, (40, 40, 200))
    ball_dot(ball)
    return frame, H


class AttributeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            cls.has_cv2 = True
        except ImportError:
            cls.has_cv2 = False

    def test_nearest_outfield_wins_distance_only(self):
        H = scaled_H()
        contact = BallDetection(frame=10, u=160.0, v=90.0, score=0.8, radius_px=6)
        players = np.array([[0.48, 0.50], [0.80, 0.50], [0.49, 0.50]], dtype=np.float64)
        team = np.array(["H", "A", "H"], dtype="<U1")
        slots = np.array(["1", "2", "24"], dtype=object)  # 24 = GK, excluded
        track = KickBallTrack(kick_frame=10, window_start=5, window_end=15, contact=contact,
                              detections=[contact])
        attr = attribute_touch(
            track, players, team, slots, H, (105.0, 68.0),
            frame_bgr=None, banks=None, gk_ids={"24", "46"},
            params=AttributeParams(min_score=0.05, w_dist=1.0, w_kit=0.0),
        )
        self.assertEqual(attr.team, "H")
        self.assertEqual(attr.slot_id, "1")
        self.assertEqual(attr.method, "dist_only")
        # GK must not appear.
        self.assertTrue(all(c.slot_id != "24" for c in attr.candidates))

    def test_kit_match_helps_choose_team(self):
        if not self.has_cv2:
            self.skipTest("cv2 required")
        frame, H = paint(ball=(0.5, 0.5), home=(0.50, 0.52), away=(0.52, 0.48))
        banks = AppearanceBanks()
        add_kit_sample(banks, frame, np.array([0.50, 0.52]), "H", H, params=KitCropParams())
        # Build away kit from a clean away-only frame
        frame_a, _ = paint(ball=(0.1, 0.1), home=(0.1, 0.9), away=(0.7, 0.5))
        add_kit_sample(banks, frame_a, np.array([0.7, 0.5]), "A", H, params=KitCropParams())

        contact = BallDetection(frame=0, u=160.0, v=90.0, score=0.7, radius_px=6)
        # Two players almost equidistant; kit should reinforce the closer blue (H) torso.
        players = np.array([[0.50, 0.52], [0.52, 0.48]], dtype=np.float64)
        team = np.array(["H", "A"], dtype="<U1")
        slots = np.array(["7", "9"], dtype=object)
        track = KickBallTrack(0, 0, 1, [contact], contact)
        attr = attribute_touch(
            track, players, team, slots, H, (105.0, 68.0),
            frame_bgr=frame, banks=banks, gk_ids={"24", "46"},
            params=AttributeParams(min_score=0.05, max_dist_m=5.0, max_dist_px=120.0),
        )
        self.assertIn(attr.team, ("H", "A"))
        self.assertGreater(attr.confidence, 0.0)
        self.assertEqual(attr.method, "dist_kit")

    def test_score_candidates_empty_when_far(self):
        H = scaled_H()
        contact = BallDetection(0, 10.0, 10.0, 0.5, 5)
        players = np.array([[0.9, 0.9]], dtype=np.float64)
        team = np.array(["H"], dtype="<U1")
        cands = score_candidates(
            contact, players, team, ["1"], H, (105.0, 68.0),
            params=AttributeParams(max_dist_m=1.0, max_dist_px=20.0),
        )
        self.assertEqual(cands, [])


class SidecarOverlayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            cls.has_cv2 = True
        except ImportError:
            cls.has_cv2 = False

    def test_sidecar_marks_contact(self):
        contact = BallDetection(100, 10.0, 20.0, 0.8, 8.0)
        track = KickBallTrack(100, 90, 110, [contact, BallDetection(101, 12, 22, 0.5, 8)], contact)
        from ball_detection.attribute import TouchAttribution
        attr = TouchAttribution(kick_frame=100, contact=contact, team="H", confidence=0.7,
                                slot_id="7", method="dist_kit")
        det = DetectExport(tracks=[track])
        att = AttributionExport(attributions=[attr])
        doc = build_sidecar(det, att)
        self.assertEqual(doc["n_contacts"], 1)
        kinds = {e["kind"] for e in doc["events"]}
        self.assertIn("contact", kinds)
        contact_ev = next(e for e in doc["events"] if e["kind"] == "contact")
        self.assertEqual(contact_ev["team"], "H")

    def test_attribution_json_roundtrip(self):
        from ball_detection.attribute import TouchAttribution
        export = AttributionExport(attributions=[
            TouchAttribution(kick_frame=1, team="A", confidence=0.5, slot_id="3", method="dist_only"),
        ])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "a.json")
            save_attributions(export, path)
            loaded = load_attributions(path)
            self.assertEqual(loaded.attributions[0].team, "A")

    def test_overlay_draws(self):
        if not self.has_cv2:
            self.skipTest("cv2 required")
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        contact = BallDetection(0, 160, 90, 0.9, 6)
        track = KickBallTrack(0, 0, 1, [contact], contact)
        from ball_detection.attribute import TouchAttribution
        attr = TouchAttribution(0, contact, team="H", confidence=0.8, slot_id="1")
        vis = draw_track_on_frame(frame, track, attr, video_frame=0)
        self.assertEqual(vis.shape, frame.shape)
        # Something non-black was drawn.
        self.assertGreater(int(vis.sum()), 0)


if __name__ == "__main__":
    unittest.main()

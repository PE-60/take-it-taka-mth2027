"""Synthetic tests for Stage B (kick windows, ROI, blob detection).

No proprietary match data is read.

    python -m ball_detection.tests.test_stage_b
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
from ball_detection.banks import BallCropParams, add_ball_sample  # noqa: E402
from ball_detection.detect import (  # noqa: E402
    DetectParams,
    choose_contact,
    detect_in_roi,
    detect_kicks_in_tracking,
    track_kick_window,
)
from ball_detection.kicks import (  # noqa: E402
    KickWindow,
    ball_prior_at,
    detect_kicks,
    kick_windows,
)
from ball_detection.results import (  # noqa: E402
    BallDetection,
    DetectExport,
    KickBallTrack,
    load_detections,
    save_detections,
)
from ball_detection.roi import PixelROI, RoiParams, search_roi  # noqa: E402


def scaled_H(width=320, height=180):
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = float(width)
    H[1, 1] = float(height)
    return H


def draw_ball_frame(width, height, ball_xy, radius=5, player_xy=None):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 140, 40)
    H = scaled_H(width, height)
    u = int(round(ball_xy[0] * width))
    v = int(round(ball_xy[1] * height))
    yy, xx = np.ogrid[:height, :width]
    frame[(xx - u) ** 2 + (yy - v) ** 2 <= radius * radius] = (235, 235, 235)
    if player_xy is not None:
        pu = int(round(player_xy[0] * width))
        pv = int(round(player_xy[1] * height))
        frame[max(0, pv - 30):pv - 5, max(0, pu - 10):pu + 10] = (200, 80, 20)
    return frame, H


class KicksTest(unittest.TestCase):
    def test_kick_windows(self):
        mask = np.zeros(40, bool)
        mask[10] = True
        mask[30] = True
        wins = kick_windows(mask, pre=3, post=4)
        self.assertEqual(len(wins), 2)
        self.assertEqual(wins[0], KickWindow(10, 7, 15))
        self.assertEqual(wins[1].start, 27)
        self.assertEqual(wins[1].end, 35)

    def test_detect_kicks_finds_accel_peak(self):
        # Stationary, then sudden jump (kick-like).
        n = 30
        ball = np.zeros((n, 2), dtype=np.float64)
        ball[:, 0] = 0.4
        ball[:, 1] = 0.5
        ball[15:, 0] = 0.55  # step change -> large accel around 15-16
        kick, accel = detect_kicks(ball, (105.0, 68.0), fps=25.0, threshold=4.0)
        self.assertTrue(kick.any())
        self.assertTrue(np.isfinite(accel).any())

    def test_ball_prior_lookback(self):
        ball = np.full((10, 2), np.nan)
        ball[3] = (0.2, 0.3)
        prior = ball_prior_at(ball, 7, lookback=10)
        np.testing.assert_allclose(prior, [0.2, 0.3])
        self.assertIsNone(ball_prior_at(ball, 2, lookback=1))


class RoiTest(unittest.TestCase):
    def test_search_roi_contains_anchor(self):
        n, S = 5, 2
        ball = np.full((n, 2), 0.5)
        players = np.zeros((n, S, 2))
        players[:, 0] = (0.52, 0.5)
        players[:, 1] = (0.9, 0.9)
        H = scaled_H(320, 180)
        roi, anchor = search_roi(
            H, ball, players[2], 2, (320, 180), (105.0, 68.0),
            slot_ids=["1", "2"], params=RoiParams(pad_px=10, min_size_px=40, player_radius_m=5.0),
        )
        self.assertIsNotNone(roi)
        self.assertIsInstance(roi, PixelROI)
        self.assertGreater(roi.width, 0)
        # Projected anchor ~ (160, 90) should lie inside ROI.
        self.assertLessEqual(roi.x0, 160)
        self.assertGreaterEqual(roi.x1, 160)
        self.assertLessEqual(roi.y0, 90)
        self.assertGreaterEqual(roi.y1, 90)


class DetectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            cls.has_cv2 = True
        except ImportError:
            cls.has_cv2 = False

    def test_detect_white_ball_in_roi(self):
        if not self.has_cv2:
            self.skipTest("cv2 required")
        w, h = 320, 180
        ball_xy = np.array([0.5, 0.5])
        frame, H = draw_ball_frame(w, h, ball_xy)
        prev, _ = draw_ball_frame(w, h, np.array([0.48, 0.5]))  # moved -> motion
        banks = AppearanceBanks()
        add_ball_sample(banks, frame, ball_xy, H,
                        params=BallCropParams(radius_px=6, min_bright_frac=0.2))
        roi = PixelROI(140, 70, 180, 110)
        det = detect_in_roi(
            frame, prev, roi, banks.ball.mean_hist(), (160.0, 90.0), frame_index=0,
            params=DetectParams(min_score=0.15, radius_px=6, foot_exclude_px=0,
                                min_bright_frac=0.2),
        )
        self.assertIsNotNone(det)
        self.assertLess(abs(det.u - 160), 12)
        self.assertLess(abs(det.v - 90), 12)

    def test_rejects_dark_shoe_blob(self):
        if not self.has_cv2:
            self.skipTest("cv2 required")
        w, h = 320, 180
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 140, 40)
        # Dark shoe-like patch near centre.
        frame[85:95, 155:165] = (30, 30, 30)
        prev = frame.copy()
        prev[85:95, 150:160] = (30, 30, 30)  # motion on the shoe
        roi = PixelROI(140, 70, 180, 110)
        det = detect_in_roi(
            frame, prev, roi, None, (160.0, 90.0), frame_index=0,
            params=DetectParams(min_score=0.1, radius_px=6, foot_exclude_px=0,
                                min_bright_frac=0.3),
        )
        self.assertIsNone(det)

    def test_track_kick_window_follows_motion(self):
        if not self.has_cv2:
            self.skipTest("cv2 required")
        w, h = 320, 180
        n = 21
        ball = np.zeros((n, 2))
        for i in range(n):
            ball[i] = (0.40 + 0.01 * i, 0.50)
        ball[10:] = ball[10:] + np.array([0.05, 0.0])
        # Keep players far so foot exclusion does not eat the ball.
        players = np.zeros((n, 1, 2))
        players[:, 0] = (0.15, 0.20)
        H = np.stack([scaled_H(w, h)] * n)
        frames = []
        for i in range(n):
            fr, _ = draw_ball_frame(w, h, ball[i], player_xy=players[i, 0])
            frames.append(fr)
        banks = AppearanceBanks()
        add_ball_sample(banks, frames[0], ball[0], H[0],
                        params=BallCropParams(radius_px=6, min_bright_frac=0.2))

        win = KickWindow(kick_index=10, start=5, end=16)
        track = track_kick_window(
            frames[win.start:win.end], win, H, ball, players, banks, (105.0, 68.0),
            frame_offset=0, slot_ids=["1"],
            params=DetectParams(min_score=0.12, radius_px=6, max_link_dist_px=50,
                                max_pred_dist_px=60, foot_exclude_px=8, min_bright_frac=0.2,
                                locked_roi_pad_px=40,
                                roi=RoiParams(pad_px=30, min_size_px=48)),
        )
        self.assertGreaterEqual(len(track.detections), 1)
        self.assertIsNotNone(track.contact)
        expected_u = ball[10, 0] * w
        self.assertLess(abs(track.contact.u - expected_u), 25)

    def test_hard_continuity_rejects_teleport(self):
        if not self.has_cv2:
            self.skipTest("cv2 required")
        w, h = 320, 180
        n = 8
        ball = np.zeros((n, 2))
        ball[:, :] = (0.3, 0.5)
        players = np.zeros((n, 1, 2))
        players[:, 0] = (0.1, 0.1)
        H = np.stack([scaled_H(w, h)] * n)
        frames = []
        for i in range(n):
            # Frames 0..3: ball on the left; 4..7: only a white blob on the far right.
            xy = (0.3, 0.5) if i < 4 else (0.85, 0.5)
            fr, _ = draw_ball_frame(w, h, xy)
            if i >= 4:
                # Also paint nothing on the left so the only bright blob teleports.
                pass
            frames.append(fr)
        banks = AppearanceBanks()
        add_ball_sample(banks, frames[0], ball[0], H[0],
                        params=BallCropParams(radius_px=6, min_bright_frac=0.2))
        win = KickWindow(kick_index=3, start=0, end=n)
        track = track_kick_window(
            frames, win, H, ball, players, banks, (105.0, 68.0),
            frame_offset=0, slot_ids=["1"],
            params=DetectParams(min_score=0.1, radius_px=6, max_link_dist_px=25,
                                max_pred_dist_px=30, foot_exclude_px=0, min_bright_frac=0.2,
                                locked_roi_pad_px=30,
                                roi=RoiParams(pad_px=40, min_size_px=40)),
        )
        # Early detections near u≈96; no accepted jump to u≈272.
        for d in track.detections:
            self.assertLess(d.u, 160)

    def test_detect_kicks_in_tracking_api(self):
        if not self.has_cv2:
            self.skipTest("cv2 required")
        w, h = 160, 90
        n = 25
        ball = np.zeros((n, 2))
        ball[:, 0] = 0.4
        ball[:, 1] = 0.5
        ball[12:, 0] = 0.55
        players = np.zeros((n, 1, 2))
        players[:] = (0.1, 0.1)
        H = np.stack([scaled_H(w, h)] * n)
        frames_by_index = {}
        for i in range(n):
            fr, _ = draw_ball_frame(w, h, ball[i], radius=4)
            frames_by_index[i] = fr
        banks = AppearanceBanks()
        add_ball_sample(banks, frames_by_index[0], ball[0], H[0],
                        params=BallCropParams(radius_px=5, min_bright_frac=0.15))
        kick_mask, _ = detect_kicks(ball, (105.0, 68.0), fps=25.0, threshold=4.0)
        tracks = detect_kicks_in_tracking(
            frames_by_index, kick_mask, H, ball, players, banks, (105.0, 68.0),
            pre=4, post=4, slot_ids=["1"], max_kicks=3,
            params=DetectParams(min_score=0.1, radius_px=5, foot_exclude_px=0,
                                min_bright_frac=0.15, max_link_dist_px=40,
                                max_pred_dist_px=50),
        )
        self.assertGreaterEqual(len(tracks), 1)

    def test_choose_contact_prefers_kick_frame(self):
        track = KickBallTrack(kick_frame=10, window_start=5, window_end=15, detections=[
            BallDetection(9, 1, 1, 0.9, 5),
            BallDetection(10, 2, 2, 0.5, 5),
            BallDetection(11, 3, 3, 0.8, 5),
        ])
        c = choose_contact(track)
        self.assertEqual(c.frame, 10)


class ResultsIoTest(unittest.TestCase):
    def test_json_roundtrip(self):
        export = DetectExport(source="tactical", banks_path="x.json", tracks=[
            KickBallTrack(
                kick_frame=100, window_start=90, window_end=110,
                detections=[BallDetection(100, 10.0, 20.0, 0.7, 8.0)],
                contact=BallDetection(100, 10.0, 20.0, 0.7, 8.0),
            ),
        ])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "det.json")
            save_detections(export, path)
            loaded = load_detections(path)
            self.assertEqual(loaded.tracks[0].kick_frame, 100)
            self.assertEqual(loaded.tracks[0].contact.u, 10.0)


if __name__ == "__main__":
    unittest.main()

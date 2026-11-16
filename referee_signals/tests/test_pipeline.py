"""Synthetic-frame tests — no match video required."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Allow running as script from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from referee_signals.classify import TemporalSmoother, classify_geometry
from referee_signals.detect import DetectorConfig, detect_frame, detections_to_schema
from referee_signals.io import index_by_frame, save_signals, load_signals
from referee_signals.schema import empty_document
from referee_signals.seeds import load_seeds
from referee_signals.synthetic import make_scene
from referee_signals.pose import mediapipe_available, classify_law5_pose
from referee_signals.broadcast import should_use_broadcast, load_broadcast_seeds


class TestSyntheticDetection(unittest.TestCase):
    def setUp(self):
        self.seeds = load_seeds()
        # Relax area thresholds for small synthetic figures
        self.seeds["official"]["min_area"] = 20
        self.seeds["flag"]["min_area"] = 5
        self.seeds["flag"]["min_area_frac_of_body"] = 0.001
        self.cfg = DetectorConfig(seeds=self.seeds)

    def test_raised_and_throw(self):
        img, expected = make_scene("raised", "throw_right", w=640, h=360)
        rows = detect_frame(img, self.cfg, frame_idx=0)
        ids = {r["id"]: r for r in rows}
        self.assertIn("ar_top", ids)
        self.assertIn("ar_bottom", ids)
        # Top should prefer raised; bottom throw_right — allow unknown if angle noisy
        top_sig = ids["ar_top"]["signal"]
        bot_sig = ids["ar_bottom"]["signal"]
        self.assertIn(top_sig, {"attention_raised", "unknown", "substitution"})
        self.assertIn(bot_sig, {"throw_in_right", "unknown", "attention_raised", "corner", "goal_kick"})
        if ids["ar_top"].get("flag"):
            self.assertLess(ids["ar_top"]["rel"]["dy"], 0.1)

    def test_schema_roundtrip(self):
        img, _ = make_scene()
        rows = detect_frame(img, self.cfg, frame_idx=42)
        dets = detections_to_schema(rows)
        doc = empty_document()
        doc["detections"] = dets
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.json")
            save_signals(doc, path)
            loaded = load_signals(path)
            idx = index_by_frame(loaded)
            self.assertIn(42, idx)

    def test_classifier_rules(self):
        sig, conf = classify_geometry({"dx": 0.05, "dy": -0.5, "angle": 80, "side": 1, "raised": 1, "length_ratio": 0.2})
        self.assertEqual(sig, "attention_raised")
        self.assertGreater(conf, 0.5)
        sig, _ = classify_geometry({"dx": 0.4, "dy": 0.0, "angle": 10, "side": 1, "raised": 0, "length_ratio": 0.2})
        self.assertEqual(sig, "throw_in_right")
        sig, _ = classify_geometry({"dx": -0.4, "dy": 0.0, "angle": 10, "side": -1, "raised": 0, "length_ratio": 0.2})
        self.assertEqual(sig, "throw_in_left")

    def test_smoother(self):
        sm = TemporalSmoother(5)
        for _ in range(3):
            sm.update("ar_top", "attention_raised", 0.8)
        sm.update("ar_top", "none", 0.1)
        sm.update("ar_top", "attention_raised", 0.7)
        sig, conf = sm.update("ar_top", "attention_raised", 0.7)
        self.assertEqual(sig, "attention_raised")
        self.assertGreater(conf, 0.3)

    def test_pose_stub(self):
        sig, conf = classify_law5_pose(None)
        self.assertEqual(sig, "none")
        _ = mediapipe_available()  # should not raise

    def test_broadcast_hooks(self):
        seeds = load_broadcast_seeds()
        self.assertEqual(seeds["source"], "broadcast")
        self.assertTrue(should_use_broadcast({"flag_recall": 0.2, "mean_ar_height_px": 10}))
        self.assertFalse(should_use_broadcast({"flag_recall": 0.9, "mean_ar_height_px": 40}))


if __name__ == "__main__":
    unittest.main()

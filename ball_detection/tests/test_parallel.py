"""Tests for memory-budgeted parallelism helpers."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ball_detection.kicks import KickWindow  # noqa: E402
from ball_detection.parallel import (  # noqa: E402
    choose_workers,
    estimate_window_mb,
    subsample_windows,
)


class ParallelHelpersTest(unittest.TestCase):
    def test_estimate_scales_with_frames(self):
        a = estimate_window_mb(1920, 1080, 10, overhead=1.0)
        b = estimate_window_mb(1920, 1080, 20, overhead=1.0)
        self.assertAlmostEqual(b / a, 2.0, places=5)

    def test_choose_workers_memory_cap(self):
        # 120 MiB/window, 400 MiB budget → 3 workers
        self.assertEqual(choose_workers(400, 120, workers=None), 3)
        self.assertEqual(choose_workers(400, 120, workers=8), 3)
        self.assertEqual(choose_workers(400, 120, workers=2), 2)
        self.assertEqual(choose_workers(50, 120, workers=None), 1)

    def test_subsample_stride_and_cap(self):
        wins = [KickWindow(i, i, i + 1) for i in range(10)]
        got = subsample_windows(wins, stride=3, max_kicks=None)
        self.assertEqual([w.kick_index for w in got], [0, 3, 6, 9])
        got2 = subsample_windows(wins, stride=2, max_kicks=3)
        self.assertEqual([w.kick_index for w in got2], [0, 2, 4])


if __name__ == "__main__":
    unittest.main()

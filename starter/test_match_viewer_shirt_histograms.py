import tempfile
import unittest
from pathlib import Path

import numpy as np

import match_viewer


class ShirtHistogramViewerTests(unittest.TestCase):
    def test_histogram_closer_to_other_team_mean_is_flagged(self):
        histograms = np.asarray([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        team_ids = np.asarray(["home", "home", "away", "away", "away"])

        mismatch, teams, means = match_viewer.histogram_team_mismatches(
            histograms, team_ids
        )

        self.assertEqual(teams.tolist(), ["away", "home"])
        self.assertEqual(mismatch.tolist(), [False, False, True, False, False])
        np.testing.assert_allclose(means.sum(axis=1), 1.0)

    def test_archive_maps_mismatches_by_frame_and_tracker_slot(self):
        histograms = np.asarray([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        frames = np.asarray([10, 11, 12, 10, 11, 12, 13], dtype=np.int32)
        track_ids = np.asarray(["1", "1", "1", "2", "2", "2", "1"])
        team_ids = np.asarray(["home"] * 3 + ["away"] * 3 + ["home"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(
                path,
                histograms=histograms,
                video_frame=frames,
                track_id=track_ids,
                team_id=team_ids,
            )
            mismatch, _, _ = match_viewer.load_shirt_mismatches(
                str(path), ["1", "2"], 10, 13
            )

        self.assertEqual(mismatch.shape, (4, 2))
        self.assertEqual(np.flatnonzero(mismatch[3]).tolist(), [0])
        self.assertFalse(mismatch[:3].any())


if __name__ == "__main__":
    unittest.main()

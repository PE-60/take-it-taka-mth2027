import unittest

import numpy as np

import match_viewer


class ChangedPossessionTimelineTests(unittest.TestCase):
    def test_changed_possession_is_included_in_timeline(self):
        n = 4
        atd = {
            "ball": np.tile([0.2, 0.5, np.nan, np.nan], (n, 1)),
            "xy": np.tile(
                np.asarray([[[0.2, 0.5], [0.8, 0.5]]], dtype=np.float32),
                (n, 1, 1),
            ),
            "ids": ["home", "away"],
            "kick": np.zeros(n, dtype=bool),
        }
        meta = {"home": {"team": "H"}, "away": {"team": "A"}}
        changed = match_viewer.compute_possession_changed(atd, meta, "H", 1.0)

        groups, _ = match_viewer.build_timeline(
            {}, None, [], None, None, changed, None, [], 0, n - 1, n
        )

        possession_rows = dict(groups)["POSSESSION"]
        row_names = [name for name, _ in possession_rows]
        self.assertIn("L&H CHANGED", row_names)
        changed_strip = dict(possession_rows)["L&H CHANGED"]
        np.testing.assert_array_equal(
            changed_strip,
            np.tile(match_viewer.LABEL_COLOR["H"], (n, 1)),
        )

    def test_derived_goalkeepers_replace_raw_goalkeeper_slots(self):
        n = 3
        atd = {
            "ball": np.tile([0.5, 0.5, np.nan, np.nan], (n, 1)),
            "xy": np.tile(
                np.asarray([[[0.9, 0.5], [0.505, 0.5], [0.9, 0.5]]], dtype=np.float32),
                (n, 1, 1),
            ),
            "ids": ["24", "10", "46"],
            "kick": np.zeros(n, dtype=bool),
        }
        meta = {
            "24": {"team": "H"},
            "10": {"team": "A"},
            "46": {"team": "A"},
        }
        derived = np.tile(
            np.asarray([[[0.5, 0.5], [0.9, 0.5]]], dtype=np.float32),
            (n, 1, 1),
        )

        raw_result = match_viewer.compute_possession_changed(atd, meta, "24", 1.0)
        derived_result = match_viewer.compute_possession_changed(
            atd, meta, "24", 1.0,
            derived_goalkeepers=derived,
            raw_goalkeeper_slots=[0, 2],
        )

        self.assertEqual(raw_result["lh"].tolist(), ["A"] * n)
        self.assertEqual(derived_result["lh"].tolist(), ["H"] * n)


if __name__ == "__main__":
    unittest.main()

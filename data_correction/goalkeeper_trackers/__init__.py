"""Corrected home/away goalkeeper tracks from the swapped tactical ATD keeper slots.

    python data_correction/goalkeeper_trackers/run.py
    # -> data_correction/goalkeeper_trackers/goalkeeper_tracks.csv
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_correction.goalkeeper_trackers.derive import (CSV_COLUMNS, DEFAULT_TRACKERS,  # noqa: E402
                                                         DerivedGoalkeepers, defending_half,
                                                         derive_goalkeeper_tracks, iter_csv_rows,
                                                         status_counts, write_csv)
from data_correction.goalkeeper_trackers.visibility import camera_visible, projective_view_valid  # noqa: E402

__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_TRACKERS",
    "DerivedGoalkeepers",
    "camera_visible",
    "defending_half",
    "derive_goalkeeper_tracks",
    "iter_csv_rows",
    "projective_view_valid",
    "status_counts",
    "write_csv",
]

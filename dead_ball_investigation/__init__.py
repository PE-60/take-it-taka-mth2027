"""Dead-ball intervals from the synchronised tactical + broadcast recordings.

    dead ball = (broadcast homography missing ∩ broadcast player count ≠ 22)
              ∪ (tactical ball missing ∩ broadcast ball missing)

    python dead_ball_investigation/run.py        # -> match_viewer/export/dead_ball.json
"""

import os
import sys

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.dead_ball import (FORMULA, DeadBall, SyncedMatch, Timeline, annotate,  # noqa: E402
                                               annotate_dead_ball, dead_ball_mask, dead_ball_terms,
                                               intervals_to_mask, mask_to_intervals, to_document)
from dead_ball_investigation.signals import SourceSignals, load_from_export, load_source, load_tags  # noqa: E402
from dead_ball_investigation.sync import SyncMap  # noqa: E402

__all__ = ["FORMULA", "DeadBall", "SourceSignals", "SyncMap", "SyncedMatch", "Timeline", "annotate",
           "annotate_dead_ball", "dead_ball_mask", "dead_ball_terms", "intervals_to_mask", "load_from_export",
           "load_source", "load_tags", "mask_to_intervals", "to_document"]

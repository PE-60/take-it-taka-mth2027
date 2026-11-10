"""Dead-ball intervals from the synchronised tactical + broadcast recordings.

    no view   = broadcast homography missing ∩ broadcast player count ≠ 22
    ball lost = tactical ball missing ∩ broadcast ball missing
    baseline  = no view ∪ (the ball-lost intervals that overlap a no-view interval)
    then      1. frames without a counterpart in the other video are dead
              2. the tactical ball off the pitch is dead (the broadcast's where the tactical ball is missing)
              3. a gap between dead intervals in which the ball stays still is dead

    python dead_ball_investigation/run.py        # -> match_viewer/export/dead_ball.json
"""

import os
import sys

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.dead_ball import (FORMULA, DeadBall, Params, SyncedMatch, Timeline, annotate,  # noqa: E402
                                               annotate_dead_ball, ball_travel, dead_ball_mask, dead_ball_terms,
                                               intervals_to_mask, mask_to_intervals, off_pitch, runs_overlapping,
                                               still_gaps, to_document)
from dead_ball_investigation.signals import (SourceSignals, load_from_export, load_source, load_tags,  # noqa: E402
                                             runs_longer_than)
from dead_ball_investigation.sync import SyncMap  # noqa: E402

__all__ = ["FORMULA", "DeadBall", "Params", "SourceSignals", "SyncMap", "SyncedMatch", "Timeline", "annotate",
           "annotate_dead_ball", "ball_travel", "dead_ball_mask", "dead_ball_terms", "intervals_to_mask",
           "load_from_export", "load_source", "load_tags", "mask_to_intervals", "off_pitch", "runs_longer_than",
           "runs_overlapping", "still_gaps", "to_document"]

"""Where things are: the repository around this package, its inputs and the results folder."""

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PACKAGE_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # games.py, dead_ball_investigation/ and evaluation/ live at the repo root

from dead_ball_investigation.paths import DEAD_BALL_JSON, EXPORT_DIR, VIEWER_DIR  # noqa: E402,F401
from dead_ball_investigation.sync import DEFAULT_SYNC  # noqa: E402,F401

RESULTS_DIR = os.path.join(REPO, "evaluation", "results")
EVALUATION_DIR = os.path.join(REPO, "evaluation")
GT_DIR = os.path.join(EVALUATION_DIR, "gt")

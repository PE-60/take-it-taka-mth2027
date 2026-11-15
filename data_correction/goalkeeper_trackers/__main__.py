import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_correction.goalkeeper_trackers.run import cli  # noqa: E402

sys.exit(cli())

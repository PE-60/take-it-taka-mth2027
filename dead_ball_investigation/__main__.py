import os
import sys

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.run import cli  # noqa: E402

sys.exit(cli())

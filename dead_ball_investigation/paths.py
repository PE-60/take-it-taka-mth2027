"""Where the repository is: the closest folder above this package that holds match_viewer/match_viewer.py."""

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_repo(start: str = PACKAGE_DIR) -> str:
    d = os.path.dirname(start)
    while True:
        if os.path.isfile(os.path.join(d, "match_viewer", "match_viewer.py")):
            return d
        up = os.path.dirname(d)
        if up == d:
            return os.path.dirname(start)
        d = up


REPO = find_repo()
VIEWER_DIR = os.path.join(REPO, "match_viewer")
EXPORT_DIR = os.path.join(VIEWER_DIR, "export")
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # games.py lives at the repo root


def require_repo():
    """Exit with a hint when the package is not inside a checkout of the repository."""
    if not os.path.isfile(os.path.join(VIEWER_DIR, "match_viewer.py")) or not os.path.isfile(os.path.join(REPO, "games.py")):
        sys.exit(f"cannot find the repository around {PACKAGE_DIR}\n"
                 "  put the dead_ball_investigation/ folder at the repo root, next to match_viewer/ and games.py")

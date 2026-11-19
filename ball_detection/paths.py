"""Repo / package paths for ball_detection. Does not open match data."""

from __future__ import annotations

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(PACKAGE_DIR, "export")


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
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def require_repo() -> str:
    if (not os.path.isfile(os.path.join(REPO, "match_viewer", "match_viewer.py"))
            or not os.path.isfile(os.path.join(REPO, "games.py"))):
        raise FileNotFoundError(
            f"cannot find the repository around {PACKAGE_DIR}\n"
            "  put ball_detection/ at the repo root, next to match_viewer/ and games.py")
    return REPO


def ensure_export_dir(path: str | None = None) -> str:
    out = path or EXPORT_DIR
    os.makedirs(out, exist_ok=True)
    return out

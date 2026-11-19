"""Optional loaders for match assets. Pure logic lives elsewhere; this only wires files.

All functions raise a clear error when gated data is missing so unit tests never need it.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

import numpy as np

from ball_detection.geometry import load_homography_table
from ball_detection.paths import REPO, require_repo


@dataclass
class TrackingSlice:
    """Minimal tracking needed to harvest appearance banks."""
    start: int
    ball: np.ndarray          # (n, 2)
    xy: np.ndarray            # (n, S, 2)
    team: np.ndarray          # (S,) "H"/"A"/"?"
    slot_ids: np.ndarray      # (S,) str
    pitch_m: tuple[float, float]
    fps: float
    source: str = "tactical"

    @property
    def n(self) -> int:
        return len(self.ball)


def data_available(source: str = "tactical") -> bool:
    """True when video + ATD + homography exist for ``source``."""
    try:
        require_repo()
        from games import asset_paths
        paths = asset_paths(source=source)
        needed = ("video", "atd_raw", "atd_meta", "homography")
        return all(os.path.isfile(paths[k]) for k in needed)
    except Exception:
        return False


def _viewer():
    require_repo()
    path = os.path.join(REPO, "match_viewer", "match_viewer.py")
    mod = None
    name = "match_viewer_ball_detection"
    if name in __import__("sys").modules:
        return __import__("sys").modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    __import__("sys").modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_tracking(source: str = "tactical", first: int | None = None, last: int | None = None) -> TrackingSlice:
    """Load ATD via match_viewer for frames first..last inclusive (defaults: whole recording)."""
    require_repo()
    from games import GAMES, GAME, asset_paths, require, source_frames

    paths = asset_paths(source=source)
    require(("atd_raw", "atd_meta"), source=source)
    n_total = int(source_frames(source))
    first = 0 if first is None else int(first)
    last = n_total - 1 if last is None else int(last)
    if first < 0 or last >= n_total or last < first:
        raise ValueError(f"bad frame range {first}..{last} for {n_total} frames")

    mv = _viewer()
    atd, players_meta, home = mv.load_atd(paths["atd_raw"], paths["atd_meta"], first, last)
    team = np.array([mv.team_colour(pid, players_meta, home)[1] for pid in atd["ids"]], dtype="<U1")
    slot_ids = np.asarray([str(pid) for pid in atd["ids"]], dtype=object)
    pitch = tuple(float(x) for x in GAMES[GAME]["pitch"])
    return TrackingSlice(
        start=first,
        ball=np.asarray(atd["ball"][:, :2], dtype=np.float64),
        xy=np.asarray(atd["xy"][:, :, :2], dtype=np.float64),
        team=team,
        slot_ids=slot_ids,
        pitch_m=pitch,
        fps=float(GAMES[GAME]["fps"]),
        source=source,
    )


def load_homography(source: str = "tactical", n_frames: int | None = None):
    require_repo()
    from games import asset_paths, require, source_frames
    paths = asset_paths(source=source)
    require(("homography",), source=source)
    n = int(source_frames(source) if n_frames is None else n_frames)
    return load_homography_table(paths["homography"], n)


def load_homography_slice(source: str = "tactical", first: int = 0, last: int | None = None):
    """Load only homography rows first..last inclusive — avoids a full-match (n,3,3) allocation.

    Returns (matrices, reprojection_error, valid) each length ``last - first + 1``.
    """
    import csv

    from ball_detection.geometry import MATRIX_COLUMNS
    from games import asset_paths, require, source_frames

    require_repo()
    paths = asset_paths(source=source)
    require(("homography",), source=source)
    n_total = int(source_frames(source))
    last = n_total - 1 if last is None else int(last)
    first = int(first)
    if first < 0 or last >= n_total or last < first:
        raise ValueError(f"bad homography slice {first}..{last} for {n_total} frames")
    n = last - first + 1
    matrices = np.full((n, 3, 3), np.nan, dtype=np.float64)
    reprojection_error = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)

    with open(paths["homography"], newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("homography CSV has no header")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"frame", "valid", "reprojection_error"} | set(MATRIX_COLUMNS)
        missing = required - set(fields)
        if missing:
            raise ValueError(f"homography missing columns {sorted(missing)}")
        for row in reader:
            frame = int(float(row[fields["frame"]]))
            if frame < first or frame > last:
                continue
            i = frame - first
            if seen[i]:
                raise ValueError(f"duplicate homography row for frame {frame}")
            seen[i] = True
            valid[i] = bool(int(float(row[fields["valid"]])))
            err_raw = row[fields["reprojection_error"]]
            reprojection_error[i] = float(err_raw) if err_raw not in ("", None) else float("nan")
            values = [float(row[fields[col]]) for col in MATRIX_COLUMNS]
            matrices[i] = np.asarray(values, dtype=np.float64).reshape(3, 3)

    if not seen.all():
        raise ValueError(f"homography slice {first}..{last} is missing {int((~seen).sum())} rows")
    return matrices, reprojection_error, valid


def open_video(source: str = "tactical"):
    """cv2.VideoCapture for the source video. Caller must release it."""
    require_repo()
    import cv2
    from games import asset_paths, require
    paths = asset_paths(source=source)
    require(("video",), source=source)
    cap = cv2.VideoCapture(paths["video"])
    if not cap.isOpened():
        raise OSError(f"could not open video {paths['video']}")
    return cap


def read_frame(cap, frame_index: int) -> np.ndarray | None:
    """Seek and read one BGR frame. Returns None on failure."""
    cap.set(1, int(frame_index))  # CAP_PROP_POS_FRAMES
    ok, frame = cap.read()
    return frame if ok else None

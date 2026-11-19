"""Pitch <-> pixel transforms and homography CSV loading.

Self-contained: no referee_signals, no match_viewer import required for the math.
Homography maps normalised pitch coordinates (0-1), the same convention as ATD, to image pixels.
"""

from __future__ import annotations

import csv
from typing import Iterable

import numpy as np

MATRIX_COLUMNS = [f"h{row}{col}" for row in range(3) for col in range(3)]


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N, 2) normalised pitch coords -> (N, 2) image pixels. NaN in -> NaN out."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H, dtype=np.float64)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    p = np.hstack([pts, ones]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def project(H: np.ndarray, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Scalars or arrays of pitch (x, y) -> pixel (u, v)."""
    pts = project_points(H, np.column_stack([np.asarray(x, dtype=np.float64),
                                             np.asarray(y, dtype=np.float64)]))
    return pts[:, 0], pts[:, 1]


def unproject_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N, 2) image pixels -> (N, 2) normalised pitch coords."""
    return project_points(np.linalg.inv(np.asarray(H, dtype=np.float64)), pts)


def pixel_in_frame(u: float, v: float, width: int, height: int, margin: int = 0) -> bool:
    return (np.isfinite(u) and np.isfinite(v)
            and margin <= u < width - margin and margin <= v < height - margin)


def pitch_distance_m(a: np.ndarray, b: np.ndarray, pitch_m: tuple[float, float]) -> np.ndarray:
    """Euclidean distance in metres between normalised pitch points.

    a: (..., 2), b: (..., 2) broadcastable. Returns metres.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.asarray(pitch_m, dtype=np.float64)
    return np.linalg.norm((a - b) * scale, axis=-1)


def load_homography_table(path: str, n_frames: int | None = None):
    """Load (matrices (n,3,3), reprojection_error (n,), valid (n,)) indexed by video frame.

    If n_frames is set, the CSV must cover frames 0 .. n_frames-1 exactly once.
    """
    matrices_list: list[np.ndarray] = []
    errors: list[float] = []
    valids: list[bool] = []
    frames: list[int] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"frame", "valid", "reprojection_error"} | set(MATRIX_COLUMNS)
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            frame = int(float(row[fields["frame"]]))
            frames.append(frame)
            valid = bool(int(float(row[fields["valid"]])))
            valids.append(valid)
            err_raw = row[fields["reprojection_error"]]
            errors.append(float(err_raw) if err_raw not in ("", None) else float("nan"))
            values = [float(row[fields[col]]) for col in MATRIX_COLUMNS]
            matrices_list.append(np.asarray(values, dtype=np.float64).reshape(3, 3))

    if not frames:
        raise ValueError(f"{path}: empty")

    if n_frames is None:
        n_frames = max(frames) + 1

    matrices = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    reprojection_error = np.full(n_frames, np.nan, dtype=np.float64)
    valid = np.zeros(n_frames, dtype=bool)
    seen = np.zeros(n_frames, dtype=bool)

    for frame, H, err, ok in zip(frames, matrices_list, errors, valids):
        if frame < 0 or frame >= n_frames:
            raise ValueError(f"{path}: frame {frame} outside 0..{n_frames - 1}")
        if seen[frame]:
            raise ValueError(f"{path}: duplicate frame {frame}")
        seen[frame] = True
        matrices[frame] = H
        reprojection_error[frame] = err
        valid[frame] = ok

    if n_frames is not None and not seen.all():
        missing_frames = int((~seen).sum())
        raise ValueError(f"{path}: missing {missing_frames} frame rows (expected 0..{n_frames - 1})")

    return matrices, reprojection_error, valid


def write_homography_table(path: str, matrices: np.ndarray, reprojection_error: Iterable[float],
                           valid: Iterable[bool]) -> None:
    """Write a minimal homography CSV (for tests / synthetic fixtures)."""
    matrices = np.asarray(matrices, dtype=np.float64)
    reprojection_error = np.asarray(list(reprojection_error), dtype=np.float64)
    valid = np.asarray(list(valid), dtype=bool)
    n = len(matrices)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", *MATRIX_COLUMNS, "reprojection_error", "valid"])
        for i in range(n):
            flat = matrices[i].reshape(-1).tolist()
            writer.writerow([i, *flat, float(reprojection_error[i]), int(bool(valid[i]))])

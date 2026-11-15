"""Camera visibility for ATD pitch positions, matching match_viewer / the notebook.

A position is camera-visible when the frame's homography is usable, the point lies in
normalized pitch bounds, and its projected image point falls inside the video frame.
"""

from __future__ import annotations

import numpy as np

PITCH_BOUNDS = (0.0, 0.0, 1.0, 1.0)


def projective_view_valid(matrix: np.ndarray, bounds=PITCH_BOUNDS) -> bool:
    """Whether the homography maps the pitch rectangle without crossing the projection plane."""
    matrix = np.asarray(matrix, dtype=np.float64)
    corners = np.asarray([
        [bounds[0], bounds[1]],
        [bounds[2], bounds[1]],
        [bounds[2], bounds[3]],
        [bounds[0], bounds[3]],
    ], dtype=np.float64)
    corner_w = matrix[2, 0] * corners[:, 0] + matrix[2, 1] * corners[:, 1] + matrix[2, 2]
    return bool(
        np.isfinite(matrix).all()
        and np.isfinite(corner_w).all()
        and np.all(np.abs(corner_w) >= 1e-8)
        and np.all(np.sign(corner_w) == np.sign(corner_w[0]))
    )


def camera_visible(positions: np.ndarray, homographies: np.ndarray, valid: np.ndarray,
                   video_size: tuple[int, int], bounds=PITCH_BOUNDS) -> np.ndarray:
    """(n, S) bool: True where each slot's pitch position projects inside the video image.

    positions     (n, S, 2) pitch fractions (NaN = absent)
    homographies  (n, 3, 3)
    valid         (n,)      CSV `valid` flag
    video_size    (width, height) in pixels
    """
    positions = np.asarray(positions, dtype=np.float64)
    homographies = np.asarray(homographies, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n, n_slots, _ = positions.shape
    width, height = int(video_size[0]), int(video_size[1])
    out = np.zeros((n, n_slots), dtype=bool)

    on_pitch = (
        np.isfinite(positions).all(axis=2)
        & (positions[:, :, 0] >= bounds[0]) & (positions[:, :, 0] <= bounds[2])
        & (positions[:, :, 1] >= bounds[1]) & (positions[:, :, 1] <= bounds[3])
    )
    for frame in np.flatnonzero(on_pitch.any(axis=1)):
        matrix = homographies[frame]
        if not valid[frame] or not projective_view_valid(matrix, bounds):
            continue
        indices = np.flatnonzero(on_pitch[frame])
        xy = positions[frame, indices]
        homogeneous = np.column_stack([xy, np.ones(len(xy))]) @ matrix.T
        denominator = homogeneous[:, 2]
        projectable = np.isfinite(homogeneous).all(axis=1) & (np.abs(denominator) >= 1e-8)
        pixels = np.full((len(indices), 2), np.nan, dtype=np.float64)
        pixels[projectable] = homogeneous[projectable, :2] / denominator[projectable, None]
        in_camera = (
            projectable
            & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        out[frame, indices[in_camera]] = True
    return out

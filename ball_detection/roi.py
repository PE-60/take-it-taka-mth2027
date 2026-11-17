"""Spatial search ROI in image pixels for a kick window frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ball_detection.geometry import pitch_distance_m, project_points
from ball_detection.gk import default_goalkeeper_slot_ids, outfield_slot_mask
from ball_detection.kicks import ball_prior_at, ball_velocity_prior


@dataclass(frozen=True)
class PixelROI:
    """Inclusive-exclusive pixel box clipped to the frame: x0:x1, y0:y1."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    def clip(self, width: int, height: int) -> "PixelROI":
        x0 = int(np.clip(self.x0, 0, width))
        y0 = int(np.clip(self.y0, 0, height))
        x1 = int(np.clip(self.x1, 0, width))
        y1 = int(np.clip(self.y1, 0, height))
        return PixelROI(x0, y0, x1, y1)

    def pad(self, px: int) -> "PixelROI":
        p = int(px)
        return PixelROI(self.x0 - p, self.y0 - p, self.x1 + p, self.y1 + p)

    def as_slice(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


@dataclass(frozen=True)
class RoiParams:
    player_radius_m: float = 5.0      # include outfield feet within this of the prior
    pad_px: int = 40                  # expand bbox in pixels
    min_size_px: int = 48             # ensure ROI is at least this wide/tall
    extrapolate_frames: int = 4       # use velocity prior this many frames ahead of last ball
    lookback_ball: int = 25


def _anchor_pitch(
        ball: np.ndarray,
        frame_index: int,
        pitch_m: tuple[float, float],
        fps: float,
        params: RoiParams,
) -> np.ndarray | None:
    prior = ball_prior_at(ball, frame_index, lookback=params.lookback_ball)
    if prior is None:
        return None
    # How many frames after the last finite ball observation is this frame?
    last_i = None
    lo = max(0, frame_index - params.lookback_ball)
    for i in range(frame_index, lo - 1, -1):
        if np.isfinite(ball[i, 0]) and np.isfinite(ball[i, 1]):
            last_i = i
            break
    if last_i is None:
        return prior
    dt = frame_index - last_i
    if dt <= 0 or dt > params.extrapolate_frames:
        return prior
    vel = ball_velocity_prior(ball, last_i, pitch_m, fps, lookback=5)
    if vel is None:
        return prior
    return prior + vel * dt


def search_roi(
        H: np.ndarray,
        ball: np.ndarray,
        players_xy: np.ndarray,
        frame_index: int,
        frame_wh: tuple[int, int],
        pitch_m: tuple[float, float],
        fps: float = 25.0,
        *,
        slot_ids=None,
        gk_ids=None,
        params: RoiParams | None = None,
) -> tuple[PixelROI | None, np.ndarray | None]:
    """Pixel ROI + pitch-fraction anchor for ``frame_index``.

    ``players_xy`` is (S, 2) for this frame. Returns (None, None) if no usable prior.
    """
    params = params or RoiParams()
    width, height = int(frame_wh[0]), int(frame_wh[1])
    anchor = _anchor_pitch(ball, frame_index, pitch_m, fps, params)
    if anchor is None:
        return None, None

    pts = [anchor]
    S = len(players_xy)
    if S:
        if slot_ids is None:
            outfield = np.ones(S, dtype=bool)
        else:
            outfield = outfield_slot_mask(
                slot_ids, gk_ids if gk_ids is not None else default_goalkeeper_slot_ids())
        for s in np.flatnonzero(outfield):
            ply = np.asarray(players_xy[s], dtype=np.float64)
            if not (np.isfinite(ply).all()):
                continue
            d = float(pitch_distance_m(anchor, ply, pitch_m))
            if d <= params.player_radius_m:
                pts.append(ply)

    pix = project_points(H, np.asarray(pts, dtype=np.float64))
    finite = np.isfinite(pix).all(axis=1)
    if not finite.any():
        return None, anchor
    pix = pix[finite]
    x0 = int(np.floor(pix[:, 0].min()))
    x1 = int(np.ceil(pix[:, 0].max())) + 1
    y0 = int(np.floor(pix[:, 1].min()))
    y1 = int(np.ceil(pix[:, 1].max())) + 1
    roi = PixelROI(x0, y0, x1, y1).pad(params.pad_px)

    # Enforce a minimum box around the projected anchor.
    ap = project_points(H, anchor.reshape(1, 2))[0]
    if np.isfinite(ap).all():
        half = max(params.min_size_px // 2, 1)
        au, av = int(round(float(ap[0]))), int(round(float(ap[1])))
        min_box = PixelROI(au - half, av - half, au + half, av + half)
        roi = PixelROI(
            min(roi.x0, min_box.x0),
            min(roi.y0, min_box.y0),
            max(roi.x1, min_box.x1),
            max(roi.y1, min_box.y1),
        )

    return roi.clip(width, height), anchor

"""Build appearance banks from BGR frames + projected ATD positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ball_detection.appearance import (
    AppearanceBanks,
    ball_color_mask_bgr,
    bgr_to_hsv,
    bright_fraction,
    circular_mask,
    grass_mask_hsv,
    hsv_histogram,
)
from ball_detection.geometry import pixel_in_frame, project


@dataclass(frozen=True)
class BallCropParams:
    radius_px: float = 6.0
    min_patch: int = 4
    exclude_grass: bool = True
    # Reject crops that are mostly shoe/kit: require enough white/light pixels in the circle.
    min_bright_frac: float = 0.35
    use_ball_color_mask: bool = True


@dataclass(frozen=True)
class KitCropParams:
    """Torso band above the projected foot/point location."""
    width_px: int = 24
    height_px: int = 36
    # Shift crop upward so we sample shirt, not feet/grass.
    up_offset_px: int = 28
    min_patch: int = 8
    exclude_grass: bool = True


def crop_square(frame: np.ndarray, cx: float, cy: float, radius: float) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Axis-aligned square crop centred on (cx, cy). Returns (patch, x0,y0,x1,y1) or None."""
    h, w = frame.shape[:2]
    r = max(1, int(round(radius)))
    x0 = int(round(cx)) - r
    y0 = int(round(cy)) - r
    x1 = x0 + 2 * r
    y1 = y0 + 2 * r
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return frame[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def crop_torso(frame: np.ndarray, cx: float, cy: float, params: KitCropParams) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Crop a torso window above the projected player point (ATD ≈ feet)."""
    h, w = frame.shape[:2]
    half_w = max(1, params.width_px // 2)
    x0 = int(round(cx)) - half_w
    x1 = x0 + params.width_px
    y1 = int(round(cy)) - max(0, params.up_offset_px - params.height_px)
    y0 = y1 - params.height_px
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return frame[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _non_grass_mask(patch: np.ndarray) -> np.ndarray:
    hsv = bgr_to_hsv(patch)
    return ~grass_mask_hsv(hsv)


def add_ball_sample(
        banks: AppearanceBanks,
        frame_bgr: np.ndarray,
        ball_xy: np.ndarray,
        H: np.ndarray,
        *,
        params: BallCropParams | None = None,
) -> bool:
    """Project ball, crop, accumulate histogram. Returns True if a sample was added."""
    params = params or BallCropParams()
    h, w = frame_bgr.shape[:2]
    u, v = project(H, ball_xy[0], ball_xy[1])
    u, v = float(np.asarray(u).reshape(())), float(np.asarray(v).reshape(()))
    if not pixel_in_frame(u, v, w, h, margin=int(params.radius_px) + 1):
        return False
    cropped = crop_square(frame_bgr, u, v, params.radius_px)
    if cropped is None:
        return False
    patch, _box = cropped
    if min(patch.shape[:2]) < params.min_patch:
        return False
    mask = circular_mask(patch.shape[0], patch.shape[1],
                         cy=(patch.shape[0] - 1) / 2.0,
                         cx=(patch.shape[1] - 1) / 2.0,
                         radius=params.radius_px)
    if params.exclude_grass:
        mask = mask & _non_grass_mask(patch)
    if params.use_ball_color_mask:
        mask = mask & ball_color_mask_bgr(patch)
    if int(mask.sum()) < params.min_patch:
        return False
    # Whole-circle bright fraction (before color mask) — shoes fail this.
    circle = circular_mask(patch.shape[0], patch.shape[1],
                           cy=(patch.shape[0] - 1) / 2.0,
                           cx=(patch.shape[1] - 1) / 2.0,
                           radius=params.radius_px)
    if bright_fraction(patch, mask=circle) < float(params.min_bright_frac):
        return False
    hist = hsv_histogram(patch, mask=mask.astype(np.uint8) * 255)
    if float(hist.sum()) <= 0:
        return False
    banks.ball.add(hist, radius_px=params.radius_px)
    return True


def add_kit_sample(
        banks: AppearanceBanks,
        frame_bgr: np.ndarray,
        player_xy: np.ndarray,
        team: str,
        H: np.ndarray,
        *,
        params: KitCropParams | None = None,
) -> bool:
    """Project player, crop torso, accumulate team kit histogram."""
    params = params or KitCropParams()
    team = str(team)
    if team not in ("H", "A"):
        return False
    h, w = frame_bgr.shape[:2]
    u, v = project(H, player_xy[0], player_xy[1])
    u, v = float(np.asarray(u).reshape(())), float(np.asarray(v).reshape(()))
    if not pixel_in_frame(u, v, w, h, margin=2):
        return False
    cropped = crop_torso(frame_bgr, u, v, params)
    if cropped is None:
        return False
    patch, _box = cropped
    if min(patch.shape[:2]) < params.min_patch:
        return False
    mask = np.ones(patch.shape[:2], dtype=bool)
    if params.exclude_grass:
        mask = _non_grass_mask(patch)
    if int(mask.sum()) < params.min_patch:
        return False
    hist = hsv_histogram(patch, mask=mask.astype(np.uint8) * 255)
    if float(hist.sum()) <= 0:
        return False
    banks.kit(team).add(hist)
    return True


def accumulate_frame(
        banks: AppearanceBanks,
        frame_bgr: np.ndarray,
        H: np.ndarray,
        *,
        ball_xy: np.ndarray | None = None,
        take_ball: bool = False,
        players_xy: np.ndarray | None = None,
        team: np.ndarray | None = None,
        kit_slot_mask: np.ndarray | None = None,
        ball_params: BallCropParams | None = None,
        kit_params: KitCropParams | None = None,
        max_kits_per_frame: int = 8,
) -> dict[str, int]:
    """Update banks from one frame. kit_slot_mask is (S,) bool for this frame."""
    counts = {"ball": 0, "kit_H": 0, "kit_A": 0}
    if take_ball and ball_xy is not None:
        if add_ball_sample(banks, frame_bgr, np.asarray(ball_xy, dtype=np.float64), H,
                           params=ball_params):
            counts["ball"] = 1
    if players_xy is None or team is None or kit_slot_mask is None:
        return counts
    players_xy = np.asarray(players_xy, dtype=np.float64)
    team = np.asarray(team)
    kit_slot_mask = np.asarray(kit_slot_mask, dtype=bool)
    added = 0
    for s in np.flatnonzero(kit_slot_mask):
        if added >= max_kits_per_frame:
            break
        t = str(team[s] if team.ndim == 1 else team[s])
        if add_kit_sample(banks, frame_bgr, players_xy[s], t, H, params=kit_params):
            counts[f"kit_{t}"] = counts.get(f"kit_{t}", 0) + 1
            added += 1
    return counts

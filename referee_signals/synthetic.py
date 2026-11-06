"""Synthetic frames for unit tests — no confidential match video required."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def make_pitch(w: int = 640, h: int = 360) -> np.ndarray:
    """Green pitch with faint white lines."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 140, 40)  # BGR grass-ish
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (220, 220, 220), 1)
    cv2.line(img, (w // 2, 20), (w // 2, h - 20), (220, 220, 220), 1)
    return img


def draw_ar(
    img: np.ndarray,
    cx: int,
    cy: int,
    scale: float = 1.0,
    flag_pose: str = "raised",
) -> Dict[str, float]:
    """Draw neon-yellow shirt AR + yellow/orange flag in a known pose. Returns expected rel hints."""
    s = scale
    # body
    shirt_w, shirt_h = int(18 * s), int(22 * s)
    short_h = int(14 * s)
    x0 = int(cx - shirt_w // 2)
    y0 = int(cy - shirt_h // 2)
    # neon yellow shirt (BGR ~ cyan-yellow in HSV)
    cv2.rectangle(img, (x0, y0), (x0 + shirt_w, y0 + shirt_h), (0, 240, 240), -1)
    cv2.rectangle(img, (x0, y0 + shirt_h), (x0 + shirt_w, y0 + shirt_h + short_h), (15, 15, 15), -1)
    # head
    cv2.circle(img, (cx, y0 - int(8 * s)), int(7 * s), (40, 60, 180), -1)

    flag_w, flag_h = int(10 * s), int(16 * s)
    torso_cy = y0 + 0.4 * shirt_h

    # Flag uses stronger orange so it separates from neon-yellow shirts in HSV
    yellow, orange = (0, 230, 255), (0, 120, 255)
    if flag_pose == "raised":
        fx, fy = cx + int(14 * s), y0 - int(36 * s)
        cv2.rectangle(img, (fx, fy), (fx + flag_w, fy + flag_h), yellow, -1)
        cv2.rectangle(img, (fx, fy + flag_h // 2), (fx + flag_w, fy + flag_h), orange, -1)
        expected = "attention_raised"
    elif flag_pose == "throw_right":
        fx, fy = cx + int(32 * s), int(torso_cy - flag_h // 2)
        cv2.rectangle(img, (fx, fy), (fx + flag_h, fy + flag_w), yellow, -1)
        cv2.rectangle(img, (fx + flag_h // 2, fy), (fx + flag_h, fy + flag_w), orange, -1)
        expected = "throw_in_right"
    elif flag_pose == "throw_left":
        fx, fy = cx - int(44 * s), int(torso_cy - flag_h // 2)
        cv2.rectangle(img, (fx, fy), (fx + flag_h, fy + flag_w), yellow, -1)
        cv2.rectangle(img, (fx + flag_h // 2, fy), (fx + flag_h, fy + flag_w), orange, -1)
        expected = "throw_in_left"
    elif flag_pose == "sub":
        fx, fy = cx - int(6 * s), int(torso_cy - flag_w // 2)
        cv2.rectangle(img, (fx, fy), (fx + int(flag_h * 0.7), fy + flag_w), orange, -1)
        expected = "substitution"
    else:
        fx, fy = cx + int(14 * s), y0 - int(36 * s)
        cv2.rectangle(img, (fx, fy), (fx + flag_w, fy + flag_h), yellow, -1)
        expected = "attention_raised"

    return {"expected": expected, "cx": cx, "cy": cy}


def make_scene(
    pose_top: str = "raised",
    pose_bottom: str = "throw_right",
    w: int = 640,
    h: int = 360,
) -> Tuple[np.ndarray, Dict[str, str]]:
    img = make_pitch(w, h)
    # top AR near top touchline band
    t = draw_ar(img, w // 3, int(0.12 * h), scale=1.2, flag_pose=pose_top)
    # bottom AR
    b = draw_ar(img, 2 * w // 3, int(0.88 * h), scale=1.2, flag_pose=pose_bottom)
    return img, {"ar_top": t["expected"], "ar_bottom": b["expected"]}

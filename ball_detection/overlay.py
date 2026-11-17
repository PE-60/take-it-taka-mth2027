"""Draw Stage B/C overlays on BGR frames (preview CLI + OpenCV QA)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ball_detection.attribute import TouchAttribution
from ball_detection.results import BallDetection, KickBallTrack

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# BGR
COLOR_BALL = (240, 240, 240)
COLOR_CONTACT = (0, 220, 255)
COLOR_H = (200, 120, 40)       # blue-ish
COLOR_A = (40, 40, 220)        # red-ish
COLOR_ROI = (80, 180, 80)
COLOR_TEXT = (255, 255, 255)


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for overlays")


def team_color(team: str) -> tuple[int, int, int]:
    if team == "H":
        return COLOR_H
    if team == "A":
        return COLOR_A
    return (180, 180, 180)


def draw_ball_detection(frame: np.ndarray, det: BallDetection, *, contact: bool = False) -> None:
    _require_cv2()
    color = COLOR_CONTACT if contact else COLOR_BALL
    u, v = int(round(det.u)), int(round(det.v))
    r = max(2, int(round(det.radius_px)))
    cv2.circle(frame, (u, v), r, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, (u, v), r, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (u, v), 2, color, -1, cv2.LINE_AA)
    label = f"{'contact' if contact else 'ball'} {det.score:.2f}"
    cv2.putText(frame, label, (u + r + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (u + r + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def draw_attribution(frame: np.ndarray, attr: TouchAttribution) -> None:
    _require_cv2()
    if attr.contact is None:
        return
    draw_ball_detection(frame, attr.contact, contact=True)
    color = team_color(attr.team)
    u, v = int(round(attr.contact.u)), int(round(attr.contact.v))
    # Line toward best candidate if we have pixel distance context: mark with label.
    team = attr.team or "?"
    label = f"touch {team} conf={attr.confidence:.2f}"
    if attr.slot_id:
        label += f" #{attr.slot_id}"
    if attr.kit_vote and attr.kit_vote != "?":
        label += f" kit={attr.kit_vote}"
    cv2.putText(frame, label, (u + 12, v + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (u + 12, v + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)

    # Rank markers for top candidates (small dots at projected positions are drawn by caller
    # when player pixels are known). Here we only annotate the contact.


def draw_candidate_pixels(
        frame: np.ndarray,
        candidates_uv: Iterable[tuple[float, float, str, float]],
) -> None:
    """candidates_uv: (u, v, team, score) for top candidates."""
    _require_cv2()
    for u, v, team, score in candidates_uv:
        color = team_color(team)
        uu, vv = int(round(u)), int(round(v))
        cv2.circle(frame, (uu, vv), 10, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (uu, vv), 10, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{team}:{score:.2f}", (uu + 12, vv + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_track_on_frame(
        frame: np.ndarray,
        track: KickBallTrack,
        attr: TouchAttribution | None = None,
        *,
        video_frame: int | None = None,
) -> np.ndarray:
    """Copy frame and draw detections active on ``video_frame`` (default: kick frame)."""
    _require_cv2()
    out = frame.copy()
    f = track.kick_frame if video_frame is None else int(video_frame)
    for det in track.detections:
        if det.frame == f:
            is_contact = (attr is not None and attr.contact is not None
                          and attr.contact.frame == det.frame
                          and abs(attr.contact.u - det.u) < 1e-3)
            if not is_contact and track.contact is not None:
                is_contact = (track.contact.frame == det.frame
                              and abs(track.contact.u - det.u) < 1e-3
                              and abs(track.contact.v - det.v) < 1e-3)
            draw_ball_detection(out, det, contact=is_contact)
    if attr is not None and attr.contact is not None and attr.contact.frame == f:
        draw_attribution(out, attr)
    # HUD
    hud = f"kick {track.kick_frame}  window {track.window_start}-{track.window_end - 1}"
    if attr is not None and attr.team:
        hud += f"  → {attr.team} ({attr.confidence:.2f})"
    cv2.rectangle(out, (8, 8), (8 + 9 * len(hud), 36), (0, 0, 0), -1)
    cv2.putText(out, hud, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    return out

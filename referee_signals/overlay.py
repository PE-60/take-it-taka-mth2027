"""Draw referee flag detections on a BGR frame (match_viewer / debug)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .schema import SIGNAL_STYLE


def draw_detections(
    frame,
    detections: Iterable[Dict[str, Any]],
    scale: float = 1.0,
    thickness: int = 2,
) -> None:
    for d in detections:
        bbox = d.get("bbox_xyxy")
        if not bbox:
            continue
        x0, y0, x1, y1 = [int(v * scale) for v in bbox]
        sig = d.get("signal") or "none"
        style = SIGNAL_STYLE.get(sig, SIGNAL_STYLE["unknown"])
        color = tuple(int(c) for c in style["color"])
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
        label = f"{d.get('official_id', '?')} {sig}"
        conf = d.get("confidence")
        if conf is not None:
            label += f" {float(conf):.2f}"
        cv2.putText(frame, label, (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * max(1.0, scale), color, 1, cv2.LINE_AA)
        fc = d.get("flag_centroid")
        if fc:
            fx, fy = int(fc[0] * scale), int(fc[1] * scale)
            cv2.circle(frame, (fx, fy), max(3, int(4 * scale)), color, -1, cv2.LINE_AA)
            # stick from torso mid to flag
            mx, my = (x0 + x1) // 2, y0 + int(0.4 * (y1 - y0))
            cv2.line(frame, (mx, my), (fx, fy), color, max(1, thickness - 1), cv2.LINE_AA)
            ang = d.get("flag_angle_deg")
            if ang is not None:
                # short orientation tick
                rad = np.radians(float(ang))
                # vertical-ish when ang~90
                dx = int(18 * scale * np.cos(np.radians(90 - float(ang))))
                dy = int(-18 * scale * np.sin(np.radians(float(ang))))
                cv2.line(frame, (fx, fy), (fx + dx, fy + dy), (255, 255, 255), 1, cv2.LINE_AA)

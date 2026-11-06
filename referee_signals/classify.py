"""Law 6 rule classifier + temporal majority smoothing."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple


def classify_geometry(rel: Optional[Dict[str, float]], cfg: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
    """Map relative flag geometry to a Law 6-ish label.

    Angles: ~90 vertical raise, ~0 horizontal. Image x increases rightward.
    """
    if not rel:
        return "none", 0.0
    cfg = cfg or {}
    raised_dy = float(cfg.get("raised_dy", -0.28))
    horiz_max = float(cfg.get("horizontal_angle_max", 35))
    vert_min = float(cfg.get("vertical_angle_min", 55))
    mid_abs = float(cfg.get("mid_torso_dy_abs", 0.22))
    corner_dy = float(cfg.get("corner_dy_min", 0.05))

    dx = rel["dx"]
    dy = rel["dy"]
    ang = rel["angle"]
    lr = rel.get("length_ratio", 0.2)

    # Vertical raise above shoulders / head → attention / offside cue.
    # Angle can be noisy on tiny tactical pixels; strong upward dy is enough.
    if dy <= raised_dy and (ang >= vert_min or (dy <= raised_dy and abs(dx) < 0.5)):
        conf = min(1.0, 0.55 + 0.3 * min(1.0, abs(dy) / 0.6) + 0.15 * (ang / 90.0))
        return "attention_raised", conf

    # Horizontal / lateral point → throw-in direction (before substitution).
    # Prefer large |dx| with mid-torso height; tolerate bad PCA angles.
    lateral = abs(dx) > 0.12 and abs(dy) < 0.45
    flatish = ang <= horiz_max or (abs(dx) > 0.25 and abs(dy) < 0.22)
    if lateral and flatish:
        if dx < 0:
            return "throw_in_left", min(0.85, 0.5 + 0.4 * min(1.0, abs(dx)))
        return "throw_in_right", min(0.85, 0.5 + 0.4 * min(1.0, abs(dx)))

    # Across mid-torso, roughly horizontal and close to body → substitution heuristic
    if abs(dy) <= mid_abs and ang <= horiz_max and abs(dx) <= 0.18:
        return "substitution", min(0.75, 0.4 + 0.3 * (1.0 - abs(dy) / max(mid_abs, 1e-3)))

    # Low angled point — weak on tactical; corner vs goal kick heuristic by side+dy
    if dy >= corner_dy and 20 <= ang <= 70 and abs(dx) > 0.15:
        # Pointing downward-outward: treat as corner; more inward/low: goal kick
        if abs(dx) > 0.35:
            return "corner", 0.35 + 0.2 * min(1.0, lr)
        return "goal_kick", 0.30 + 0.2 * min(1.0, lr)

    if lr < 0.02:
        return "none", 0.2
    return "unknown", 0.25


class TemporalSmoother:
    """Per-official majority vote over a sliding window of raw (signal, conf)."""

    def __init__(self, window: int = 9):
        self.window = max(1, int(window))
        self._bufs: Dict[str, Deque[Tuple[str, float]]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, official_id: str, signal: str, confidence: float) -> Tuple[str, float]:
        buf = self._bufs[official_id]
        buf.append((signal, confidence))
        counts = Counter(s for s, _ in buf)
        # Prefer non-none if present with enough support
        best, n = counts.most_common(1)[0]
        if best == "none" and len(counts) > 1:
            alt = [(s, c) for s, c in counts.items() if s != "none"]
            alt.sort(key=lambda t: -t[1])
            if alt and alt[0][1] >= max(2, self.window // 3):
                best, n = alt[0]
        confs = [c for s, c in buf if s == best]
        mean_c = sum(confs) / max(1, len(confs))
        # Down-weight if unstable
        stability = n / len(buf)
        return best, float(mean_c * (0.5 + 0.5 * stability))

    def flush_ids(self) -> List[str]:
        return list(self._bufs.keys())

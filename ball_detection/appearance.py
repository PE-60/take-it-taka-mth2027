"""HSV histogram appearance models for the ball and team kits.

Stage A output: JSON banks that later stages load for kick-local detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in environments without OpenCV
    cv2 = None


# OpenCV HSV: H 0-180, S 0-255, V 0-255
DEFAULT_HSV_BINS = (18, 8, 8)
DEFAULT_HSV_RANGES = [ (0, 180), (0, 256), (0, 256) ]


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for appearance histograms")


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    _require_cv2()
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected BGR image HxWx3")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def hsv_histogram(
        bgr: np.ndarray,
        mask: np.ndarray | None = None,
        bins: tuple[int, int, int] = DEFAULT_HSV_BINS,
        ranges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Normalised 3D HSV histogram, shape (h_bins, s_bins, v_bins), sums to 1 (or 0 if empty)."""
    _require_cv2()
    ranges = ranges or DEFAULT_HSV_RANGES
    hsv = bgr_to_hsv(bgr)
    hist_mask = None
    if mask is not None:
        hist_mask = np.asarray(mask, dtype=np.uint8)
        if hist_mask.ndim != 2:
            raise ValueError("mask must be HxW")
    hist = cv2.calcHist([hsv], [0, 1, 2], hist_mask, list(bins),
                        [ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1],
                         ranges[2][0], ranges[2][1]])
    hist = np.asarray(hist, dtype=np.float64)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya-friendly intersection score in [0, 1]."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("histogram shapes differ")
    return float(np.minimum(a, b).sum())


def circular_mask(h: int, w: int, cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(radius) ** 2


@dataclass
class HistogramBank:
    """Running mean of normalised HSV histograms plus optional scalar stats."""
    name: str
    bins: tuple[int, int, int] = DEFAULT_HSV_BINS
    hist_sum: np.ndarray | None = None
    n_samples: int = 0
    radius_px_sum: float = 0.0
    n_radius: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, hist: np.ndarray, radius_px: float | None = None) -> None:
        hist = np.asarray(hist, dtype=np.float64)
        if self.hist_sum is None:
            self.hist_sum = np.zeros_like(hist, dtype=np.float64)
        if hist.shape != self.hist_sum.shape:
            raise ValueError(f"{self.name}: hist shape {hist.shape} != {self.hist_sum.shape}")
        self.hist_sum += hist
        self.n_samples += 1
        if radius_px is not None and np.isfinite(radius_px):
            self.radius_px_sum += float(radius_px)
            self.n_radius += 1

    def mean_hist(self) -> np.ndarray | None:
        if self.hist_sum is None or self.n_samples == 0:
            return None
        out = self.hist_sum / float(self.n_samples)
        total = float(out.sum())
        if total > 0:
            out = out / total
        return out

    def mean_radius_px(self) -> float | None:
        if self.n_radius == 0:
            return None
        return self.radius_px_sum / float(self.n_radius)

    def to_dict(self) -> dict[str, Any]:
        hist = self.mean_hist()
        return {
            "name": self.name,
            "bins": list(self.bins),
            "n_samples": int(self.n_samples),
            "mean_hist": None if hist is None else hist.tolist(),
            "mean_radius_px": self.mean_radius_px(),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistogramBank":
        bins = tuple(int(x) for x in data.get("bins", DEFAULT_HSV_BINS))
        bank = cls(name=str(data["name"]), bins=bins, meta=dict(data.get("meta") or {}))
        hist = data.get("mean_hist")
        n = int(data.get("n_samples") or 0)
        if hist is not None and n > 0:
            arr = np.asarray(hist, dtype=np.float64)
            bank.hist_sum = arr * n
            bank.n_samples = n
        radius = data.get("mean_radius_px")
        if radius is not None and n > 0:
            bank.radius_px_sum = float(radius) * n
            bank.n_radius = n
        return bank


@dataclass
class AppearanceBanks:
    """Ball + per-team kit banks for one match/source."""
    ball: HistogramBank = field(default_factory=lambda: HistogramBank(name="ball"))
    kits: dict[str, HistogramBank] = field(default_factory=dict)
    version: int = 1
    source: str = "tactical"
    game: str = "BAR-ATH"
    gk_excluded: list[str] = field(default_factory=list)

    def kit(self, team: str) -> HistogramBank:
        team = str(team)
        if team not in self.kits:
            self.kits[team] = HistogramBank(name=f"kit_{team}")
        return self.kits[team]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "game": self.game,
            "source": self.source,
            "gk_excluded": list(self.gk_excluded),
            "ball": self.ball.to_dict(),
            "kits": {k: v.to_dict() for k, v in sorted(self.kits.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppearanceBanks":
        banks = cls(
            version=int(data.get("version") or 1),
            game=str(data.get("game") or "BAR-ATH"),
            source=str(data.get("source") or "tactical"),
            gk_excluded=[str(x) for x in (data.get("gk_excluded") or [])],
        )
        if data.get("ball"):
            banks.ball = HistogramBank.from_dict(data["ball"])
        for team, payload in (data.get("kits") or {}).items():
            banks.kits[str(team)] = HistogramBank.from_dict(payload)
        return banks


def save_banks(banks: AppearanceBanks, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(banks.to_dict(), fh, indent=2)
        fh.write("\n")


def load_banks(path: str) -> AppearanceBanks:
    with open(path, encoding="utf-8") as fh:
        return AppearanceBanks.from_dict(json.load(fh))


def grass_mask_hsv(hsv: np.ndarray,
                   h_low: int = 35, h_high: int = 95,
                   s_low: int = 40, v_low: int = 40) -> np.ndarray:
    """True where pixels look like grass (to exclude from ball/kit histograms)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= h_low) & (h <= h_high) & (s >= s_low) & (v >= v_low)


def ball_color_mask_hsv(
        hsv: np.ndarray,
        *,
        v_low: int = 175,
        s_high: int = 55,
        v_dark: int = 90,
) -> np.ndarray:
    """True for white / light-grey ball pixels; rejects dark shoe leather and saturated kit.

    OpenCV HSV: H 0-180, S/V 0-255. Specular shoe dots may still pass; combine with shape
    and foot-exclusion at detection time.
    """
    s, v = hsv[..., 1], hsv[..., 2]
    return (v >= v_low) & (s <= s_high) & (v >= v_dark)


def ball_color_mask_bgr(bgr: np.ndarray, **kwargs) -> np.ndarray:
    return ball_color_mask_hsv(bgr_to_hsv(bgr), **kwargs)


def bright_fraction(bgr: np.ndarray, mask: np.ndarray | None = None, **kwargs) -> float:
    """Fraction of (masked) pixels that look ball-coloured."""
    ballish = ball_color_mask_bgr(bgr, **kwargs)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        n = int(m.sum())
        if n <= 0:
            return 0.0
        return float((ballish & m).sum()) / float(n)
    n = ballish.size
    return float(ballish.sum()) / float(n) if n else 0.0

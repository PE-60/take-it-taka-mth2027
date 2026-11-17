"""Kick peaks and short temporal windows for Stage B.

Two detectors are available:

* ``vector_peak`` (default) — local maxima of ‖Δv_xy‖·fps at/above a threshold
  (same rule as ``lh_tbp`` / match_viewer when that module is importable).
* ``extrema`` — local minima **and** maxima of the signed pitch-axis acceleration
  components, filtered by prominence (vertical separation) and optional width
  (broadness). Captures both deflections and accelerations along a fixed direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KICK_METHODS = ("vector_peak", "extrema")


@dataclass(frozen=True)
class KickWindow:
    """Half-open frame window [start, end) around a kick, indices on the same axis as ``ball``."""
    kick_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _ball_m(ball_xy: np.ndarray, pitch_m: tuple[float, float]) -> np.ndarray:
    return np.asarray(ball_xy[..., :2], dtype=np.float64) * np.asarray(pitch_m, dtype=np.float64)


def _velocity_m_s(ball_m: np.ndarray, fps: float) -> np.ndarray:
    n = len(ball_m)
    vel = np.full((n, 2), np.nan, dtype=np.float64)
    if n < 2:
        return vel
    both = np.isfinite(ball_m[:-1]).all(1) & np.isfinite(ball_m[1:]).all(1)
    delta = np.diff(ball_m, axis=0) * float(fps)
    vel[1:] = np.where(both[:, None], delta, np.nan)
    return vel


def _accel_vec_m_s2(vel_m_s: np.ndarray, fps: float) -> np.ndarray:
    """Signed xy acceleration (n, 2) in m/s²: (v[t] − v[t−1]) · fps."""
    n = len(vel_m_s)
    acc = np.full((n, 2), np.nan, dtype=np.float64)
    if n < 2:
        return acc
    both = np.isfinite(vel_m_s[:-1]).all(1) & np.isfinite(vel_m_s[1:]).all(1)
    delta = np.diff(vel_m_s, axis=0) * float(fps)
    acc[1:] = np.where(both[:, None], delta, np.nan)
    return acc


def _accel_mag(acc_xy: np.ndarray) -> np.ndarray:
    mag = np.full(len(acc_xy), np.nan, dtype=np.float64)
    ok = np.isfinite(acc_xy).all(axis=1)
    mag[ok] = np.linalg.norm(acc_xy[ok], axis=1)
    return mag


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware centered moving average; ``win`` odd preferred."""
    x = np.asarray(x, dtype=np.float64)
    if win <= 1 or len(x) == 0:
        return x.copy()
    w = int(win)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    kernel = np.ones(w, dtype=np.float64)
    vals = np.where(np.isfinite(x), x, 0.0)
    counts = np.where(np.isfinite(x), 1.0, 0.0)
    num = np.convolve(vals, kernel, mode="full")[pad:pad + len(x)]
    den = np.convolve(counts, kernel, mode="full")[pad:pad + len(x)]
    out = np.full_like(x, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _local_extrema(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of strict-ish local maxima and minima (plateau peaks keep the middle)."""
    n = len(y)
    maxima = np.zeros(n, bool)
    minima = np.zeros(n, bool)
    a = np.where(np.isfinite(y), y, np.nan)
    i = 1
    while i < n - 1:
        if not np.isfinite(a[i]):
            i += 1
            continue
        # Expand plateau of equal values.
        j = i
        while j + 1 < n - 1 and np.isfinite(a[j + 1]) and a[j + 1] == a[i]:
            j += 1
        left = a[i - 1]
        right = a[j + 1] if j + 1 < n else np.nan
        if not (np.isfinite(left) and np.isfinite(right)):
            i = j + 1
            continue
        mid = (i + j) // 2
        if a[i] >= left and a[j] >= right and (a[i] > left or a[j] > right):
            maxima[mid] = True
        if a[i] <= left and a[j] <= right and (a[i] < left or a[j] < right):
            minima[mid] = True
        i = j + 1
    return maxima, minima


def _prominence_and_width(y: np.ndarray, peak: int, *, as_max: bool) -> tuple[float, float]:
    """Contour prominence and half-prominence width (frames) for one extremum.

    For a minimum, the series is negated so the same max-logic applies.
    """
    n = len(y)
    if peak <= 0 or peak >= n - 1 or not np.isfinite(y[peak]):
        return 0.0, 0.0
    s = y if as_max else -y
    h = float(s[peak])

    left_min = h
    left_bound = 0
    for i in range(peak - 1, -1, -1):
        if not np.isfinite(s[i]):
            break
        if s[i] > h:
            left_bound = i + 1
            break
        left_min = min(left_min, float(s[i]))
        left_bound = i
    else:
        left_bound = 0

    right_min = h
    right_bound = n - 1
    for i in range(peak + 1, n):
        if not np.isfinite(s[i]):
            break
        if s[i] > h:
            right_bound = i - 1
            break
        right_min = min(right_min, float(s[i]))
        right_bound = i
    else:
        right_bound = n - 1

    base = max(left_min, right_min)
    prom = h - base
    if prom <= 0:
        return 0.0, 0.0

    # Width at half-prominence (linear interp to neighbors).
    level = h - 0.5 * prom
    left_i = peak
    for i in range(peak, left_bound - 1, -1):
        if not np.isfinite(s[i]):
            break
        if s[i] <= level:
            left_i = i
            break
        left_i = i
    right_i = peak
    for i in range(peak, right_bound + 1):
        if not np.isfinite(s[i]):
            break
        if s[i] <= level:
            right_i = i
            break
        right_i = i
    width = float(max(0, right_i - left_i))
    return float(prom), width


def _apply_refractory(kick: np.ndarray, refractory: int) -> np.ndarray:
    if refractory <= 0:
        return kick
    out = kick.copy()
    last = -10**9
    for i in np.flatnonzero(kick):
        if i - last < refractory:
            out[i] = False
        else:
            last = int(i)
    return out


def detect_kicks_vector_peak(
        ball_xy: np.ndarray,
        pitch_m: tuple[float, float],
        fps: float = 25.0,
        threshold: float = 4.0,
        refractory_s: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of ‖Δv‖·fps at/above ``threshold`` (legacy / lh_tbp rule)."""
    try:
        from possesion_algorithms.algorithms.lh_tbp import detect_ball_kicks
        return detect_ball_kicks(ball_xy, pitch_m, fps, threshold=threshold,
                                 refractory_s=refractory_s)
    except Exception:
        return _detect_kicks_vector_peak_local(ball_xy, pitch_m, fps, threshold, refractory_s)


def _detect_kicks_vector_peak_local(ball_xy, pitch_m, fps, threshold, refractory_s):
    ball = _ball_m(ball_xy, pitch_m)
    vel = _velocity_m_s(ball, fps)
    acc_xy = _accel_vec_m_s2(vel, fps)
    accel = _accel_mag(acc_xy)
    kick = np.zeros(len(accel), bool)
    a = np.where(np.isfinite(accel), accel, -np.inf)
    for i in range(1, len(a) - 1):
        if a[i] >= threshold and a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            kick[i] = True
    refractory = int(round(float(refractory_s) * float(fps)))
    kick = _apply_refractory(kick, refractory)
    return kick, accel.astype(np.float32)


def detect_kicks_extrema(
        ball_xy: np.ndarray,
        pitch_m: tuple[float, float],
        fps: float = 25.0,
        *,
        prominence: float = 3.0,
        min_width_s: float = 0.0,
        max_width_s: float | None = 0.40,
        smooth_s: float = 0.08,
        refractory_s: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Kicks from local min/max of signed pitch-axis acceleration.

    Uses ``a_x`` and ``a_y`` (m/s²). A frame is a kick if either component has a
    local maximum or minimum whose **prominence** (vertical separation from the
    surrounding contour) is ≥ ``prominence``, and whose half-prominence **width**
    lies in ``[min_width_s, max_width_s]`` (seconds; ``max_width_s=None`` disables
    the upper bound). Optional ``smooth_s`` applies a short moving average first
    so single-frame jitter is less likely to form extrema.

    Returns ``(kick_mask, accel_mag)`` where ``accel_mag`` is ‖a_xy‖ for diagnostics.
    """
    ball = _ball_m(ball_xy, pitch_m)
    vel = _velocity_m_s(ball, fps)
    acc_xy = _accel_vec_m_s2(vel, fps)
    accel_mag = _accel_mag(acc_xy)

    smooth_w = max(1, int(round(float(smooth_s) * float(fps))))
    min_w = float(min_width_s) * float(fps)
    max_w = None if max_width_s is None else float(max_width_s) * float(fps)
    prom_min = float(prominence)

    kick = np.zeros(len(ball), bool)
    for axis in (0, 1):
        series = acc_xy[:, axis]
        if smooth_w > 1:
            series = _moving_average(series, smooth_w)
        maxima, minima = _local_extrema(series)
        for idxs, as_max in ((maxima, True), (minima, False)):
            for i in np.flatnonzero(idxs):
                prom, width = _prominence_and_width(series, int(i), as_max=as_max)
                if prom < prom_min:
                    continue
                if width < min_w:
                    continue
                if max_w is not None and width > max_w:
                    continue
                kick[int(i)] = True

    refractory = int(round(float(refractory_s) * float(fps)))
    kick = _apply_refractory(kick, refractory)
    return kick, accel_mag.astype(np.float32)


def detect_kicks(
        ball_xy: np.ndarray,
        pitch_m: tuple[float, float],
        fps: float = 25.0,
        threshold: float = 4.0,
        refractory_s: float = 0.3,
        *,
        method: str = "vector_peak",
        prominence: float = 3.0,
        min_width_s: float = 0.0,
        max_width_s: float | None = 0.40,
        smooth_s: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(kick_mask, accel_m_s2)`` each shaped (n,).

    ``method``:
      * ``vector_peak`` — thresholded local maxima of ‖Δv‖·fps (default).
      * ``extrema`` — prominence/width-filtered local min/max of signed a_x, a_y.
    """
    method = str(method or "vector_peak").strip().lower()
    if method not in KICK_METHODS:
        raise ValueError(f"unknown kick method {method!r}; choose from {KICK_METHODS}")
    if method == "extrema":
        return detect_kicks_extrema(
            ball_xy, pitch_m, fps,
            prominence=prominence,
            min_width_s=min_width_s,
            max_width_s=max_width_s,
            smooth_s=smooth_s,
            refractory_s=refractory_s,
        )
    return detect_kicks_vector_peak(
        ball_xy, pitch_m, fps, threshold=threshold, refractory_s=refractory_s)


def kick_windows(
        kick_mask: np.ndarray,
        *,
        pre: int = 8,
        post: int = 12,
        n_frames: int | None = None,
) -> list[KickWindow]:
    """Build a temporal window around each True kick index."""
    kick_mask = np.asarray(kick_mask, dtype=bool)
    n = int(n_frames if n_frames is not None else len(kick_mask))
    out: list[KickWindow] = []
    for k in np.flatnonzero(kick_mask):
        ki = int(k)
        if ki < 0 or ki >= n:
            continue
        start = max(0, ki - int(pre))
        end = min(n, ki + int(post) + 1)
        out.append(KickWindow(kick_index=ki, start=start, end=end))
    return out


def ball_prior_at(
        ball: np.ndarray,
        index: int,
        *,
        lookback: int = 25,
) -> np.ndarray | None:
    """Last finite ball position at or before ``index``, within ``lookback`` frames."""
    ball = np.asarray(ball, dtype=np.float64)
    lo = max(0, int(index) - int(lookback))
    for i in range(int(index), lo - 1, -1):
        if np.isfinite(ball[i, 0]) and np.isfinite(ball[i, 1]):
            return ball[i, :2].copy()
    return None


def ball_velocity_prior(
        ball: np.ndarray,
        index: int,
        pitch_m: tuple[float, float],
        fps: float,
        *,
        lookback: int = 5,
) -> np.ndarray | None:
    """Rough pitch-fraction velocity (frac/frame) from recent finite samples."""
    ball = np.asarray(ball, dtype=np.float64)
    pts = []
    lo = max(0, int(index) - int(lookback))
    for i in range(lo, int(index) + 1):
        if np.isfinite(ball[i, 0]) and np.isfinite(ball[i, 1]):
            pts.append((i, ball[i, 0], ball[i, 1]))
    if len(pts) < 2:
        return None
    (i0, x0, y0), (i1, x1, y1) = pts[-2], pts[-1]
    dt = i1 - i0
    if dt <= 0:
        return None
    return np.array([(x1 - x0) / dt, (y1 - y0) / dt], dtype=np.float64)

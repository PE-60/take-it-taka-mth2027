"""Memory-aware parallelism helpers for kick-window detection."""

from __future__ import annotations


def estimate_window_mb(
        width: int,
        height: int,
        frames_per_window: int,
        *,
        overhead: float = 1.5,
) -> float:
    """Rough RAM (MiB) to hold one kick window of BGR frames in memory."""
    bytes_raw = max(1, int(width)) * max(1, int(height)) * 3 * max(1, int(frames_per_window))
    return (bytes_raw * float(overhead)) / (1024.0 * 1024.0)


def choose_workers(
        max_memory_mb: float,
        window_mb: float,
        *,
        workers: int | None = None,
        hard_cap: int = 32,
) -> int:
    """How many kick windows may run concurrently under a memory budget.

    ``workers`` is an optional explicit request; the memory budget always caps it.
    """
    by_mem = max(1, int(float(max_memory_mb) // max(float(window_mb), 1e-6)))
    by_mem = min(by_mem, max(1, int(hard_cap)))
    if workers is None or int(workers) <= 0:
        return by_mem
    return max(1, min(int(workers), by_mem))


def subsample_windows(windows: list, stride: int, max_kicks: int | None) -> list:
    """Keep every ``stride``-th window, then optionally cap the count."""
    stride = max(1, int(stride))
    out = list(windows)[::stride]
    if max_kicks is not None:
        out = out[: max(0, int(max_kicks))]
    return out

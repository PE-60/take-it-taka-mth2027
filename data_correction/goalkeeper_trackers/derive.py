"""Derive team-specific goalkeeper tracks from the two swapped ATD keeper slots.

A keeper is assigned to the team defending the physical pitch half they stand on. When both
source trackers are camera-visible on the same half, raw ATD speed picks the source only when
exactly one tracker has any finite speed in that contiguous overlap; that source is reused on
speed-missing frames inside the overlap. Overlaps with zero or two speed-bearing sources stay
unresolved. Source ATD slots are never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDES = ("home", "away")
DEFAULT_TRACKERS = ("24", "46")

STATUS_NO_VISIBLE = "no_visible_tracker"
STATUS_BY_HALF = "assigned_by_pitch_half"
STATUS_SPEED = "speed_disambiguated"
STATUS_SPEED_INFERRED = "speed_inferred_within_overlap"
STATUS_BOTH_ACTIVE = "unresolved_same_half_both_active"
STATUS_NO_SPEED = "unresolved_same_half_no_speed"
STATUS_NO_LABEL = "no_label_in_defending_half"


@dataclass
class DerivedGoalkeepers:
    """Per-team goalkeeper tracks on video frames 0 .. n - 1."""
    position: dict          # side -> (n, 2) float32
    speed: dict             # side -> (n,) float32
    source_track_id: dict   # side -> (n,) unicode
    candidate_track_ids: dict
    status: dict
    pitch_half: dict
    team_names: dict        # side -> str
    tracker_ids: tuple
    fps: float

    @property
    def n(self) -> int:
        return len(next(iter(self.position.values())))


def defending_half(attack_direction_first_half: str, second_half: np.ndarray) -> np.ndarray:
    """Physical pitch half (0=left, 1=right) the team defends on each frame."""
    if attack_direction_first_half not in ("left_to_right", "right_to_left"):
        raise ValueError(f"unknown first-half attack direction: {attack_direction_first_half!r}")
    first = 0 if attack_direction_first_half == "left_to_right" else 1
    return np.where(second_half, 1 - first, first).astype(np.int8)


def derive_goalkeeper_tracks(
        positions: np.ndarray,
        speeds: np.ndarray,
        visible: np.ndarray,
        home_defending_half: np.ndarray,
        away_defending_half: np.ndarray,
        tracker_ids: tuple[str, str] = DEFAULT_TRACKERS,
        team_names: dict | None = None,
        fps: float = 25.0,
) -> DerivedGoalkeepers:
    """Assign two source keeper slots to home/away defending halves.

    positions  (n, 2, 2)  pitch fractions for the two trackers
    speeds     (n, 2)     raw ATD speeds (NaN where missing)
    visible    (n, 2)     True where the tracker is camera-visible
    home_defending_half / away_defending_half  (n,)  0=left, 1=right
    """
    positions = np.asarray(positions, dtype=np.float32)
    speeds = np.asarray(speeds, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    home_defending_half = np.asarray(home_defending_half, dtype=np.int8)
    away_defending_half = np.asarray(away_defending_half, dtype=np.int8)
    n = len(positions)
    if positions.shape != (n, 2, 2) or speeds.shape != (n, 2) or visible.shape != (n, 2):
        raise ValueError("positions, speeds and visible must cover two trackers over n frames")
    if home_defending_half.shape != (n,) or away_defending_half.shape != (n,):
        raise ValueError("defending-half arrays must have length n")
    if len(tracker_ids) != 2:
        raise ValueError("exactly two source tracker ids are required")
    if not np.all(home_defending_half + away_defending_half == 1):
        raise ValueError("home and away must defend opposite pitch halves on every frame")

    team_names = dict(team_names or {"home": "home", "away": "away"})
    pitch_half_code = np.where(positions[:, :, 0] <= 0.5, 0, 1).astype(np.int8)
    target = {"home": home_defending_half, "away": away_defending_half}

    out = {
        side: {
            "position": np.full((n, 2), np.nan, dtype=np.float32),
            "speed": np.full(n, np.nan, dtype=np.float32),
            "source_track_id": np.full(n, "", dtype="<U8"),
            "candidate_track_ids": np.full(n, "", dtype="<U16"),
            "status": np.full(n, STATUS_NO_VISIBLE, dtype="<U48"),
            "pitch_half": np.full(n, "", dtype="<U4"),
        }
        for side in SIDES
    }

    def team_for_half(frame: int, half_code: int) -> str:
        matches = [side for side in SIDES if int(target[side][frame]) == int(half_code)]
        if len(matches) != 1:
            raise ValueError(f"could not uniquely map pitch half at frame {frame}")
        return matches[0]

    def assign(frame: int, side: str, source_index: int, status: str) -> None:
        track = out[side]
        track["position"][frame] = positions[frame, source_index]
        track["speed"][frame] = speeds[frame, source_index]
        track["source_track_id"][frame] = tracker_ids[source_index]
        track["candidate_track_ids"][frame] = tracker_ids[source_index]
        track["status"][frame] = status
        track["pitch_half"][frame] = "left" if pitch_half_code[frame, source_index] == 0 else "right"

    same_half_overlap = (
        visible[:, 0] & visible[:, 1]
        & (pitch_half_code[:, 0] == pitch_half_code[:, 1])
    )
    for frame in np.flatnonzero(visible.any(axis=1) & ~same_half_overlap):
        for source_index in np.flatnonzero(visible[frame]):
            half_code = int(pitch_half_code[frame, source_index])
            assign(int(frame), team_for_half(int(frame), half_code), int(source_index), STATUS_BY_HALF)

    edges = np.flatnonzero(np.diff(np.r_[False, same_half_overlap, False].astype(np.int8)))
    for interval_start, interval_end in zip(edges[::2], edges[1::2]):
        speed_present = np.isfinite(speeds[interval_start:interval_end]).any(axis=0)
        for frame in range(int(interval_start), int(interval_end)):
            half_code = int(pitch_half_code[frame, 0])
            side = team_for_half(frame, half_code)
            other = "away" if side == "home" else "home"
            out[other]["status"][frame] = STATUS_NO_LABEL
            active = np.flatnonzero(np.isfinite(speeds[frame]))
            if len(active) == 1:
                assign(frame, side, int(active[0]), STATUS_SPEED)
            elif len(active) == 2:
                track = out[side]
                track["status"][frame] = STATUS_BOTH_ACTIVE
                track["candidate_track_ids"][frame] = ";".join(tracker_ids)
            elif int(speed_present.sum()) == 1:
                source_index = int(np.flatnonzero(speed_present)[0])
                assign(frame, side, source_index, STATUS_SPEED_INFERRED)
            else:
                track = out[side]
                track["status"][frame] = STATUS_NO_SPEED
                track["candidate_track_ids"][frame] = ";".join(tracker_ids)

    return DerivedGoalkeepers(
        position={side: out[side]["position"] for side in SIDES},
        speed={side: out[side]["speed"] for side in SIDES},
        source_track_id={side: out[side]["source_track_id"] for side in SIDES},
        candidate_track_ids={side: out[side]["candidate_track_ids"] for side in SIDES},
        status={side: out[side]["status"] for side in SIDES},
        pitch_half={side: out[side]["pitch_half"] for side in SIDES},
        team_names=team_names,
        tracker_ids=tuple(tracker_ids),
        fps=float(fps),
    )


CSV_COLUMNS = (
    "frame", "time_seconds", "team_side", "team_name", "x", "y", "speed_m_s",
    "source_track_id", "candidate_track_ids", "pitch_half", "status",
)


def _fmt_float(values: np.ndarray) -> np.ndarray:
    """Stringify floats; empty string for non-finite values."""
    out = np.full(values.shape, "", dtype=object)
    ok = np.isfinite(values)
    out[ok] = np.char.mod("%.8g", values[ok].astype(np.float64))
    return out


def iter_csv_rows(derived: DerivedGoalkeepers):
    """Yield one dict per (frame, team_side), matching the notebook CSV columns."""
    for side in SIDES:
        name = derived.team_names[side]
        pos = derived.position[side]
        speed = derived.speed[side]
        source = derived.source_track_id[side]
        candidates = derived.candidate_track_ids[side]
        half = derived.pitch_half[side]
        status = derived.status[side]
        for frame in range(derived.n):
            yield {
                "frame": frame,
                "time_seconds": frame / derived.fps,
                "team_side": side,
                "team_name": name,
                "x": pos[frame, 0],
                "y": pos[frame, 1],
                "speed_m_s": speed[frame],
                "source_track_id": source[frame],
                "candidate_track_ids": candidates[frame],
                "pitch_half": half[frame],
                "status": status[frame],
            }


def write_csv(derived: DerivedGoalkeepers, path: str) -> int:
    """Write the long-format correction CSV. Returns the number of data rows."""
    import csv

    frames = np.arange(derived.n, dtype=np.int64)
    time_s = _fmt_float(frames / derived.fps)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for side in SIDES:
            name = derived.team_names[side]
            x = _fmt_float(derived.position[side][:, 0])
            y = _fmt_float(derived.position[side][:, 1])
            speed = _fmt_float(derived.speed[side])
            source = derived.source_track_id[side]
            candidates = derived.candidate_track_ids[side]
            half = derived.pitch_half[side]
            status = derived.status[side]
            for i, frame in enumerate(frames):
                writer.writerow([
                    frame, time_s[i], side, name, x[i], y[i], speed[i],
                    source[i], candidates[i], half[i], status[i],
                ])
    return derived.n * len(SIDES)


def status_counts(derived: DerivedGoalkeepers) -> dict:
    return {
        side: {str(k): int(v) for k, v in zip(*np.unique(derived.status[side], return_counts=True))}
        for side in SIDES
    }

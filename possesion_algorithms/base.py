"""What a possession algorithm is: a name, two choices, and a function from tracking to labels.

    class MyAlgorithm(PossessionAlgorithm):
        name = "mine"                       # -> evaluation/results/mine.csv
        description = "one line for the viewer's algorithm picker"
        dead_ball = True                    # label only the alive intervals of dead_ball.json, D on the rest
        source = TACTICAL_THEN_BROADCAST    # TACTICAL, BROADCAST or TACTICAL_THEN_BROADCAST
        params = {"radius_m": 1.0}          # defaults; run.py --param radius_m=1.5 overrides them

        def label(self, track):             # Tracking -> one label per frame of track
            ...

`label` sees one stretch of frames at a time and starts from scratch on each: the whole covered
video, or with dead_ball = True each alive interval on its own. Frames the source has no data for
are never passed in and come out X.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TACTICAL = "tactical"
BROADCAST = "broadcast"
TACTICAL_THEN_BROADCAST = "tactical+broadcast"
SOURCES = (TACTICAL, BROADCAST, TACTICAL_THEN_BROADCAST)
ORIGIN = {TACTICAL: 0, BROADCAST: 1}       # Tracking.origin codes; -1 = no data
LABELS = ("H", "A", "D", "X")


@dataclass
class Tracking:
    """Ball and players on tactical video frames start .. start + n - 1 (the submission timeline),
    from whichever recording `origin` says.

    ball    (n, 2)     pitch fractions, NaN where the ball is not tracked
    xy      (n, S, 2)  pitch fractions per track slot, NaN where the slot is empty
    team    (S,)       'H', 'A' or '?' per slot
    kick    (n,)       ball acceleration peak >= 4 m/s² (match_viewer's KICK row)
    origin  (n,)       ORIGIN code of the recording each frame comes from, -1 where there is none

    A slot is a tracklet of one recording, not a person. With both recordings, the broadcast's
    slots come after the tactical ones, so a slot index never jumps between recordings.
    """
    start: int
    ball: np.ndarray
    xy: np.ndarray
    team: np.ndarray
    kick: np.ndarray
    origin: np.ndarray
    pitch: tuple
    fps: float = 25.0

    @property
    def n(self) -> int:
        return len(self.ball)

    @property
    def n_slots(self) -> int:
        return self.xy.shape[1]

    def slice(self, a: int, b: int) -> "Tracking":
        """Rows a .. b - 1 (relative to start)."""
        return Tracking(self.start + a, self.ball[a:b], self.xy[a:b], self.team, self.kick[a:b],
                        self.origin[a:b], self.pitch, self.fps)

    def ball_distances(self) -> np.ndarray:
        """(n, S) ball-to-slot distance in metres, inf where the ball or the slot is missing."""
        m = np.asarray(self.pitch, dtype=np.float32)
        d = np.hypot(*np.moveaxis((self.xy - self.ball[:, None, :]) * m, -1, 0))
        return np.where(np.isfinite(d), d, np.inf).astype(np.float32)


class PossessionAlgorithm:
    """Subclass it, set the class attributes, implement label(). One subclass is one CSV; a subclass
    of a subclass that only changes `name`, `dead_ball`, `source` or `params` is a variant."""

    name: str | None = None
    description: str = ""
    dead_ball: bool = False
    source: str = TACTICAL
    params: dict = {}

    def __init__(self, **params):
        unknown = sorted(set(params) - set(type(self).params))
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {unknown}; it has {sorted(type(self).params)}")
        if self.source not in SOURCES:
            raise ValueError(f"{self.name}: source must be one of {SOURCES}, got {self.source!r}")
        self.params = {**type(self).params, **params}

    def label(self, track: Tracking) -> np.ndarray:
        """One label per frame of `track`: 'H', 'A', 'D' or 'X' ('' counts as X)."""
        raise NotImplementedError

    def settings(self) -> dict:
        return {"algorithm": self.name, "class": type(self).__name__, "description": self.description,
                "dead_ball": bool(self.dead_ball), "source": self.source, "params": dict(self.params)}

"""What an algorithm reads: tracking per recording, the sync between them, and the dead-ball intervals.

    SourceTrack   one recording on its own video frames, from the raw files or a viewer export
    MatchInputs   everything for one run, loaded on first use and cached:
                    .tracking(source)   Tracking on the tactical timeline for TACTICAL, BROADCAST
                                        or TACTICAL_THEN_BROADCAST
                    .dead_ball()        bool mask over the tactical frames (dead_ball.json)

Raw files go through match_viewer.load_atd, exactly as the viewer reads them (slow: the whole ATD
is parsed). Exports are match_viewer/export/<source>_<mode>/ (meta.json + data.bin) and load in
seconds, but only cover the frames that were exported.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from possesion_algorithms.base import BROADCAST, ORIGIN, TACTICAL, TACTICAL_THEN_BROADCAST, Tracking
from possesion_algorithms.paths import DEAD_BALL_JSON, DEFAULT_SYNC, EXPORT_DIR, VIEWER_DIR


@dataclass
class SourceTrack:
    """One recording's tracking on its own video frames start .. start + n - 1 (see Tracking)."""
    source: str
    start: int
    ball: np.ndarray
    xy: np.ndarray
    team: np.ndarray
    kick: np.ndarray
    pitch: tuple
    fps: float = 25.0
    origin_path: str = ""
    slot_ids: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.ball)

    @property
    def end(self) -> int:
        return self.start + self.n - 1


def load_raw(source: str) -> SourceTrack:
    """The whole recording from the raw ATD, through match_viewer.load_atd."""
    from dead_ball_investigation.signals import find_inputs, viewer
    from games import source_frames
    mv = viewer()
    atd_raw, atd_meta, _hom = find_inputs(source)
    n = source_frames(source)
    print(f"--- {source}: parsing {os.path.relpath(atd_raw)} (frames 0..{n - 1})")
    atd, players_meta, home = mv.load_atd(atd_raw, atd_meta, 0, n - 1)
    team = np.array([mv.team_colour(pid, players_meta, home)[1] for pid in atd["ids"]], dtype="<U1")
    slot_ids = np.asarray([str(pid) for pid in atd["ids"]], dtype=object)
    return SourceTrack(source, 0, atd["ball"][:, :2], atd["xy"][:, :, :2], team, np.asarray(atd["kick"], bool),
                       tuple(mv.PITCH_M), float(mv.FPS), atd_raw, slot_ids)


def resolve_export(path: str) -> str:
    """An export folder given as a path, a path relative to match_viewer/, or a name in match_viewer/export/."""
    for cand in (path, os.path.join(VIEWER_DIR, path), os.path.join(EXPORT_DIR, path)):
        if os.path.isfile(os.path.join(cand, "meta.json")):
            return os.path.abspath(cand)
    raise IOError(f"no export at {path} (looked for meta.json there, under {VIEWER_DIR} and {EXPORT_DIR})\n"
                  "  make one:  python match_viewer/match_viewer.py --export --source tactical")


def find_export(source: str) -> str:
    """The widest export of `source` in match_viewer/export/."""
    best, best_n = None, -1
    if os.path.isdir(EXPORT_DIR):
        for name in sorted(os.listdir(EXPORT_DIR)):
            p = os.path.join(EXPORT_DIR, name, "meta.json")
            if not os.path.isfile(p):
                continue
            with open(p) as fh:
                meta = json.load(fh)
            if meta.get("source", TACTICAL) == source and int(meta.get("n", 0)) > best_n:
                best, best_n = os.path.dirname(p), int(meta["n"])
    if best is None:
        raise IOError(f"no {source} export in {EXPORT_DIR}\n"
                      f"  make one:  python match_viewer/match_viewer.py --export --source {source} --range all")
    return best


def load_export(path: str) -> SourceTrack:
    """A match_viewer export: data.bin holds `block` float32 per frame, 9 homography values, 6 ball
    values (x, y, ...), then 4 per slot (x, y, ...); kicks are meta.events.kick (export rows)."""
    path = resolve_export(path)
    with open(os.path.join(path, "meta.json")) as fh:
        meta = json.load(fh)
    n, block, S = int(meta["n"]), int(meta["block"]), int(meta["n_slots"])
    data = np.fromfile(os.path.join(path, "data.bin"), dtype=np.float32)
    if data.size != n * block or block != 15 + 4 * S:
        raise ValueError(f"{path}: data.bin has {data.size} floats, expected {n} x {block} with {S} slots")
    data = data.reshape(n, block)
    xy = np.stack([data[:, 15::4], data[:, 16::4]], axis=-1)
    team = np.array([s.get("team") or "?" for s in meta["slots"]], dtype="<U1")
    kick = np.zeros(n, bool)
    rows = np.asarray((meta.get("events") or {}).get("kick") or [], dtype=np.int64)
    kick[rows[(rows >= 0) & (rows < n)]] = True
    slot_ids = np.asarray([str(s.get("id") or "") for s in meta["slots"]], dtype=object)
    return SourceTrack(meta.get("source", TACTICAL), int(meta["start"]), data[:, 9:11].copy(), xy, team, kick,
                       tuple(meta["pitch"]), float(meta.get("fps", 25.0)), path, slot_ids)


def _runs(mask):
    """[(a, b)] half-open runs of True in mask."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def compose(n_frames: int, parts, fps: float = 25.0) -> Tracking:
    """Tracking on tactical frames from per-recording pieces.

    parts: [(SourceTrack, rows, take)] where rows (n_frames,) is the row of that track for each
    tactical frame (-1 = none) and take (n_frames,) says which frames come from it; later parts
    win where several take the same frame. The result spans the first to the last frame taken.
    """
    taken = np.zeros(n_frames, bool)
    for _, rows, take in parts:
        taken |= take & (rows >= 0)
    hit = np.flatnonzero(taken)
    lo, hi = (int(hit[0]), int(hit[-1]) + 1) if len(hit) else (0, 0)
    m, S = hi - lo, sum(t.xy.shape[1] for t, _, _ in parts)
    ball = np.full((m, 2), np.nan, np.float32)
    xy = np.full((m, S, 2), np.nan, np.float32)
    kick = np.zeros(m, bool)
    origin = np.full(m, -1, np.int8)
    team, slot_ids, off, pitch = [], [], 0, None
    for t, rows, take in parts:
        s = t.xy.shape[1]
        sel = np.flatnonzero((take & (rows >= 0))[lo:hi])
        r = rows[lo:hi][sel]
        ball[sel] = t.ball[r]
        xy[sel] = np.nan
        xy[sel, off:off + s] = t.xy[r]
        kick[sel] = t.kick[r]
        origin[sel] = ORIGIN[t.source]
        team.extend(t.team.tolist())
        if t.slot_ids is not None:
            slot_ids.extend(str(x) for x in t.slot_ids.tolist())
        else:
            slot_ids.extend([""] * s)
        off += s
        pitch = pitch or t.pitch
    ids = np.asarray(slot_ids, dtype=object) if any(slot_ids) else None
    return Tracking(lo, ball, xy, np.array(team, dtype="<U1"), kick, origin, pitch, fps, ids)


class MatchInputs:
    """Everything a run needs, loaded on first use.

    from_exports   read viewer exports instead of the raw files
    exports        {source: export folder}; default the widest export of each source
    tracks         {source: SourceTrack} to use as they are (tests, notebooks)
    """

    def __init__(self, from_exports=False, exports=None, tracks=None, sync_path=DEFAULT_SYNC,
                 dead_ball_path=DEAD_BALL_JSON, n_frames=None, sync=None, dead_ball=None):
        from games import source_frames
        self.from_exports, self.exports = from_exports, dict(exports or {})
        self.sync_path, self.dead_ball_path = sync_path, dead_ball_path
        self.n_frames = int(n_frames or source_frames(TACTICAL))
        self._tracks = dict(tracks or {})
        self._sync, self._dead = sync, dead_ball
        self._dead_doc = None
        self._tracking = {}

    def track(self, source: str) -> SourceTrack:
        if source not in self._tracks:
            if self.from_exports or source in self.exports:
                self._tracks[source] = load_export(self.exports.get(source) or find_export(source))
            else:
                self._tracks[source] = load_raw(source)
            t = self._tracks[source]
            print(f"{source}: frames {t.start}..{t.end}, {t.xy.shape[1]} slots, "
                  f"ball on {100 * np.isfinite(t.ball[:, 0]).mean():.0f}% of them"
                  + (f"  ({os.path.relpath(t.origin_path)})" if t.origin_path else ""))
        return self._tracks[source]

    def sync(self):
        if self._sync is None:
            from dead_ball_investigation.sync import SyncMap
            self._sync = SyncMap.from_json(self.sync_path)
        return self._sync

    def _tactical_rows(self):
        t = self.track(TACTICAL)
        f = np.arange(self.n_frames)
        return t, np.where((f >= t.start) & (f <= t.end), f - t.start, -1)

    def _broadcast_rows(self):
        b = self.track(BROADCAST)
        return b, self.sync().index(0, self.n_frames - 1, TACTICAL, b.start, b.end)

    def tracking(self, source: str) -> Tracking:
        """Tracking on the tactical timeline. TACTICAL_THEN_BROADCAST takes a frame from the
        broadcast where the tactical has no row or lost the ball and the synced broadcast has it."""
        if source in self._tracking:
            return self._tracking[source]
        if source == TACTICAL:
            t, rows = self._tactical_rows()
            parts = [(t, rows, rows >= 0)]
        elif source == BROADCAST:
            b, rows = self._broadcast_rows()
            parts = [(b, rows, rows >= 0)]
        elif source == TACTICAL_THEN_BROADCAST:
            t, tr = self._tactical_rows()
            b, br = self._broadcast_rows()
            t_ball = np.zeros(self.n_frames, bool)
            t_ball[tr >= 0] = np.isfinite(t.ball[tr[tr >= 0], 0])
            b_ball = np.zeros(self.n_frames, bool)
            b_ball[br >= 0] = np.isfinite(b.ball[br[br >= 0], 0])
            use_b = (br >= 0) & ((tr < 0) | (~t_ball & b_ball))
            parts = [(t, tr, (tr >= 0) & ~use_b), (b, br, use_b)]
        else:
            raise ValueError(f"unknown source {source!r}")
        fps = parts[0][0].fps
        trk = compose(self.n_frames, parts, fps)
        if source == TACTICAL_THEN_BROADCAST:
            print(f"tactical+broadcast: {int((trk.origin == ORIGIN[BROADCAST]).sum())} frames from the broadcast")
        self._tracking[source] = trk
        return trk

    def dead_ball(self) -> np.ndarray:
        """Dead-ball mask over the tactical frames, from dead_ball.json."""
        if self._dead is None:
            from dead_ball_investigation.dead_ball import intervals_to_mask
            if not os.path.isfile(self.dead_ball_path):
                raise IOError(f"no dead-ball intervals at {self.dead_ball_path}\n"
                              "  make them:  python dead_ball_investigation/run.py   (or --from-exports)")
            with open(self.dead_ball_path) as fh:
                self._dead_doc = json.load(fh)
            tl = (self._dead_doc.get("timelines") or {}).get(TACTICAL)
            if not tl:
                raise ValueError(f"{self.dead_ball_path} has no tactical timeline")
            self._dead = intervals_to_mask(tl["dead_ball"], 0, self.n_frames)
        return np.asarray(self._dead, bool)

    def describe(self, source: str, dead_ball: bool) -> dict:
        """Where the inputs of one run came from, for the .meta.json next to the CSV."""
        used = [TACTICAL] if source == TACTICAL else [BROADCAST] if source == BROADCAST else [TACTICAL, BROADCAST]
        out = {}
        for s in used:
            t = self._tracks.get(s)
            if t is not None:
                kind = "export" if (self.from_exports or s in self.exports) else "raw data" if t.origin_path else "in memory"
                out[s] = {"from": kind, "path": os.path.relpath(t.origin_path) if t.origin_path else None,
                          "frames": [t.start, t.end]}
        if BROADCAST in used and self._sync is not None:
            out["sync"] = {"path": os.path.relpath(self.sync_path),
                           "anchors": [[a["tactical"], a["broadcast"]] for a in self._sync.anchors]}
        if dead_ball:
            out["dead_ball"] = {"path": os.path.relpath(self.dead_ball_path),
                                "frames": int(self.dead_ball().sum())}
            if self._dead_doc:
                out["dead_ball"]["created"] = self._dead_doc.get("created")
        return out

    def runs(self, track: Tracking, dead_ball: bool):
        """[(a, b)] half-open row ranges of `track` an algorithm labels, each on its own: the frames
        with data, cut further at the dead balls when dead_ball is set."""
        live = track.origin >= 0
        if dead_ball:
            live = live & ~self.dead_ball()[track.start:track.start + track.n]
        return _runs(live)

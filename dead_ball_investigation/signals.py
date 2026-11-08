"""Per-frame data-quality signals for one recording, gathered exactly as match_viewer.py does.

    ball missing        ~isfinite(atd["ball"][:, 0])                      timeline row BALL LOST
    player count        atd["count"]  (track slots with a position)       timeline row COUNT != 22
    homography missing  no row, valid == 0, or a non-finite matrix        "homography  none"

Two ways to get them:
  * load_source()       from the raw files, through match_viewer.load_atd / load_homography
  * load_from_export()  from an existing match_viewer export (meta.json + data.bin), much faster
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dead_ball_investigation.paths import REPO, VIEWER_DIR, require_repo  # noqa: E402,F401

PLAYERS_ON_PITCH = 22
_MV = None


def viewer():
    """match_viewer/match_viewer.py as a module (imported on first use: it pulls in cv2)."""
    global _MV
    if _MV is None:
        require_repo()
        mod = sys.modules.get("match_viewer")
        if mod is None or not hasattr(mod, "load_atd"):
            spec = importlib.util.spec_from_file_location("match_viewer", os.path.join(VIEWER_DIR, "match_viewer.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["match_viewer"] = mod
            spec.loader.exec_module(mod)
        _MV = mod
    return _MV


@dataclass
class SourceSignals:
    """Signals for video frames start .. start + len - 1 of one recording."""
    source: str
    start: int
    ball_missing: np.ndarray      # bool
    count: np.ndarray             # int, tracked players per frame
    hom_missing: np.ndarray       # bool

    def __post_init__(self):
        self.ball_missing = np.asarray(self.ball_missing, dtype=bool)
        self.count = np.asarray(self.count, dtype=np.int64)
        self.hom_missing = np.asarray(self.hom_missing, dtype=bool)
        if not (len(self.ball_missing) == len(self.count) == len(self.hom_missing)):
            raise ValueError(f"{self.source}: signal lengths differ")

    @property
    def n(self) -> int:
        return len(self.count)

    @property
    def end(self) -> int:
        return self.start + self.n - 1

    @property
    def count_off(self) -> np.ndarray:
        return self.count != PLAYERS_ON_PITCH


def homography_missing(hom, start: int, end: int) -> np.ndarray:
    """Per frame start..end: True where match_viewer draws "homography  none"."""
    idx, Hmat, _err, valid = hom
    out = np.ones(end - start + 1, dtype=bool)
    finite = np.isfinite(Hmat.reshape(len(Hmat), -1)).all(axis=1)
    for f, i in idx.items():
        if start <= f <= end and valid[i] and finite[i]:
            out[f - start] = False
    return out


def signals_from_viewer_data(source: str, start: int, end: int, atd, hom) -> SourceSignals:
    """Signals from what match_viewer.load_atd / load_homography return."""
    n = end - start + 1
    if atd:
        ball_missing = ~np.isfinite(atd["ball"][:, 0])
        count = atd["count"]
    else:
        ball_missing, count = np.ones(n, dtype=bool), np.zeros(n, dtype=np.int64)
    hom_missing = homography_missing(hom, start, end) if hom is not None else np.ones(n, dtype=bool)
    return SourceSignals(source, start, ball_missing, count, hom_missing)


def find_inputs(source: str):
    """(atd raw, atd xml, homography) paths, found the way match_viewer.py finds them."""
    mv = viewer()
    atd = mv.find_file(f"*{source}_atd*.txt", source=source)
    meta = mv.find_file(f"*{source}_atd*.xml", required=False, source=source)
    hom = mv.find_file(f"*{source}_homography*.csv", source=source)
    return atd, meta, hom


def load_source(source: str, start: int = 0, end: int | None = None) -> SourceSignals:
    """Signals for one recording from the raw data files (slow: parses the whole ATD)."""
    from games import source_frames
    mv = viewer()
    end = source_frames(source) - 1 if end is None else end
    atd_raw, atd_meta, hom_path = find_inputs(source)
    print(f"--- {source}: frames {start}..{end}")
    atd, _meta, _home = mv.load_atd(atd_raw, atd_meta, start, end)
    hom = mv.load_homography(hom_path)
    return signals_from_viewer_data(source, start, end, atd, hom)


def _tag(a, b, code, type_, team):
    return {"a": int(a), "b": int(b), "code": code, "type": type_, "team": team}


def load_tags(path: str | None = None):
    """Smart Tagging instances on tactical frames a..b, via match_viewer.load_tags."""
    mv = viewer()
    path = path or mv.find_file("*smart_tagging*.xml", source="tactical")
    return [_tag(e["a"], e["b"], e["code"], e["tags"].get("Type"), e["tags"].get("Team")) for e in mv.load_tags(path)]


def load_tags_from_export(export_dir: str):
    """The Smart Tagging instances a tactical export carries in meta.json (those overlapping it)."""
    if not os.path.isabs(export_dir):
        export_dir = os.path.join(VIEWER_DIR, export_dir)
    with open(os.path.join(export_dir, "meta.json")) as fh:
        meta = json.load(fh)
    return [_tag(e["a"], e["b"], e.get("code"), e.get("type"), e.get("team")) for e in meta.get("tags") or []]


def load_from_export(export_dir: str) -> SourceSignals:
    """Signals from a match_viewer export folder (what match_viewer.html itself reads).

    data.bin holds `block` float32 per frame: 9 homography values (NaN where there is none), then
    ball x, y, ... So ball missing is a NaN ball x, homography missing is a NaN first matrix entry,
    and the player count is meta["count"].
    """
    if not os.path.isabs(export_dir):
        export_dir = os.path.join(VIEWER_DIR, export_dir)
    with open(os.path.join(export_dir, "meta.json")) as fh:
        meta = json.load(fh)
    n, block = int(meta["n"]), int(meta["block"])
    data = np.fromfile(os.path.join(export_dir, "data.bin"), dtype=np.float32)
    if data.size != n * block:
        raise ValueError(f"{export_dir}: data.bin has {data.size} floats, expected {n} x {block}")
    data = data.reshape(n, block)
    count = np.asarray(meta["count"], dtype=np.int64) if meta.get("count") else np.zeros(n, dtype=np.int64)
    return SourceSignals(meta["source"], int(meta["start"]), ~np.isfinite(data[:, 9]), count, ~np.isfinite(data[:, 0]))

"""Load tactical ATD keeper slots, attack directions, homography and video size."""

from __future__ import annotations

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_correction.goalkeeper_trackers.derive import DEFAULT_TRACKERS, defending_half  # noqa: E402
from data_correction.goalkeeper_trackers.paths import VIEWER_DIR, require_repo  # noqa: E402
from data_correction.goalkeeper_trackers.visibility import camera_visible  # noqa: E402

_MV = None


def viewer():
    """match_viewer/match_viewer.py as a module (imported on first use: it pulls in cv2)."""
    global _MV
    if _MV is None:
        require_repo()
        mod = sys.modules.get("match_viewer")
        if mod is None or not hasattr(mod, "load_atd"):
            spec = importlib.util.spec_from_file_location(
                "match_viewer", os.path.join(VIEWER_DIR, "match_viewer.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["match_viewer"] = mod
            spec.loader.exec_module(mod)
        _MV = mod
    return _MV


def _local_name(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def parse_team_meta(xml_path: str) -> dict:
    """Teams, slot list and first-half attack directions from the ATD metadata XML."""
    root = ET.parse(xml_path).getroot()
    teams: dict[str, str] = {}
    attack: dict[str, str] = {}
    slots = []
    for element in root.iter():
        tag = _local_name(element)
        if tag == "Team":
            team_id = element.get("id") or element.get("teamId")
            if not team_id:
                continue
            name = next(((c.text or "").strip() for c in element if _local_name(c) == "Name"), team_id)
            teams[team_id] = name
            for parameter in element.iter():
                if _local_name(parameter) != "ProviderParameter":
                    continue
                pname = next((c.text or "" for c in parameter if _local_name(c) == "Name"), "")
                if pname.strip() != "attack_direction_first_half":
                    continue
                attack[team_id] = next(
                    (c.text or "" for c in parameter if _local_name(c) == "Value"), "").strip()
        elif tag == "Player":
            player_id = element.get("id") or element.get("playerId")
            if not player_id:
                continue
            player_name = next(
                ((c.text or "").strip() for c in element if _local_name(c) == "Name"),
                f"Track {player_id}",
            )
            slots.append({
                "id": player_id,
                "name": player_name,
                "team_id": element.get("teamId", ""),
            })
    if len(teams) != 2 or not slots:
        raise ValueError("expected two teams and at least one player slot in the ATD metadata")
    unknown = sorted({slot["team_id"] for slot in slots} - set(teams))
    if unknown:
        raise ValueError(f"player slots reference unknown teams: {unknown}")
    team_ids = list(teams)
    team_side = {team_ids[0]: "home", team_ids[1]: "away"}
    for slot in slots:
        slot["side"] = team_side[slot["team_id"]]
    return {
        "teams": teams,
        "team_ids": team_ids,
        "team_side": team_side,
        "attack": attack,
        "slots": slots,
        "team_names_by_side": {team_side[tid]: teams[tid] for tid in team_ids},
        "attack_by_side": {team_side[tid]: attack.get(tid, "") for tid in team_ids},
    }


def second_half_mask(n_frames: int, game: str | None = None) -> np.ndarray:
    """True on second-half video frames (and after), using games.py half anchors."""
    from games import GAME, GAMES
    game = game or GAME
    second_start = GAMES[game]["halves"][2][0] - GAMES[game]["offsets"][2]
    return np.arange(n_frames) >= second_start


def video_size(video_path: str) -> tuple[int, int]:
    import cv2
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(f"could not open tactical video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid video size {width}x{height} from {video_path}")
    return width, height


def load_homography_table(path: str, n_frames: int):
    """(homographies (n,3,3), valid (n,)) aligned to video frames 0 .. n-1."""
    import csv

    matrix_columns = [f"h{row}{col}" for row in range(3) for col in range(3)]
    matrices = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    valid = np.zeros(n_frames, dtype=bool)
    seen = np.zeros(n_frames, dtype=bool)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("homography CSV has no header")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"frame", "valid"} | set(matrix_columns)
        missing = required - set(fields)
        if missing:
            raise ValueError(f"homography is missing columns: {sorted(missing)}")
        for row in reader:
            frame = int(float(row[fields["frame"]]))
            if frame < 0 or frame >= n_frames:
                raise ValueError(f"homography frame {frame} outside 0..{n_frames - 1}")
            if seen[frame]:
                raise ValueError(f"duplicate homography row for frame {frame}")
            seen[frame] = True
            valid[frame] = bool(int(float(row[fields["valid"]])))
            values = [float(row[fields[col]]) for col in matrix_columns]
            matrices[frame] = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if not seen.all():
        raise ValueError("homography rows do not match zero-based video frames")
    return matrices, valid


@dataclass
class KeeperInputs:
    """Everything derive_goalkeeper_tracks needs for one match."""
    positions: np.ndarray       # (n, 2, 2)
    speeds: np.ndarray          # (n, 2)
    visible: np.ndarray         # (n, 2)
    home_defending_half: np.ndarray
    away_defending_half: np.ndarray
    tracker_ids: tuple
    team_names: dict
    fps: float
    n_frames: int
    game: str
    source: str = "tactical"


def load_keeper_inputs(
        tracker_ids: tuple[str, str] = DEFAULT_TRACKERS,
        game: str | None = None,
        source: str = "tactical",
        video_wh: tuple[int, int] | None = None,
) -> KeeperInputs:
    """Read tactical ATD + homography (+ video size) and build keeper inputs."""
    from games import GAME, GAMES, asset_paths, require, source_frames

    require_repo()
    game = game or GAME
    paths = asset_paths(game=game, source=source)
    require(("atd_raw", "atd_meta", "homography"), game=game, source=source)
    if video_wh is None:
        require(("video",), game=game, source=source)
        video_wh = video_size(paths["video"])

    n_frames = int(source_frames(source, game))
    fps = float(GAMES[game]["fps"])

    meta = parse_team_meta(paths["atd_meta"])
    missing = [tid for tid in tracker_ids if tid not in {s["id"] for s in meta["slots"]}]
    if missing:
        raise ValueError(f"missing goalkeeper tracker slots: {missing}")

    mv = viewer()
    print(f"--- {source}: loading ATD frames 0..{n_frames - 1}")
    atd, _players_meta, _home = mv.load_atd(paths["atd_raw"], paths["atd_meta"], 0, n_frames - 1)
    id_to_col = {pid: i for i, pid in enumerate(atd["ids"])}
    cols = [id_to_col[tid] for tid in tracker_ids]
    positions = atd["xy"][:, cols, :2].astype(np.float32)
    # Raw ATD speed channel (same as the notebook); not the frame-differenced pspeed_f.
    speeds = atd["xy"][:, cols, 2].astype(np.float32)

    print(f"--- {source}: loading homography")
    homographies, valid = load_homography_table(paths["homography"], n_frames)
    print(f"--- {source}: classifying camera visibility for trackers {tracker_ids}")
    visible = camera_visible(positions, homographies, valid, video_wh)

    second_half = second_half_mask(n_frames, game)
    home_dir = meta["attack_by_side"]["home"]
    away_dir = meta["attack_by_side"]["away"]
    return KeeperInputs(
        positions=positions,
        speeds=speeds,
        visible=visible,
        home_defending_half=defending_half(home_dir, second_half),
        away_defending_half=defending_half(away_dir, second_half),
        tracker_ids=tuple(tracker_ids),
        team_names=meta["team_names_by_side"],
        fps=fps,
        n_frames=n_frames,
        game=game,
        source=source,
    )

"""Readers and the command line, run against the small synthetic match in the scratchpad (skipped when absent)."""
import os
import json
import numpy as np
import pytest
import yaml

from ballq import io as bio
from ballq.cli import paths_for, main
from ballq.contract import read_table, FRAME_COLUMNS, FRAME_EXTRA

from ballq.cli import find_data_root
import glob

# the real match when the package sits in the data folder, a synthetic one via BALLQ_TEST_DATA, else skipped
DATA = os.environ.get("BALLQ_TEST_DATA") or find_data_root()
needs_data = pytest.mark.skipif(not glob.glob(os.path.join(DATA, "**", "*tactical_atd*.txt"), recursive=True), reason="no tactical ATD found under " + DATA)


@pytest.fixture(scope="module")
def paths():
    return paths_for(DATA, "tactical")


@needs_data
def test_read_atd_shapes_and_teams(paths):
    w = bio.read_atd(paths["atd_txt"], paths["atd_xml"], 11185, 11284)
    assert w["xy"].shape == (100, len(w["slot_ids"]), 2) and w["ball_obs"].shape == (100, 2)
    assert set(w["team"]) == {"H", "A"}
    assert all(isinstance(s, str) and s for s in w["slot_ids"])
    assert set(w["team_names"]) == {"H", "A"} and all(w["team_names"].values())
    fin = w["xy"][np.isfinite(w["xy"])]
    assert fin.min() > -0.05 and fin.max() < 1.05


@needs_data
def test_read_atd_fails_on_empty_window(paths):
    with pytest.raises(ValueError):
        bio.read_atd(paths["atd_txt"], paths["atd_xml"], 10 ** 8, 10 ** 8 + 10)


@needs_data
def test_homography_tags_gt(paths):
    idx, H, err, valid = bio.read_homography(paths["homography"])
    assert H.shape[1:] == (3, 3) and len(idx) == len(err) == len(valid)
    tags = bio.read_tags(paths["tags"], 25)
    assert all(t["a"] <= t["b"] for t in tags)
    gt = bio.read_gt(paths["gt"])
    assert set(gt.values()) <= {"H", "A", "D", "X"}


@needs_data
def test_load_world_has_everything_the_filter_needs(paths):
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
    w = bio.load_world(paths, cfg, 11185, 11284)
    for k in ("xy", "ball_obs", "team", "fps", "pitch", "hom_err", "dead", "tags", "restarts"):
        assert k in w, k
    assert w["hom_err"].shape == (100,) and w["dead"].shape == (100,)


@needs_data
def test_gt_never_feeds_the_filter_by_default(paths):
    """rule 1: the ground truth is read for the agreement number only"""
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
    assert cfg["features"]["dead_from_gt"] is False
    w = bio.load_world(paths, cfg, 11185, 11284)
    from ballq.features import dead_windows
    assert np.array_equal(w["dead"], dead_windows(100, 11185, None, w["tags"], cfg["fps"]))


@needs_data
def test_cli_writes_the_contract(paths, tmp_path):
    main(["run", "--source", "tactical", "--data", DATA, "--start", "11185", "--end", "11234", "--out", str(tmp_path),
          "--config", os.path.join(os.path.dirname(__file__), "..", "config.yaml")])
    out = tmp_path / "tactical"
    frames = read_table(out / "ball_frames")
    assert list(frames.columns) == FRAME_COLUMNS + FRAME_EXTRA
    assert len(frames) == 50 and frames["frame"].iloc[0] == 11185
    players = read_table(out / "ball_players"); grid = read_table(out / "ball_grid")
    assert players["p_within_1m"].between(0, 1).all() and grid["p"].gt(0).all()
    q = json.load(open(out / "ball_quality.json"))
    assert q["n_frames"] == 50 and q["config_hash"] and "gt_agreement_lh_posterior" in q
    # the binary grid holds the same cells as the table
    idx = np.fromfile(out / "ball_grid_index.bin", np.uint64); raw = np.fromfile(out / "ball_grid.bin", np.uint16).reshape(-1, 3)
    assert len(idx) == 51 and idx[-1] == raw.nbytes and len(raw) == len(grid)
    f0 = raw[: int(idx[1]) // 6]; g0 = grid[grid.frame == 11185]
    assert len(f0) == len(g0) and abs(f0[:, 2].astype(float).sum() / 65535 - g0.p.sum()) < 0.01
    # the same input gives the same output: no seed anywhere
    main(["run", "--source", "tactical", "--data", DATA, "--start", "11185", "--end", "11234", "--out", str(tmp_path / "again"),
          "--config", os.path.join(os.path.dirname(__file__), "..", "config.yaml")])
    again = read_table(tmp_path / "again" / "tactical" / "ball_frames")
    assert np.allclose(frames[["mean_x", "mean_y", "entropy_bits"]].to_numpy(), again[["mean_x", "mean_y", "entropy_bits"]].to_numpy())


def test_find_prefers_data_folder_and_skips_overlays(tmp_path):
    from ballq.cli import find
    (tmp_path / "data" / "tactical").mkdir(parents=True)
    (tmp_path / "data" / "tactical" / "X_tactical.mp4").write_bytes(b"")
    (tmp_path / "tactical_overlay_gt.mp4").write_bytes(b"")
    (tmp_path / "export").mkdir(); (tmp_path / "export" / "a_tactical.mp4").write_bytes(b"")
    assert find(str(tmp_path), "*tactical*.mp4").endswith(os.path.join("data", "tactical", "X_tactical.mp4"))

import json
import numpy as np
import pandas as pd
from ballq.submit import read_dead_intervals, dead_mask, build_labels, score_against_gt, write_submission


def test_dead_intervals_from_team_json_and_csv(tmp_path):
    p = tmp_path / "dead.json"
    p.write_text(json.dumps({"timelines": {"tactical": {"dead_ball": [[10, 12], [3, 4]]}, "broadcast": {"dead_ball": [[0, 1]]}}}))
    assert read_dead_intervals(p, "tactical") == [(3, 4), (10, 12)]
    assert read_dead_intervals(p, "broadcast") == [(0, 1)]
    c = tmp_path / "dead.csv"; c.write_text("first,last\n5,6\n")
    assert read_dead_intervals(c) == [(5, 6)]
    m = dead_mask([(3, 4), (10, 12)], 15)
    assert m.sum() == 5 and m[3] and m[4] and not m[5] and m[12] and not m[13]


def test_labels_combine_dead_and_lh_and_abstain_elsewhere():
    frames = pd.DataFrame({"frame": [2, 3, 4, 5], "lh_post_team": ["H", "A", "X", None], "lh_raw_team": ["A", "A", "A", "A"]})
    dead = dead_mask([(3, 3)], 8)
    lab = build_labels(frames, 8, dead, "lh_post_team")
    assert list(lab) == ["X", "X", "H", "D", "X", "X", "X", "X"]
    lab2 = build_labels(frames, 8, dead, "lh_raw_team")
    assert list(lab2[2:6]) == ["A", "D", "A", "A"]


def test_score_and_file_format(tmp_path):
    lab = np.array(list("XXHDAX"), dtype=object)
    gt = {0: "X", 2: "H", 3: "D", 4: "H", 5: "A"}
    s = score_against_gt(lab, gt)
    assert s["frames"] == 4 and abs(s["accuracy"] - 0.5) < 1e-9 and s["D"]["recall"] == 1.0 and s["H"]["recall"] == 0.5
    p = write_submission(lab, tmp_path / "sub.csv")
    text = open(p).read()
    assert text.startswith("frame,label\n0,X\n1,X\n2,H\n3,D\n") and text.count("\n") == 7

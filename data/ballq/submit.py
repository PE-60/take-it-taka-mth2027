"""The submission file: frame,label for every tactical video frame (SUBMISSION.md).

label = D inside one of our dead-ball intervals, otherwise the Link and Hoernig team label from
ball_frames (filtered ball by default, raw on request), X where neither says anything.

read_dead_intervals(path, source)  -> [(first, last)] inclusive, from the team's dead-ball json
dead_mask(intervals, n)            -> bool (n,)
build_labels(frames_df, n, dead, column) -> array of 'H' 'A' 'D' 'X'
score_against_gt(labels, gt)       -> accuracy over GT frames, plus per-label agreement
"""
import json
import numpy as np
import pandas as pd


def read_dead_intervals(path, source="tactical"):
    """accepts the team json ({"timelines": {source: {"dead_ball": [[a, b], ...]}}}),
    a bare list of [a, b], or a csv with columns first,last (or start,end)"""
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path); df.columns = [c.strip().lower() for c in df.columns]
        a = "first" if "first" in df.columns else "start"; b = "last" if "last" in df.columns else "end"
        return [(int(x), int(y)) for x, y in zip(df[a], df[b])]
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        iv = data
    elif "timelines" in data:
        if source not in data["timelines"]:
            raise ValueError(f"{path}: no '{source}' timeline, has {list(data['timelines'])}")
        iv = data["timelines"][source]["dead_ball"]
    elif "dead_ball" in data:
        iv = data["dead_ball"]
    else:
        raise ValueError(f"{path}: expected timelines.<source>.dead_ball or a list of [first, last]")
    out = sorted((int(a), int(b)) for a, b in iv)
    for a, b in out:
        if b < a:
            raise ValueError(f"interval {a}..{b} ends before it starts")
    return out


def dead_mask(intervals, n, offset=0):
    """frames offset..offset+n-1 -> True inside an interval"""
    m = np.zeros(n, bool)
    for a, b in intervals:
        lo, hi = max(a - offset, 0), min(b - offset + 1, n)
        if hi > lo:
            m[lo:hi] = True
    return m


def build_labels(frames, n, dead, column="lh_post_team"):
    """frames: ball_frames table (any subset of 0..n-1); dead: bool (n,). Returns object array of labels."""
    lab = np.full(n, "X", dtype=object)
    if frames is not None and len(frames):
        f = frames["frame"].to_numpy(int); v = frames[column].fillna("X").astype(str).str.upper().to_numpy()
        ok = (f >= 0) & (f < n)
        lab[f[ok]] = np.where(np.isin(v[ok], ["H", "A"]), v[ok], "X")
    lab[dead] = "D"
    return lab


def score_against_gt(labels, gt):
    """gt: {frame: label}. Accuracy on frames where GT is H, A or D, and a per-label breakdown."""
    rows = [(labels[f], g) for f, g in gt.items() if g in ("H", "A", "D") and 0 <= f < len(labels)]
    if not rows:
        return None
    pred = np.array([p for p, _ in rows]); true = np.array([g for _, g in rows])
    out = {"frames": len(rows), "accuracy": float(np.mean(pred == true)), "abstained": float(np.mean(pred == "X"))}
    for lab in ("H", "A", "D"):
        t = true == lab; p = pred == lab
        tp = int((t & p).sum())
        out[lab] = {"gt": int(t.sum()), "pred": int(p.sum()), "recall": tp / max(1, t.sum()), "precision": tp / max(1, p.sum())}
    return out


def write_submission(labels, path):
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False, lineterminator="\n")
    return path

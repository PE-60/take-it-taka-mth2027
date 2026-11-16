"""Run one algorithm over the match and write evaluation/results/<name>.csv (+ <name>.meta.json).

The CSV is the submission format evaluation/run_evaluation.py scores: header `frame,label`, one row
per tactical video frame 0 .. N-1, labels H / A / D / X. The .meta.json next to it says how it was
made (algorithm, dead-ball mode, source, parameters, inputs); the viewers show it in their
algorithm picker, run_evaluation.py ignores it.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import sys

import numpy as np

from possesion_algorithms.base import LABELS, ORIGIN, PossessionAlgorithm
from possesion_algorithms.inputs import MatchInputs
from possesion_algorithms.paths import EVALUATION_DIR, GT_DIR, RESULTS_DIR


def _metrics():
    if EVALUATION_DIR not in sys.path:
        sys.path.insert(0, EVALUATION_DIR)
    import possession_metrics
    return possession_metrics


def clean_labels(raw, n: int, who: str) -> np.ndarray:
    """An algorithm's answer as H/A/D/X: '' and '?' (unknown team) become X."""
    out = np.asarray(raw).astype("<U1")
    if out.shape != (n,):
        raise ValueError(f"{who}: label() returned shape {out.shape} for {n} frames")
    out = np.char.upper(out)
    out[(out == "") | (out == "?")] = "X"
    bad = sorted(set(np.unique(out).tolist()) - set(LABELS))
    if bad:
        raise ValueError(f"{who}: label() returned {bad}; labels are {LABELS} ('' or '?' for none)")
    return out


def label_match(algo: PossessionAlgorithm, inputs: MatchInputs) -> tuple[np.ndarray, dict]:
    """Labels for every tactical frame, and a summary of what was labelled how."""
    n = inputs.n_frames
    labels = np.full(n, "X", dtype="<U1")
    track = inputs.tracking(algo.source)
    runs = inputs.runs(track, algo.dead_ball)
    live = np.zeros(track.n, bool)
    for a, b in runs:
        live[a:b] = True
    algo.prepare(track, live)
    for a, b in runs:
        part = track.slice(a, b)
        labels[part.start:part.start + part.n] = clean_labels(algo.label(part), part.n, algo.name)
    if algo.dead_ball:
        labels[inputs.dead_ball()] = "D"
    covered = np.zeros(n, bool)
    covered[track.start:track.start + track.n] = track.origin >= 0
    origin = np.full(n, -1, np.int8)
    origin[track.start:track.start + track.n] = track.origin
    summary = {
        "frames": n,
        "covered": int(covered.sum()),
        "labelled_runs": len(runs),
        "from_broadcast": int((origin == ORIGIN["broadcast"]).sum()),
        "counts": {k: int((labels == k).sum()) for k in LABELS},
    }
    return labels, summary


def write_csv(path: str, labels: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.write("".join(f"{i},{l}\n" for i, l in enumerate(labels.tolist())))


def validate(path: str, n: int):
    """Raise if run_evaluation.py would skip the file."""
    pm = _metrics()
    reason = pm.validate_submission(pm.read_label_csv(path), n)
    if reason:
        raise ValueError(f"{path} is not a valid submission: {reason}")


def public_gt():
    hits = sorted(glob.glob(os.path.join(GT_DIR, "*_possession_gt_public.csv")))
    return hits[0] if hits else None


def score(path: str, gt_path: str | None = None) -> dict | None:
    """The headline numbers on the public ground truth (None when there is none)."""
    gt_path = gt_path or public_gt()
    if not gt_path or not os.path.isfile(gt_path):
        return None
    pm = _metrics()
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    pred = pm.label_codes(pm.read_label_csv(path))
    if len(gt) != len(pred):
        return None
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return {"gt": os.path.basename(gt_path), "spell_f1_mean": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"], "n_graded": r["n_graded"]}


def run(algo: PossessionAlgorithm, inputs: MatchInputs, out_dir: str = RESULTS_DIR, name: str | None = None,
        do_score: bool = True) -> dict:
    """Label, write <out_dir>/<name>.csv and its .meta.json, validate, score. Returns the meta."""
    name = name or algo.name
    labels, summary = label_match(algo, inputs)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    write_csv(csv_path, labels)
    validate(csv_path, inputs.n_frames)
    meta = {"name": name, **algo.settings(), "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "inputs": inputs.describe(algo.source, algo.dead_ball), **summary}
    if do_score:
        meta["public_gt"] = score(csv_path)
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta

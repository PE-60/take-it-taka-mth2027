"""Spell smoothing for the submission labels.

The scorer (EVALUATION.md) treats a possession as one object: a maximal run of H or of A, matched
to the true possessions by temporal IoU. A short false spell therefore costs twice: it is an
unmatched prediction, and it cuts a true possession into two halves that may each fail the IoU
thresholds. X is free inside a possession (the scorer forward-fills it) and only harmful at a real
hand-over. So the right policy is: change the team only on evidence, and never for a moment.

absorb_short_spells(labels, fps, min_s): a run of H or A shorter than min_s that is bordered by the
same team on both sides (a one-touch interception that came straight back, or a pass grazing an
opponent) takes the surrounding team. Runs next to a dead ball or at a real change of team are kept,
whatever their length: cutting those would move a transition, and a second late costs a third of the
score. min_s defaults to 1 s, the hold Link and Hoernig use before a possession ends; it is not tuned
on the ground truth.
"""
import numpy as np


def runs(labels):
    """[(start, end_exclusive, label)] over the whole array"""
    out = []; n = len(labels)
    if n == 0:
        return out
    s = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[s]:
            out.append((s, i, labels[s])); s = i
    return out


def absorb_short_spells(labels, fps, min_s=1.0, max_rounds=3):
    """labels: sequence of 'H' 'A' 'D' 'X'. Returns a new object array."""
    lab = np.array(list(labels), dtype=object)
    min_len = int(round(min_s * fps))
    for _ in range(max_rounds):
        changed = False
        rs = runs(lab)
        # team runs with X folded in: an X stretch inside a possession is still that possession for the scorer
        for k, (a, b, v) in enumerate(rs):
            if v not in ("H", "A") or b - a >= min_len:
                continue
            prev = next((rs[j][2] for j in range(k - 1, -1, -1) if rs[j][2] != "X"), None)
            nxt = next((rs[j][2] for j in range(k + 1, len(rs)) if rs[j][2] != "X"), None)
            if prev in ("H", "A") and prev == nxt and prev != v:
                lab[a:b] = prev; changed = True
        if not changed:
            break
    return lab


def spell_stats(labels, fps):
    rs = [r for r in runs(labels) if r[2] in ("H", "A")]
    if not rs:
        return {"possessions": 0, "mean_s": None, "min_s": None}
    d = np.array([(b - a) / fps for a, b, _ in rs])
    return {"possessions": len(rs), "mean_s": float(d.mean()), "min_s": float(d.min()), "under_1s": int((d < 1).sum())}

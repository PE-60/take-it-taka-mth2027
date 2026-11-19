"""Build the submission: L&H labels, changed to lgbm_shape's team only where lgbm_shape is sure.

    python make_submission.py LH_RESTARTGAP.csv LGBM_SHAPE.csv LGBM_PROBA.csv submission.csv

LH_RESTARTGAP.csv  frame,label from the L&H run, with the restart gaps already filled
LGBM_SHAPE.csv     frame,label from lgbm_shape
LGBM_PROBA.csv     frame,p_home,p_home_oof from lgbm_shape

Where the two label files name different teams, each stretch is looked at by itself: if lgbm's
mean probability for its own team over the stretch is at least SURE, the stretch takes lgbm's
team; otherwise the L&H label stays. Dead ball is never changed. No randomness: the same inputs always give the same file.
"""
import sys
import numpy as np
import pandas as pd

SURE = 0.7                  # how sure lgbm_shape must be before it overrules L&H
COLUMN = "p_home_oof"       # probabilities from the fold that did not train on the frame


def read_labels(path):
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()


def sure_override(base, lgbm, p_home, sure):
    out = base.copy()
    differ = np.isin(base, ["H", "A"]) & np.isin(lgbm, ["H", "A"]) & (base != lgbm)
    own = np.where(lgbm == "H", p_home, 1 - p_home)          # lgbm's probability for the team it names
    edge = np.diff(np.r_[0, differ.astype(int), 0])
    changed = frames = 0
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        # one stretch has one L&H team and one lgbm team: cut where either of them switches
        cuts = [a] + [i for i in range(a + 1, b) if lgbm[i] != lgbm[i - 1] or base[i] != base[i - 1]] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            c = own[s:e]; c = c[np.isfinite(c)]
            if len(c) and c.mean() >= sure:
                out[s:e] = lgbm[s:e]; changed += 1; frames += int(e - s)
    return out, changed, frames


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    lh_path, shape_path, proba_path, out_path = sys.argv[1:]
    lh, lgbm = read_labels(lh_path), read_labels(shape_path)
    if len(lh) != len(lgbm):
        sys.exit(f"the two label files have different lengths: {len(lh)} and {len(lgbm)}")
    p_home = pd.read_csv(proba_path).set_index("frame")[COLUMN].reindex(range(len(lh))).to_numpy(float)
    out, changed, frames = sure_override(lh, lgbm, p_home, SURE)
    print(f"lgbm sure at {SURE}: {changed} stretches changed, {frames} frames")
    pd.DataFrame({"frame": np.arange(len(out)), "label": out}).to_csv(out_path, index=False, lineterminator="\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

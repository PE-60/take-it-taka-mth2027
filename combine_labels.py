"""Build the submission from two label files that the repository already produces.

    python combine_labels.py LH_LABELS.csv LGBM_SHAPE.csv OUT.csv [--max-s 2.5]

Two steps, in this order:
  1. restart gap: an X stretch that follows a dead ball takes the team that comes next
  2. short-run veto: a team run shorter than --max-s is given to the other team when
     lgbm_shape labels more than half of its frames as the other team
No randomness: the same two inputs always give the same output file.
"""
import argparse
import numpy as np
import pandas as pd


def read_labels(path):
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()


def runs(labels):
    """[(start, end_exclusive, label)] of the constant stretches"""
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    b = np.concatenate([[0], cut, [len(labels)]])
    return [(int(b[i]), int(b[i + 1]), labels[b[i]]) for i in range(len(b) - 1)]


def fill_restart_gap(labels):
    out = labels.copy(); r = runs(labels)
    for i, (a, b, k) in enumerate(r):
        before = r[i - 1][2] if i else "D"
        if k == "X" and i + 1 < len(r) and before == "D" and r[i + 1][2] in "HA":
            out[a:b] = r[i + 1][2]
    return out


def short_run_veto(labels, reference, max_frames):
    out = labels.copy(); flipped = 0
    for a, b, k in runs(labels):
        if k not in "HA" or b - a >= max_frames:
            continue
        other = "A" if k == "H" else "H"
        if (reference[a:b] == other).mean() > 0.5:
            out[a:b] = other; flipped += 1
    return out, flipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("shape"); ap.add_argument("out")
    ap.add_argument("--max-s", type=float, default=2.5, help="runs shorter than this may be overruled")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    lab, ref = read_labels(args.labels), read_labels(args.shape)
    if len(lab) != len(ref):
        raise SystemExit(f"the two files have different lengths: {len(lab)} and {len(ref)}")
    step1 = fill_restart_gap(lab)
    step2, flipped = short_run_veto(step1, ref, args.max_s * args.fps)
    print(f"restart gap: {int((step1 != lab).sum())} frames filled; short-run veto under {args.max_s} s: {flipped} runs overruled")
    pd.DataFrame({"frame": np.arange(len(step2)), "label": step2}).to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Combine L&H labels with the lgbm_shape probabilities.

    python combine_proba.py LH_LABELS.csv lgbm_shape_proba.csv OUT.csv [--conf 0.7] [--max-s 0] [--column p_home_oof]

Two steps, in this order:
  1. restart gap: an X stretch that follows a dead ball takes the team that comes next
  2. confidence veto: a team run is given to the other team when the lgbm model's mean
     probability for the other team over that run is at least --conf. No duration test
     unless --max-s is given.
Prints how many runs were flipped, by length, so the effect can be seen before scoring.
"""
import argparse
import numpy as np
import pandas as pd


def runs(labels):
    """[(start, end_exclusive, label)] of the constant stretches"""
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    b = np.concatenate([[0], cut, [len(labels)]])
    return [(int(b[i]), int(b[i + 1]), labels[b[i]]) for i in range(len(b) - 1)]


def fill_restart_gap(labels):
    out = labels.copy(); r = runs(labels)
    for i, (a, b, k) in enumerate(r):
        if k == "X" and 0 < i < len(r) - 1 and r[i - 1][2] == "D" and r[i + 1][2] in "HA":
            out[a:b] = r[i + 1][2]
    return out


def confidence_veto(labels, p_home, conf, max_frames=0):
    out = labels.copy(); flipped = []
    for a, b, k in runs(labels):
        if k not in "HA" or (max_frames and b - a >= max_frames):
            continue
        p = p_home[a:b]; p = p[np.isfinite(p)]
        if not len(p):
            continue
        other = 1 - p.mean() if k == "H" else p.mean()      # mean probability of the other team
        if other >= conf:
            out[a:b] = "A" if k == "H" else "H"; flipped.append((a, b - 1, k, round(float(other), 3)))
    return out, flipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("proba"); ap.add_argument("out")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--column", default="p_home_oof", help="p_home_oof (default, not trained on that frame) or p_home")
    ap.add_argument("--max-s", type=float, default=0.0, help="only runs shorter than this may be flipped; 0 = any length")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    lab = pd.read_csv(args.labels)["label"].astype(str).str.strip().str.upper().to_numpy()
    pr = pd.read_csv(args.proba).set_index("frame")[args.column].reindex(range(len(lab))).to_numpy(float)
    step1 = fill_restart_gap(lab)
    step2, flipped = confidence_veto(step1, pr, args.conf, int(round(args.max_s * args.fps)))
    print(f"restart gap: {int((step1 != lab).sum())} frames filled")
    secs = np.array([(b - a + 1) / args.fps for a, b, _, _ in flipped])
    print(f"confidence veto at {args.conf}: {len(flipped)} runs flipped; "
          + ", ".join(f"{name} {int(((secs >= lo) & (secs < hi)).sum())}" for name, lo, hi in
                      [("under 1 s", 0, 1), ("1 to 2.5 s", 1, 2.5), ("2.5 to 5 s", 2.5, 5), ("5 s and over", 5, 1e9)]))
    for a, b, k, o in flipped:
        print(f"  frames {a} to {b} ({(b - a + 1) / args.fps:.1f} s): {k} -> {'A' if k == 'H' else 'H'}, other team's mean probability {o}")
    pd.DataFrame({"frame": np.arange(len(step2)), "label": step2}).to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

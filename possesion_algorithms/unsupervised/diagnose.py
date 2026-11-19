"""Inspect a GeometryFit against labels (public GT or synthetic) without tuning on them.

Reports whether the possession axis and individual anti features actually separate H from A,
whether the HMM sign is flipped, how anomalies behave on graded frames, and how spellF1 changes
under ablations (near_gap-only, flipped axis). Designed so a synthetic win that does not transfer
is visible as "features correlate on synthetic, not on public GT".
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from possesion_algorithms.algorithms.geometry_hmm import GeometryFit, decode_sticky_axis
from possesion_algorithms.unsupervised.features import ANTI, NAMES, NEEDS_BALL


@dataclass
class FeatureSep:
    name: str
    anti: bool
    needs_ball: bool
    mean_H: float
    mean_A: float
    gap: float              # mean_H - mean_A (want > 0 for anti features)
    corr: float             # corr with signed GT (+1 H, -1 A) on finite frames
    n_H: int
    n_A: int
    finite_frac: float


def _signed_gt(labels: np.ndarray) -> np.ndarray:
    """+1 on H, -1 on A, NaN elsewhere."""
    out = np.full(len(labels), np.nan, dtype=np.float64)
    out[labels == "H"] = 1.0
    out[labels == "A"] = -1.0
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def feature_separation(values: np.ndarray, names: tuple, anti: np.ndarray,
                       needs_ball: np.ndarray, gt: np.ndarray,
                       mask: np.ndarray | None = None) -> list[FeatureSep]:
    """Per-feature H/A means and correlation with signed GT, on ``mask`` (default: all)."""
    if mask is None:
        mask = np.ones(len(gt), bool)
    signed = _signed_gt(gt)
    out = []
    for i, name in enumerate(names):
        col = values[:, i]
        use = mask & np.isfinite(col)
        h = use & (gt == "H")
        a = use & (gt == "A")
        mean_h = float(np.nanmean(col[h])) if h.any() else float("nan")
        mean_a = float(np.nanmean(col[a])) if a.any() else float("nan")
        out.append(FeatureSep(
            name=name, anti=bool(anti[i]), needs_ball=bool(needs_ball[i]),
            mean_H=mean_h, mean_A=mean_a,
            gap=mean_h - mean_a if np.isfinite(mean_h) and np.isfinite(mean_a) else float("nan"),
            corr=_corr(col, signed),
            n_H=int(h.sum()), n_A=int(a.sum()),
            finite_frac=float(np.isfinite(col[mask]).mean()) if mask.any() else 0.0,
        ))
    return out


def axis_report(poss: np.ndarray, gt: np.ndarray, live: np.ndarray | None = None) -> dict:
    """Does the possession axis point the right way on graded H/A frames?"""
    if live is None:
        live = np.ones(len(gt), bool)
    graded = live & np.isin(gt, ("H", "A"))
    signed = _signed_gt(gt)
    h = graded & (gt == "H")
    a = graded & (gt == "A")
    mean_h = float(np.nanmean(poss[h])) if h.any() else float("nan")
    mean_a = float(np.nanmean(poss[a])) if a.any() else float("nan")
    # Sign agreement: positive axis -> H
    agree = graded & np.isfinite(poss)
    if agree.any():
        pred_sign = np.where(poss[agree] >= 0, "H", "A")
        sign_acc = float((pred_sign == gt[agree]).mean())
    else:
        sign_acc = float("nan")
    flipped = bool(np.isfinite(mean_h) and np.isfinite(mean_a) and mean_h < mean_a)
    return {
        "mean_H": mean_h,
        "mean_A": mean_a,
        "gap_H_minus_A": mean_h - mean_a if np.isfinite(mean_h) and np.isfinite(mean_a) else None,
        "corr_signed_gt": _corr(poss, signed),
        "sign_accuracy": sign_acc,
        "likely_flipped": flipped,
        "n_graded_HA": int(graded.sum()),
    }


def confusion_ha(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """2x2 on frames where GT is H or A (and mask)."""
    if mask is None:
        mask = np.ones(len(gt), bool)
    use = mask & np.isin(gt, ("H", "A"))
    g, p = gt[use], pred[use]
    # Map blank/X to a third bucket for visibility
    cells = {}
    for true in ("H", "A"):
        for lab in ("H", "A", "D", "X", ""):
            cells[f"gt{true}_pred{lab or 'blank'}"] = int(((g == true) & (p == lab)).sum())
    n = int(use.sum())
    correct = int(((g == p) & np.isin(p, ("H", "A"))).sum())
    return {"n": n, "accuracy_HA": correct / n if n else None, "cells": cells}


def score_labels(gt: np.ndarray, pred: np.ndarray) -> dict:
    """spellF1 / macroF1 via evaluation/possession_metrics (same as the board)."""
    import sys
    from possesion_algorithms.paths import EVALUATION_DIR
    if EVALUATION_DIR not in sys.path:
        sys.path.insert(0, EVALUATION_DIR)
    import possession_metrics as pm
    g = pm.label_codes(gt)
    p = pm.label_codes(np.where(np.isin(pred, list("HADX")), pred, "X"))
    r = pm.evaluate(g, p, bootstrap_n=0)
    cm = pm.confusion(g, p)
    return {
        "spell_f1": float(r["spell_f1_mean"]),
        "macro_f1": float(r["macro_f1"]),
        "accuracy": float(r["accuracy"]),
        "n_graded": int(r["n_graded"]),
        "confusion": cm.tolist(),
    }


def place_on_timeline(track_start: int, n_frames: int, series: np.ndarray) -> np.ndarray:
    """Pad/crop a track-relative series onto tactical frames 0 .. n_frames-1."""
    out = np.full(n_frames, np.nan if np.issubdtype(series.dtype, np.floating) else "",
                  dtype=series.dtype if series.dtype.kind != "U" else "<U1")
    a = max(0, track_start)
    b = min(n_frames, track_start + len(series))
    if b > a:
        out[a:b] = series[a - track_start:b - track_start]
    return out


def ascii_timeline(gt: np.ndarray, pred: np.ndarray, axis: np.ndarray,
                   every: int = 25, width: int = 80) -> str:
    """Compact public-window sketch: one char per `every` frames."""
    n = len(gt)
    lines = []

    def pack_labels(labels):
        chars = []
        for i in range(0, n, every):
            chunk = labels[i:i + every]
            for lab in ("H", "A", "D", "X"):
                if (chunk == lab).sum() >= len(chunk) / 2:
                    chars.append(lab)
                    break
            else:
                chars.append(".")
        return "".join(chars)[:width]

    def pack_signs(values):
        chars = []
        for i in range(0, n, every):
            chunk = values[i:i + every]
            ok = chunk[np.isfinite(chunk)]
            if not len(ok):
                chars.append(".")
            elif np.mean(ok >= 0) >= 0.5:
                chars.append("+")
            else:
                chars.append("-")
        return "".join(chars)[:width]

    lines.append(f"gt   {pack_labels(gt)}")
    lines.append(f"pred {pack_labels(pred)}")
    lines.append(f"axis {pack_signs(axis)}")
    lines.append(f"      (1 char = {every} frames ≈ {every / 25:.1f}s; + = poss≥0 → model H)")
    return "\n".join(lines)


def anomaly_on_window(fit: GeometryFit, gt: np.ndarray, track_start: int,
                      window: slice | None = None) -> dict:
    """How often the ball is anomaly-masked on graded frames in the window."""
    n = len(gt)
    ball = place_on_timeline(track_start, n, fit.anomaly.ball_mask.astype(np.float32))
    if window is None:
        window = slice(0, n)
    w = np.zeros(n, bool)
    w[window] = True
    graded = w & (gt != "X")
    masked = graded & (ball > 0.5)
    return {
        "graded_frames": int(graded.sum()),
        "ball_anomaly_on_graded": int(masked.sum()),
        "ball_anomaly_rate_graded": float(masked.mean()) if graded.any() else None,
        "gk_flips": list(fit.anomaly.gk_flips),
        "n_ball_anomalies_track": fit.anomaly.n_ball,
    }


def build_report(fit: GeometryFit, gt_full: np.ndarray, pred_full: np.ndarray,
                 live_full: np.ndarray | None = None,
                 baselines: dict[str, np.ndarray] | None = None,
                 window: slice | None = None,
                 fps: float = 25.0) -> dict:
    """Full inspection dict for one GeometryFit vs a full-length GT label array."""
    n = len(gt_full)
    if window is None:
        # Default: every graded (non-X) frame.
        graded = np.flatnonzero(gt_full != "X")
        window = slice(int(graded[0]), int(graded[-1]) + 1) if len(graded) else slice(0, n)
    wmask = np.zeros(n, bool)
    wmask[window] = True

    # Align fit series onto full timeline.
    poss = place_on_timeline(fit.track_start, n, fit.poss)
    feat_vals = np.full((n, fit.features.values.shape[1]), np.nan, dtype=np.float32)
    a0 = max(0, fit.track_start)
    b0 = min(n, fit.track_start + fit.n)
    if b0 > a0:
        feat_vals[a0:b0] = fit.features.values[a0 - fit.track_start:b0 - fit.track_start]

    live = live_full if live_full is not None else place_on_timeline(
        fit.track_start, n, fit.live.astype(np.float32)) > 0.5

    needs = np.array([NEEDS_BALL[NAMES.index(nm)] if nm in NAMES else False
                      for nm in fit.features.names])
    seps = feature_separation(feat_vals, fit.features.names, fit.features.anti, needs,
                              gt_full, mask=wmask & live)

    anti_seps = [s for s in seps if s.anti]
    anti_seps.sort(key=lambda s: abs(s.corr) if np.isfinite(s.corr) else -1, reverse=True)

    axis = axis_report(poss, gt_full, live & wmask)
    conf = confusion_ha(gt_full, pred_full, wmask)
    scores = {"geom_hmm": score_labels(gt_full, pred_full)}

    # Ablations on the tracking stretch, then placed on full timeline.
    def _scaled_axis(name: str) -> np.ndarray:
        col = fit.features.column(name)
        med = np.nanmedian(col[fit.live])
        mad = 1.4826 * np.nanmedian(np.abs(col[fit.live] - med))
        mad = mad if np.isfinite(mad) and mad > 1e-6 else 1.0
        axis = np.clip((col - med) / mad, -3, 3)
        return np.where(np.isfinite(axis), axis, 0.0)

    def _ablate(name: str, key: str) -> None:
        lab = decode_sticky_axis(_scaled_axis(name), fit.live, fps,
                                 sticky_s=float(fit.params.get("sticky_s", 10.0)),
                                 min_spell_s=float(fit.params.get("min_spell_s", 0.8)))
        full = place_on_timeline(fit.track_start, n, lab).copy()
        full[pred_full == "D"] = "D"
        full[(full == "") | (full == "?")] = "X"
        scores[key] = score_labels(gt_full, full)

    _ablate("near_gap", "near_gap_only")
    if "control_gap" in fit.features.names:
        _ablate("control_gap", "control_gap_only")
    for name, key in (("recv_gap", "recv_gap_only"), ("lead_gap", "lead_gap_only")):
        if name in fit.features.names and np.isfinite(fit.features.column(name)).any():
            _ablate(name, key)

    # Flipped possession axis: if the model is sign-flipped, this should jump.
    flip_lab = decode_sticky_axis(-fit.poss, fit.live, fps,
                                  sticky_s=float(fit.params.get("sticky_s", 10.0)),
                                  min_spell_s=float(fit.params.get("min_spell_s", 0.8)))
    flip_full = place_on_timeline(fit.track_start, n, flip_lab).copy()
    flip_full[pred_full == "D"] = "D"
    flip_full[(flip_full == "") | (flip_full == "?")] = "X"
    scores["poss_axis_flipped"] = score_labels(gt_full, flip_full)

    if baselines:
        for name, labs in baselines.items():
            scores[name] = score_labels(gt_full, labs)

    timeline = ascii_timeline(gt_full[window], pred_full[window], poss[window])

    return {
        "window": {"start": window.start, "stop": window.stop},
        "diag": fit.diag,
        "anomaly": anomaly_on_window(fit, gt_full, fit.track_start, window),
        "axis": axis,
        "confusion_HA": conf,
        "scores": scores,
        "anti_features_ranked": [asdict(s) for s in anti_seps],
        "all_features": [asdict(s) for s in seps],
        "timeline": timeline,
        "verdict": _verdict(axis, anti_seps, scores),
    }


def _verdict(axis: dict, anti_seps: list[FeatureSep], scores: dict) -> list[str]:
    notes = []
    if axis.get("likely_flipped"):
        notes.append("Possession axis mean is higher on A than H — labels may be globally swapped. "
                     "Check scores['poss_axis_flipped'].")
    corr = axis.get("corr_signed_gt")
    if corr is not None and np.isfinite(corr) and abs(corr) < 0.1:
        notes.append(f"Possession axis barely correlates with GT (|corr|={abs(corr):.3f}). "
                     "Collective geometry may not carry possession on this data.")
    if anti_seps:
        top = anti_seps[0]
        if np.isfinite(top.corr) and abs(top.corr) > 0.15:
            notes.append(f"Strongest anti feature on this window: {top.name} "
                         f"(corr={top.corr:.3f}, gap={top.gap:.3f}).")
        near = next((s for s in anti_seps if s.name == "near_gap"), None)
        ctrl = next((s for s in anti_seps if s.name == "control_gap"), None)
        if near and np.isfinite(near.corr):
            notes.append(f"near_gap corr={near.corr:.3f}; compare near_gap_only / control_gap_only "
                         "vs geom_hmm.")
        if ctrl and near and np.isfinite(ctrl.corr) and np.isfinite(near.corr):
            if abs(ctrl.corr) + 0.02 < abs(near.corr):
                notes.append("control_gap correlates less than near_gap on this window — "
                             "press penalty may be too strong or too weak.")
    g = scores.get("geom_hmm", {})
    for key in ("near_gap_only", "control_gap_only", "lead_gap_only", "recv_gap_only"):
        alt = scores.get(key, {})
        if g and alt and alt.get("spell_f1", 0) > g.get("spell_f1", 0) + 0.04:
            notes.append(f"{key} beats geom_hmm on spellF1 — check poss_features / stickiness.")
    lead = scores.get("lead_gap_only", {})
    if lead and g and lead.get("spell_f1", 0) > g.get("spell_f1", 0) + 0.04:
        notes.append(
            "lead_gap_only wins partly because non-flight frames are filled with 0 → the HMM "
            "emits equally for H/A and stickiness carries the last team. If geom is worse, "
            "ground control_gap is over-flipping; raise sticky_s or ground_deadzone."
        )
    f = scores.get("poss_axis_flipped", {})
    if g and f and f.get("spell_f1", 0) > g.get("spell_f1", 0) + 0.04:
        notes.append("Flipping the possession axis improves spellF1 — fix H/A sign assignment.")
    if not notes:
        notes.append("No automatic red flags from axis/feature correlations on this window.")
    return notes


def write_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, allow_nan=True)


def write_series_csv(path: str, gt: np.ndarray, pred: np.ndarray, poss: np.ndarray,
                     near_gap: np.ndarray, live: np.ndarray, ball_anom: np.ndarray,
                     window: slice) -> None:
    """Per-frame series for the inspection window (for plotting elsewhere)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,gt,pred,poss,near_gap,live,ball_anomaly\n")
        for i in range(window.start, window.stop):
            fh.write(f"{i},{gt[i]},{pred[i]},"
                     f"{'' if not np.isfinite(poss[i]) else f'{poss[i]:.6f}'},"
                     f"{'' if not np.isfinite(near_gap[i]) else f'{near_gap[i]:.6f}'},"
                     f"{int(bool(live[i]))},{int(bool(ball_anom[i]))}\n")


def format_report(report: dict) -> str:
    """Human-readable summary for the terminal."""
    lines = []
    w = report["window"]
    lines.append(f"=== geom_hmm inspection  frames {w['start']}..{w['stop'] - 1} ===")
    lines.append("")
    lines.append("-- anomaly --")
    an = report["anomaly"]
    lines.append(f"  ball anomalies on graded: {an['ball_anomaly_on_graded']}/"
                 f"{an['graded_frames']} ({(an['ball_anomaly_rate_graded'] or 0) * 100:.1f}%)")
    lines.append(f"  gk_flips: {an['gk_flips']}")
    lines.append("")
    lines.append("-- possession axis --")
    ax = report["axis"]
    lines.append(f"  mean on GT H/A: {ax['mean_H']:.4f} / {ax['mean_A']:.4f}  "
                 f"gap={ax['gap_H_minus_A']}")
    lines.append(f"  corr vs signed GT: {ax['corr_signed_gt']:.4f}  "
                 f"sign accuracy: {ax['sign_accuracy']:.3f}  "
                 f"likely_flipped: {ax['likely_flipped']}")
    lines.append("")
    lines.append("-- anti features by |corr| --")
    lines.append(f"  {'name':16} {'corr':>7} {'gap':>8} {'meanH':>8} {'meanA':>8} {'finite':>7}")
    for s in report["anti_features_ranked"][:12]:
        lines.append(f"  {s['name']:16} {s['corr']:7.3f} {s['gap']:8.3f} "
                     f"{s['mean_H']:8.3f} {s['mean_A']:8.3f} {s['finite_frac']:7.2f}")
    lines.append("")
    lines.append("-- scores (public / provided GT) --")
    lines.append(f"  {'model':22} {'spellF1':>8} {'macroF1':>8}")
    for name, sc in report["scores"].items():
        lines.append(f"  {name:22} {sc['spell_f1']:8.4f} {sc['macro_f1']:8.4f}")
    lines.append("")
    lines.append("-- H/A confusion (GT H/A frames in window) --")
    lines.append(f"  accuracy: {report['confusion_HA']['accuracy_HA']}")
    lines.append("")
    lines.append("-- timeline --")
    lines.append(report["timeline"])
    lines.append("")
    lines.append("-- verdict --")
    for note in report["verdict"]:
        lines.append(f"  * {note}")
    return "\n".join(lines)

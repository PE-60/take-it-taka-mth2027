"""Why a submission loses points: fragmentation and timing, taken apart.

Uses the organisers' own functions (possession_metrics.py) for runs, spells and transitions, so
every number here is counted the way the scorer counts it. Reads the ground truth for inspection
only; nothing is fitted.

analyse(...) returns four tables:
  spells       one row per true possession: how it was predicted and why it was or was not matched
  transitions  one row per true change of label: the nearest predicted change and the signed offset
  intrusions   one row per predicted stretch of another label inside a true possession, with the
               touch that caused it
  gains        one row per gain the model made in the graded window: right or wrong, with its evidence
report_html(...) draws them on one page: overview timeline, offsets, evidence.
"""
import html
import numpy as np
import pandas as pd

NAMES = "HADX"


def load_codes(path):
    lab = pd.read_csv(path)["label"].astype(str).str.strip().str.upper()
    return lab.map({c: i for i, c in enumerate(NAMES)}).fillna(3).to_numpy(int)


def _kind(frm, to):
    return "to_dead" if to == 2 else ("restart" if frm == 2 else "turnover")


def analyse(pm, gt, pred, touches=None, track=None, fps=25, window=75):
    runs = pm.valid_runs(gt); filled = pm.fill_abstentions(pred, runs)
    gt_sp = pm.possession_spells(gt, runs); pr_sp = pm.possession_spells(filled, runs)
    pr_seg = pm.segments(filled, runs)
    ov = pm.spell_overlaps(gt_sp, pr_sp)

    # ---- spells
    rows = []
    for i, (a, b, k) in enumerate(gt_sp):
        mine = sorted(((iou, j) for iou, gi, j in ov if gi == i), reverse=True)
        best_iou, bj = mine[0] if mine else (0.0, -1)
        other = [(max(a, s), min(b, e), c) for s, e, c in pr_seg if c != k and s < b and e > a]
        wrong = sum(e - s for s, e, _ in other)
        cause = "matched" if best_iou >= 0.5 else "missed" if not mine else "split" if len(mine) >= 2 else "shifted"
        rows.append({"gt_start": a, "gt_end": b - 1, "team": NAMES[k], "duration_s": (b - a) / fps, "pred_spells_inside": len(mine), "best_iou": round(best_iou, 3),
                     "start_offset_f": pr_sp[bj][0] - a if bj >= 0 else None, "end_offset_f": pr_sp[bj][1] - b if bj >= 0 else None,
                     "intrusions": len(other), "wrong_frames": wrong, "wrong_share": round(wrong / (b - a), 3), "cause": cause})
    spells = pd.DataFrame(rows)

    # ---- intrusions, with the touch that opened each one
    gains = touches[touches.kind == "gain"] if touches is not None else None
    irows = []
    for a, b, k in gt_sp:
        for s, e, c in pr_seg:
            if c == k or s >= b or e <= a:
                continue
            s2, e2 = max(a, s), min(b, e)
            r = {"gt_start": a, "gt_team": NAMES[k], "start": s2, "end": e2 - 1, "duration_s": (e2 - s2) / fps, "pred": NAMES[c],
                 "position": "head" if s2 == a else "tail" if e2 == b else "inside"}
            if gains is not None and c in (0, 1):
                g = gains[(gains.frame_start >= s - 2) & (gains.frame_start <= e) & (gains.team == NAMES[c])]
                if len(g):
                    t = g.iloc[0]; r.update({"touch_frame": int(t.frame_start), "slot": t.slot, "dv_ms": round(t.dv_ms, 1), "touch_s": round(t.duration_s, 2), "duel": bool(t.duel), "p": round(t.p, 2)})
            irows.append(r)
    intr = pd.DataFrame(irows)

    # ---- transitions: nearest predicted change to the same label, signed offset (negative = predicted early)
    gt_tr = pm.transitions(gt, runs); pr_tr = pm.transitions(filled, runs)
    trows = []
    for f, frm, to in gt_tr:
        cand = [p for p in pr_tr if p[2] == to and abs(p[0] - f) <= window]
        best = min(cand, key=lambda p: abs(p[0] - f)) if cand else None
        r = {"gt_frame": f, "change": f"{NAMES[frm]}>{NAMES[to]}", "kind": _kind(frm, to), "pred_frame": best[0] if best else None, "offset_f": best[0] - f if best else None,
             "within_12": bool(best and abs(best[0] - f) <= 12), "within_25": bool(best and abs(best[0] - f) <= 25)}
        if touches is not None and to in (0, 1):
            near = touches[(touches.team == NAMES[to]) & (touches.frame_end >= f - window) & (touches.frame_start <= f + window)]
            if len(near):
                t = near.iloc[(near.frame_start - f).abs().argsort().iloc[0]]
                r.update({"touch_frame": int(t.frame_start), "touch_end": int(t.frame_end), "touch_offset_f": int(t.frame_start - f), "touch_end_offset_f": int(t.frame_end - f),
                          "touch_kind": t.kind, "dv_ms": round(t.dv_ms, 1), "touch_s": round(t.duration_s, 2), "p": round(t.p, 2)})
        if track is not None:
            r["ball_at_gt"] = track.source.iloc[f] if f < len(track) else ""
        trows.append(r)
    trans = pd.DataFrame(trows)

    # ---- every gain the model made in the graded window: does the truth agree half a second later?
    grows = []
    if gains is not None:
        for t in gains.itertuples():
            if not any(a <= t.frame_start < b for a, b in runs):
                continue
            chk = min(int(t.frame_end) + 12, len(gt) - 1)
            g = gt[chk]
            grows.append({"frame": int(t.frame_start), "slot": t.slot, "team": t.team, "dv_ms": round(t.dv_ms, 1), "touch_s": round(t.duration_s, 2), "duel": bool(t.duel), "p": round(t.p, 2),
                          "gt_after": NAMES[g], "verdict": "right" if NAMES[g] == t.team else "dead" if g == 2 else "wrong"})
    gain_tab = pd.DataFrame(grows)
    return {"spells": spells, "transitions": trans, "intrusions": intr, "gains": gain_tab, "runs": runs, "filled": filled,
            "counts": {"gt_spells": len(gt_sp), "pred_spells": len(pr_sp), "gt_segments": len(pm.segments(gt, runs)), "pred_segments": len(pr_seg), "gt_transitions": len(gt_tr), "pred_transitions": len(pr_tr)}}


def summary_lines(name, res):
    """the statistics, as text someone can paste"""
    sp, tr, it, ga, c = res["spells"], res["transitions"], res["intrusions"], res["gains"], res["counts"]
    out = [f"== {name}: {c['pred_spells']} predicted possessions for {c['gt_spells']} true; segments {c['pred_segments']} / {c['gt_segments']}; transitions {c['pred_transitions']} / {c['gt_transitions']}"]
    out.append("true possessions by outcome: " + ", ".join(f"{k} {v}" for k, v in sp.cause.value_counts().items()))
    if len(it):
        out.append(f"intrusions (another label predicted inside a true possession): {len(it)}, by position " + ", ".join(f"{k} {v}" for k, v in it.position.value_counts().items())
                   + "; by predicted label " + ", ".join(f"{k} {v}" for k, v in it.pred.value_counts().items()))
        d = it.duration_s
        out.append(f"  duration: median {d.median():.1f} s, quartiles {d.quantile(.25):.1f} to {d.quantile(.75):.1f} s, under 1 s {int((d < 1).sum())}, 1 to 3 s {int(((d >= 1) & (d < 3)).sum())}, over 3 s {int((d >= 3).sum())}")
        ins = it[it.position == "inside"]
        if len(ins):
            out.append(f"  inside (these split a possession): {len(ins)}, median {ins.duration_s.median():.1f} s; " + ("with a touch: dv median %.1f m/s, touch median %.2f s, duel %d of %d" % (ins.dv_ms.median(), ins.touch_s.median(), int(ins.duel.sum()), int(ins.dv_ms.notna().sum())) if "dv_ms" in ins and ins.dv_ms.notna().any() else "no gaining touch found"))
    for kind in ("turnover", "restart", "to_dead"):
        t = tr[tr.kind == kind]
        if not len(t):
            continue
        o = t.offset_f.dropna()
        line = f"transitions {kind}: {len(t)} true, {int(t.within_12.sum())} within 12 f, {int(t.within_25.sum())} within 25 f, {int(t.offset_f.isna().sum())} with no predicted change within 3 s"
        if len(o):
            line += f"; offset median {o.median():+.0f} f, quartiles {o.quantile(.25):+.0f} to {o.quantile(.75):+.0f} (negative = predicted early)"
        out.append(line)
        if "touch_offset_f" in t and t.touch_offset_f.notna().any():
            out.append(f"  nearest touch of the gaining team: starts {t.touch_offset_f.median():+.0f} f, ends {t.touch_end_offset_f.median():+.0f} f relative to the true change (medians); kinds " + ", ".join(f"{k} {v}" for k, v in t.touch_kind.value_counts().items()))
    if len(ga):
        out.append("gains made in the graded window: " + ", ".join(f"{k} {v}" for k, v in ga.verdict.value_counts().items()))
        for v in ("right", "wrong"):
            g = ga[ga.verdict == v]
            if len(g):
                out.append(f"  {v}: dv median {g.dv_ms.median():.1f} m/s, touch median {g.touch_s.median():.2f} s, p median {g.p.median():.2f}, duel {int(g.duel.sum())} of {len(g)}")
    return out


# ---------------------------------------------------------------- the page
COL = {"H": "#2e7dd1", "A": "#d1452e", "D": "#8a8f98", "X": "#e8e8e8"}
KIND_COL = {"gain": "#111", "keep": "#6b9e3f", "taker": "#b07cc6", "passed_by": "#c9c9c9"}
SRC_COL = {"seen": "#3c9d5b", "flight": "#e0a03c", "held": "#2e7dd1", "interp": "#b9d98a", "unknown": "#dddddd"}


def _strip(codes, a, b, y, h, scale, label, title=""):
    out = [f'<text x="-8" y="{y + h * 0.7:.0f}" text-anchor="end" class="lab">{html.escape(label)}</text>']
    seg = codes[a:b]; cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1; bounds = np.concatenate([[0], cut, [len(seg)]])
    for i in range(len(bounds) - 1):
        s, e = int(bounds[i]), int(bounds[i + 1]); c = NAMES[seg[s]]
        out.append(f'<rect x="{s * scale:.1f}" y="{y}" width="{max((e - s) * scale, 0.6):.1f}" height="{h}" fill="{COL[c]}"><title>{label}: {c}, frames {a + s} to {a + e - 1}, {(e - s) / 25:.1f} s</title></rect>')
    return out


def _hist(values, lo, hi, step, w=420, h=110, title=""):
    edges = np.arange(lo, hi + step, step); counts, _ = np.histogram(np.clip(values, lo, hi - 1e-9), edges)
    m = max(counts.max(), 1); bw = w / len(counts)
    out = [f'<svg width="{w + 40}" height="{h + 34}" class="hist"><g transform="translate(30,6)">']
    for i, c in enumerate(counts):
        bh = h * c / m
        out.append(f'<rect x="{i * bw + 1:.1f}" y="{h - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" fill="{"#d1452e" if edges[i] < 0 else "#2e7dd1"}"><title>{edges[i]:+.0f} to {edges[i + 1]:+.0f} frames: {c}</title></rect>')
        if c:
            out.append(f'<text x="{i * bw + bw / 2:.1f}" y="{h - bh - 3:.1f}" text-anchor="middle" class="tick">{c}</text>')
    zero = (0 - lo) / (hi - lo) * w
    out.append(f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="0" y2="{h}" stroke="#111" stroke-dasharray="3 3"/>')
    for v in (lo, 0, hi):
        out.append(f'<text x="{(v - lo) / (hi - lo) * w:.1f}" y="{h + 14}" text-anchor="middle" class="tick">{v:+d}</text>')
    out.append(f'<text x="{w / 2}" y="{h + 28}" text-anchor="middle" class="tick">{html.escape(title)}</text></g></svg>')
    return "".join(out)


def report_html(gt, results, touches=None, track=None, fps=25, px_per_s=6.0):
    """results: {name: analyse(...) result}. One section per graded run, the same x axis for every row."""
    first = next(iter(results.values())); runs = first["runs"]; scale = px_per_s / fps
    parts = ["""<!doctype html><meta charset="utf-8"><title>Possession diagnosis</title><style>
body{font:13px/1.45 system-ui,sans-serif;margin:20px;color:#1c1c1c;background:#fff}h1{font-size:18px}h2{font-size:15px;margin-top:26px}
.wrap{overflow-x:auto;border:1px solid #ddd;padding:8px 8px 8px 130px}.lab{font-size:11px;fill:#333}.tick{font-size:10px;fill:#555}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ddd;padding:3px 7px;text-align:right;font-variant-numeric:tabular-nums}th{background:#f4f4f4}td:first-child,th:first-child{text-align:left}
.key span{display:inline-block;margin-right:14px}.key i{display:inline-block;width:11px;height:11px;margin-right:4px;vertical-align:-1px}.note{color:#555;max-width:900px}
.bad{background:#fde8e4}.ok{background:#e7f3e4}</style>
<h1>Possession diagnosis: where the possessions break and where the changes land</h1>
<p class="note">Every strip shares one time axis. Hover any mark for frames and values, then seek to that frame in the viewer. Counting follows the organisers' scorer: X is forward-filled, possessions are runs of H or A.</p>
<div class="key">""" + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in (("home", COL["H"]), ("away", COL["A"]), ("dead", COL["D"]), ("touch: gain", KIND_COL["gain"]), ("keep", KIND_COL["keep"]), ("taker", KIND_COL["taker"]), ("passed by", KIND_COL["passed_by"]))) + "</div>"]
    for a, b in runs:
        W = (b - a) * scale; y = 18; rows = [f'<g>']
        for s in range(0, (b - a) // fps + 1, 10):
            rows.append(f'<line x1="{s * px_per_s:.0f}" x2="{s * px_per_s:.0f}" y1="12" y2="2000" stroke="#eee"/><text x="{s * px_per_s:.0f}" y="9" class="tick">{a + s * fps}</text>')
        rows += _strip(gt, a, b, y, 16, scale, "ground truth"); y += 20
        for name, res in results.items():
            rows += _strip(res["filled"], a, b, y, 16, scale, name); y += 18
            tr = res["transitions"]; tr = tr[(tr.gt_frame >= a) & (tr.gt_frame < b)]
            rows.append(f'<text x="-8" y="{y + 9}" text-anchor="end" class="lab">change offset</text>')
            for t in tr.itertuples():
                x0 = (t.gt_frame - a) * scale
                if pd.isna(t.offset_f):
                    rows.append(f'<text x="{x0:.1f}" y="{y + 10}" text-anchor="middle" fill="#d1452e" class="tick">×<title>{t.change} at {t.gt_frame}: no predicted change within 3 s</title></text>')
                else:
                    x1 = x0 + t.offset_f * scale; col = "#3c9d5b" if abs(t.offset_f) <= 12 else "#e0a03c" if abs(t.offset_f) <= 25 else "#d1452e"
                    rows.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y + 6}" y2="{y + 6}" stroke="{col}" stroke-width="3"><title>{t.change} at {t.gt_frame}: predicted at {int(t.pred_frame)}, {int(t.offset_f):+d} frames</title></line><circle cx="{x0:.1f}" cy="{y + 6}" r="2.5" fill="#111"/>')
            y += 18
        if touches is not None:
            rows.append(f'<text x="-8" y="{y + 20}" text-anchor="end" class="lab">touches (height = p)</text>')
            tt = touches[(touches.frame_end >= a) & (touches.frame_start < b)]
            for t in tt.itertuples():
                f0c, f1c = max(t.frame_start, a), min(t.frame_end, b - 1)                 # drawn inside the window only
                x = (f0c - a) * scale; w = max((f1c - f0c + 1) * scale, 1.5); hh = 4 + 24 * float(t.p)
                rows.append(f'<rect x="{x:.1f}" y="{y + 30 - hh:.1f}" width="{w:.1f}" height="{hh:.1f}" fill="{KIND_COL.get(t.kind, "#999")}" stroke="{COL[t.team]}" stroke-width="1.2"><title>{t.kind} by {t.team} #{t.slot}, frames {t.frame_start} to {t.frame_end}: dv {t.dv_ms:.1f} m/s, {t.duration_s:.2f} s, p {t.p:.2f}{", duel" if t.duel else ""}</title></rect>')
            y += 36
        if track is not None:
            rows.append(f'<text x="-8" y="{y + 9}" text-anchor="end" class="lab">ball source</text>')
            src = track.source.to_numpy()[a:b]; cut = np.flatnonzero(src[1:] != src[:-1]) + 1; bounds = np.concatenate([[0], cut, [len(src)]])
            for i in range(len(bounds) - 1):
                s, e = int(bounds[i]), int(bounds[i + 1])
                rows.append(f'<rect x="{s * scale:.1f}" y="{y}" width="{max((e - s) * scale, 0.6):.1f}" height="10" fill="{SRC_COL.get(src[s], "#999")}"><title>ball {src[s]}, frames {a + s} to {a + e - 1}</title></rect>')
            y += 16
        rows.append("</g>")
        parts.append(f'<h2>Frames {a} to {b - 1}</h2><div class="wrap"><svg width="{W + 20:.0f}" height="{y + 10}" style="overflow:visible">' + "".join(rows) + "</svg></div>")
        if track is not None:
            parts.append('<div class="key" style="margin-top:4px">ball source: ' + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in SRC_COL.items()) + "</div>")
    for name, res in results.items():
        parts.append(f"<h2>{html.escape(name)}</h2><pre>" + html.escape("\n".join(summary_lines(name, res)[1:])) + "</pre>")
        tr = res["transitions"]
        for kind in ("turnover", "restart", "to_dead"):
            o = tr[tr.kind == kind].offset_f.dropna().to_numpy()
            if len(o):
                parts.append(_hist(o, -75, 75, 10, title=f"{kind}: predicted minus true, frames"))
        for key, cols in (("spells", None), ("intrusions", None), ("transitions", None)):
            df = res[key]
            if len(df):
                parts.append(f"<h3>{key}</h3>" + df.to_html(index=False, na_rep="", border=0, float_format=lambda v: f"{v:g}"))
    return "\n".join(parts)

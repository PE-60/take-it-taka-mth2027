#!/usr/bin/env python3
"""
match_viewer.py: see what the data sees, on the tactical or the broadcast video.

Rendered video (same command shape for both sources; --mode gt exists for tactical only):
    python match_viewer.py --mode gt                          # tactical, public 5-minute window
    python match_viewer.py --mode full                        # tactical, whole match
    python match_viewer.py --source broadcast --mode full     # broadcast, whole match

Interactive viewer (match_viewer.html + viewer/ next to this script):
    python match_viewer.py --mode gt --export --serve
    python match_viewer.py --mode full --export --serve
    python match_viewer.py --source broadcast --mode full --export --serve
    python match_viewer.py --serve                            # reopen without recomputing

AR flag overlays (optional): put export/referee_signals_tactical.json next to the repo
(or pass --referee-signals PATH). Built by: python -m referee_signals --video ... --out ...
    
Data layout (the repository's): data/tactical/ and data/broadcast/ next to this script.
`--inspect` prints how the ATD is parsed.  See MATCH_VIEWER.md.
"""
import argparse
import glob
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

# ----------------------------------------------------------------------------
# self-install
# ----------------------------------------------------------------------------
REQUIRED = {"cv2": "opencv-python", "pandas": "pandas", "numpy": "numpy", "PIL": "pillow"}


def ensure_packages():
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"installing missing packages: {', '.join(missing)}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--user", *missing])
    if r.returncode != 0:
        r = subprocess.run([sys.executable, "-m", "pip", "install", *missing])
    if r.returncode != 0:
        sys.exit("automatic install failed, run:  pip install --user " + " ".join(missing))
    importlib.invalidate_caches()


ensure_packages()
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# where to look for the data: this folder, the folder above it (the zip unpacks into a subfolder,
# so the data may sit one level up), and MTAH_DATA_DIR if set
SEARCH_ROOTS = [HERE, os.path.dirname(HERE)] + ([os.environ["MTAH_DATA_DIR"]] if os.environ.get("MTAH_DATA_DIR") else [])
PITCH_M = (105.3, 68.0)                      # metres, from games.py
HALVES = {1: (11185, 90651), 2: (110808, 190197)}   # tactical graded blocks (replaced for broadcast)
HALVES_EST = False
FPS = 25
BALL_RADIUS_M = 0.11                          # a size 5 ball is about 22 cm across
BOLT_MS = 12.4                                # Usain Bolt's peak speed, m/s

# colour system (BGR). One hue per team for every team related mark.
# ---- colour system (BGR). One hue per team for every team-related mark.
HOME, AWAY = (70, 70, 235), (235, 170, 50)
DEAD, NONE = (190, 190, 190), (110, 110, 110)
WHITE, BLACK, MUTED = (255, 255, 255), (0, 0, 0), (170, 170, 170)
GOOD, WARN, BAD = (90, 220, 90), (60, 215, 245), (60, 60, 240)
RING = (0, 230, 255)                          # ball radius ring, yellow
EV_RESTART, EV_LOST, EV_BOLT = (90, 220, 90), (60, 60, 240), (230, 80, 230)
EV_KICK, EV_JUMP, EV_CHURN, EV_AIR = (0, 200, 255), (255, 80, 200), (200, 140, 255), (255, 220, 120)
# set-piece types that end a dead ball (SET PIECES tag, Type group), each with its own hue
RESTART_TYPES = {"throw": ("Throw-In", (255, 200, 60), "diamond"), "corner": ("Corner", (60, 200, 255), "circle"),
                 "goal kick": ("Goal Kick", (200, 120, 255), "square"), "free": ("Free Kick", (90, 220, 90), "tri_down"),
                 "kick off": ("Kick-Off", (255, 255, 255), "star"), "penalty": ("Penalty", (120, 60, 255), "cross")}
GT_ALIVE_GLYPH = ("GT dead→alive", EV_RESTART, "tri_up")


def draw_glyph(img, kind, cx, cy, r, color):
    """Small filled marker shapes used on the restart row and in the legend."""
    cx, cy, r = int(cx), int(cy), max(2, int(r))
    if kind == "circle":
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    elif kind == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    elif kind == "diamond":
        cv2.fillPoly(img, [np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)], color, cv2.LINE_AA)
    elif kind == "tri_up":
        cv2.fillPoly(img, [np.array([[cx, cy - r], [cx + r, cy + r], [cx - r, cy + r]], np.int32)], color, cv2.LINE_AA)
    elif kind == "tri_down":
        cv2.fillPoly(img, [np.array([[cx, cy + r], [cx + r, cy - r], [cx - r, cy - r]], np.int32)], color, cv2.LINE_AA)
    elif kind == "star":
        t = np.linspace(-np.pi / 2, 1.5 * np.pi, 11)[:-1]
        rr = np.where(np.arange(10) % 2 == 0, r * 1.3, r * 0.55)
        cv2.fillPoly(img, [np.c_[cx + rr * np.cos(t), cy + rr * np.sin(t)].astype(np.int32)], color, cv2.LINE_AA)
    elif kind == "cross":
        cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 2 * r + 2, 2, cv2.LINE_AA)


def restart_type(e):
    """-> (name, colour, glyph) for a SET PIECES tag"""
    t = (e["tags"].get("Type") or "").lower()
    for key, (name, col, glyph) in RESTART_TYPES.items():
        if key in t:
            return name, col, glyph
    return (e["tags"].get("Type") or "set piece"), EV_RESTART, "circle"
PANEL = (18, 18, 18)
LABEL_NAME = {"H": "HOME  Barcelona", "A": "AWAY  Athletic", "D": "DEAD BALL", "X": "no label"}
LABEL_COLOR = {"H": HOME, "A": AWAY, "D": DEAD, "X": NONE}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------------
# file discovery
# ----------------------------------------------------------------------------
def find_file(pattern, required=True, source=None):
    """Prefer data/<source>/, then anything under the search roots (shallowest first)."""
    hits = []
    if source:
        hits += [h for root in SEARCH_ROOTS for h in glob.glob(os.path.join(root, "data", source, pattern)) if os.path.isfile(h)]
    hits += [h for root in SEARCH_ROOTS
             for h in glob.glob(os.path.join(root, "**", pattern), recursive=True) if os.path.isfile(h)]
    if hits:
        hits = sorted(set(hits), key=lambda h: (0 if f"{os.sep}data{os.sep}" in h else 1, len(h)))
        return hits[0]
    if required:
        sys.exit(f"could not find '{pattern}'.\n  looked under: " + "\n                ".join(SEARCH_ROOTS) +
                 "\n  expected layout:  data/tactical/BAR-ATH_tactical.mp4 ...   data/broadcast/BAR-ATH_broadcast.mp4 ..."
                 "\n  the data folder may be next to this script or one level above it; loose files anywhere below those work too.")
    return None


# ----------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------
def load_labels(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "frame" not in df.columns or "label" not in df.columns:
        sys.exit(f"{path}: expected columns 'frame,label', got {list(df.columns)}")
    return dict(zip(df["frame"].astype(int), df["label"].astype(str).str.strip().str.upper()))


def labelled_window(gt_path, gt):
    meta = os.path.splitext(gt_path)[0] + "_meta.json"
    if os.path.exists(meta):
        with open(meta) as f:
            w = json.load(f).get("public_window", {})
        if "first_video_frame" in w:
            return int(w["first_video_frame"]), int(w["last_video_frame"])
    real = [f for f, l in gt.items() if l != "X"]
    if not real:
        sys.exit("ground truth has no non-X labels")
    return min(real), max(real)


def half_and_clock(f):
    if not HALVES:
        return "", "--:--"
    for h, (a, b) in HALVES.items():
        if a <= f <= b:
            t = (f - a) / FPS + (45 * 60 if h == 2 else 0)
            return f"{h}{'st' if h == 1 else 'nd'} HALF", f"{int(t // 60):02d}:{int(t % 60):02d}"
    if f < HALVES[1][0]:
        return "PRE-MATCH", "--:--"
    if f < HALVES[2][0]:
        return "HALFTIME", "--:--"
    return "FULL TIME", "--:--"


# ----------------------------------------------------------------------------
# homography
# ----------------------------------------------------------------------------
def load_homography(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.set_index("frame")
    H = df[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy(dtype=float).reshape(-1, 3, 3)
    return dict(zip(df.index.astype(int), range(len(df)))), H, \
        df["reprojection_error"].to_numpy(dtype=float), df["valid"].to_numpy(dtype=int)


def project(Hm, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def good_px(px):
    return np.all(np.isfinite(px)) and np.all(np.abs(px) < 1e5)


_t = np.linspace(0, 2 * np.pi, 48)
PITCH_LINES = [np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float),
               np.array([[0.5, 0], [0.5, 1]], float),
               np.c_[0.5 + (9.15 / PITCH_M[0]) * np.cos(_t), 0.5 + (9.15 / PITCH_M[1]) * np.sin(_t)]]
for depth, width in ((16.5, 40.32), (5.5, 18.32)):
    d, hw = depth / PITCH_M[0], (width / 2) / PITCH_M[1]
    PITCH_LINES.append(np.array([[0, 0.5 - hw], [d, 0.5 - hw], [d, 0.5 + hw], [0, 0.5 + hw]], float))
    PITCH_LINES.append(np.array([[1, 0.5 - hw], [1 - d, 0.5 - hw], [1 - d, 0.5 + hw], [1, 0.5 + hw]], float))


def draw_pitch(img, Hm, s, color, thick):
    for line in PITCH_LINES:
        px = project(Hm, line) * s
        if good_px(px):
            cv2.polylines(img, [px.astype(np.int32).reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# ATD: Metrica FIFA/EPTS raw text + metadata XML
# ----------------------------------------------------------------------------
def _tag(el):
    return el.tag.split("}", 1)[-1]


def parse_atd_meta(xml_path):
    root = ET.parse(xml_path).getroot()
    teams, players, channels, spec, attack = [], {}, {}, None, {}

    def walk(el):
        nonlocal spec
        t = _tag(el)
        if t == "Team":
            name = next((c.text for c in el if _tag(c) == "Name"), None)
            tid = el.get("id") or el.get("teamId")
            teams.append((tid, (name or "").strip()))
            for pp in el.iter():
                if _tag(pp) == "ProviderParameter":
                    pn = next((c.text for c in pp if _tag(c) == "Name"), "")
                    pv = next((c.text for c in pp if _tag(c) == "Value"), "")
                    if (pn or "").strip() == "attack_direction_first_half":
                        attack[tid] = (pv or "").strip()
        elif t == "Player":
            pid = el.get("id") or el.get("playerId")
            name, shirt = pid, None
            for c in el:
                if _tag(c) == "Name":
                    name = (c.text or "").strip()
                elif _tag(c) == "ShirtNumber":
                    shirt = (c.text or "").strip()
            players[pid] = {"name": name, "team": el.get("teamId"), "slot": shirt}
        elif t == "PlayerChannel":
            channels[el.get("id")] = (el.get("playerId"), el.get("channelId"))
        elif t == "DataFormatSpecification":
            spec = el
            return
        for c in el:
            walk(c)

    walk(root)
    if teams:
        print("teams in ATD metadata:", " | ".join(
            f"{n} ({'H' if i == 0 else 'A'}, 1st half {attack.get(tid, '?')})" for i, (tid, n) in enumerate(teams)))
    home_id = teams[0][0] if teams else None
    hd = attack.get(home_id, "")
    home_dir = 1 if hd == "left_to_right" else (-1 if hd == "right_to_left" else 0)
    return players, home_id, channels, spec, home_dir


def ball_channels(spec):
    return [el.get("channelId") for el in spec.iter() if _tag(el) == "BallChannelRef"] if spec is not None else []


def build_decoder(spec, channels):
    def compile_el(el):
        t = _tag(el)
        if t == "SplitRegister":
            return ("split", el.get("separator"), [compile_el(c) for c in el], None)
        if t == "StringRegister":
            return ("str", None, None, el.get("name"))
        if t == "PlayerChannelRef":
            return ("pch", None, None, channels.get(el.get("playerChannelId"), (None, None)))
        if t == "BallChannelRef":
            return ("bch", None, None, el.get("channelId"))
        return ("skip", None, None, None)

    # the top-level separator lives on the DataFormatSpecification element itself (e.g. ':')
    tree = ("split", spec.get("separator"), [compile_el(c) for c in spec], None)

    def fill(node, text, out):
        kind, sep, kids, ref = node
        if kind == "split":
            for kid, part in zip(kids, text.split(sep) if sep else [text]):
                fill(kid, part, out)
        elif kind == "str":
            out["str"][ref] = text
        elif kind == "pch":
            pid, ch = ref
            if pid is not None and ch in ("x", "y", "s"):
                try:
                    out["p"].setdefault(pid, {})[ch] = float(text)
                except ValueError:
                    pass
        elif kind == "bch" and ref:
            try:
                out["b"][ref] = float(text)
            except ValueError:
                pass

    def decode(line):
        out = {"str": {}, "p": {}, "b": {}}
        fill(tree, line.rstrip("\r\n"), out)
        pl = {pid: (v["x"], v["y"], v.get("s", np.nan)) for pid, v in out["p"].items()
              if "x" in v and "y" in v and np.isfinite(v["x"]) and np.isfinite(v["y"])}
        b = out["b"]
        ball = None
        if "x" in b and "y" in b and np.isfinite(b["x"]) and np.isfinite(b["y"]):
            # any channel that looks like a speed: 's', 'speed', 'v'
            sp = next((b[k] for k in b if k and k.lower() in ("s", "speed", "v", "velocity")), np.nan)
            z = next((b[k] for k in b if k and k.lower() in ("z", "height", "h")), np.nan)
            ball = (b["x"], b["y"], sp, z)
        return pl, ball

    return decode


_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|NaN")


def heuristic_decode(line):
    parts = line.rstrip("\r\n").split(":")
    nums = _NUM.findall(parts[-1])
    ball = None
    if len(nums) >= 2:
        x, y = float(nums[0]), float(nums[1])
        if np.isfinite(x) and np.isfinite(y):
            ball = (x, y, np.nan, np.nan)
    pl, k = {}, 0
    for chunk in parts[1:-1]:
        for ent in chunk.split(";"):
            k += 1
            n = _NUM.findall(ent)
            if len(n) >= 2:
                x, y = float(n[0]), float(n[1])
                if np.isfinite(x) and np.isfinite(y):
                    pl[f"T{k}"] = (x, y, float(n[2]) if len(n) > 2 else np.nan)
    return pl, ball


def spec_summary(el, depth=0, out=None):
    out = [] if out is None else out
    t = _tag(el)
    kids = list(el)
    if t == "SplitRegister":
        kinds = {}
        for c in kids:
            kinds[_tag(c)] = kinds.get(_tag(c), 0) + 1
        out.append("  " * depth + f"SplitRegister sep={el.get('separator')!r}  children={len(kids)} {kinds}")
        seen = set()
        for c in kids:
            if _tag(c) not in seen:
                seen.add(_tag(c))
                spec_summary(c, depth + 1, out)
    elif t == "StringRegister":
        out.append("  " * depth + f"StringRegister name={el.get('name')!r}")
    elif t == "PlayerChannelRef":
        out.append("  " * depth + f"PlayerChannelRef playerChannelId={el.get('playerChannelId')!r}")
    elif t == "BallChannelRef":
        out.append("  " * depth + f"BallChannelRef channelId={el.get('channelId')!r}")
    else:
        out.append("  " * depth + f"<{t}> attrs={dict(el.attrib)}")
        for c in kids:
            spec_summary(c, depth + 1, out)
    return out


def inspect_atd(raw_path, decode, players_meta, channels, spec):
    print("\n=== ATD INSPECT ===")
    print(f"track slots in xml: {len(players_meta)}   player channels: {len(channels)}   ball channels: {ball_channels(spec)}")
    if players_meta:
        pid, m = next(iter(players_meta.items()))
        print(f"first player: id={pid!r} meta={m}")
    if spec is not None:
        print("DataFormatSpecification:")
        for line in spec_summary(spec):
            print("   ", line)
    with open(raw_path) as fh:
        for n, line in enumerate(fh):
            if n >= 2:
                break
            line = line.rstrip("\r\n")
            print(f"raw line {n}: {len(line)} chars")
            print("   ", line[:200] + (" ..." if len(line) > 200 else ""))
            pl, b = decode(line)
            print(f"    decoded -> {len(pl)} players, ball={b}")
    print("=== end inspect ===\n")


def load_atd(raw_path, meta_path, first, last, inspect=False):
    players_meta, home, decode, home_dir = {}, None, heuristic_decode, 0
    channels, spec = {}, None
    if meta_path:
        try:
            players_meta, home, channels, spec, home_dir = parse_atd_meta(meta_path)
            if spec is not None and channels:
                decode = build_decoder(spec, channels)
                print(f"ATD metadata: {len(players_meta)} track slots, format spec parsed, ball channels {ball_channels(spec)}")
            else:
                print("ATD metadata: no DataFormatSpecification, heuristic parser")
        except Exception as e:  # noqa: BLE001
            print(f"could not parse ATD xml ({e}), heuristic parser")
    if inspect:
        inspect_atd(raw_path, decode, players_meta, channels, spec)
    if decode is not heuristic_decode:
        got = 0
        with open(raw_path) as fh:
            for n, line in enumerate(fh):
                if n >= 2000:
                    break
                pl, b = decode(line)
                got += len(pl) + (b is not None)
        if got == 0:
            print("warning: spec-based decoder produced no positions on the first 2000 lines, "
                  "falling back to the heuristic parser (teams unknown). Run with --inspect and share the output.")
            decode = heuristic_decode
            players_meta = {}

    n = last - first + 1
    ids = list(players_meta) if (players_meta and decode is not heuristic_decode) else [f"T{k}" for k in range(1, 61)]
    col = {pid: k for k, pid in enumerate(ids)}
    xy = np.full((n, len(ids), 3), np.nan, dtype=np.float32)      # x, y, data speed
    ball = np.full((n, 4), np.nan, dtype=np.float32)               # x, y, data speed, z
    seen = 0
    with open(raw_path) as fh:
        for line in fh:
            sep = line.find(":")
            if sep < 0:
                continue
            try:
                vf = int(line[:sep]) - 1
            except ValueError:
                continue
            if vf < first:
                continue
            if vf > last:
                break
            pl, b = decode(line)
            r = vf - first
            seen += 1
            for pid, p in pl.items():
                k = col.get(pid)
                if k is not None:
                    xy[r, k] = p
            if b is not None:
                ball[r] = b
    n_ball = int(np.isfinite(ball[:, 0]).sum())
    per = np.isfinite(xy[:, :, 0]).sum(axis=1)
    print(f"ATD: {seen} frames in range, ball on {n_ball} ({100 * n_ball / max(1, seen):.1f}%), "
          f"mean {per[per > 0].mean() if (per > 0).any() else 0:.1f} players/frame")
    fin = xy[:, :, :2][np.isfinite(xy[:, :, :2])]
    if fin.size:
        frac_out = float(np.mean((fin < -0.05) | (fin > 1.05)))
        if frac_out > 0.25:
            print(f"warning: {100 * frac_out:.0f}% of coordinates are outside 0..1, the raw format was not parsed as expected")

    # ---- speeds from frames (m/s), 3-frame smoothed, NaN where a position is missing
    def frame_speed(pos):                       # pos: (n, ..., 2)
        d = np.diff(pos, axis=0) * np.array(PITCH_M, dtype=np.float32)
        sp = np.hypot(d[..., 0], d[..., 1]) * FPS
        sp = np.concatenate([np.full_like(sp[:1], np.nan), sp], axis=0)
        k = np.ones(3, dtype=np.float32) / 3
        valid = np.isfinite(sp)
        num = np.apply_along_axis(lambda v: np.convolve(np.nan_to_num(v), k, "same"), 0, sp)
        den = np.apply_along_axis(lambda v: np.convolve(v.astype(np.float32), k, "same"), 0, valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = num / np.maximum(den, 1e-9)
        sm[~valid] = np.nan
        return sm.astype(np.float32)

    bspeed_f = frame_speed(ball[:, :2])
    pspeed_f = frame_speed(xy[:, :, :2])
    data_ball_speed = bool(np.isfinite(ball[:, 2]).any())
    has_z = bool(np.isfinite(ball[:, 3]).any())
    with np.errstate(invalid="ignore"):
        pmax_data = np.nanmax(np.where(np.isfinite(xy[:, :, 2]), xy[:, :, 2], -1), axis=1) if len(ids) else np.zeros(n)

    # ---- data-quality diagnostics, per frame
    M = np.array(PITCH_M, dtype=np.float32)
    dpl = np.diff(xy[:, :, :2], axis=0) * M                       # per-slot displacement in metres per frame
    pjump_m = np.hypot(dpl[..., 0], dpl[..., 1])                 # (n-1, slots)
    pjump_m = np.concatenate([np.full((1, len(ids)), np.nan, np.float32), pjump_m])
    p_jump = np.where(np.isfinite(pjump_m), pjump_m, 0) > 1.0    # > 1 m in 1/25 s  (= 25 m/s)
    dbl = np.diff(ball[:, :2], axis=0) * M
    bjump_m = np.concatenate([[np.nan], np.hypot(dbl[:, 0], dbl[:, 1])])
    b_jump = np.where(np.isfinite(bjump_m), bjump_m, 0) > 1.6    # > 1.6 m/frame (= 40 m/s, beyond any shot)
    present = np.isfinite(xy[:, :, 0])
    count = present.sum(axis=1)
    prev_present = np.concatenate([present[:1], present[:-1]])
    appear_raw = present & ~prev_present
    vanish_raw = ~present & prev_present
    # a slot that is gone for < 0.5 s is a dropout (tracker flicker); longer gaps are real vanish/appear events
    min_gap = int(0.5 * FPS)
    appear = np.zeros_like(appear_raw)
    vanish = np.zeros_like(vanish_raw)
    dropout = np.zeros_like(vanish_raw)
    for k in range(len(ids)):
        col_v = np.flatnonzero(vanish_raw[:, k])
        for v in col_v:
            nxt = np.flatnonzero(present[v:, k])
            gap = int(nxt[0]) if len(nxt) else n - v
            if gap >= min_gap:
                vanish[v, k] = True
                if len(nxt):
                    appear[v + gap, k] = True
            else:
                dropout[v:v + gap, k] = True
        # slots appearing at the start of the range or after a long gap already covered above
        if present[0, k] is False and appear_raw[:, k].any():
            first_app = int(np.flatnonzero(appear_raw[:, k])[0])
            if first_app >= min_gap:
                appear[first_app, k] = True
    off = present & ((xy[:, :, 0] < -0.02) | (xy[:, :, 0] > 1.02) | (xy[:, :, 1] < -0.02) | (xy[:, :, 1] > 1.02))
    # ball acceleration from frame speed (Link & Hoernig: a kick is a local max >= 4 m/s^2)
    acc = np.concatenate([[np.nan], np.diff(bspeed_f)]) * FPS
    kick = np.zeros(n, bool)
    a = np.where(np.isfinite(acc), acc, -np.inf)
    for i in range(1, n - 1):
        if a[i] >= 4.0 and a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            kick[i] = True
    # thin kicks: keep one per 0.3 s
    last = -999
    for i in np.flatnonzero(kick):
        if i - last < int(0.3 * FPS):
            kick[i] = False
        else:
            last = i
    print(f"diagnostics: player jumps {int(p_jump.any(axis=1).sum())} frames, ball jumps {int(b_jump.sum())}, "
          f"slots vanishing >=0.5s {int(vanish.sum())} / reappearing {int(appear.sum())}, short dropouts {int(dropout.any(axis=1).sum())} frames, "
          f"frames with != 22 tracked {int((count != 22).sum())}, kicks {int(kick.sum())}, ball z channel: {has_z}")
    return {"first": first, "xy": xy, "ball": ball, "ids": ids, "home_dir": home_dir,
            "bspeed_f": bspeed_f, "pspeed_f": pspeed_f, "pmax_data": pmax_data,
            "has_ball_speed": data_ball_speed, "has_z": has_z,
            "p_jump": p_jump, "pjump_m": pjump_m, "b_jump": b_jump, "bjump_m": bjump_m,
            "count": count, "appear": appear, "vanish": vanish, "dropout": dropout, "off": off, "present": present,
            "acc": acc, "kick": kick}, players_meta, home


def estimate_halves(atd, min_gap_s=120):
    cnt = np.isfinite(atd["xy"][:, :, 0]).sum(axis=1)
    empty = cnt < 5
    best, run, start = (0, 0, 0), 0, 0
    for i, e in enumerate(empty):
        if e:
            if run == 0:
                start = i
            run += 1
            if run > best[0]:
                best = (run, start, i)
        else:
            run = 0
    if best[0] < min_gap_s * FPS:
        return None
    gap_a, gap_b = best[1] + atd["first"], best[2] + atd["first"]
    play = np.flatnonzero(~empty)
    if not len(play):
        return None
    return {1: (int(play[0]) + atd["first"], gap_a - 1), 2: (gap_b + 1, int(play[-1]) + atd["first"])}


def frame_tracking(atd, f):
    """players {pid: (x, y, s_data, s_frames)}, ball (x, y)|None, ball speeds (frames, data), heading"""
    if not atd:
        return {}, None, (np.nan, np.nan), None
    r = f - atd["first"]
    if r < 0 or r >= len(atd["xy"]):
        return {}, None, (np.nan, np.nan), None
    row = atd["xy"][r]
    ok = np.isfinite(row[:, 0]) & np.isfinite(row[:, 1])
    players = {atd["ids"][k]: (float(row[k, 0]), float(row[k, 1]), float(row[k, 2]), float(atd["pspeed_f"][r, k]))
               for k in np.flatnonzero(ok)}
    b = atd["ball"][r]
    ball = (float(b[0]), float(b[1])) if np.isfinite(b[0]) and np.isfinite(b[1]) else None
    prev = atd["ball"][r - 1] if r > 0 else None
    heading = None
    if ball is not None and prev is not None and np.all(np.isfinite(prev[:2])):
        heading = (float(b[0] - prev[0]), float(b[1] - prev[1]))
    return players, ball, (float(atd["bspeed_f"][r]), float(b[2])), heading


def attack_dir(home_dir, f):
    if not home_dir or not HALVES:
        return 0
    return home_dir if f < HALVES[2][0] else -home_dir


def ball_zone(ball, hdir):
    if ball is None or not hdir:
        return None
    x = ball[0] if hdir > 0 else 1 - ball[0]
    if x < 1 / 3:
        return "H defensive / A attacking third", "Hdef"
    if x > 2 / 3:
        return "H attacking / A defensive third", "Hatt"
    return "middle third", "mid"


def metres(a, b):
    return float(np.hypot((a[0] - b[0]) * PITCH_M[0], (a[1] - b[1]) * PITCH_M[1]))


# ----------------------------------------------------------------------------
# possession heuristics, after Link & Hoernig (2017)
#   naive : the nearest player, if within T_P of the ball
#   L&H   : naive gain, kept while the 1 s-ahead ball prediction stays within T_A of the
#           holder's 1 s-ahead position; a kick (accel >= 4 m/s^2) re-opens the gain test
# ----------------------------------------------------------------------------
def compute_possession(atd, meta, home, t_p, t_a=2.5, horizon_s=1.0):
    n = len(atd["ball"])
    xy, ball = atd["xy"], atd["ball"]
    M = np.array(PITCH_M, dtype=np.float32)
    side_of = np.array([team_colour(pid, meta, home)[1] for pid in atd["ids"]])
    naive = np.full(n, "", dtype="<U1")
    naive_pid = np.full(n, -1, int)
    lh = np.full(n, "", dtype="<U1")
    lh_pid = np.full(n, -1, int)
    holder, last_team = -1, ""
    h = int(horizon_s * FPS)
    for r in range(n):
        if not np.isfinite(ball[r, 0]):
            holder = -1
            lh[r] = last_team                         # team is sticky through tracking loss
            continue
        d = np.hypot(*((xy[r, :, :2] - ball[r, :2]) * M).T)
        d = np.where(np.isfinite(d), d, np.inf)
        k = int(np.argmin(d))
        if d[k] <= t_p:
            naive[r], naive_pid[r] = side_of[k], k
        # L&H: does the current holder keep the ball?
        if holder >= 0 and np.isfinite(xy[r, holder, 0]):
            r2 = min(n - 1, r + h)
            if np.isfinite(ball[r2, 0]) and np.isfinite(xy[r2, holder, 0]):
                dist_future = float(np.hypot(*((ball[r2, :2] - xy[r2, holder, :2]) * M)))
            else:
                dist_future = d[holder]
            if dist_future > t_a or (atd["kick"][r] and d[k] <= t_p and k != holder):
                holder = -1
        else:
            holder = -1
        if holder < 0 and d[k] <= t_p:
            holder = k
        if holder >= 0:
            last_team, lh_pid[r] = side_of[holder], holder
        lh[r] = last_team                             # challenge rule: a loose ball stays with the last team
    print(f"possession heuristics: naive covers {100 * (naive != '').mean():.0f}% of frames, "
          f"L&H covers {100 * (lh != '').mean():.0f}%")
    return {"naive": naive, "naive_pid": naive_pid, "lh": lh, "lh_pid": lh_pid}


# ----------------------------------------------------------------------------
# Smart Tagging (SportsCode XML, seconds from video start)
# ----------------------------------------------------------------------------
def tag_side(team_text):
    t = (team_text or "").lower()
    if "barc" in t:
        return "H"
    if "atl" in t or "ath" in t or "bilbao" in t:
        return "A"
    return None


def load_tags(path):
    out = []
    for inst in ET.parse(path).iter("instance"):
        try:
            s0, s1 = float(inst.findtext("start")), float(inst.findtext("end"))
        except (TypeError, ValueError):
            continue
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append({"code": (inst.findtext("code") or "").strip(), "a": int(round(s0 * FPS)),
                    "b": int(round(s1 * FPS)), "tags": tags, "side": tag_side(tags.get("Team"))})
    out.sort(key=lambda e: e["a"])
    print(f"smart tagging: {len(out)} instances")
    return out


def tag_line(e):
    extras = [v for k, v in e["tags"].items() if k not in ("Team", "Half") and v]
    team = e["tags"].get("Team") or "-"
    return f"{e['code']}  ·  {team}" + (f"  ·  {' · '.join(extras)}" if extras else "")


# ----------------------------------------------------------------------------
# presentation layer
#
# Design rules, after Omnioculars (Chen et al., IEEE VIS 2022) and iBall (Chen et al., CHI 2023):
#   * keep the pitch clear: numbers live in a courtside band at the top, overview at the bottom
#   * anything data-bearing on the pitch is anchored to the ground under its player
#   * highlight only the key actors (ball carrier, closest opponent); everyone else is a bare mark
#   * one key stat per on-pitch element; the rest goes to the panels
#   * one hue per team for every team-related mark; data-quality and events use their own hues
# ----------------------------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:                                    # pillow is optional; cv2 text is the fallback
    _PIL = False

FONT_CANDIDATES = {
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Helvetica.ttc"],
    "mono": ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
             "/System/Library/Fonts/Menlo.ttc"],
}


class Text:
    """Queues text and draws it once per frame through Pillow (proper typeface, anti-aliased)."""

    def __init__(self, u):
        self.u = u
        self.q = []
        self.fonts = {}
        self.paths = {}
        for kind, cands in FONT_CANDIDATES.items():
            self.paths[kind] = next((p for p in cands if os.path.exists(p)), None)
        self.ok = _PIL and self.paths["regular"] is not None
        if not self.ok:
            print("note: Pillow or a system font not found, using OpenCV text")

    def font(self, kind, px):
        key = (kind, px)
        if key not in self.fonts:
            p = self.paths.get(kind) or self.paths["regular"]
            self.fonts[key] = ImageFont.truetype(p, px)
        return self.fonts[key]

    def put(self, s, x, y, px, color, kind="regular"):
        """x,y = top-left of the text box, px = font pixel size (already scaled)."""
        self.q.append((s, int(x), int(y), int(px), color, kind))

    def shift(self, dx, dy):
        self.q = [(t, x + dx, y + dy, px, c, k) for t, x, y, px, c, k in self.q]

    def width(self, s, px, kind="regular"):
        if self.ok:
            f = self.font(kind, int(px))
            return f.getlength(s)
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, px / 28, 1)[0][0]

    def flush(self, frame):
        if not self.q:
            return frame
        if self.ok:
            img = Image.fromarray(frame[:, :, ::-1])
            d = ImageDraw.Draw(img)
            for s, x, y, px, color, kind in self.q:
                d.text((x, y), s, font=self.font(kind, px), fill=(color[2], color[1], color[0]))
            frame = np.asarray(img)[:, :, ::-1].copy()
        else:
            for s, x, y, px, color, kind in self.q:
                cv2.putText(frame, s, (x, y + int(px * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, px / 28, color,
                            2 if kind == "bold" else 1, cv2.LINE_AA)
        self.q = []
        return frame


def panel(img, x, y, w, h, alpha=0.72, color=None):
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color or PANEL), alpha, 0, roi)


def fill_poly_alpha(img, pts, color, alpha, outline=None, thick=1):
    pts = np.asarray(pts).astype(np.int32)
    x0, y0 = pts.min(axis=0) - 2
    x1, y1 = pts.max(axis=0) + 3
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.fillPoly(over, [pts - [x0, y0]], color, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, roi)
    if outline is not None:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, outline, thick, cv2.LINE_AA)


def line_alpha(img, a, b, color, alpha, thick=1):
    x0, y0 = max(0, min(a[0], b[0]) - 2), max(0, min(a[1], b[1]) - 2)
    x1, y1 = min(img.shape[1], max(a[0], b[0]) + 3), min(img.shape[0], max(a[1], b[1]) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.line(over, (a[0] - x0, a[1] - y0), (b[0] - x0, b[1] - y0), color, thick, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, roi)


def ground_circle(Hm, centre, radius_m, n=36):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[centre[0] + (radius_m / PITCH_M[0]) * np.cos(t), centre[1] + (radius_m / PITCH_M[1]) * np.sin(t)]
    return project(Hm, pts)


def team_colour(pid, meta, home):
    t = meta.get(pid, {}).get("team")
    if t is None or home is None:
        return MUTED, "?"
    return (HOME, "H") if t == home else (AWAY, "A")


def slot_label(pid, meta):
    m = meta.get(pid)
    if not m:
        return pid.lstrip("T")
    return m.get("slot") or re.sub(r"\D", "", m.get("name", "")) or pid


class Band:
    """A courtside panel block in the top band. Rows are (text, colour, kind)."""

    def __init__(self, tx, u):
        self.tx, self.u = tx, u
        self.h_title = int(13 * u)
        self.h_row = int(17 * u)
        self.gap = int(5 * u)
        self.pad = int(10 * u)

    def draw(self, img, x, y, w, h, title, rows, accent=None):
        panel(img, x, y, w, h)
        if accent:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + max(2, 3 * self.u))), accent, -1)
        self.tx.put(title.upper(), x + self.pad, y + self.pad * 0.7, self.h_title, MUTED, "bold")
        yy = y + self.pad * 0.7 + self.h_title + self.gap
        for r in rows:
            if r is None:
                yy += self.gap
                continue
            s, col, kind = (r + ("regular",))[:3]
            self.tx.put(s, x + self.pad, yy, self.h_row, col or WHITE, kind)
            yy += self.h_row + self.gap


def draw_minimap(img, tx, u, players, ball, meta, home, radius, x, y, w, h, carrier=None, hull=False, regions=True):
    panel(img, x, y, w, h, 0.85)
    ix, iy, iw, ih = x + int(8 * u), y + int(8 * u), w - int(16 * u), h - int(16 * u)
    lc = (75, 120, 75)
    if regions and players:
        # dominant regions (Stein et al.): which team reaches each cell first, by plain distance
        gw, gh = 42, 27
        gx, gy = np.meshgrid((np.arange(gw) + 0.5) / gw, (np.arange(gh) + 0.5) / gh)
        best = {"H": np.full((gh, gw), np.inf), "A": np.full((gh, gw), np.inf)}
        for pid, p in players.items():
            _, side = team_colour(pid, meta, home)
            if side in best:
                d = np.hypot((gx - p[0]) * PITCH_M[0], (gy - p[1]) * PITCH_M[1])
                best[side] = np.minimum(best[side], d)
        dom = np.where(best["H"] < best["A"], 0, np.where(best["A"] < best["H"], 1, 2)).astype(np.uint8)
        pal = np.array([HOME, AWAY, (0, 0, 0)], np.uint8)
        tile = cv2.resize(pal[dom], (iw, ih), interpolation=cv2.INTER_NEAREST)
        roi = img[iy:iy + ih, ix:ix + iw]
        cv2.addWeighted(roi, 0.72, tile, 0.28, 0, roi)
    cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), lc, 1)
    cv2.line(img, (ix + iw // 2, iy), (ix + iw // 2, iy + ih), lc, 1)
    cv2.circle(img, (ix + iw // 2, iy + ih // 2), int(9.15 / PITCH_M[0] * iw), lc, 1)
    for side in (0, 1):
        d, hw = int(16.5 / PITCH_M[0] * iw), int(20.16 / PITCH_M[1] * ih)
        xx = ix if side == 0 else ix + iw - d
        cv2.rectangle(img, (xx, iy + ih // 2 - hw), (xx + d, iy + ih // 2 + hw), lc, 1)

    def to_px(p):
        return int(ix + p[0] * iw), int(iy + p[1] * ih)

    pts = {"H": [], "A": []}
    for pid, p in players.items():
        col, side = team_colour(pid, meta, home)
        r = max(2, int(4 * u)) if pid == carrier else max(1, int(2.5 * u))
        cv2.circle(img, to_px(p), r, col, -1, cv2.LINE_AA)
        if side in pts:
            pts[side].append(to_px(p))
    if hull:
        for side, col in (("H", HOME), ("A", AWAY)):
            if len(pts[side]) >= 3:
                cv2.polylines(img, [cv2.convexHull(np.array(pts[side], np.int32))], True, col, 1, cv2.LINE_AA)
    if ball is not None:
        bp = to_px(ball)
        cv2.ellipse(img, bp, (max(1, int(radius / PITCH_M[0] * iw)), max(1, int(radius / PITCH_M[1] * ih))),
                    0, 0, 360, RING, 1, cv2.LINE_AA)
        cv2.circle(img, bp, max(1, int(2.5 * u)), WHITE, -1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# timeline: GT strip + event rows
# ----------------------------------------------------------------------------
def build_timeline(gt, pred, tags, atd, poss, hom, start, end, width):
    """Returns (groups, flags). groups: [(group_name, [(row_name, strip)])]; flags: [(frame, colour, label)]."""
    n = end - start + 1
    cols = np.linspace(start, end + 1, width + 1).astype(int)
    BG = (34, 34, 34)

    def bucket_any(mask, color, bg=BG):
        strip = np.full((width, 3), bg, np.uint8)
        for i in range(width):
            a, b = cols[i] - start, max(cols[i] + 1, cols[i + 1]) - start
            if mask[a:b].any():
                strip[i] = color
        return strip

    def label_strip(lab, cmap):
        strip = np.full((width, 3), BG, np.uint8)
        for i in range(width):
            a, b = cols[i] - start, max(cols[i] + 1, cols[i + 1]) - start
            vals, cnt = np.unique(lab[a:b], return_counts=True)
            strip[i] = cmap.get(vals[cnt.argmax()], BG)
        return strip

    def intensity_strip(vals, vmax, color, bad=None, bad_thresh=None):
        strip = np.full((width, 3), BG, np.uint8)
        for i in range(width):
            a, b = cols[i] - start, max(cols[i] + 1, cols[i + 1]) - start
            seg = vals[a:b]
            seg = seg[np.isfinite(seg)]
            if not len(seg):
                continue
            v = float(seg.max())
            if bad is not None and v > bad_thresh:
                strip[i] = bad
            else:
                k = min(1.0, v / vmax)
                strip[i] = tuple(int(BG[j] + (color[j] - BG[j]) * k) for j in range(3))
        return strip

    groups, flags = [], []
    cmap = dict(LABEL_COLOR)
    cmap["X"] = BG
    cmap[""] = BG

    # --- possession
    rows = []
    restart = np.zeros(n, bool)
    if gt:
        lab = np.array([gt.get(f, "X") for f in range(start, end + 1)])
        rows.append(("GT", label_strip(lab, cmap)))
        prev = np.concatenate([["X"], lab[:-1]])
        for i in np.flatnonzero((prev == "D") & np.isin(lab, ["H", "A"])):
            restart[i:min(n, i + 2)] = True
            flags.append((start + i, EV_RESTART, "GT alive", "tri_up"))
    if pred:
        rows.append(("PRED", label_strip(np.array([pred.get(f, "X") for f in range(start, end + 1)]), cmap)))
    if poss:
        rows.append(("NAIVE", label_strip(poss["naive"], cmap)))
        rows.append(("L&H", label_strip(poss["lh"], cmap)))
    if rows:
        groups.append(("POSSESSION", rows))

    # --- events
    rows = []
    n_sp = 0
    if gt or tags:
        # one row for everything that ends a dead ball: set-piece tag windows as bars (by type colour),
        # GT dead->alive frames as bright green ticks on top
        sp = np.full((width, 3), BG, np.uint8)
        for e in tags or []:
            if not e["code"].upper().startswith("SET PIECE"):
                continue
            a, b = max(e["a"], start), min(e["b"], end + 1)
            if b <= a:
                continue
            n_sp += 1
            name, col, glyph = restart_type(e)
            i0 = int((a - start) / n * width)
            i1 = max(i0 + 2, int((b - start) / n * width))
            sp[i0:i1] = tuple(int(c * 0.55) for c in col)
            flags.append((e["a"], col, name, glyph))
        for i in range(width):
            a, b = cols[i] - start, max(cols[i] + 1, cols[i + 1]) - start
            if restart[a:b].any():
                sp[i] = EV_RESTART
        rows.append(("RESTART", sp))
    if atd:
        rows.append(("KICK", bucket_any(atd["kick"], EV_KICK)))
    if rows:
        groups.append(("EVENTS", rows))

    # --- data quality
    rows = []
    if atd:
        lost = ~np.isfinite(atd["ball"][:, 0])
        rows.append(("BALL LOST", bucket_any(lost, EV_LOST)))
        rows.append(("BALL SPEED", intensity_strip(atd["bspeed_f"], 30.0, WHITE, bad=BAD, bad_thresh=40.0)))
        if atd["has_z"]:
            rows.append(("BALL AIR", intensity_strip(atd["ball"][:, 3], 3.0, EV_AIR)))
        rows.append(("BALL JUMP", bucket_any(atd["b_jump"], EV_JUMP)))
        rows.append(("PLAYER JUMP", bucket_any(atd["p_jump"].any(axis=1), EV_JUMP)))
        rows.append(("DROPOUT <0.5s", bucket_any(atd["dropout"].any(axis=1), (120, 120, 120))))
        rows.append(("VANISH/APPEAR", bucket_any(atd["appear"].any(axis=1) | atd["vanish"].any(axis=1), EV_CHURN)))
        cnt = atd["count"]
        cstrip = np.full((width, 3), BG, np.uint8)
        for i in range(width):
            a, b = cols[i] - start, max(cols[i] + 1, cols[i + 1]) - start
            seg = cnt[a:b]
            lo, hi = int(seg.min()), int(seg.max())
            if hi > 22:
                cstrip[i] = EV_CHURN
            elif lo < 16:
                cstrip[i] = BAD
            elif lo < 20:
                cstrip[i] = WARN
            elif lo < 22:
                cstrip[i] = (90, 90, 90)
        rows.append(("COUNT != 22", cstrip))
        rows.append((">BOLT", bucket_any(atd["pmax_data"] > BOLT_MS, EV_BOLT)))
    if hom is not None:
        idx, Hmat, err, valid = hom
        hs = np.full((width, 3), BG, np.uint8)
        for i in range(width):
            fr = range(cols[i], max(cols[i] + 1, cols[i + 1]))
            es = [err[idx[f]] if (f in idx and valid[idx[f]]) else np.inf for f in fr]
            e = max(es) if es else np.inf
            hs[i] = BAD if not np.isfinite(e) else (GOOD if e < 1 else (WARN if e < 3 else BAD))
        rows.append(("HOMOGRAPHY", hs))
    if rows:
        groups.append(("DATA QUALITY", rows))

    if atd:
        print(f"events in range: GT restarts {len([f for f in flags if f[2] == 'GT alive'])}, set-piece tags {n_sp}, "
              f"kicks {int(atd['kick'].sum())}, ball lost {100 * (~np.isfinite(atd['ball'][:, 0])).mean():.1f}%")
    return groups, flags


# ----------------------------------------------------------------------------
# export for the interactive viewer (match_viewer.html)
# ----------------------------------------------------------------------------
def export_data(out_dir, src, video, start, end, fps, W, Hh, gt, pred, tags, atd, meta, home, hom, poss, flags, referee_doc=None):
    os.makedirs(out_dir, exist_ok=True)
    n = end - start + 1
    ids = atd["ids"] if atd else []
    S = len(ids)
    # per-frame block: H[9] | ball x,y,z,sp_data,sp_frames,acc [6] | slots S x (x,y,sp_data,sp_frames)
    blk = 9 + 6 + 4 * S
    buf = np.full((n, blk), np.nan, dtype=np.float32)
    if hom is not None:
        idx, Hmat, err, valid = hom
        for r, f in enumerate(range(start, end + 1)):
            i = idx.get(f)
            if i is not None and valid[i]:
                buf[r, :9] = Hmat[i].ravel()
    if atd:
        b = atd["ball"]
        buf[:, 9:11] = b[:, :2]
        buf[:, 11] = b[:, 3]
        buf[:, 12] = b[:, 2]
        buf[:, 13] = atd["bspeed_f"]
        buf[:, 14] = atd["acc"]
        xy = atd["xy"]
        buf[:, 15::4][:, :S] = xy[:, :, 0]
        buf[:, 16::4][:, :S] = xy[:, :, 1]
        buf[:, 17::4][:, :S] = xy[:, :, 2]
        buf[:, 18::4][:, :S] = atd["pspeed_f"]
    buf.tofile(os.path.join(out_dir, "data.bin"))

    def ev_list(mask2d):
        out = []
        for r, k in zip(*np.nonzero(mask2d)):
            out.append([int(r), int(k)])
        return out

    hom_err = None
    if hom is not None:
        idx, Hmat, err, valid = hom
        hom_err = [float(err[idx[f]]) if (f in idx and valid[idx[f]] and np.isfinite(err[idx[f]])) else -1 for f in range(start, end + 1)]
    m = {
        "source": src, "video": os.path.relpath(video, HERE).replace(os.sep, "/"), "fps": fps,
        "start": start, "end": end, "n": n, "width": W, "height": Hh, "pitch": PITCH_M,
        "block": blk, "n_slots": S,
        "slots": [{"id": pid, "num": slot_label(pid, meta), "team": team_colour(pid, meta, home)[1]} for pid in ids],
        "home_dir": atd.get("home_dir", 0) if atd else 0,
        "halves": {str(k): list(v) for k, v in (HALVES or {}).items()},
        "gt": "".join(gt.get(f, "X") for f in range(start, end + 1)) if gt else None,
        "pred": "".join(pred.get(f, "X") for f in range(start, end + 1)) if pred else None,
        "tags": [{"a": e["a"], "b": e["b"], "code": e["code"], "team": e["tags"].get("Team"), "type": e["tags"].get("Type"),
                  "side": e["side"], "restart": (list(restart_type(e)) if e["code"].upper().startswith("SET PIECE") else None)}
                 for e in tags if e["b"] >= start and e["a"] <= end],
        "flags": [{"f": int(f), "name": name, "glyph": glyph, "color": list(col)} for f, col, name, glyph in flags],
        "events": {
            "kick": [int(r) for r in np.flatnonzero(atd["kick"])] if atd else [],
            "bjump": [[int(r), float(atd["bjump_m"][r])] for r in np.flatnonzero(atd["b_jump"])] if atd else [],
            "pjump": [[int(r), int(k), float(atd["pjump_m"][r, k])] for r, k in zip(*np.nonzero(atd["p_jump"]))] if atd else [],
            "vanish": ev_list(atd["vanish"]) if atd else [],
            "appear": ev_list(atd["appear"]) if atd else [],
        },
        "count": atd["count"].astype(int).tolist() if atd else None,
        "dropout": atd["dropout"].any(axis=1).astype(int).tolist() if atd else None,
        "pmax": [float(v) if np.isfinite(v) else -1 for v in atd["pmax_data"]] if atd else None,
        "hom_err": hom_err,
        "has_z": bool(atd and atd["has_z"]), "has_ball_speed": bool(atd and atd["has_ball_speed"]),
        "bolt": BOLT_MS, "ball_radius": BALL_RADIUS_M,
    }
    if referee_doc:
        try:
            from referee_signals.io import signal_events_for_meta
            from referee_signals.schema import SIGNAL_STYLE
            m["referee"] = {
                "source": referee_doc.get("source", src),
                "fps": referee_doc.get("fps", fps),
                "officials": referee_doc.get("officials") or [],
                "events": signal_events_for_meta(referee_doc, start, end),
                "styles": SIGNAL_STYLE,
            }
            # Sidecar full document for debugging / re-import
            with open(os.path.join(out_dir, "referee_signals.json"), "w") as fh:
                json.dump(referee_doc, fh)
        except Exception as e:
            print(f"warning: could not attach referee signals to export: {e}")
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(m, fh)
    size = os.path.getsize(os.path.join(out_dir, "data.bin")) / 1e6
    print(f"exported {n} frames -> {out_dir}  (data.bin {size:.1f} MB, meta.json)")


def server_root():
    """The folder the server has to serve from: the common parent of this script and the video.
    Written by --export into export/latest.json; defaults to this folder."""
    try:
        with open(os.path.join(HERE, "export", "latest.json")) as fh:
            j = json.load(fh)
        root = j.get("root")
        if root and os.path.isdir(root):
            return root
    except (OSError, ValueError):
        pass
    return HERE


def serve(root, port):
    """Static file server with HTTP Range support (needed for video seeking); opens match_viewer.html."""
    import http.server
    import socketserver
    import webbrowser
    if not os.path.exists(os.path.join(HERE, "match_viewer.html")):
        sys.exit("match_viewer.html must sit next to match_viewer.py")

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=root, **k)

        def send_head(self):
            path = self.translate_path(self.path)
            if os.path.isdir(path):
                return super().send_head()
            rng = self.headers.get("Range")
            if not rng or not os.path.isfile(path):
                return super().send_head()
            size = os.path.getsize(path)
            try:
                a, b = rng.replace("bytes=", "").split("-")
                a = int(a) if a else 0
                b = int(b) if b else size - 1
            except ValueError:
                return super().send_head()
            b = min(b, size - 1)
            if a > b:
                self.send_error(416)
                return None
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            fh = open(path, "rb")
            fh.seek(a)
            self._range = (fh, b - a + 1)
            return fh

        def copyfile(self, source, outputfile):
            r = getattr(self, "_range", None)
            try:
                if r is None:
                    return super().copyfile(source, outputfile)
                fh, left = r
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    outputfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass          # the browser cancels range requests whenever it seeks; that is normal
            finally:
                self._range = None

        def handle(self):
            try:
                super().handle()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True

    class Srv(socketserver.ThreadingTCPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            import sys as _s
            et = _s.exc_info()[0]
            if et not in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                super().handle_error(request, client_address)

    with Srv(("127.0.0.1", port), H) as httpd:
        page = os.path.relpath(os.path.join(HERE, "match_viewer.html"), root).replace(os.sep, "/")
        url = f"http://127.0.0.1:{port}/{page}"
        print(f"serving {root}\nopen   {url}   (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["gt", "full"], default="full")
    ap.add_argument("--source", choices=["tactical", "broadcast"], default="tactical")
    ap.add_argument("--second-half-start", type=int, help="broadcast: video frame of the 2nd-half kickoff")
    ap.add_argument("--inspect", action="store_true", help="print how the ATD files are parsed, then exit")
    ap.add_argument("--export", action="store_true", help="write export/<source>_<mode>/ for match_viewer.html instead of a video")
    ap.add_argument("--serve", action="store_true", help="start a local server and open match_viewer.html")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--video")
    ap.add_argument("--gt")
    ap.add_argument("--atd")
    ap.add_argument("--atd-meta")
    ap.add_argument("--homography")
    ap.add_argument("--tags")
    ap.add_argument("--pred")
    ap.add_argument("--out")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--scale", type=float, default=1.0, help="output resolution factor")
    ap.add_argument("--layout", choices=["boxed", "overlay"], default="boxed",
                    help="boxed = video shrunk into the middle with the panels outside it (default); overlay = panels on the video")
    ap.add_argument("--ui", type=float, default=0.6, help="overlay size factor (1.0 = large, 0.45 = tiny)")
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--trail", type=float, default=1.5, help="ball trail length in seconds (0 = off)")
    ap.add_argument("--shape", action="store_true")
    ap.add_argument("--referee-signals", help="path to referee_signals.json (AR flag detections)")
    ap.add_argument("--no-referee", action="store_true", help="skip referee flag overlays even if a JSON is found")
    for flag in ("players", "ball", "pitch", "minimap", "tags", "timeline", "regions"):
        ap.add_argument(f"--no-{flag}", action="store_true")
    args = ap.parse_args()
    global HALVES, HALVES_EST
    if args.serve and not args.export:
        serve(server_root(), args.port)
        return
    src = args.source
    broadcast = src == "broadcast"
    out_path = args.out or os.path.join(HERE, f"{src}_overlay_{'full' if broadcast else args.mode}.mp4")
    if broadcast and args.mode == "gt":
        print("broadcast has no ground truth window, rendering the whole video")
        args.mode = "full"

    video = args.video or find_file(f"*{src}*.mp4", source=src)
    gt_path = None if broadcast else (args.gt or find_file("*possession_gt_public.csv"))
    hom_path = args.homography or find_file(f"*{src}_homography*.csv", required=False, source=src)
    want_atd = not (args.no_players and args.no_ball)
    atd_path = (args.atd or find_file(f"*{src}_atd*.txt", required=False, source=src)) if want_atd else None
    atd_meta = (args.atd_meta or find_file(f"*{src}_atd*.xml", required=False, source=src)) if atd_path else None
    tags_path = None if (args.no_tags or broadcast) else (args.tags or find_file("*smart_tagging*.xml", required=False, source=src))
    if broadcast and args.pred:
        print("--pred ignored: submissions are indexed to the tactical video, not the broadcast")
        args.pred = None

    referee_doc = None
    referee_by_frame = {}
    if not args.no_referee:
        try:
            from referee_signals.io import find_signals_path, index_by_frame, load_signals
            ref_path = find_signals_path(HERE, args.referee_signals, source=src)
            if ref_path:
                referee_doc = load_signals(ref_path)
                referee_by_frame = index_by_frame(referee_doc)
                print(f"referee     : {ref_path}  ({len(referee_doc.get('detections') or [])} detections)")
            else:
                print("referee     : (not found, skipped)")
        except Exception as e:
            print(f"referee     : (load failed: {e})")

    for k, v in (("video", video), ("ground truth", gt_path), ("homography", hom_path),
                 ("ATD raw", atd_path), ("ATD xml", atd_meta), ("smart tags", tags_path)):
        print(f"{k:13s}: {v or '(not found, skipped)'}")

    gt = load_labels(gt_path) if gt_path else {}
    pred = load_labels(args.pred) if args.pred else None
    hom = load_homography(hom_path) if hom_path else None
    tags = load_tags(tags_path) if tags_path else []

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if gt and n_frames and abs(len(gt) - n_frames) > 2:
        print(f"warning: video has {n_frames} frames, GT has {len(gt)} rows, is this the tactical video?")

    if args.mode == "gt":
        w0, w1 = labelled_window(gt_path, gt)
    else:
        w0, w1 = 0, (n_frames - 1 if n_frames else max(gt, default=0))
    start = args.start if args.start is not None else w0
    end = args.end if args.end is not None else w1
    if n_frames:
        end = min(end, n_frames - 1)
    if args.inspect:
        if not atd_path:
            sys.exit("no ATD raw file found to inspect")
        i0 = labelled_window(gt_path, gt)[0] if gt else start
        load_atd(atd_path, atd_meta, i0, min(end, i0 + 500), inspect=True)
        return
    atd, meta, home = load_atd(atd_path, atd_meta, start, end) if atd_path else ({}, {}, None)
    if atd and not atd["has_ball_speed"]:
        print("note: the ATD ball speed channel is empty (NaN) in this range, ball speed is computed from frames only")

    if broadcast:
        HALVES = None
        if args.second_half_start is not None:
            HALVES = {1: (start, args.second_half_start - 1), 2: (args.second_half_start, end)}
        elif atd:
            HALVES = estimate_halves(atd)
            HALVES_EST = HALVES is not None
        if HALVES:
            print(f"halves on broadcast timeline{' (estimated from tracking gap)' if HALVES_EST else ''}: "
                  f"1st {HALVES[1][0]}..{HALVES[1][1]}, 2nd {HALVES[2][0]}..{HALVES[2][1]}")
        else:
            print("could not determine halves on the broadcast, no match clock / attack direction")

    s = args.scale
    u = s * args.ui
    out_w, out_h = int(W * s), int(Hh * s)
    tx = Text(u)
    band = Band(tx, u)
    writer = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(start):
            if not cap.grab():
                sys.exit("video shorter than requested start frame")

    # ---- layout (all in output pixels)
    m = max(6, int(22 * u))                       # outer margin
    gap = max(3, int(8 * u))
    top_y, top_h = m, int(150 * u)
    # bottom band: timeline rows (left) + minimap (right)
    poss = compute_possession(atd, meta, home, args.radius) if atd else None
    tl_h, tl_gap, grp_gap = max(5, int(12 * u)), max(1, int(2 * u)), max(3, int(8 * u))
    mm_w, mm_h = int(250 * u), int(162 * u)
    tl_label_w = int(110 * u)
    tl_x = m + tl_label_w
    tl_w = out_w - m - mm_w - gap - tl_x if (atd and not args.no_minimap) else out_w - m - tl_x
    tl_groups, tl_flags = build_timeline(gt, pred, tags, atd, poss, hom, start, end, tl_w) if not args.no_timeline else ([], [])
    if args.export:
        out_dir = os.path.join(HERE, "export", f"{src}_{'full' if broadcast else args.mode}")
        export_data(out_dir, src, video, start, end, src_fps, W, Hh, gt, pred, tags, atd, meta, home, hom, poss, tl_flags, referee_doc=referee_doc)
        root = os.path.commonpath([HERE, os.path.dirname(os.path.abspath(video))])
        with open(os.path.join(HERE, "export", "latest.json"), "w") as fh:
            json.dump({"dir": os.path.relpath(out_dir, HERE).replace(os.sep, "/"), "root": root}, fh)
        cap.release()
        if args.serve:
            serve(root, args.port)
        else:
            print("now run:  python match_viewer.py --serve")
        return
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (out_w, out_h))
    n_rows = sum(len(r) for _, r in tl_groups)
    tl_total = n_rows * (tl_h + tl_gap) + max(0, len(tl_groups) - 1) * grp_gap
    tl_y = out_h - m - tl_total
    flag_h = int(14 * u)
    recent = []                                   # (frame, colour, text) for the ticker
    boxed = args.layout == "boxed"
    if boxed:
        vy0 = top_y + top_h + gap
        vy1 = tl_y - flag_h - int(20 * u) - gap
        avail_w, avail_h = out_w - 2 * m, max(50, vy1 - vy0)
        sv = min(avail_w / W, avail_h / Hh)
        vw, vh = int(W * sv), int(Hh * sv)
        ox, oy = (out_w - vw) // 2, vy0 + (avail_h - vh) // 2
        print(f"boxed layout: video {vw}x{vh} at ({ox},{oy}) inside {out_w}x{out_h}")
    else:
        sv, vw, vh, ox, oy = s, out_w, out_h, 0, 0

    total = end - start + 1
    print(f"rendering frames {start}..{end} ({total} frames, {total / src_fps / 60:.1f} min) -> {out_path}")

    t1 = 1
    t2 = max(1, int(round(2 * u)))
    agree = disagree = 0
    prev_label, seg_start, last_change = None, start, -999
    ti = 0
    trail_n = int(args.trail * src_fps)
    trail = []
    future_n = int(1.0 * src_fps)
    fade = {}                                     # on-pitch marks that fade: key -> (until_frame, kind, data)

    for f in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            print(f"stream ended at frame {f}")
            break
        if sv != 1.0:
            frame = cv2.resize(frame, (vw, vh))
        s = sv                                   # on-pitch projections use the video scale

        # ---------------- homography
        Hm, hq, htxt = None, None, None
        if hom is not None:
            idx, Hmat, err, valid = hom
            i = idx.get(f)
            if i is None or valid[i] == 0 or not np.all(np.isfinite(Hmat[i])):
                hq, htxt = BAD, "homography  none"
            else:
                Hm, e = Hmat[i], err[i]
                hq = GOOD if e < 1.0 else (WARN if e < 3.0 else BAD)
                htxt = f"homography  err {e:.2f}"
                if not args.no_pitch:
                    # pitch lines stay quiet (muted white); only a bad fit turns them red
                    draw_pitch(frame, Hm, s, (215, 215, 215) if e < 3.0 else BAD, t1)

        hdir = attack_dir(atd.get("home_dir", 0), f) if atd else 0
        if Hm is not None and hdir:
            for col, x_from, x_to in ((HOME, 0.34, 0.44), (AWAY, 0.66, 0.56)):
                if hdir < 0:
                    x_from, x_to = 1 - x_from, 1 - x_to
                a, b = project(Hm, [[x_from, -0.035], [x_to, -0.035]]) * s
                if good_px(a) and good_px(b):
                    cv2.arrowedLine(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, t2, cv2.LINE_AA, tipLength=0.3)

        # ---------------- referee flag overlays (pixel space; independent of ATD)
        if referee_by_frame:
            try:
                from referee_signals.overlay import draw_detections
                dets = referee_by_frame.get(f) or []
                if dets:
                    draw_detections(frame, dets, scale=s, thickness=max(1, t1))
            except Exception:
                pass


        # ---------------- tracking
        players, ball, (bsp_f, bsp_d), heading = frame_tracking(atd, f)
        if args.no_players:
            players = {}
        if args.no_ball:
            ball = None
        sides = {pid: team_colour(pid, meta, home) for pid in players}

        # proximity: who is close to the ball, per team
        inside = {"H": 0, "A": 0}
        nearest = {"H": None, "A": None}
        carrier = None
        if ball is not None:
            for pid, p in players.items():
                d = metres(ball, p)
                _, side = sides[pid]
                if side in inside:
                    if d <= args.radius:
                        inside[side] += 1
                    if nearest[side] is None or d < nearest[side][0]:
                        nearest[side] = (d, pid, p[2], p[3])
            r0 = f - atd["first"]
            k = poss["lh_pid"][r0] if poss else -1
            carrier = atd["ids"][k] if k >= 0 else None
            opp_side = None
            if carrier is not None:
                opp_side = "A" if sides[carrier][1] == "H" else "H"
            trail.append(ball)
        else:
            trail.append(None)
        trail = trail[-max(1, trail_n):]

        # ---- events on this frame -> fading on-pitch marks + ticker
        if atd:
            r0 = f - atd["first"]
            if atd["kick"][r0] and ball is not None:
                fade[("kick", f)] = (f + int(0.4 * src_fps), "kick", ball)
                recent.append((f, EV_KICK, f"KICK  ball {atd['bspeed_f'][r0]:.0f} m/s"))
            if atd["b_jump"][r0] and r0 > 0:
                fade[("bjump", f)] = (f + int(0.8 * src_fps), "jump", (atd["ball"][r0 - 1, :2], atd["ball"][r0, :2], WHITE))
                recent.append((f, EV_JUMP, f"BALL JUMP  {atd['bjump_m'][r0]:.1f} m in one frame"))
            for k in np.flatnonzero(atd["p_jump"][r0]):
                pid = atd["ids"][k]
                fade[("pjump", f, pid)] = (f + int(0.8 * src_fps), "jump", (atd["xy"][r0 - 1, k, :2], atd["xy"][r0, k, :2], team_colour(pid, meta, home)[0]))
                recent.append((f, EV_JUMP, f"PLAYER JUMP  #{slot_label(pid, meta)}  {atd['pjump_m'][r0, k]:.1f} m"))
            for k in np.flatnonzero(atd["vanish"][r0]):
                pid = atd["ids"][k]
                fade[("vanish", f, pid)] = (f + int(1.0 * src_fps), "vanish", (atd["xy"][r0 - 1, k, :2], pid))
                recent.append((f, EV_CHURN, f"VANISHED  #{slot_label(pid, meta)}"))
            for k in np.flatnonzero(atd["appear"][r0]):
                pid = atd["ids"][k]
                fade[("appear", f, pid)] = (f + int(0.6 * src_fps), "appear", (atd["xy"][r0, k, :2], pid))
                recent.append((f, EV_CHURN, f"APPEARED  #{slot_label(pid, meta)}"))
        for fl in tl_flags:
            if fl[0] == f:
                recent.append((f, fl[1], f"RESTART  {fl[2]}"))
        recent = [e for e in recent if f - e[0] < 2.5 * src_fps][-4:]
        fade = {k: v for k, v in fade.items() if v[0] >= f}

        # ---------------- on-pitch layer, ground-anchored
        if Hm is not None:
            # ball trail (fading, thin)
            if trail_n > 1 and len(trail) > 2:
                pts = [project(Hm, b)[0] * s if b is not None else None for b in trail]
                for k in range(1, len(pts)):
                    if pts[k] is None or pts[k - 1] is None or not (good_px(pts[k]) and good_px(pts[k - 1])):
                        continue
                    a = 0.15 + 0.6 * k / len(pts)
                    line_alpha(frame, tuple(pts[k - 1].astype(int)), tuple(pts[k].astype(int)), WHITE, a, t1)

            # future ball path (1 s ahead, dotted), Stein et al. show past and future with transparency
            if ball is not None and future_n > 1:
                r0 = f - atd["first"]
                fut = atd["ball"][r0:r0 + future_n:3, :2]
                fut = fut[np.isfinite(fut[:, 0])]
                if len(fut) > 1:
                    fp = project(Hm, fut) * s
                    if good_px(fp):
                        for k, p in enumerate(fp):
                            cv2.circle(frame, tuple(p.astype(int)), 1, WHITE, -1, cv2.LINE_AA)

            # fading event marks
            for key, (until, kind, data) in fade.items():
                a = max(0.15, (until - f) / max(1, until - key[1]))
                if kind == "kick":
                    rp = ground_circle(Hm, data, 0.6 + 1.2 * (1 - a)) * s
                    if good_px(rp):
                        cv2.polylines(frame, [rp.astype(np.int32).reshape(-1, 1, 2)], True, EV_KICK, t1, cv2.LINE_AA)
                elif kind == "jump":
                    p0, p1, col = data
                    if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                        a0, a1 = project(Hm, p0)[0] * s, project(Hm, p1)[0] * s
                        if good_px(a0) and good_px(a1):
                            line_alpha(frame, tuple(a0.astype(int)), tuple(a1.astype(int)), EV_JUMP, a, t2)
                            cv2.drawMarker(frame, tuple(a0.astype(int)), EV_JUMP, cv2.MARKER_TILTED_CROSS, int(12 * u), t1, cv2.LINE_AA)
                elif kind == "vanish":
                    p0, pid = data
                    if np.all(np.isfinite(p0)):
                        c0 = project(Hm, p0)[0] * s
                        if good_px(c0):
                            c0 = tuple(c0.astype(int))
                            cv2.circle(frame, c0, max(3, int(8 * u)), EV_CHURN, t1, cv2.LINE_AA)
                            cv2.drawMarker(frame, c0, EV_CHURN, cv2.MARKER_TILTED_CROSS, int(10 * u), t1, cv2.LINE_AA)
                            tx.put(f"lost #{slot_label(pid, meta)}", c0[0] + int(10 * u), c0[1] - int(14 * u), int(10 * u), EV_CHURN)
                elif kind == "appear":
                    p0, pid = data
                    rp = ground_circle(Hm, p0, 0.5 + 1.0 * (1 - a)) * s
                    if good_px(rp):
                        cv2.polylines(frame, [rp.astype(np.int32).reshape(-1, 1, 2)], True, EV_CHURN, t1, cv2.LINE_AA)

            # possession ring: tinted by whoever is inside it
            if ball is not None:
                rp = ground_circle(Hm, ball, args.radius) * s
                if good_px(rp):
                    tint = HOME if inside["H"] and not inside["A"] else (AWAY if inside["A"] and not inside["H"] else None)
                    if tint is not None:
                        fill_poly_alpha(frame, rp, tint, 0.16, outline=RING, thick=t1)
                    else:
                        cv2.polylines(frame, [rp.astype(np.int32).reshape(-1, 1, 2)], True, RING, t1, cv2.LINE_AA)

            # players: bare marks; ball carrier gets a ground disc; closest opponent gets a line to the ball
            if players:
                pids = list(players)
                px = project(Hm, [players[p][:2] for p in pids]) * s
                hull_pts = {"H": [], "A": []}
                r = max(3, int(8 * u))
                for pid, (cx, cy) in zip(pids, px):
                    if not (np.isfinite(cx) and np.isfinite(cy)) or abs(cx) > 1e5 or abs(cy) > 1e5:
                        continue
                    col, side = sides[pid]
                    c = (int(cx), int(cy))
                    if side in hull_pts:
                        hull_pts[side].append(c)
                    if pid == carrier:
                        gp = ground_circle(Hm, players[pid][:2], 0.6) * s
                        if good_px(gp):
                            fill_poly_alpha(frame, gp, col, 0.38, outline=col, thick=t1)
                    cv2.circle(frame, c, r, BLACK, t2 + 1, cv2.LINE_AA)
                    cv2.circle(frame, c, r, col, t1, cv2.LINE_AA)
                    lab = slot_label(pid, meta)
                    if pid == carrier and ball is not None:
                        lab = f"{lab}  {metres(ball, players[pid]):.1f} m"
                    tx.put(lab, c[0] + r + 3, c[1] - r - int(12 * u), int(11 * u), col, "bold" if pid == carrier else "regular")
                if ball is not None and carrier is not None and opp_side and nearest[opp_side]:
                    d, opid, _, _ = nearest[opp_side]
                    op = project(Hm, players[opid][:2])[0] * s
                    bp0 = project(Hm, ball)[0] * s
                    if good_px(op) and good_px(bp0):
                        line_alpha(frame, tuple(op.astype(int)), tuple(bp0.astype(int)), sides[opid][0], 0.7, t1)
                        tx.put(f"{d:.1f} m", (op[0] + bp0[0]) / 2 + 4, (op[1] + bp0[1]) / 2 - int(14 * u), int(10 * u), sides[opid][0])
                if args.shape:
                    for side, col in (("H", HOME), ("A", AWAY)):
                        if len(hull_pts[side]) >= 3:
                            hp = cv2.convexHull(np.array(hull_pts[side], np.int32))
                            fill_poly_alpha(frame, hp.reshape(-1, 2), col, 0.08, outline=col, thick=t1)

            # the ball itself: real size, translucent, with a short heading arrow
            if ball is not None:
                bp = ground_circle(Hm, ball, BALL_RADIUS_M, 16) * s
                if good_px(bp):
                    fill_poly_alpha(frame, bp, WHITE, 0.6, outline=BLACK, thick=1)
                    c = tuple(bp.mean(axis=0).astype(int))
                    if heading is not None and np.isfinite(bsp_f) and bsp_f > 1.0:
                        ap_ = project(Hm, (ball[0] + heading[0] * 10, ball[1] + heading[1] * 10))[0] * s
                        if good_px(ap_):
                            cv2.arrowedLine(frame, c, (int(ap_[0]), int(ap_[1])), WHITE, t1, cv2.LINE_AA, tipLength=0.25)

        # ---------------- compose: video into the canvas, panels outside it
        if boxed:
            canvas = np.full((out_h, out_w, 3), (10, 10, 10), np.uint8)
            canvas[oy:oy + vh, ox:ox + vw] = frame
            cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + vw, oy + vh), (60, 60, 60), 1)
            tx.shift(ox, oy)
            frame = canvas
        s = args.scale

        # ---------------- top band: courtside panels
        g = gt.get(f, "X") if gt else None
        if g is not None and g != prev_label:
            if prev_label is not None:
                last_change = f
            seg_start, prev_label = f, g
        x = m
        # 1 POSSESSION
        w1 = int(360 * u)
        panel(frame, x, top_y, w1, top_h)
        if g is None:
            tx.put("POSSESSION", x + band.pad, top_y + band.pad * 0.7, band.h_title, MUTED, "bold")
            tx.put("no ground truth on the", x + band.pad, top_y + int(45 * u), band.h_row, MUTED)
            tx.put("broadcast timeline", x + band.pad, top_y + int(68 * u), band.h_row, MUTED)
        else:
            gcol = LABEL_COLOR.get(g, WHITE)
            cv2.rectangle(frame, (x, top_y), (x + w1, top_y + max(2, int(3 * u))), gcol, -1)
            if (f - last_change) < src_fps / 2:
                cv2.rectangle(frame, (x, top_y), (x + w1, top_y + top_h), WHITE, t1)
            tx.put("POSSESSION  ·  GROUND TRUTH", x + band.pad, top_y + band.pad * 0.7, band.h_title, MUTED, "bold")
            tx.put(g, x + band.pad, top_y + int(22 * u), int(78 * u), gcol, "bold")
            held = (f - seg_start) / src_fps
            tx.put(LABEL_NAME.get(g, g), x + int(72 * u), top_y + int(38 * u), band.h_row, WHITE, "bold")
            if g != "X":
                tx.put(f"held {held:.1f} s", x + int(72 * u), top_y + int(60 * u), band.h_row, MUTED)
            if pred is not None:
                p = pred.get(f, "X")
                same = p == g
                if g != "X":
                    agree += same
                    disagree += not same
                pc = GOOD if same else BAD
                tx.put("PREDICTION", x + int(230 * u), top_y + band.pad * 0.7, band.h_title, MUTED, "bold")
                tx.put(p, x + int(230 * u), top_y + int(22 * u), int(78 * u), pc, "bold")
                if agree + disagree:
                    tx.put(f"acc {agree / (agree + disagree):.3f}", x + int(285 * u), top_y + int(60 * u), band.h_row, pc)
            tx.put(f"within {args.radius:g} m   H {inside['H']}   A {inside['A']}" if ball is not None else "ball not tracked",
                   x + band.pad, top_y + int(120 * u), band.h_row,
                   (HOME if inside["H"] and not inside["A"] else AWAY if inside["A"] and not inside["H"] else WHITE) if ball else BAD)
        x += w1 + gap

        if atd:
            # 2 BALL & PROXIMITY
            rows = []
            if ball is None:
                rows.append(("ball not tracked", BAD, "bold"))
            else:
                rows.append((f"ball  x {ball[0]:.3f}   y {ball[1]:.3f}", WHITE))
                for side, col in (("H", HOME), ("A", AWAY)):
                    nn = nearest[side]
                    if nn:
                        rows.append((f"nearest {side}   #{slot_label(nn[1], meta)}   {nn[0]:.2f} m", col))
                opp_near = 0
                if carrier is not None:
                    cside = sides[carrier][1]
                    opp_near = sum(1 for pid, p in players.items() if sides[pid][1] not in (cside, "?") and metres(players[carrier], p) <= 3.0)
                    rows.append((f"holder #{slot_label(carrier, meta)}   pressure {opp_near} opp < 3 m", sides[carrier][0]))
                zrow = atd["ball"][f - atd["first"], 3] if atd["has_z"] else np.nan
                z = ball_zone(ball, hdir)
                ztxt = f"   z {zrow:.2f} m" if np.isfinite(zrow) else ""
                if z:
                    rows.append((z[0] + ztxt, HOME if z[1] == "Hatt" else (AWAY if z[1] == "Hdef" else MUTED)))
            nh = sum(1 for v in sides.values() if v[1] == "H")
            na = sum(1 for v in sides.values() if v[1] == "A")
            rows.append((f"tracked   H {nh}   A {na}   all {len(players)}", MUTED))
            w2 = int(300 * u)
            band.draw(frame, x, top_y, w2, top_h, "ball & proximity", rows)
            x += w2 + gap

            # 3 SPEED  (frames vs data)
            def fmt(v):
                return f"{v:4.1f}" if np.isfinite(v) else " n/a"
            rows = [("            frames    data", MUTED)]
            if ball is not None:
                rows.append((f"ball        {fmt(bsp_f)}      {fmt(bsp_d)}  m/s", WHITE, "mono"))
            cand = [nn for nn in nearest.values() if nn]
            if cand:
                nn = min(cand, key=lambda q: q[0])
                dcol = WHITE
                if np.isfinite(nn[2]) and np.isfinite(nn[3]):
                    dcol = GOOD if abs(nn[3] - nn[2]) < 1.0 else WARN
                rows.append((f"#{slot_label(nn[1], meta):<4}      {fmt(nn[3])}      {fmt(nn[2])}  m/s", dcol, "mono"))
            if players:
                fast = max(players.items(), key=lambda kv: (kv[1][2] if np.isfinite(kv[1][2]) else -1))
                fs = fast[1][2]
                if np.isfinite(fs):
                    bolt = fs > BOLT_MS
                    rows.append((f"fastest  #{slot_label(fast[0], meta)}  {fs:.1f} m/s" + ("   > BOLT" if bolt else ""),
                                 EV_BOLT if bolt else MUTED, "bold" if bolt else "regular"))
            w3 = int(300 * u)
            band.draw(frame, x, top_y, w3, top_h, "speed", rows)
            x += w3 + gap

        # 4 EVENTS & TAGS
        if tags or atd:
            rows = []
            for ef, col, txt in reversed(recent):
                age = (f - ef) / src_fps
                rows.append((f"{txt}   {age:.1f} s", col, "bold"))
            act = []
            if tags:
                while ti < len(tags) and tags[ti]["b"] <= f:
                    ti += 1
                act = [e for e in tags[ti:ti + 40] if e["a"] <= f < e["b"]]
            for e in act[:max(1, 5 - len(rows))]:
                if e["code"].upper().startswith("SET PIECE"):
                    name, col, _ = restart_type(e)
                    rows.append((f"restart  ·  {name}  ·  {e['tags'].get('Team') or '-'}", col, "bold"))
                else:
                    rows.append((tag_line(e)[:40], LABEL_COLOR.get(e["side"], MUTED)))
            rows = rows or [("none", MUTED)]
            w4 = int(330 * u)
            band.draw(frame, x, top_y, w4, top_h, "events  ·  tags", rows[:5])
            x += w4 + gap

        # 5 MATCH (right-aligned)
        half, clock = half_and_clock(f)
        t = f / src_fps
        rows = []
        if half:
            rows.append((f"{half}   {clock}" + ("  est." if HALVES_EST else ""), WHITE, "bold"))
        rows.append((f"frame {f}   ·   {int(t // 60):02d}:{t % 60:05.2f}", MUTED, "mono"))
        if hdir:
            rows.append((f"attack   BAR {'→' if hdir > 0 else '←'}   ATH {'←' if hdir > 0 else '→'}", WHITE))
        if htxt:
            rows.append((htxt, hq))
        w5 = int(250 * u)
        band.draw(frame, out_w - m - w5, top_y, w5, top_h, "match" + ("  ·  broadcast" if broadcast else ""), rows)
        # legend as a single row of dots under the match block
        lx, ly = out_w - m - w5 + band.pad, top_y + top_h - int(20 * u)
        for name, col in (("BAR", HOME), ("ATH", AWAY), ("dead", DEAD), ("ring", RING)):
            cv2.circle(frame, (int(lx + 4 * u), int(ly + 7 * u)), max(2, int(4 * u)), col, -1, cv2.LINE_AA)
            tx.put(name, lx + int(12 * u), ly, int(11 * u), MUTED)
            lx += int(12 * u) + tx.width(name, int(11 * u)) + int(10 * u)

        # ---------------- bottom band: timeline + minimap
        if tl_groups:
            yy = tl_y
            for gname, rows in tl_groups:
                g0 = yy
                for name, strip in rows:
                    frame[yy:yy + tl_h, tl_x:tl_x + tl_w] = strip[None, :, :]
                    tx.put(name, m + int(5 * u), yy - int(1 * u), int(11 * u), MUTED)
                    yy += tl_h + tl_gap
                cv2.line(frame, (m, g0), (m, yy - tl_gap), (90, 90, 90), 1)
                yy += grp_gap
            # restart flags: a full-height line + a small triangle, so dead-ball endings are unmissable
            for ff, col, lab, glyph in tl_flags:
                if start <= ff <= end:
                    fx = tl_x + int((ff - start) / max(1, end - start) * (tl_w - 1))
                    line_alpha(frame, (fx, tl_y - flag_h), (fx, tl_y + tl_total), col, 0.5, 1)
                    draw_glyph(frame, glyph, fx, tl_y - flag_h + int(5 * u), 4.5 * u, col)
            cx = tl_x + int((f - start) / max(1, end - start) * (tl_w - 1))
            cv2.line(frame, (cx, tl_y - flag_h - int(2 * u)), (cx, tl_y + tl_total), WHITE, t1)
            lx, ly = tl_x, tl_y - flag_h - int(16 * u)
            for name, col, glyph in [GT_ALIVE_GLYPH] + \
                    [(n_, c_, g_) for n_, c_, g_ in RESTART_TYPES.values() if n_ != "Penalty"] + \
                    [("kick", EV_KICK, None), ("ball lost", EV_LOST, None), ("jump", EV_JUMP, None),
                     ("vanish/appear", EV_CHURN, None), (f"> Bolt {BOLT_MS} m/s", EV_BOLT, None)]:
                if glyph:
                    draw_glyph(frame, glyph, lx + 4 * u, ly + 6 * u, 4 * u, col)
                else:
                    cv2.rectangle(frame, (int(lx), int(ly + 3 * u)), (int(lx + 8 * u), int(ly + 9 * u)), col, -1)
                tx.put(name, lx + int(13 * u), ly - int(2 * u), int(10 * u), MUTED)
                lx += int(13 * u) + tx.width(name, int(10 * u)) + int(12 * u)
        if atd and not args.no_minimap:
            draw_minimap(frame, tx, u, players, ball, meta, home, args.radius,
                         out_w - m - mm_w, out_h - m - mm_h, mm_w, mm_h, carrier=carrier, hull=args.shape,
                         regions=not args.no_regions)

        frame = tx.flush(frame)
        writer.write(frame)
        if (f - start) % 250 == 0:
            print(f"  {f - start}/{total}", end="\r", flush=True)

    writer.release()
    cap.release()
    print()
    if pred is not None and (agree + disagree):
        print(f"frame agreement on labelled frames: {agree}/{agree + disagree} = {agree / (agree + disagree):.4f}")
    print("done:", out_path)


if __name__ == "__main__":
    main()

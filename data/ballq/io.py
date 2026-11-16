"""Readers for the raw files. The only module that opens a file. Each reader validates what it
reads and fails loudly before any computation.

read_atd(txt, xml)        -> dict with xy (n, S, 2), ball (n, 2), speed_data (n, S), slot ids, teams, home id, attack direction
read_homography(csv)      -> (index {frame: row}, H (m, 3, 3), err (m,), valid (m,))
read_tags(xml)            -> [{a, b, code, team, type}] frames from video start
read_gt(csv)              -> {frame: label}
load_world(paths, cfg, start, end) -> the dict the filter consumes
"""
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|NaN")


def _tag(el):
    return el.tag.split("}", 1)[-1]


def parse_atd_meta(xml_path):
    """teams, players (track slots), channels and the format spec from the Metrica EPTS xml"""
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
            shirt = next((c.text for c in el if _tag(c) == "ShirtNumber"), None)
            players[pid] = {"team": el.get("teamId"), "slot": (shirt or pid).strip()}
        elif t == "PlayerChannel":
            channels[el.get("id")] = (el.get("playerId"), el.get("channelId"))
        elif t == "DataFormatSpecification":
            spec = el
            return
        for c in el:
            walk(c)

    walk(root)
    if not teams or len(teams) < 2:
        raise ValueError(f"{xml_path}: expected two teams, found {len(teams)}")
    if spec is None or not channels:
        raise ValueError(f"{xml_path}: no DataFormatSpecification or PlayerChannels")
    return {"teams": teams, "players": players, "channels": channels, "spec": spec, "attack": attack}


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
        return out["p"], out["b"]

    return decode


def read_atd(txt_path, xml_path, start, end):
    """positions for video frames start..end (frame = line id - 1)"""
    meta = parse_atd_meta(xml_path)
    decode = build_decoder(meta["spec"], meta["channels"])
    ids = list(meta["players"]); col = {pid: k for k, pid in enumerate(ids)}
    n, S = end - start + 1, len(ids)
    xy = np.full((n, S, 2), np.nan, np.float32); sp = np.full((n, S), np.nan, np.float32); ball = np.full((n, 2), np.nan, np.float32)
    seen = 0
    with open(txt_path) as fh:
        for line in fh:
            sep = line.find(":")
            if sep < 0:
                continue
            try:
                vf = int(line[:sep]) - 1
            except ValueError:
                continue
            if vf < start:
                continue
            if vf > end:
                break
            pl, b = decode(line); r = vf - start; seen += 1
            for pid, v in pl.items():
                k = col.get(pid)
                if k is not None and "x" in v and "y" in v and np.isfinite(v["x"]) and np.isfinite(v["y"]):
                    xy[r, k] = (v["x"], v["y"]); sp[r, k] = v.get("s", np.nan)
            if "x" in b and "y" in b and np.isfinite(b["x"]) and np.isfinite(b["y"]):
                ball[r] = (b["x"], b["y"])
    if seen == 0:
        raise ValueError(f"{txt_path}: no frames in {start}..{end}")
    home = meta["teams"][0][0]
    team = np.array(["H" if meta["players"][pid]["team"] == home else "A" for pid in ids])
    fin = xy[np.isfinite(xy)]
    if fin.size and np.mean((fin < -0.05) | (fin > 1.05)) > 0.25:
        raise ValueError("more than a quarter of positions outside 0..1: not the expected normalised ATD")
    return {"xy": xy, "ball_obs": ball, "speed_data": sp, "team": team, "slot_ids": [meta["players"][p]["slot"] for p in ids],
            "team_names": {"H": meta["teams"][0][1], "A": meta["teams"][1][1]}, "frames_seen": seen,
            "attack_first_half": meta["attack"].get(home)}


def read_homography(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["frame", "reprojection_error", "valid"] + [f"h{i}{j}" for i in range(3) for j in range(3)]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    H = df[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy(float).reshape(-1, 3, 3)
    return {int(f): i for i, f in enumerate(df["frame"])}, H, df["reprojection_error"].to_numpy(float), df["valid"].to_numpy(int)


def read_tags(xml_path, fps=25):
    out = []
    for inst in ET.parse(xml_path).iter("instance"):
        try:
            s0, s1 = float(inst.findtext("start")), float(inst.findtext("end"))
        except (TypeError, ValueError):
            continue
        labels = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append({"a": int(round(s0 * fps)), "b": int(round(s1 * fps)), "code": (inst.findtext("code") or "").strip(),
                    "team": labels.get("Team"), "type": labels.get("Type")})
    return sorted(out, key=lambda e: e["a"])


def read_gt(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "frame" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path}: expected frame,label")
    return dict(zip(df["frame"].astype(int), df["label"].astype(str).str.strip().str.upper()))


RESTART_SPOTS = {  # pitch units for the restart prior; which end comes from the attack direction and half
    "corner": None, "goal kick": None, "kick off": (0.5, 0.5), "penalty": None,
}


def load_world(paths, cfg, start, end):
    """paths: {atd_txt, atd_xml, homography?, tags?, gt?}"""
    w = read_atd(paths["atd_txt"], paths["atd_xml"], start, end)
    n = end - start + 1
    w["fps"] = cfg["fps"]; w["pitch"] = tuple(cfg["pitch_m"]); w["start"] = start; w["end"] = end
    w["hom_err"] = np.full(n, np.nan)
    if paths.get("homography"):
        idx, H, err, valid = read_homography(paths["homography"])
        for r in range(n):
            i = idx.get(start + r)
            w["hom_err"][r] = err[i] if (i is not None and valid[i]) else np.inf
        w["homography"] = (idx, H, valid)
    w["tags"] = read_tags(paths["tags"], cfg["fps"]) if paths.get("tags") else []
    w["gt"] = read_gt(paths["gt"]) if paths.get("gt") else None
    from .features import dead_windows
    # dead-ball windows come from the set-piece tags. The GT labels are hand labels on this game and
    # rule 1 keeps them out of the pipeline; they are loaded only to report agreement. The switch below
    # exists for experiments on other games where that rule does not apply.
    gt_for_dead = w["gt"] if cfg["features"].get("dead_from_gt", False) else None
    w["dead"] = dead_windows(n, start, gt_for_dead, w["tags"], cfg["fps"])
    # restart priors: the centre spot at a kick-off tag end; corners and goal kicks need the end, left to the possession side
    w["restarts"] = []
    for e in w["tags"]:
        if e["code"].upper().startswith("SET PIECE") and e.get("type") and "kick off" in e["type"].lower().replace("-", " "):
            r = e["b"] - start
            if 0 <= r < n:
                w["restarts"].append((r, 0.5, 0.5))
    return w

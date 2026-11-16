"""Option 3: kit colours from outside the video. A provider answers "which kits can this team wear"
by team name, as read from the ATD xml. Two providers:

  FileKitDB   a yaml file next to the config, {team name: {home: '#rrggbb', away: ..., third: ...}}
              the sane default: offline, reviewable, extend by hand for a new club
  HttpKitDB   a public sports database (TheSportsDB free endpoint), used when a file has no entry
              and the network allows; any failure returns nothing rather than an error

resolve_kits combines the two sources with the measured colours (option 2):
  the measured team colour picks the closest listed kit, which tells the kit name and confirms light/dark;
  when the video gives nothing (no homography, no video) the database alone decides the light team,
  by the kit the provider lists first (home for the home team, away for the away team).
"""
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import yaml


def _norm(name):
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def hex_to_lab(hx):
    """'#rrggbb' -> (L, a, b) on the OpenCV scale, so it compares with the measured colours"""
    import cv2
    hx = hx.lstrip("#")
    r, g, b = (int(hx[i:i + 2], 16) for i in (0, 2, 4))
    lab = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0, 0]
    return tuple(float(v) for v in lab)


class FileKitDB:
    def __init__(self, path):
        self.kits = {}
        if path and os.path.exists(path):
            raw = yaml.safe_load(open(path)) or {}
            self.kits = {_norm(k): v for k, v in raw.items()}

    def lookup(self, team_name):
        """-> {kit name: '#rrggbb'} or {}"""
        hit = self.kits.get(_norm(team_name))
        if hit:
            return dict(hit)
        # tolerate 'FC Barcelona' vs 'Barcelona'
        n = _norm(team_name)
        for k, v in self.kits.items():
            if n and (n in k or k in n):
                return dict(v)
        return {}


class HttpKitDB:
    URL = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={}"

    def lookup(self, team_name, timeout=5):
        try:
            with urllib.request.urlopen(self.URL.format(urllib.parse.quote(team_name)), timeout=timeout) as r:
                data = json.load(r)
        except Exception:  # noqa: BLE001  network is optional
            return {}
        teams = data.get("teams") or []
        if not teams:
            return {}
        t = teams[0]
        out = {}
        for k, name in (("strKitColour1", "home"), ("strKitColour2", "away"), ("strKitColour3", "third")):
            v = t.get(k)
            if v and re.fullmatch(r"#?[0-9a-fA-F]{6}", v):
                out[name] = "#" + v.lstrip("#")
        return out


def lookup_kits(team_names, providers):
    """team_names: {'H': name, 'A': name}. First provider with an answer wins, per team."""
    out = {}
    for side, name in team_names.items():
        kits = {}
        for p in providers:
            kits = p.lookup(name)
            if kits:
                break
        out[side] = kits
    return out


def resolve_kits(db_kits, measured_lab, white_lightness_min, white_chroma_max):
    """db_kits: {'H': {kit: hex}, 'A': {...}}; measured_lab: {'H': [L,a,b] or None, 'A': ...}
    -> {'white_team', 'kit_used': {'H': name or None, ...}, 'source': 'video'|'database'|'none', 'notes': [...]}"""
    from .colours import is_light
    kit_used, notes, light_by_db = {}, [], {}
    for side in ("H", "A"):
        kits = db_kits.get(side) or {}
        labs = {k: hex_to_lab(v) for k, v in kits.items()}
        m = measured_lab.get(side)
        if m is not None and labs:
            best = min(labs, key=lambda k: float(np.linalg.norm(np.array(labs[k]) - np.array(m))))
            kit_used[side] = best
            d = float(np.linalg.norm(np.array(labs[best]) - np.array(m)))
            if d > 60:
                notes.append(f"{side}: measured colour is {d:.0f} Lab units from the closest listed kit ({best}); the kit list may be stale")
        elif labs:
            # no measurement: assume the usual kit, home for the home side, away for the away side
            default = "home" if side == "H" else "away"
            kit_used[side] = default if default in labs else next(iter(labs))
        else:
            kit_used[side] = None
        if kit_used[side]:
            light_by_db[side] = is_light(labs[kit_used[side]], white_lightness_min, white_chroma_max)
    measured_light = [s for s in ("H", "A") if measured_lab.get(s) is not None and is_light(measured_lab[s], white_lightness_min, white_chroma_max)]
    if len(measured_light) == 1:
        white, source = measured_light[0], "video"
        if light_by_db and light_by_db.get(white) is False and any(light_by_db.values()):
            notes.append("video says the light team is " + white + " but the database kit for it is dark; trusting the video")
    elif len(measured_light) == 2:
        white, source = None, "video"
        notes.append("both teams measure light: shirt sampling is off or the kits clash; set override_white_team")
    else:
        db_light = [s for s, v in light_by_db.items() if v]
        white, source = (db_light[0], "database") if len(db_light) == 1 else (None, "none")
    return {"white_team": white, "kit_used": kit_used, "source": source, "notes": notes}

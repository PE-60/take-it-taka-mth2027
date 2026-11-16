"""Shirt colours from the video, so the pipeline knows which team wears the light kit on this
recording and which tracked slot does not look like its team. Works for any two teams: nothing
here knows a club name.

shirt_point     where a player's shirt is in the image, from the ground point and the homography
sample_patch_lab median Lab colour of a small patch
measure_slots    run over sampled frames -> per slot median colour
classify_kits    light team, per slot mislabel flags
"""
import numpy as np


def shirt_point(H, x, y, shirt_height_m, pitch, frame_h):
    """Project the ground point, then go up the image by the pixel size of shirt_height_m at that spot.
    The vertical scale is taken from how far one metre along y moves in the image at that point:
    a cheap stand-in for the camera height, good enough to land on the torso."""
    def proj(px, py):
        w = H[2, 0] * px + H[2, 1] * py + H[2, 2]
        return (H[0, 0] * px + H[0, 1] * py + H[0, 2]) / w, (H[1, 0] * px + H[1, 1] * py + H[1, 2]) / w
    fx, fy = proj(x, y)
    _, fy2 = proj(x, y + 1.0 / pitch[1])              # one metre further along y, on the ground
    px_per_m = abs(fy2 - fy)
    return fx, max(0.0, fy - 0.8 * shirt_height_m * px_per_m)


def sample_patch_lab(img_bgr, fx, fy, patch_px):
    import cv2
    h, w = img_bgr.shape[:2]
    x0, x1 = int(max(0, fx - patch_px / 2)), int(min(w, fx + patch_px / 2))
    y0, y1 = int(max(0, fy - patch_px / 2)), int(min(h, fy + patch_px / 2))
    if x1 <= x0 or y1 <= y0:
        return None
    lab = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3)
    med = np.median(lab, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def measure_slots(video_path, frames, xy, hom, start, slot_ids, pitch, ccfg, progress=None):
    """frames: export-relative indices to sample. Returns {slot_id: (L, a, b, n)} medians."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples = {sid: [] for sid in slot_ids}
    for i, r in enumerate(frames):
        H = hom(r)
        if H is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + r)
        ok, img = cap.read()
        if not ok:
            continue
        for k, sid in enumerate(slot_ids):
            if not np.isfinite(xy[r, k, 0]):
                continue
            fx, fy = shirt_point(H, float(xy[r, k, 0]), float(xy[r, k, 1]), ccfg["shirt_height_m"], pitch, fh)
            s = sample_patch_lab(img, fx, fy, ccfg["patch_px"])
            if s is not None:
                samples[sid].append(s)
        if progress:
            progress(i + 1, len(frames))
    cap.release()
    out = {}
    for sid, L in samples.items():
        if L:
            a = np.array(L); med = np.median(a, axis=0); out[sid] = (float(med[0]), float(med[1]), float(med[2]), len(L))
    return out


def is_light(lab, white_lightness_min, white_chroma_max):
    """bright and grey in Lab (OpenCV scale: L 0..255, a and b centred on 128)"""
    return lab[0] >= white_lightness_min and float(np.hypot(lab[1] - 128, lab[2] - 128)) <= white_chroma_max


def classify_kits(slot_lab, slot_team, white_lightness_min, white_chroma_max, third_dist=60.0):
    """slot_lab: {slot: (L, a, b[, n])}; slot_team: {slot: 'H'|'A'}.
    Team colour = median over its slots. The light team is the one whose median is bright and grey.
    A slot is a mislabel when it is closer to the other team's colour than to its own by a clear margin.
    A slot far from both team colours is a third kit: a goalkeeper, the referee, or a bad track."""
    teams = {}
    for t in ("H", "A"):
        pts = np.array([slot_lab[s][:3] for s in slot_lab if slot_team.get(s) == t])
        teams[t] = np.median(pts, axis=0) if len(pts) else None
    white = None
    for t, c in teams.items():
        if c is not None and is_light(c, white_lightness_min, white_chroma_max):
            white = t
    slots = {}
    for s, lab in slot_lab.items():
        t = slot_team.get(s); o = "A" if t == "H" else "H"
        own = float(np.linalg.norm(np.array(lab[:3]) - teams[t])) if teams.get(t) is not None else np.nan
        oth = float(np.linalg.norm(np.array(lab[:3]) - teams[o])) if teams.get(o) is not None else np.nan
        mis = bool(np.isfinite(own) and np.isfinite(oth) and oth < own * 0.6)
        third = bool(np.isfinite(own) and np.isfinite(oth) and min(own, oth) > third_dist)
        slots[s] = {"L": lab[0], "a": lab[1], "b": lab[2], "d_own": own, "d_other": oth, "mislabel": mis, "third": third, "n": lab[3] if len(lab) > 3 else None}
    return {"white_team": white, "team_lab": {t: (None if c is None else [float(v) for v in c]) for t, c in teams.items()}, "slots": slots}

"""Per frame features from the raw positions. Pure numpy; the filter and the possession rules read this."""
import numpy as np


def ball_speed_frames(ball, pitch, fps, smooth=3):
    """m/s from consecutive positions, light box smoothing, NaN where a position is missing"""
    d = np.diff(ball, axis=0) * np.array(pitch, dtype=np.float32)
    sp = np.hypot(d[:, 0], d[:, 1]) * fps
    sp = np.concatenate([[np.nan], sp])
    k = np.ones(smooth) / smooth
    valid = np.isfinite(sp)
    num = np.convolve(np.nan_to_num(sp), k, "same")
    den = np.convolve(valid.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = num / np.maximum(den, 1e-9)
    sm[~valid] = np.nan
    return sm


def ball_accel(speed, fps):
    return np.concatenate([[np.nan], np.diff(speed)]) * fps


def ball_accel_xy(ball, pitch, fps, smooth=3):
    """magnitude of the xy acceleration in m/s^2 (Link and Hoernig use the vector, so a deflection at
    constant speed counts as a ball action too); NaN where a position is missing"""
    p = ball * np.array(pitch, dtype=np.float32)
    v = np.diff(p, axis=0) * fps
    k = np.ones(smooth) / smooth
    for c in range(2):
        valid = np.isfinite(v[:, c]); col = np.where(valid, v[:, c], 0.0)
        num = np.convolve(col, k, mode="same"); den = np.convolve(valid.astype(float), k, mode="same")
        v[:, c] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    a = np.diff(v, axis=0) * fps
    mag = np.hypot(a[:, 0], a[:, 1])
    return np.concatenate([[np.nan, np.nan], mag])


def kicks_from_accel(speed, fps, accel_min, min_gap_s, accel=None):
    """Link and Hoernig: a kick is a local maximum of acceleration above the threshold.
    accel: xy acceleration magnitude per frame (ball_accel_xy); falls back to the speed derivative."""
    a = accel if accel is not None else ball_accel(speed, fps)
    a = np.where(np.isfinite(a), a, -np.inf)
    kicks, last = [], -10 ** 9
    for i in range(1, len(a) - 1):
        if a[i] >= accel_min and a[i] >= a[i - 1] and a[i] >= a[i + 1] and i - last >= int(min_gap_s * fps):
            kicks.append(i); last = i
    return kicks


def nearest_player_m(ball, xy, pitch):
    """distance in metres to the nearest tracked player per frame, and its slot (-1 when none)"""
    n, S = xy.shape[:2]
    out_d = np.full(n, np.inf); out_k = np.full(n, -1, int)
    for r in range(n):
        if not np.all(np.isfinite(ball[r])):
            continue
        d = np.hypot((xy[r, :, 0] - ball[r, 0]) * pitch[0], (xy[r, :, 1] - ball[r, 1]) * pitch[1])
        d = np.where(np.isfinite(d), d, np.inf)
        k = int(np.argmin(d))
        if np.isfinite(d[k]):
            out_d[r], out_k[r] = d[k], k
    return out_d, out_k


def air_ball_flags(speed, accel, nearest_d, speed_cap, accel_min, near_m):
    """implied speed beyond any shot, or a kick-sized acceleration with nobody near the ball"""
    sp = np.nan_to_num(speed, nan=0.0); ac = np.nan_to_num(accel, nan=0.0)
    return (sp > speed_cap) | ((np.abs(ac) >= accel_min) & (nearest_d > near_m))


def dead_windows(n, start, gt=None, tags=None, fps=25):
    """frames inside a dead ball: GT 'D' where we have it, else inside a set-piece tag window"""
    dead = np.zeros(n, bool)
    if gt is not None:
        for r in range(n):
            if gt.get(start + r) == "D":
                dead[r] = True
    for e in tags or []:
        if str(e.get("code", "")).upper().startswith("SET PIECE"):
            a, b = max(e["a"] - start, 0), min(e["b"] - start, n)
            if b > a:
                dead[a:b] = True
    return dead


def white_blend_flags(nearest_d, nearest_k, slot_team, white_team, radius_m):
    if white_team is None:
        return np.zeros(len(nearest_d), bool)
    out = np.zeros(len(nearest_d), bool)
    for r in range(len(nearest_d)):
        k = nearest_k[r]
        if k >= 0 and nearest_d[r] <= radius_m and slot_team[k] == white_team:
            out[r] = True
    return out

"""Stage C: which outfield team's player touched the ball at a kick contact."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ball_detection.appearance import AppearanceBanks, grass_mask_hsv, hist_intersection, hsv_histogram, bgr_to_hsv
from ball_detection.banks import KitCropParams, crop_torso
from ball_detection.geometry import pitch_distance_m, project_points, unproject_points
from ball_detection.gk import default_goalkeeper_slot_ids, outfield_slot_mask
from ball_detection.results import BallDetection, KickBallTrack


@dataclass(frozen=True)
class AttributeParams:
    max_dist_m: float = 3.0
    max_dist_px: float = 90.0
    w_dist: float = 0.55
    w_kit: float = 0.45
    min_score: float = 0.15
    # Softmax-ish margin: confidence grows when best beats second-best.
    margin_scale: float = 0.25
    kit: KitCropParams = KitCropParams()


@dataclass
class PlayerCandidate:
    slot_id: str
    team: str                 # ATD team label
    dist_m: float
    dist_px: float
    kit_H: float = 0.0
    kit_A: float = 0.0
    kit_vote: str = "?"       # H / A / ?
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "team": self.team,
            "dist_m": float(self.dist_m),
            "dist_px": float(self.dist_px),
            "kit_H": float(self.kit_H),
            "kit_A": float(self.kit_A),
            "kit_vote": self.kit_vote,
            "score": float(self.score),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerCandidate":
        return cls(
            slot_id=str(data.get("slot_id") or ""),
            team=str(data.get("team") or "?"),
            dist_m=float(data.get("dist_m") or 0.0),
            dist_px=float(data.get("dist_px") or 0.0),
            kit_H=float(data.get("kit_H") or 0.0),
            kit_A=float(data.get("kit_A") or 0.0),
            kit_vote=str(data.get("kit_vote") or "?"),
            score=float(data.get("score") or 0.0),
        )


@dataclass
class TouchAttribution:
    kick_frame: int
    contact: BallDetection | None = None
    team: str = ""              # H / A / "" 
    confidence: float = 0.0
    slot_id: str = ""
    kit_vote: str = "?"
    method: str = "none"        # dist_kit | dist_only | none
    candidates: list[PlayerCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kick_frame": self.kick_frame,
            "contact": None if self.contact is None else self.contact.to_dict(),
            "team": self.team,
            "confidence": float(self.confidence),
            "slot_id": self.slot_id,
            "kit_vote": self.kit_vote,
            "method": self.method,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TouchAttribution":
        contact = data.get("contact")
        return cls(
            kick_frame=int(data["kick_frame"]),
            contact=None if contact is None else BallDetection.from_dict(contact),
            team=str(data.get("team") or ""),
            confidence=float(data.get("confidence") or 0.0),
            slot_id=str(data.get("slot_id") or ""),
            kit_vote=str(data.get("kit_vote") or "?"),
            method=str(data.get("method") or "none"),
            candidates=[PlayerCandidate.from_dict(c) for c in (data.get("candidates") or [])],
        )


def _kit_hist_at(
        frame_bgr: np.ndarray,
        player_xy: np.ndarray,
        H: np.ndarray,
        params: KitCropParams,
) -> np.ndarray | None:
    pix = project_points(H, np.asarray(player_xy, dtype=np.float64).reshape(1, 2))[0]
    if not np.isfinite(pix).all():
        return None
    u, v = float(pix[0]), float(pix[1])
    cropped = crop_torso(frame_bgr, u, v, params)
    if cropped is None:
        return None
    patch, _ = cropped
    mask = np.ones(patch.shape[:2], dtype=bool)
    hsv = bgr_to_hsv(patch)
    mask = mask & ~grass_mask_hsv(hsv)
    if int(mask.sum()) < params.min_patch:
        mask = np.ones(patch.shape[:2], dtype=bool)
    hist = hsv_histogram(patch, mask=mask.astype(np.uint8) * 255)
    return hist if float(hist.sum()) > 0 else None


def _proximity(dist: float, max_dist: float) -> float:
    if not np.isfinite(dist) or max_dist <= 0:
        return 0.0
    return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))


def score_candidates(
        contact: BallDetection,
        players_xy: np.ndarray,
        team: np.ndarray,
        slot_ids,
        H: np.ndarray,
        pitch_m: tuple[float, float],
        *,
        frame_bgr: np.ndarray | None = None,
        banks: AppearanceBanks | None = None,
        gk_ids=None,
        params: AttributeParams | None = None,
) -> list[PlayerCandidate]:
    """Rank nearby outfield players for a contact detection."""
    params = params or AttributeParams()
    players_xy = np.asarray(players_xy, dtype=np.float64)
    team = np.asarray(team)
    S = len(players_xy)
    if team.ndim != 1 or len(team) != S:
        raise ValueError("team must be shape (S,)")
    outfield = outfield_slot_mask(
        slot_ids if slot_ids is not None else [str(i) for i in range(S)],
        gk_ids if gk_ids is not None else default_goalkeeper_slot_ids(),
    )

    contact_pitch = None
    if np.isfinite(H).all():
        try:
            contact_pitch = unproject_points(
                H, np.array([[contact.u, contact.v]], dtype=np.float64))[0]
            if not np.isfinite(contact_pitch).all():
                contact_pitch = None
        except np.linalg.LinAlgError:
            contact_pitch = None

    hist_H = banks.kits["H"].mean_hist() if banks and "H" in banks.kits and banks.kits["H"].n_samples else None
    hist_A = banks.kits["A"].mean_hist() if banks and "A" in banks.kits and banks.kits["A"].n_samples else None
    use_kit = frame_bgr is not None and (hist_H is not None or hist_A is not None)

    cands: list[PlayerCandidate] = []
    for s in np.flatnonzero(outfield):
        ply = players_xy[s]
        if not np.isfinite(ply).all():
            continue
        t = str(team[s])
        if t not in ("H", "A"):
            continue

        dist_m = float("inf")
        if contact_pitch is not None:
            dist_m = float(pitch_distance_m(contact_pitch, ply, pitch_m))

        pix = project_points(H, ply.reshape(1, 2))[0]
        if np.isfinite(pix).all():
            dist_px = float(np.hypot(pix[0] - contact.u, pix[1] - contact.v))
        else:
            dist_px = float("inf")

        if dist_m > params.max_dist_m and dist_px > params.max_dist_px:
            continue

        kit_H = kit_A = 0.0
        kit_vote = "?"
        if use_kit:
            hist = _kit_hist_at(frame_bgr, ply, H, params.kit)
            if hist is not None:
                if hist_H is not None:
                    kit_H = hist_intersection(hist, hist_H)
                if hist_A is not None:
                    kit_A = hist_intersection(hist, hist_A)
                if kit_H > kit_A + 0.02:
                    kit_vote = "H"
                elif kit_A > kit_H + 0.02:
                    kit_vote = "A"

        prox_m = _proximity(dist_m, params.max_dist_m)
        prox_px = _proximity(dist_px, params.max_dist_px)
        prox = 0.5 * (prox_m + prox_px) if np.isfinite(dist_m) and np.isfinite(dist_px) else max(prox_m, prox_px)

        if use_kit:
            # Prefer kit agreement with ATD team; still allow kit-only signal.
            kit_for_team = kit_H if t == "H" else kit_A
            kit_best = max(kit_H, kit_A)
            kit_term = 0.7 * kit_for_team + 0.3 * kit_best
            score = params.w_dist * prox + params.w_kit * kit_term
        else:
            score = prox

        cands.append(PlayerCandidate(
            slot_id=str(slot_ids[s]) if slot_ids is not None else str(s),
            team=t,
            dist_m=dist_m if np.isfinite(dist_m) else 1e9,
            dist_px=dist_px if np.isfinite(dist_px) else 1e9,
            kit_H=kit_H,
            kit_A=kit_A,
            kit_vote=kit_vote,
            score=float(score),
        ))

    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def attribute_touch(
        track: KickBallTrack,
        players_xy: np.ndarray,
        team: np.ndarray,
        slot_ids,
        H: np.ndarray,
        pitch_m: tuple[float, float],
        *,
        frame_bgr: np.ndarray | None = None,
        banks: AppearanceBanks | None = None,
        gk_ids=None,
        params: AttributeParams | None = None,
) -> TouchAttribution:
    """Attribute a kick track's contact to an outfield team/player."""
    params = params or AttributeParams()
    out = TouchAttribution(kick_frame=track.kick_frame, contact=track.contact)
    if track.contact is None:
        return out

    cands = score_candidates(
        track.contact, players_xy, team, slot_ids, H, pitch_m,
        frame_bgr=frame_bgr, banks=banks, gk_ids=gk_ids, params=params,
    )
    out.candidates = cands[:8]
    if not cands or cands[0].score < params.min_score:
        return out

    best = cands[0]
    second = cands[1].score if len(cands) > 1 else 0.0
    margin = max(0.0, best.score - second)
    conf = float(np.clip(best.score * (0.6 + 0.4 * (margin / max(params.margin_scale, 1e-6))), 0.0, 1.0))

    use_kit = frame_bgr is not None and banks is not None and any(
        k in banks.kits and banks.kits[k].n_samples > 0 for k in ("H", "A"))
    # If kit vote strongly disagrees with ATD team, soft-penalize confidence.
    if use_kit and best.kit_vote in ("H", "A") and best.kit_vote != best.team:
        conf *= 0.7

    out.team = best.team
    out.slot_id = best.slot_id
    out.kit_vote = best.kit_vote
    out.confidence = conf
    out.method = "dist_kit" if use_kit else "dist_only"
    return out


def attribute_tracks(
        tracks: list[KickBallTrack],
        *,
        frames_by_video_frame: dict[int, np.ndarray] | None,
        players_xy_by_index: dict[int, np.ndarray],
        team: np.ndarray,
        slot_ids,
        H_by_index: dict[int, np.ndarray],
        pitch_m: tuple[float, float],
        track_index_of_video_frame,
        banks: AppearanceBanks | None = None,
        gk_ids=None,
        params: AttributeParams | None = None,
) -> list[TouchAttribution]:
    """Attribute each track. ``track_index_of_video_frame(f) -> tracking row or None``."""
    out: list[TouchAttribution] = []
    for tr in tracks:
        contact_frame = tr.contact.frame if tr.contact is not None else tr.kick_frame
        idx = track_index_of_video_frame(contact_frame)
        if idx is None:
            out.append(TouchAttribution(kick_frame=tr.kick_frame, contact=tr.contact))
            continue
        ply = players_xy_by_index.get(idx)
        H = H_by_index.get(idx)
        if ply is None or H is None:
            out.append(TouchAttribution(kick_frame=tr.kick_frame, contact=tr.contact))
            continue
        frame = None if frames_by_video_frame is None else frames_by_video_frame.get(contact_frame)
        out.append(attribute_touch(
            tr, ply, team, slot_ids, H, pitch_m,
            frame_bgr=frame, banks=banks, gk_ids=gk_ids, params=params,
        ))
    return out

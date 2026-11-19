"""Stage B result schema (JSON-serialisable)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BallDetection:
    frame: int                 # absolute video frame when known; else window-local index
    u: float                   # pixel x
    v: float                   # pixel y
    score: float
    radius_px: float
    hist_score: float = 0.0
    motion_score: float = 0.0
    prior_score: float = 0.0
    source: str = "blob"       # blob | miss

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BallDetection":
        return cls(
            frame=int(data["frame"]),
            u=float(data["u"]),
            v=float(data["v"]),
            score=float(data["score"]),
            radius_px=float(data.get("radius_px") or 0.0),
            hist_score=float(data.get("hist_score") or 0.0),
            motion_score=float(data.get("motion_score") or 0.0),
            prior_score=float(data.get("prior_score") or 0.0),
            source=str(data.get("source") or "blob"),
        )


@dataclass
class KickBallTrack:
    kick_frame: int
    window_start: int
    window_end: int
    detections: list[BallDetection] = field(default_factory=list)
    # Detection nearest the kick frame (or best score if kick frame missed).
    contact: BallDetection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kick_frame": self.kick_frame,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "detections": [d.to_dict() for d in self.detections],
            "contact": None if self.contact is None else self.contact.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KickBallTrack":
        dets = [BallDetection.from_dict(d) for d in (data.get("detections") or [])]
        contact = data.get("contact")
        return cls(
            kick_frame=int(data["kick_frame"]),
            window_start=int(data["window_start"]),
            window_end=int(data["window_end"]),
            detections=dets,
            contact=None if contact is None else BallDetection.from_dict(contact),
        )


@dataclass
class DetectExport:
    version: int = 1
    game: str = "BAR-ATH"
    source: str = "tactical"
    banks_path: str = ""
    tracks: list[KickBallTrack] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "game": self.game,
            "source": self.source,
            "banks_path": self.banks_path,
            "n_kicks": len(self.tracks),
            "tracks": [t.to_dict() for t in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectExport":
        return cls(
            version=int(data.get("version") or 1),
            game=str(data.get("game") or "BAR-ATH"),
            source=str(data.get("source") or "tactical"),
            banks_path=str(data.get("banks_path") or ""),
            tracks=[KickBallTrack.from_dict(t) for t in (data.get("tracks") or [])],
        )


def save_detections(export: DetectExport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(export.to_dict(), fh, indent=2)
        fh.write("\n")


def load_detections(path: str) -> DetectExport:
    with open(path, encoding="utf-8") as fh:
        return DetectExport.from_dict(json.load(fh))


@dataclass
class AttributionExport:
    """Stage C output: one touch attribution per kick track."""
    version: int = 1
    game: str = "BAR-ATH"
    source: str = "tactical"
    banks_path: str = ""
    detections_path: str = ""
    attributions: list[Any] = field(default_factory=list)  # TouchAttribution; typed loosely to avoid cycle

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "game": self.game,
            "source": self.source,
            "banks_path": self.banks_path,
            "detections_path": self.detections_path,
            "n_attributions": len(self.attributions),
            "attributions": [a.to_dict() for a in self.attributions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttributionExport":
        from ball_detection.attribute import TouchAttribution
        return cls(
            version=int(data.get("version") or 1),
            game=str(data.get("game") or "BAR-ATH"),
            source=str(data.get("source") or "tactical"),
            banks_path=str(data.get("banks_path") or ""),
            detections_path=str(data.get("detections_path") or ""),
            attributions=[TouchAttribution.from_dict(a) for a in (data.get("attributions") or [])],
        )


def save_attributions(export: AttributionExport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(export.to_dict(), fh, indent=2)
        fh.write("\n")


def load_attributions(path: str) -> AttributionExport:
    with open(path, encoding="utf-8") as fh:
        return AttributionExport.from_dict(json.load(fh))

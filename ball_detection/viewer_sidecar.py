"""Flatten Stage B/C exports into a match_viewer-friendly sidecar JSON.

The HTML viewer loads ``ball_detection.json`` (absolute video frames) the same way it
loads referee sidecars: pixel overlays + candidate stepping on kick contacts.
"""

from __future__ import annotations

import json
from typing import Any

from ball_detection.attribute import TouchAttribution
from ball_detection.results import AttributionExport, DetectExport, KickBallTrack


TEAM_HEX = {"H": "#3c78c8", "A": "#c82828", "": "#ffdc00", "?": "#ffdc00"}


def track_events(track: KickBallTrack, attr: TouchAttribution | None = None) -> list[dict[str, Any]]:
    """One event per detection in the window, plus a contact summary on the kick frame."""
    events: list[dict[str, Any]] = []
    contact_uv = None
    if attr is not None and attr.contact is not None:
        contact_uv = (attr.contact.u, attr.contact.v)
    elif track.contact is not None:
        contact_uv = (track.contact.u, track.contact.v)

    for det in track.detections:
        is_contact = (contact_uv is not None
                      and det.frame == (attr.contact.frame if attr and attr.contact else track.kick_frame)
                      and abs(det.u - contact_uv[0]) < 0.5
                      and abs(det.v - contact_uv[1]) < 0.5)
        team = (attr.team if attr and is_contact else "") or ""
        events.append({
            "f": int(det.frame),
            "kind": "contact" if is_contact else "ball",
            "u": float(det.u),
            "v": float(det.v),
            "radius_px": float(det.radius_px),
            "score": float(det.score),
            "team": team,
            "confidence": float(attr.confidence) if attr and is_contact else float(det.score),
            "slot_id": attr.slot_id if attr and is_contact else "",
            "kit_vote": attr.kit_vote if attr and is_contact else "",
            "kick_frame": int(track.kick_frame),
            "hex": TEAM_HEX.get(team, TEAM_HEX[""]),
        })

    # Ensure a contact marker exists even when blob detection missed the kick frame.
    if attr is not None and attr.contact is not None:
        has_contact = any(e["kind"] == "contact" for e in events)
        if not has_contact:
            c = attr.contact
            events.append({
                "f": int(c.frame),
                "kind": "contact",
                "u": float(c.u),
                "v": float(c.v),
                "radius_px": float(c.radius_px),
                "score": float(c.score),
                "team": attr.team,
                "confidence": float(attr.confidence),
                "slot_id": attr.slot_id,
                "kit_vote": attr.kit_vote,
                "kick_frame": int(track.kick_frame),
                "hex": TEAM_HEX.get(attr.team, TEAM_HEX[""]),
            })
    return events


def build_sidecar(
        detections: DetectExport,
        attributions: AttributionExport | None = None,
) -> dict[str, Any]:
    attrs_by_kick: dict[int, TouchAttribution] = {}
    if attributions is not None:
        for a in attributions.attributions:
            attrs_by_kick[int(a.kick_frame)] = a

    events: list[dict[str, Any]] = []
    contact_frames: list[int] = []
    for track in detections.tracks:
        attr = attrs_by_kick.get(int(track.kick_frame))
        evs = track_events(track, attr)
        events.extend(evs)
        for e in evs:
            if e["kind"] == "contact":
                contact_frames.append(int(e["f"]))

    contact_frames = sorted(set(contact_frames))
    return {
        "version": 1,
        "game": detections.game,
        "source": detections.source,
        "n_events": len(events),
        "n_contacts": len(contact_frames),
        "contact_frames": contact_frames,
        "events": events,
    }


def save_sidecar(doc: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def load_sidecar(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

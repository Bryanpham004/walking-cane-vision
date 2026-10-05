"""Turning a stream of detections into a sparse stream of useful alerts.

The hard problem is not detecting obstacles, it is *not* reporting them. A
detector firing at 15fps will produce 15 alerts per second for the same lamp
post, which is worse than silence because the user stops trusting it. Three
mechanisms keep the channel quiet:

1. **Confirmation** - a track must persist ``confirm_frames`` frames before it
   can alert, which kills single-frame false positives.
2. **Cooldown** - one alert per zone per ``cooldown_s``, so a static obstacle
   is announced once as you approach it rather than continuously.
3. **Escalation override** - cooldown is bypassed when urgency *rises*, so a
   warning that becomes critical is never suppressed by its own earlier alert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .config import AlertConfig
from .geometry import Detection, Urgency, Zone, time_to_contact

_URGENCY_RANK = {Urgency.INFO: 0, Urgency.WARN: 1, Urgency.CRITICAL: 2}


@dataclass(frozen=True)
class Alert:
    """A single thing worth telling the user about."""

    urgency: Urgency
    zone: Zone
    label: str
    distance_m: float
    message: str
    time_to_contact_s: Optional[float] = None
    distance_is_calibrated: bool = True


@dataclass
class _Track:
    """Per-zone memory of what we have been seeing and saying."""

    frames_seen: int = 0
    last_distance_m: Optional[float] = None
    last_seen_t: Optional[float] = None
    last_alert_t: Optional[float] = None
    last_urgency: Optional[Urgency] = None


def classify_urgency(
    distance_m: float, ttc_s: Optional[float], config: AlertConfig
) -> Urgency:
    """Urgency from proximity, escalated by how fast the gap is closing."""
    if distance_m <= config.critical_distance_m:
        return Urgency.CRITICAL
    if ttc_s is not None and ttc_s <= config.critical_ttc_s:
        return Urgency.CRITICAL
    if distance_m <= config.warn_distance_m:
        return Urgency.WARN
    return Urgency.INFO


def describe(label: str, zone: Zone, distance_m: float, speak_labels: bool) -> str:
    """Short spoken phrase. Brevity matters: it is heard while walking."""
    noun = label if (speak_labels and label != "obstacle") else "obstacle"
    where = "ahead" if zone is Zone.CENTER else f"to your {zone.value}"
    return f"{noun} {where}, {distance_m:.1f} metres"


class AlertPolicy:
    """Stateful reducer from per-frame detections to emitted alerts."""

    def __init__(self, config: AlertConfig) -> None:
        self._config = config
        self._tracks: dict[Zone, _Track] = {zone: _Track() for zone in Zone}

    def _nearest_per_zone(
        self, detections: Sequence[Detection], frame_width: int
    ) -> dict[Zone, Detection]:
        """Collapse a frame to at most one obstacle per zone - the closest.

        Reporting every detection would flood the user; the nearest obstacle in
        a direction is the only one that changes what they should do.
        """
        best: dict[Zone, Detection] = {}
        for det in detections:
            if det.distance_m is None or det.distance_m > self._config.max_range_m:
                continue
            zone = det.zone(frame_width)
            incumbent = best.get(zone)
            if incumbent is None or det.distance_m < (incumbent.distance_m or 1e9):
                best[zone] = det
        return best

    def update(
        self, detections: Sequence[Detection], frame_width: int, now: float
    ) -> list[Alert]:
        """Fold one frame of detections into zero or more alerts."""
        candidates = self._nearest_per_zone(detections, frame_width)
        alerts: list[Alert] = []

        for zone in Zone:
            track = self._tracks[zone]
            det = candidates.get(zone)

            if det is None:
                # Zone cleared: reset confirmation so a reappearance must be
                # confirmed again, but keep cooldown history.
                track.frames_seen = 0
                track.last_distance_m = None
                track.last_urgency = None
                continue

            distance = float(det.distance_m or 0.0)
            dt = (now - track.last_seen_t) if track.last_seen_t is not None else 0.0
            ttc = time_to_contact(distance, track.last_distance_m, dt)

            track.frames_seen += 1
            track.last_distance_m = distance
            track.last_seen_t = now

            if track.frames_seen < self._config.confirm_frames:
                continue

            urgency = classify_urgency(distance, ttc, self._config)
            if self._suppressed(track, urgency, now):
                continue

            track.last_alert_t = now
            track.last_urgency = urgency
            alerts.append(
                Alert(
                    urgency=urgency,
                    zone=zone,
                    label=det.label,
                    distance_m=round(distance, 2),
                    message=describe(
                        det.label, zone, distance, self._config.speak_labels
                    ),
                    time_to_contact_s=round(ttc, 2) if ttc is not None else None,
                    distance_is_calibrated=det.distance_is_calibrated,
                )
            )

        # Most urgent first, then nearest - the consumer may only act on one.
        alerts.sort(key=lambda a: (-_URGENCY_RANK[a.urgency], a.distance_m))
        return alerts

    def _suppressed(self, track: _Track, urgency: Urgency, now: float) -> bool:
        """Whether cooldown should swallow this alert."""
        if track.last_alert_t is None:
            return False
        escalated = (
            track.last_urgency is not None
            and _URGENCY_RANK[urgency] > _URGENCY_RANK[track.last_urgency]
        )
        if escalated:
            return False
        return (now - track.last_alert_t) < self._config.cooldown_s

    def reset(self) -> None:
        self._tracks = {zone: _Track() for zone in Zone}

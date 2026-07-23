"""
Per-zone decaying occupancy tracking for the Frigate <-> UniFi Protect bridge.

See protocol spec Section 8.5: zone occupancy in real UniFi Protect devices is
a smoothed/decaying aggregate signal, not a hard per-object boolean -- once an
occupying object leaves a zone, the zone's reported "level" decays gradually
over several ticks before the status flips from "moving" to "leave". These
constants approximate that behavior; tune DECAY_PER_TICK against your own
captured traffic if needed.
"""
from dataclasses import dataclass
from typing import Any

# observed: zonesStatus.level runs ~10 below confidenceLevel
ZONE_LEVEL_CONFIDENCE_OFFSET = 10
ZONE_DECAY_PER_TICK = 8


@dataclass
class ZoneState:
    """Per-zone decaying occupancy state. One instance per configured Protect zone ID."""

    level: float = 0.0
    status: str = "none"  # "none" | "enter" | "moving" | "leave"
    was_active: bool = False

    def update(self, occupied_this_tick: bool, best_confidence: float) -> None:
        if occupied_this_tick:
            # Reflect THIS tick's occupancy directly. Do not max against
            # self.level here -- that previously created a ratchet where a
            # zone's level could never drop while occupied, even when the
            # current occupant's confidence was much lower than an earlier
            # tick's value. Cross-occupant maxing (so two simultaneous
            # objects in one zone don't cause flicker) already happens one
            # level up, in ZoneStatusTracker._recompute()'s `occupancy` dict.
            self.level = min(100.0, max(0.0, best_confidence - ZONE_LEVEL_CONFIDENCE_OFFSET))
            self.status = "moving" if self.was_active else "enter"
            self.was_active = True
        elif self.was_active:
            self.level = max(0.0, self.level - ZONE_DECAY_PER_TICK)
            if self.level <= 0.0:
                self.status = "leave"
                self.was_active = False
            else:
                self.status = "moving"  # still decaying, not fully cleared yet
        else:
            self.status = "none"
            self.level = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"level": int(self.level), "status": self.status}


class ZoneStatusTracker:
    """
    Maintains per-zone decaying occupancy across ALL concurrently-active tracks
    on one camera. This is intentionally decoupled from any single Frigate
    event's lifecycle: `zonesStatus` in the real protocol reflects the union of
    every currently-active track's zone membership, not one track's state, and
    a track's own `leave`/closing-summary can reference a completely different
    zone state than what's reported here on the same tick (protocol spec
    Section 7.4/8.5).
    """

    def __init__(self, zone_ids: list[int]) -> None:
        self._zones: dict[int, ZoneState] = {zid: ZoneState() for zid in zone_ids}
        # occupancy contributed by each active track, keyed by tracker_id
        self._track_zone_confidence: dict[int, dict[int, float]] = {}

    def update_track(self, tracker_id: int, zone_ids: list[int], confidence: float) -> None:
        self._track_zone_confidence[tracker_id] = {zid: confidence for zid in zone_ids}
        self._recompute()

    def remove_track(self, tracker_id: int) -> None:
        self._track_zone_confidence.pop(tracker_id, None)
        self._recompute()

    def _recompute(self) -> None:
        # Union of all currently-active tracks' zone membership, using max
        # confidence per zone across occupants so one dipping tracker doesn't
        # flicker a zone that another occupant is still solidly holding.
        occupancy: dict[int, float] = {}
        for zone_conf in self._track_zone_confidence.values():
            for zid, conf in zone_conf.items():
                occupancy[zid] = max(occupancy.get(zid, 0.0), conf)

        for zid, zone_state in self._zones.items():
            if zid in occupancy:
                zone_state.update(True, occupancy[zid])
            else:
                zone_state.update(False, 0.0)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        # Every configured zone must be present on every message, not just
        # the ones that changed (protocol spec Section 3).
        return {str(zid): state.as_dict() for zid, state in self._zones.items()}

    @staticmethod
    def mark_all_departed(zones_status: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Force every non-'none' zone in a zonesStatus dict to 'leave'.

        Used when a track ends: any zone it was still occupying should be
        reported as departed in that track's final update, regardless of the
        tracker's own decay state (which may still say "moving").
        """
        for zone in zones_status.values():
            if zone.get("status") != "none":
                zone["status"] = "leave"
        return zones_status

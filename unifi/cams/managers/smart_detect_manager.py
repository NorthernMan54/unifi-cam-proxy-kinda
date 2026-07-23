"""
Owns the EventSmartDetect (start/update/stationary/stop) lifecycle that used
to live directly on UnifiCamBase.

Extracted in refactor Phase 2. The manager doesn't know how to talk to the
websocket itself -- it's handed `send` and `gen_response` callables from the
cam so it stays testable without a live connection. Similarly `get_uptime`
is injected rather than the manager tracking its own clock, since the wire
clock must stay in lockstep with the cam's single init time.

Cross-manager coupling: when a smart-detect event starts while a motion
analytics event is active, the real protocol links the two (the analytics
event's payload references which smart-detect event IDs occurred during it).
That's MotionAnalyticsManager's data to own, so instead of reaching into its
internals directly, this manager calls an injected `on_event_started`
callback and lets the analytics manager decide what to do with it.
"""
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from unifi.cams.handlers.snapshot_utils import calculate_snapshot_dimensions, get_image_dimensions

SendFn = Callable[[dict[str, Any]], Awaitable[None]]
GenResponseFn = Callable[..., dict[str, Any]]
GetUptimeFn = Callable[[], float]


class SmartDetectObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PACKAGE = "package"


class SmartDetectEventManager:
    def __init__(
        self,
        logger: logging.Logger,
        send: SendFn,
        gen_response: GenResponseFn,
        get_uptime: GetUptimeFn,
        detected_resolutions: dict[str, tuple[int, int]],
        on_event_started: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.logger = logger
        self._send = send
        self._gen_response = gen_response
        self._get_uptime = get_uptime
        self._detected_resolutions = detected_resolutions
        self._on_event_started = on_event_started

        self._event_id_counter: int = 0
        self.active_events: dict[int, dict[str, Any]] = {}

        # Tracks the most recently touched object type/descriptor so legacy
        # callers (trigger_motion_update/stop with no object_type) can infer
        # which event they meant. See LegacyMotionAPI.
        self.last_object_type: Optional[SmartDetectObjectType] = None
        self.last_descriptor: Optional[dict[str, Any]] = None
        self.last_event_ts: Optional[float] = None

    # -- lifecycle -----------------------------------------------------

    def _cleanup_old_events(self) -> None:
        current_time = time.time()
        sixty_minutes_ago = current_time - 3600

        to_remove = [
            event_id
            for event_id, event_data in self.active_events.items()
            if event_data.get("end_time") and event_data["end_time"] < sixty_minutes_ago
        ]

        for event_id in to_remove:
            event_data = self.active_events[event_id]
            for key in ["snapshot_crop_path", "snapshot_fov_path", "heatmap_path"]:
                path = event_data.get(key)
                if path and path.exists():
                    try:
                        path.unlink()
                        self.logger.debug(f"Deleted cached snapshot: {path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cached snapshot {path}: {e}")
            del self.active_events[event_id]

        if to_remove:
            self.logger.info(
                f"Cleaned up {len(to_remove)} old smart detect events. "
                f"Remaining: {len(self.active_events)}"
            )

    async def trigger_start(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zones_status: Optional[dict[str, Any]] = None,
    ) -> int:
        current_time = time.time()
        epoch_ms = int(time.time() * 1000)
        event_id = epoch_ms * 1000 + (self._event_id_counter % 1000)
        self._event_id_counter += 1

        if event_id in self.active_events:
            existing = self.active_events[event_id]
            self.logger.warning(
                f"Smart detect event {event_id} already active "
                f"(type: {existing['object_type'].value}, "
                f"started: {current_time - existing['start_time']:.1f}s ago). "
                f"Ignoring duplicate start for {object_type.value}."
            )
            return event_id

        descriptors = [custom_descriptor] if custom_descriptor else []

        payload: dict[str, Any] = {
            "clockMonotonic": int(self._get_uptime() * 1000),
            "clockStream": int(self._get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "descriptors": descriptors,
            "displayTimeoutMSec": 10000,
            "edgeType": "enter",
            "eventId": self._event_id_counter,
            "objectTypes": [object_type.value],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zones_status,
        }

        self.logger.info(
            f"Starting smart detect event {event_id} for {object_type.value} "
            f"(active smart events: {len(self.active_events)})"
        )

        await self._send(self._gen_response("EventSmartDetect", payload=payload))

        self._cleanup_old_events()

        self.active_events[event_id] = {
            "object_type": object_type,
            "start_time": current_time,
            "end_time": None,
            "event_timestamp": event_timestamp,
            "last_descriptor": custom_descriptor,
            "descriptor_history": [],
            "snapshot_crop_path": None,
            "snapshot_fov_path": None,
            "heatmap_path": None,
            "snapshot_width": None,
            "snapshot_height": None,
            # Per-trackerID snapshots: {tracker_id: {crop: Path, fov: Path}}.
            "tracker_snapshots": {},
        }

        if custom_descriptor:
            self.active_events[event_id]["descriptor_history"].append({
                "descriptor": custom_descriptor,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self._get_uptime() * 1000),
                "snapshot_width": 640,
                "snapshot_height": 360,
            })

        if self._on_event_started:
            self._on_event_started(event_id)

        self.last_event_ts = current_time
        self.last_object_type = object_type
        self.last_descriptor = custom_descriptor

        return event_id

    async def trigger_update(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zones_status: Optional[dict[str, Any]] = None,
        event_id: Optional[int] = None,
    ) -> None:
        """
        Args:
            event_id: Specific event to update. If omitted, falls back to the
                first active event matching object_type -- ambiguous when
                multiple concurrent tracks share a type, so pass event_id
                explicitly whenever the caller already knows it.
        """
        target_event_id = event_id
        if target_event_id is not None and target_event_id not in self.active_events:
            self.logger.warning(
                f"trigger_update called with event_id={target_event_id} "
                f"but it is not active. Falling back to object_type lookup."
            )
            target_event_id = None

        if target_event_id is None:
            for eid, event in self.active_events.items():
                if event["object_type"] == object_type:
                    target_event_id = eid
                    break

        if target_event_id is None:
            self.logger.warning(
                f"trigger_update called for {object_type.value} but no active "
                f"event found. Event may have already ended or never started. Ignoring."
            )
            return

        active_event = self.active_events[target_event_id]
        descriptors = []
        if custom_descriptor:
            descriptors = [custom_descriptor]
            active_event["last_descriptor"] = custom_descriptor
            self.last_descriptor = custom_descriptor
            active_event["descriptor_history"].append({
                "descriptor": custom_descriptor,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self._get_uptime() * 1000),
                "snapshot_width": 640,
                "snapshot_height": 360,
            })

        self._event_id_counter += 1
        payload: dict[str, Any] = {
            "clockMonotonic": int(self._get_uptime() * 1000),
            "clockStream": int(self._get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "descriptors": descriptors,
            "displayTimeoutMSec": 10000,
            "edgeType": "moving",
            "eventId": self._event_id_counter,
            "objectTypes": [object_type.value],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zones_status,
        }

        self.logger.debug(f"Updating smart detect event {target_event_id} for {object_type.value}")
        await self._send(self._gen_response("EventSmartDetect", payload=payload))

    async def trigger_stationary(
        self,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zones_status: Optional[dict[str, Any]] = None,
        event_id: Optional[int] = None,
    ) -> None:
        """
        Emit edgeType='none' heartbeat for a stationary background tracker
        (Frigate stationary=true): still visible, not in any zone, not moving.
        """
        if event_id is not None and event_id not in self.active_events:
            self.logger.debug(
                f"trigger_stationary called with event_id={event_id} "
                f"but it is not active. Skipping."
            )
            return

        self._event_id_counter += 1
        descriptors = [custom_descriptor] if custom_descriptor else []

        payload: dict[str, Any] = {
            "clockMonotonic": int(self._get_uptime() * 1000),
            "clockStream": int(self._get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "descriptors": descriptors,
            "displayTimeoutMSec": 300,
            "edgeType": "none",
            "eventId": self._event_id_counter,
            "objectTypes": [],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zones_status,
        }

        self.logger.debug(
            "Sending stationary heartbeat (edgeType=none) for trackerID="
            f"{custom_descriptor.get('trackerID') if custom_descriptor else 'unknown'}"
        )
        await self._send(self._gen_response("EventSmartDetect", payload=payload))

    def attach_tracker_snapshot(
        self, event_id: int, tracker_id: int, crop: Optional[Any] = None, fov: Optional[Any] = None
    ) -> None:
        """
        Record a per-trackerID snapshot for an in-progress event, so the
        eventual 'leave' payload can reference the right Frigate crop for
        each tracker instead of the shared event-level snapshot.

        Replaces FrigateCam reaching directly into
        active_events[id]['tracker_snapshots'] -- keep the dict private.
        """
        if event_id not in self.active_events:
            self.logger.debug(
                f"attach_tracker_snapshot called for inactive event {event_id}, ignoring."
            )
            return
        entry = self.active_events[event_id]["tracker_snapshots"].setdefault(tracker_id, {})
        if crop is not None:
            entry["crop"] = crop
        if fov is not None:
            entry["fov"] = fov

    async def trigger_stop(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        event_id: Optional[int] = None,
        frame_time_ms: Optional[int] = None,
        zones_status: Optional[dict[str, Any]] = None,
    ) -> None:
        target_event_id = event_id

        if target_event_id is None:
            for eid, event in self.active_events.items():
                if event["object_type"] == object_type:
                    target_event_id = eid
                    break

        if target_event_id is None:
            self.logger.warning(
                f"trigger_stop called for {object_type.value} but no active "
                f"event found. Event may have already ended or never started. Ignoring."
            )
            return

        if target_event_id not in self.active_events:
            self.logger.warning(
                f"trigger_stop called for event {target_event_id} but it is "
                f"not in active events list. Ignoring."
            )
            return

        active_event = self.active_events[target_event_id]

        # NOTE: FrigateCam sends its own explicit final trigger_update() call
        # immediately before calling trigger_stop() (see handle_detection_event's
        # "end" branch), so no final-update-before-stop logic lives here. If
        # that responsibility ever moves back into this manager, remove the
        # equivalent call from FrigateCam -- don't run both.

        smart_detect_snapshots = []
        tracker_id_attr_map = {}

        descriptors_to_process = active_event.get("descriptor_history", [])
        if not descriptors_to_process and custom_descriptor:
            w, h = calculate_snapshot_dimensions(
                custom_descriptor, self._detected_resolutions, logger=self.logger
            )
            descriptors_to_process = [{
                "descriptor": custom_descriptor,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self._get_uptime() * 1000),
                "snapshot_width": w,
                "snapshot_height": h,
            }]
        elif not descriptors_to_process and active_event.get("last_descriptor"):
            last_desc = active_event["last_descriptor"]
            w, h = calculate_snapshot_dimensions(
                last_desc, self._detected_resolutions, logger=self.logger
            )
            descriptors_to_process = [{
                "descriptor": last_desc,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self._get_uptime() * 1000),
                "snapshot_width": w,
                "snapshot_height": h,
            }]

        best_by_tracker: dict[int, dict[str, Any]] = {}
        latest_by_tracker: dict[int, dict[str, Any]] = {}

        for desc_entry in descriptors_to_process:
            descriptor = desc_entry["descriptor"]
            tracker_id = descriptor.get("trackerID", 1)
            confidence = descriptor.get("confidenceLevel", 0)

            latest_by_tracker[tracker_id] = desc_entry

            if tracker_id not in best_by_tracker:
                best_by_tracker[tracker_id] = desc_entry
            else:
                existing_conf = best_by_tracker[tracker_id]["descriptor"].get("confidenceLevel", 0)
                if confidence > existing_conf:
                    best_by_tracker[tracker_id] = desc_entry

        # Stationary bystanders (still visible at departure) go into
        # descriptors[]; active/departing trackers go into
        # smartDetectSnapshots + trackerIDAttrMap (protocol spec Section 7).
        bystander_descriptors: list[dict[str, Any]] = []

        snapshot_height = 360
        snapshot_width = 640

        for tracker_id, desc_entry in best_by_tracker.items():
            descriptor = desc_entry["descriptor"]
            latest_entry = latest_by_tracker.get(tracker_id, desc_entry)
            is_stationary = latest_entry["descriptor"].get("stationary", False)

            if is_stationary:
                bystander_descriptors.append(latest_entry["descriptor"])
                continue

            zones = descriptor.get("zones", [1])
            descriptor_object_type = descriptor.get("objectType", object_type.value)

            snapshot_width = desc_entry.get("snapshot_width") or active_event.get("snapshot_width") or 640
            snapshot_height = desc_entry.get("snapshot_height") or active_event.get("snapshot_height") or 360

            tracker_snapshots = active_event.get("tracker_snapshots", {})
            per_tracker = tracker_snapshots.get(tracker_id, {})
            snapshot_crop_path = per_tracker.get("crop") or active_event.get("snapshot_crop_path")
            snapshot_filename = (
                str(snapshot_crop_path)
                if snapshot_crop_path
                else f"smartdetectsnap_zone_{tracker_id}_{desc_entry['timestamp_ms']}.jpg"
            )

            smart_detect_snapshots.append({
                "clockBestMonotonic": desc_entry["monotonic"],
                "clockBestWall": desc_entry["timestamp_ms"],
                "smartDetectSnapshot": snapshot_filename,
                "smartDetectSnapshotHeight": snapshot_height,
                "smartDetectSnapshotName": descriptor.get("name", ""),
                "smartDetectSnapshotType": descriptor_object_type,
                "smartDetectSnapshotWidth": snapshot_width,
                "trackerID": tracker_id,
            })

            tracker_id_attr_map[str(tracker_id)] = {
                "objectType": descriptor_object_type,
                "zone": zones if zones else [1],
            }

        if not smart_detect_snapshots:
            default_tracker_id = 1
            default_timestamp_ms = event_timestamp or int(round(time.time() * 1000))
            default_monotonic = int(self._get_uptime() * 1000)

            snapshot_crop_path = active_event.get("snapshot_crop_path")
            snapshot_filename = (
                str(snapshot_crop_path)
                if snapshot_crop_path
                else f"smartdetectsnap_zone_{default_tracker_id}_{default_timestamp_ms}.jpg"
            )

            smart_detect_snapshots.append({
                "clockBestMonotonic": default_monotonic,
                "clockBestWall": default_timestamp_ms,
                "smartDetectSnapshot": snapshot_filename,
                "smartDetectSnapshotHeight": snapshot_height,
                "smartDetectSnapshotName": "",
                "smartDetectSnapshotType": object_type.value,
                "smartDetectSnapshotWidth": snapshot_width,
                "trackerID": default_tracker_id,
            })
            tracker_id_attr_map[str(default_tracker_id)] = {
                "objectType": object_type.value,
                "zone": [1],
            }

        snapshot_fov_path = active_event.get("snapshot_fov_path")
        fov_filename = str(snapshot_fov_path) if snapshot_fov_path else f"smartdetectsnap_{target_event_id}_fullfov.jpg"

        if snapshot_fov_path:
            fov_width, fov_height = get_image_dimensions(snapshot_fov_path, self.logger)
        else:
            fov_width = active_event.get("snapshot_width") or 640
            fov_height = active_event.get("snapshot_height") or 360

        self._event_id_counter += 1
        payload: dict[str, Any] = {
            "clockMonotonic": int(self._get_uptime() * 1000),
            "clockStream": int(self._get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "descriptors": bystander_descriptors,
            "displayTimeoutMSec": 2000,
            "edgeType": "leave",
            "eventId": self._event_id_counter,
            # objectTypes clears to [] in lockstep with descriptors on the
            # terminal 'leave' message -- confirmed against real captures
            # (protocol spec Section 3).
            "objectTypes": [],
            "smartDetectSnapshotFullFoV": fov_filename,
            "smartDetectSnapshotFullFoVHeight": fov_height,
            "smartDetectSnapshotFullFoVWidth": fov_width,
            "smartDetectSnapshots": smart_detect_snapshots,
            "trackerIDAttrMap": tracker_id_attr_map,
            "zonesStatus": zones_status,
        }

        duration = time.time() - active_event["start_time"]
        active_count = len([e for e in self.active_events.values() if e.get("end_time") is None])
        self.logger.info(
            f"Stopping smart detect event {target_event_id} for {object_type.value} "
            f"(duration: {duration:.1f}s, active smart events: {active_count})"
        )

        await self._send(self._gen_response("EventSmartDetect", payload=payload))

        active_event["end_time"] = time.time()

        still_active = [e for e in self.active_events.values() if e.get("end_time") is None]
        if not still_active:
            self.last_object_type = None
            self.last_descriptor = None
            # last_event_ts is cleared by the caller if no analytics event is
            # active either -- see LegacyMotionAPI / base.py wiring.

    # -- queries used by SnapshotHandlers / summaries -------------------

    def has_active_events(self) -> bool:
        return len(self.active_events) > 0

    def get_snapshot_paths(self, event_id: int) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Return (crop, fov, heatmap) paths for an event, or (None, None, None)."""
        event = self.active_events.get(event_id)
        if not event:
            return (None, None, None)
        return (
            event.get("snapshot_crop_path"),
            event.get("snapshot_fov_path"),
            event.get("heatmap_path"),
        )

    def find_snapshot_by_path(self, snapshot_type: str, filename: str, original_filename: str) -> Optional[Any]:
        """Look up a cached snapshot path matching a GetRequest's filename."""
        if snapshot_type != "smartDetectZoneSnapshot":
            return None
        for event_data in self.active_events.values():
            cached_path = event_data.get("snapshot_crop_path")
            if cached_path and cached_path.exists():
                if (filename and str(cached_path) == filename) or (
                    original_filename and str(cached_path) == original_filename
                ):
                    return cached_path
        return None

    def summary(self) -> dict[int, dict[str, Any]]:
        return {
            event_id: {
                "object_type": event["object_type"].value,
                "duration": time.time() - event["start_time"],
                "has_descriptor": event["last_descriptor"] is not None,
                "has_snapshot_crop": event["snapshot_crop_path"] is not None,
                "has_snapshot_fov": event["snapshot_fov_path"] is not None,
                "has_heatmap": event["heatmap_path"] is not None,
            }
            for event_id, event in self.active_events.items()
        }

    async def force_stop_all(self) -> None:
        """Force-stop every currently active smart detect event."""
        for event_id in list(self.active_events.keys()):
            event = self.active_events[event_id]
            self.logger.info(
                f"Force stopping smart detect event {event_id} ({event['object_type'].value})"
            )
            try:
                await self.trigger_stop(object_type=event["object_type"])
            except Exception as e:
                self.logger.error(f"Error stopping smart detect event {event_id}: {e}")

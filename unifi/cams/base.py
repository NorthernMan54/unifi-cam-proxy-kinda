import argparse
import asyncio
import atexit
import json
import logging
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone

import aiohttp
import websockets

from unifi.core import RetryableError
from unifi.cams.handlers import ProtocolHandlers, VideoStreamHandlers, SnapshotHandlers

AVClientRequest = AVClientResponse = dict[str, Any]


class SmartDetectObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PACKAGE = "package"


class UnifiCamBase(ProtocolHandlers, VideoStreamHandlers, SnapshotHandlers, metaclass=ABCMeta):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger

        self._msg_id: int = 0
        self._init_time: float = time.time()
        self._streams: dict[str, str] = {}

        self._motion_snapshot: Optional[Path] = None
        self._motion_snapshot_crop: Optional[Path] = None
        self._motion_snapshot_fov: Optional[Path] = None
        self._motion_heatmap: Optional[Path] = None

        # FIXED: real UniFi Protect cameras run INDEPENDENT eventId sequences per
        # function -- EventSmartDetect and EventSmartMotion each increment their
        # own counter (confirmed from real captures: EventSmartDetect ran
        # 36150->36203 while EventSmartMotion ran 3464->3467 concurrently over
        # the same session).
        self._smart_detect_event_id: int = 0  # For EventSmartDetect messages
        self._motion_event_id: int = 0        # For EventSmartMotion messages

        self._analytics_event_history: dict[int, dict[str, Any]] = {}
        self._active_analytics_event_id: Optional[int] = None
        self._active_smart_events: dict[int, dict[str, Any]] = {}

        self._motion_event_ts: Optional[float] = None
        self._motion_object_type: Optional[SmartDetectObjectType] = None
        self._motion_last_descriptor: Optional[dict[str, Any]] = None

        self._ffmpeg_handles: dict[str, subprocess.Popen] = {}

        self._detected_resolutions: dict[str, tuple[int, int]] = {
            "video1": (2560, 1920),
            "video2": (1280, 704),
            "video3": (640, 360),
        }

        self.lingerEventStart: int = 1000
        self._analytics_start_task: Optional[asyncio.Task] = None
        self._motion_pulse_task: Optional[asyncio.Task] = None

        # Motion zone configuration: map zone IDs to names, default to zone "1"
        self._motion_zone_id: str = "1"
        self._motion_zone_config: dict[str, str] = {"1": "Default"}

        # Motion confidence level for pulse events
        self._motion_confidence_level: int = 50

        self.motionEvents: bool = True

        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        self._ssl_context.load_cert_chain(args.cert, args.cert)
        self._session: Optional[websockets.client.WebSocketClientProtocol] = None
        atexit.register(self.close_streams)

    @classmethod
    def add_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ffmpeg-args",
            "-f",
            default="-c:v copy -ar 32000 -ac 1 -codec:a aac -b:a 32k",
            help="Transcoding args for `ffmpeg -i <src> <args> <dst>`",
        )
        parser.add_argument(
            "--ffmpeg-base-args",
            "-b",
            help="Base args for `ffmpeg <base_args> -i <src> <args> <dst>",
            type=str,
        )
        parser.add_argument(
            "--rtsp-transport",
            default="tcp",
            choices=["tcp", "udp", "http", "udp_multicast"],
            help="RTSP transport protocol used by stream",
        )
        parser.add_argument(
            "--timestamp-modifier",
            type=int,
            default="90",
            help="Modify the timestamp correction factor (default: 90)",
        )
        parser.add_argument(
            "--loglevel",
            default="error",
            choices=["trace", "debug", "verbose", "info", "warning", "error", "fatal", "panic", "quiet"],
            help="Set the ffmpeg log level",
        )
        parser.add_argument(
            "--format",
            default="flv",
            help="Set the ffpmeg output format",
        )

    async def _run(self, ws) -> None:
        self._session = ws
        await self.init_adoption()
        while True:
            try:
                msg = await ws.recv()
            except websockets.exceptions.ConnectionClosedError:
                self.logger.info(f"Connection to {self.args.host} was closed.")
                raise RetryableError()

            if msg is not None:
                force_reconnect = await self.process(msg)
                if force_reconnect:
                    self.logger.info("Reconnecting...")
                    raise RetryableError()

    async def run(self) -> None:
        return

    async def get_video_settings(self) -> dict[str, Any]:
        return {}

    async def change_video_settings(self, options) -> None:
        return

    @abstractmethod
    async def get_snapshot(self) -> Path:
        raise NotImplementedError("You need to write this!")

    @abstractmethod
    async def get_stream_source(self, stream_index: str) -> str:
        raise NotImplementedError("You need to write this!")

    async def fetch_snapshots_for_event(
        self, event_id: int, event_type: str = "analytics"
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        snapshot = await self.get_snapshot()
        return (snapshot, snapshot, snapshot)

    def update_snapshot_dimensions_from_file(self, event_id: int, snapshot_path: Optional[Path]) -> None:
        if event_id not in self._active_smart_events:
            return

        if snapshot_path:
            width, height = self._get_image_dimensions(snapshot_path)

            self._active_smart_events[event_id]["snapshot_width"] = width
            self._active_smart_events[event_id]["snapshot_height"] = height

            descriptor_history = self._active_smart_events[event_id]["descriptor_history"]
            for desc_entry in descriptor_history:
                desc_entry["snapshot_width"] = width
                desc_entry["snapshot_height"] = height

            self.logger.debug(
                f"Updated snapshot dimensions for event {event_id}: {width}x{height} from file {snapshot_path} "
                f"(updated {len(descriptor_history)} descriptor entries)"
            )

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            "mic": True,
            "aec": [],
            "videoMode": ["default"],
            "motionDetect": ["enhanced"],
            "hotplug":{"extender":{"attached":False}}
        }

    ###
    # Payload structure reference for motion events:
    # clockBestMonotonic, clockBestWall, clockMonotonic, clockStream,
    # clockStreamRate, clockWall, edgeType (start/stop/unknown),
    # eventId, eventType (motion/pulse), levels, motionHeatmap,
    # motionSnapshot, motionSnapshotFullFoV, etc.
    #
    # NOTE: this schema matches EventSmartMotion in every real device capture
    # reviewed. The functionName actually sent below is "EventAnalytics" -- no
    # real capture has ever shown that function name on the wire. Flagging this
    # as a likely mismatch; verify against your controller before relying on
    # this event path (see _send_analytics_start_event / trigger_analytics_stop
    # below).
    ###

    # API for subclasses - Smart Detect Events
    def _cleanup_old_analytics_events(self) -> None:
        current_time = time.time()
        one_hour_ago = current_time - 3600

        events_to_remove = []
        for event_id, event_data in self._analytics_event_history.items():
            event_time = event_data.get('end_time') or event_data.get('start_time', 0)
            if event_time < one_hour_ago:
                events_to_remove.append(event_id)

        for event_id in events_to_remove:
            event_data = self._analytics_event_history[event_id]

            for snapshot_key in ['snapshot_crop_path', 'snapshot_fov_path', 'heatmap_path']:
                snapshot_path = event_data.get(snapshot_key)
                if snapshot_path and isinstance(snapshot_path, Path) and snapshot_path.exists():
                    try:
                        snapshot_path.unlink()
                        self.logger.debug(f"Deleted cached snapshot: {snapshot_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cached snapshot {snapshot_path}: {e}")

            del self._analytics_event_history[event_id]
            self.logger.debug(
                f"Cleaned up analytics event {event_id} "
                f"(age: {(current_time - event_data.get('end_time', current_time)) / 60:.1f} minutes)"
            )

        if events_to_remove:
            self.logger.info(
                f"Cleaned up {len(events_to_remove)} old analytics events. "
                f"Remaining in history: {len(self._analytics_event_history)}"
            )

    def _cleanup_old_smart_events(self) -> None:
        current_time = time.time()
        sixty_minutes_ago = current_time - 3600

        events_to_remove = []
        for event_id, event_data in self._active_smart_events.items():
            end_time = event_data.get('end_time')
            if end_time and end_time < sixty_minutes_ago:
                events_to_remove.append(event_id)

        for event_id in events_to_remove:
            event_data = self._active_smart_events[event_id]

            for snapshot_key in ['snapshot_crop_path', 'snapshot_fov_path', 'heatmap_path']:
                snapshot_path = event_data.get(snapshot_key)
                if snapshot_path and isinstance(snapshot_path, Path) and snapshot_path.exists():
                    try:
                        snapshot_path.unlink()
                        self.logger.debug(f"Deleted cached snapshot: {snapshot_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cached snapshot {snapshot_path}: {e}")

            del self._active_smart_events[event_id]
            if end_time is not None:
                self.logger.debug(
                    f"Cleaned up smart detect event {event_id} "
                    f"(age: {(current_time - end_time) / 60:.1f} minutes)"
                )
            else:
                self.logger.debug(
                    f"Cleaned up smart detect event {event_id} "
                    "(age: unknown, end_time is null)"
                )

        if events_to_remove:
            self.logger.info(
                f"Cleaned up {len(events_to_remove)} old smart detect events. "
                f"Remaining: {len(self._active_smart_events)}"
            )

    def _get_image_dimensions(self, image_path: Optional[Path]) -> tuple[int, int]:
        if not image_path or not image_path.exists():
            return (640, 360)

        try:
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    return img.size
            except ImportError:
                with image_path.open('rb') as f:
                    header = f.read(24)
                    if header[:8] == b'\x89PNG\r\n\x1a\n':
                        f.seek(16)
                        width_bytes = f.read(4)
                        height_bytes = f.read(4)
                        width = int.from_bytes(width_bytes, byteorder='big')
                        height = int.from_bytes(height_bytes, byteorder='big')
                        return (width, height)
                    elif header[:2] == b'\xff\xd8':
                        f.seek(0)
                        self.logger.debug("JPEG format detected but dimensions parsing not implemented without PIL")
                        return (640, 360)
                    else:
                        self.logger.debug(f"Unknown image format for {image_path}")
                        return (640, 360)
        except Exception as e:
            self.logger.debug(f"Could not read image dimensions from {image_path}: {e}")
            return (640, 360)

    def _calculate_snapshot_dimensions(self, descriptor: dict[str, Any], snapshot_path: Optional[Path] = None) -> tuple[int, int]:
        if snapshot_path:
            width, height = self._get_image_dimensions(snapshot_path)
            if (width, height) != (640, 360):
                return (width, height)

        coord = descriptor.get("coord")
        if coord and len(coord) >= 4:
            try:
                x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]
                stream_width, stream_height = self._detected_resolutions.get("video3", (640, 360))
                bbox_width = abs(x2 - x1) * stream_width
                bbox_height = abs(y2 - y1) * stream_height
                width = max(int(bbox_width), 100)
                height = max(int(bbox_height), 100)
                return (width, height)
            except (ValueError, TypeError, IndexError) as e:
                self.logger.debug(f"Could not parse coord from descriptor: {e}")

        return (640, 360)

    # API for subclasses - Smart Detect Events
    async def trigger_smart_detect_start(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zonesStatus: Optional[dict[str, Any]] = None
    ) -> int:
        current_time = time.time()

        epoch_ms = int(time.time() * 1000)
        event_id = epoch_ms * 1000 + (self._smart_detect_event_id % 1000)
        self._smart_detect_event_id += 1
        
        # Check if we already have an active smart detect event with this event_id
        if event_id in self._active_smart_events:
            existing_event = self._active_smart_events[event_id]
            self.logger.warning(
                f"Smart detect event {event_id} already active "
                f"(type: {existing_event['object_type'].value}, "
                f"started: {current_time - existing_event['start_time']:.1f}s ago). "
                f"Ignoring duplicate start for {object_type.value}."
            )
            return event_id

        descriptors = []
        if custom_descriptor:
            descriptors = [custom_descriptor]

        payload: dict[str, Any] = {
            # FIX: get_uptime() returns fractional SECONDS; clockStreamRate=1000
            # means these clocks are expected in MILLISECONDS. Previously this
            # was `int(self.get_uptime())`, which truncated to whole seconds and
            # made clockMonotonic/clockStream advance ~1000x slower than real
            # time -- this is what caused the "frozen clock" symptom seen across
            # consecutive test messages (91960 -> 91960 -> 91960 -> 91961 despite
            # ~1s of real elapsed time between them).
            "clockMonotonic": int(self.get_uptime() * 1000),
            "clockStream": int(self.get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "descriptors": descriptors,
            "displayTimeoutMSec": 10000,
            "edgeType": "enter",
            "eventId": self._smart_detect_event_id,
            "objectTypes": [object_type.value],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zonesStatus,
        }

        self.logger.info(
            f"Starting smart detect event {event_id} for {object_type.value} "
            f"(active smart events: {len(self._active_smart_events)})"
        )

        await self.send(
            self.gen_response("EventSmartDetect", payload=payload)
        )

        self._cleanup_old_smart_events()

        self._active_smart_events[event_id] = {
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
            # Per-trackerID snapshots: {tracker_id: {crop: Path, fov: Path}}
            # Populated by _fetch_and_cache_frigate_event_snapshot so each
            # trackerID in the stop payload uses its own Frigate event snapshot.
            "tracker_snapshots": {},
        }

        if custom_descriptor:
            snapshot_width, snapshot_height = (640, 360)

            self._active_smart_events[event_id]["descriptor_history"].append({
                "descriptor": custom_descriptor,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self.get_uptime() * 1000),
                "snapshot_width": snapshot_width,
                "snapshot_height": snapshot_height,
            })

        if self._active_analytics_event_id is not None:
            active_analytics = self._analytics_event_history.get(self._active_analytics_event_id)
            if active_analytics:
                active_analytics["smart_detect_event_ids"].append(event_id)
                self.logger.debug(
                    f"Associated smart detect event {event_id} ({object_type.value}) "
                    f"with analytics event {self._active_analytics_event_id}. "
                    f"Total smart detects for this analytics event: "
                    f"{len(active_analytics['smart_detect_event_ids'])}"
                )

        self._motion_event_ts = current_time
        self._motion_object_type = object_type
        self._motion_last_descriptor = custom_descriptor

        return event_id

    async def trigger_smart_detect_update(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zonesStatus: Optional[dict[str, Any]] = None,
        event_id: Optional[int] = None,
    ) -> None:
        """
        Send a smart detect update (moving) event with updated descriptor information.

        Args:
            event_id: Optional specific event ID to update. If not provided,
                falls back to finding the first active event matching
                object_type -- NOTE: that fallback is ambiguous when multiple
                concurrent tracks share the same object_type (e.g. two
                vehicles); pass event_id explicitly whenever the caller
                already knows it (e.g. FrigateCam has unifi_event_id from
                trigger_smart_detect_start).
        """
        target_event_id = event_id
        if target_event_id is not None and target_event_id not in self._active_smart_events:
            self.logger.warning(
                f"trigger_smart_detect_update called with event_id={target_event_id} "
                f"but it is not active. Falling back to object_type lookup."
            )
            target_event_id = None

        if target_event_id is None:
            for eid, event in self._active_smart_events.items():
                if event["object_type"] == object_type:
                    target_event_id = eid
                    break

        if target_event_id is None:
            self.logger.warning(
                f"trigger_smart_detect_update called for {object_type.value} "
                f"but no active event found. Event may have already ended or never started. Ignoring."
            )
            return

        active_event = self._active_smart_events[target_event_id]
        descriptors = []
        if custom_descriptor:
            descriptors = [custom_descriptor]
            active_event["last_descriptor"] = custom_descriptor
            self._motion_last_descriptor = custom_descriptor
            snapshot_width, snapshot_height = (640, 360)

            active_event["descriptor_history"].append({
                "descriptor": custom_descriptor,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self.get_uptime() * 1000),
                "snapshot_width": snapshot_width,
                "snapshot_height": snapshot_height,
            })

        self._smart_detect_event_id += 1
        payload: dict[str, Any] = {
            "clockMonotonic": int(self.get_uptime() * 1000),
            "clockStream": int(self.get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "descriptors": descriptors,
            "displayTimeoutMSec": 10000,
            "edgeType": "moving",
            "eventId": self._smart_detect_event_id,
            "objectTypes": [object_type.value],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zonesStatus,
        }

        self.logger.debug(
            f"Updating smart detect event {target_event_id} for {object_type.value}"
        )

        await self.send(
            self.gen_response("EventSmartDetect", payload=payload)
        )

    async def trigger_smart_detect_stationary(
        self,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zonesStatus: Optional[dict[str, Any]] = None,
        event_id: Optional[int] = None,
    ) -> None:
        """
        Emit edgeType='none' for a stationary background tracker.

        Used when Frigate reports stationary=true on an update: the object is
        still visible but is not occupying any zone and has not moved.  Real
        devices use this heartbeat roughly every ~5 minutes; objectTypes is
        empty and all zonesStatus entries are at level 0 / status 'none'
        (protocol spec Section 7).
        """
        if event_id is not None and event_id not in self._active_smart_events:
            self.logger.debug(
                f"trigger_smart_detect_stationary called with event_id={event_id} "
                f"but it is not active. Skipping."
            )
            return

        self._smart_detect_event_id += 1
        descriptors = [custom_descriptor] if custom_descriptor else []

        payload: dict[str, Any] = {
            "clockMonotonic": int(self.get_uptime() * 1000),
            "clockStream": int(self.get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "descriptors": descriptors,
            "displayTimeoutMSec": 300,
            "edgeType": "none",
            "eventId": self._smart_detect_event_id,
            "objectTypes": [],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zonesStatus,
        }

        self.logger.debug(
            f"Sending stationary heartbeat (edgeType=none) for "
            f"trackerID={custom_descriptor.get('trackerID') if custom_descriptor else 'unknown'}"
        )

        await self.send(
            self.gen_response("EventSmartDetect", payload=payload)
        )

    async def trigger_smart_detect_stop(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        event_id: Optional[int] = None,
        frame_time_ms: Optional[int] = None,
        zonesStatus: Optional[dict[str, Any]] = None
    ) -> None:
        target_event_id = event_id

        if target_event_id is None:
            for eid, event in self._active_smart_events.items():
                if event["object_type"] == object_type:
                    target_event_id = eid
                    break

        if target_event_id is None:
            self.logger.warning(
                f"trigger_smart_detect_stop called for {object_type.value} "
                f"but no active event found. Event may have already ended or never started. Ignoring."
            )
            return

        if target_event_id not in self._active_smart_events:
            self.logger.warning(
                f"trigger_smart_detect_stop called for event {target_event_id} "
                f"but it is not in active events list. Ignoring."
            )
            return

        active_event = self._active_smart_events[target_event_id]

        # NOTE: this block is intentionally left disabled. FrigateCam now sends
        # its own explicit final trigger_smart_detect_update() call immediately
        # before calling trigger_smart_detect_stop() (see handle_detection_event's
        # "end" branch). Re-enabling this would double-send the final update.
        # If you move the "final update" responsibility back into the base
        # class, remove the equivalent logic from FrigateCam instead of running
        # both.
#        if custom_descriptor:
#            self.logger.debug(
#                f"Sending final update with custom descriptor before stopping event {target_event_id}"
#            )
#            await self.trigger_smart_detect_update(
#                object_type,
#                custom_descriptor,
#                frame_time_ms or event_timestamp
#            )

        smart_detect_snapshots = []
        tracker_id_attr_map = {}

        descriptors_to_process = active_event.get("descriptor_history", [])
        if not descriptors_to_process and custom_descriptor:
            snapshot_width, snapshot_height = self._calculate_snapshot_dimensions(custom_descriptor)
            descriptors_to_process = [{
                "descriptor": custom_descriptor,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self.get_uptime() * 1000),
                "snapshot_width": snapshot_width,
                "snapshot_height": snapshot_height,
            }]
        elif not descriptors_to_process and active_event.get("last_descriptor"):
            last_desc = active_event["last_descriptor"]
            snapshot_width, snapshot_height = self._calculate_snapshot_dimensions(last_desc)
            descriptors_to_process = [{
                "descriptor": last_desc,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self.get_uptime() * 1000),
                "snapshot_width": snapshot_width,
                "snapshot_height": snapshot_height,
            }]

        best_descriptors_by_tracker: dict[int, dict[str, Any]] = {}
        latest_descriptors_by_tracker: dict[int, dict[str, Any]] = {}

        for desc_entry in descriptors_to_process:
            descriptor = desc_entry["descriptor"]
            tracker_id = descriptor.get("trackerID", 1)
            confidence = descriptor.get("confidenceLevel", 0)

            # Track the latest update for each tracker (for stationary state)
            latest_descriptors_by_tracker[tracker_id] = desc_entry

            if tracker_id not in best_descriptors_by_tracker:
                best_descriptors_by_tracker[tracker_id] = desc_entry
            else:
                existing_confidence = best_descriptors_by_tracker[tracker_id]["descriptor"].get("confidenceLevel", 0)
                if confidence > existing_confidence:
                    best_descriptors_by_tracker[tracker_id] = desc_entry

        # Split: stationary bystanders go into descriptors[], active/departing
        # trackers go into smartDetectSnapshots + trackerIDAttrMap.
        # Protocol spec Section 7: stationary objects that are still visible at
        # the time of departure appear as descriptors only -- they are NOT
        # recorded as event participants.
        bystander_descriptors: list[dict[str, Any]] = []

        for tracker_id, desc_entry in best_descriptors_by_tracker.items():
            descriptor = desc_entry["descriptor"]

            # Use the LATEST state for the stationary check (a tracker might
            # have started moving then gone stationary, or vice versa).
            latest_entry = latest_descriptors_by_tracker.get(tracker_id, desc_entry)
            is_stationary = latest_entry["descriptor"].get("stationary", False)

            if is_stationary:
                # Carry-along bystander: include in descriptors, skip snapshot/attrmap
                bystander_descriptors.append(latest_entry["descriptor"])
                continue

            zones = descriptor.get("zones", [1])
            descriptor_object_type = descriptor.get("objectType", object_type.value)

            snapshot_width = desc_entry.get("snapshot_width") or active_event.get("snapshot_width") or 640
            snapshot_height = desc_entry.get("snapshot_height") or active_event.get("snapshot_height") or 360

            # Prefer the per-tracker snapshot (keyed by trackerID, fetched from
            # /api/events/{frigate_id}/snapshot.jpg for the specific Frigate track).
            # Fall back to the shared smart-event snapshot if not available.
            tracker_snapshots = active_event.get("tracker_snapshots", {})
            per_tracker = tracker_snapshots.get(tracker_id, {})
            snapshot_crop_path = per_tracker.get("crop") or active_event.get("snapshot_crop_path")
            snapshot_filename = str(snapshot_crop_path) if snapshot_crop_path else f"smartdetectsnap_zone_{tracker_id}_{desc_entry['timestamp_ms']}.jpg"

            smart_detect_snapshots.append({
                "clockBestMonotonic": desc_entry["monotonic"],
                "clockBestWall": desc_entry["timestamp_ms"],
                "smartDetectSnapshot": snapshot_filename,
                "smartDetectSnapshotHeight": snapshot_height,
                "smartDetectSnapshotName": descriptor.get("name", ""),
                "smartDetectSnapshotType": descriptor_object_type,
                "smartDetectSnapshotWidth": snapshot_width,
                "trackerID": tracker_id
            })

            tracker_id_attr_map[str(tracker_id)] = {
                "objectType": descriptor_object_type,
                "zone": zones if zones else [1]
            }

        if not smart_detect_snapshots:
            default_tracker_id = 1
            default_timestamp_ms = event_timestamp or int(round(time.time() * 1000))
            default_monotonic = int(self.get_uptime() * 1000)

            snapshot_crop_path = active_event.get("snapshot_crop_path")
            snapshot_filename = str(snapshot_crop_path) if snapshot_crop_path else f"smartdetectsnap_zone_{default_tracker_id}_{default_timestamp_ms}.jpg"

            smart_detect_snapshots.append({
                "clockBestMonotonic": default_monotonic,
                "clockBestWall": default_timestamp_ms,
                "smartDetectSnapshot": snapshot_filename,
                "smartDetectSnapshotHeight": snapshot_height,
                "smartDetectSnapshotName": "",
                "smartDetectSnapshotType": object_type.value,
                "smartDetectSnapshotWidth": snapshot_width,
                "trackerID": default_tracker_id
            })
            tracker_id_attr_map[str(default_tracker_id)] = {
                "objectType": object_type.value,
                "zone": [1]
            }

        snapshot_fov_path = active_event.get("snapshot_fov_path")
        fov_filename = str(snapshot_fov_path) if snapshot_fov_path else f"smartdetectsnap_{target_event_id}_fullfov.jpg"

        if snapshot_fov_path:
            fov_width, fov_height = self._get_image_dimensions(snapshot_fov_path)
        else:
            fov_width = active_event.get("snapshot_width") or 640
            fov_height = active_event.get("snapshot_height") or 360

        self._smart_detect_event_id += 1
        payload: dict[str, Any] = {
            "clockMonotonic": int(self.get_uptime() * 1000),
            "clockStream": int(self.get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            # Stationary bystanders (still visible at departure time) go here;
            # the departing active tracker(s) are in smartDetectSnapshots instead.
            "descriptors": bystander_descriptors,
            "displayTimeoutMSec": 2000,
            "edgeType": "leave",
            "eventId": self._smart_detect_event_id,
            # FIX: real devices clear objectTypes to [] in lockstep with
            # descriptors on the terminal "leave" message -- confirmed against
            # multiple real captures (protocol spec Section 3). Previously this
            # stayed populated as [object_type.value] even though descriptors
            # was already empty above.
            "objectTypes": [],
            "smartDetectSnapshotFullFoV": fov_filename,
            "smartDetectSnapshotFullFoVHeight": fov_height,
            "smartDetectSnapshotFullFoVWidth": fov_width,
            "smartDetectSnapshots": smart_detect_snapshots,
            "trackerIDAttrMap": tracker_id_attr_map,
            "zonesStatus": zonesStatus,
        }

        duration = time.time() - active_event["start_time"]
        self.logger.info(
            f"Stopping smart detect event {target_event_id} for {object_type.value} "
            f"(duration: {duration:.1f}s, active smart events: "
            f"{len([e for e in self._active_smart_events.values() if e.get('end_time') is None])})"
        )

        await self.send(
            self.gen_response("EventSmartDetect", payload=payload)
        )

        active_event["end_time"] = time.time()

        active_smart_events = [e for e in self._active_smart_events.values() if e.get("end_time") is None]
        if not active_smart_events:
            self._motion_object_type = None
            self._motion_last_descriptor = None
            if self._active_analytics_event_id is None:
                self._motion_event_ts = None

    # API for subclasses - Analytics (Motion) Events
    async def _send_motion_start_event(
        self,
        event_id: int,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """Send EventSmartMotion start message with motion zone confidence levels."""
        if event_id not in self._analytics_event_history:
            self.logger.debug(
                f"Motion event {event_id} was stopped before linger period elapsed. "
                f"Not sending start event."
            )
            return

        active_event = self._analytics_event_history[event_id]

        if active_event.get("end_time") is not None:
            self.logger.debug(
                f"Motion event {event_id} already ended. Not sending start event."
            )
            return

        self._motion_event_id += 1
        active_event["motion_event_id"] = self._motion_event_id

        # Use motion_levels from event history (initialized with default zone)
        motion_levels = active_event.get("motion_levels", {self._motion_zone_id: 50})

        # Generate motion snapshot filenames
        motion_heatmap_filename = f"heatmap_{self._motion_event_id:08d}.png"
        motion_snapshot_filename = f"motionsnap_{self._motion_event_id:08d}.jpg"
        motion_snapshot_fov_filename = f"motionsnap_{self._motion_event_id:08d}_fullfov.jpg"
        motion_raw_heatmap_npz = f"motion_raw_heatmap_{int(round(event_timestamp or time.time() * 1000))}.npz"

        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self.get_uptime() * 1000),
            "clockStream": int(self.get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "start",
            "eventId": self._motion_event_id,
            "eventType": "motion",
            "levels": motion_levels,
            "motionHeatmap": motion_heatmap_filename,
            "motionHeatmapHeight": 90,
            "motionHeatmapWidth": 160,
            "motionSnapshot": motion_snapshot_filename,
            "motionSnapshotFullFoV": motion_snapshot_fov_filename,
            "motionSnapshotFullFoVHeight": 360,
            "motionSnapshotFullFoVWidth": 640,
            "motionSnapshotHeight": 360,
            "motionSnapshotWidth": 360,
            "motionRawHeatmapNPZ": motion_raw_heatmap_npz,
        }

        self.logger.info(
            f"Sending motion start event {self._motion_event_id} (analytics {event_id}) "
            f"with levels: {motion_levels}, pulse task running: {self._motion_pulse_task is not None and not self._motion_pulse_task.done()}"
        )

        await self.send(
            self.gen_response("EventSmartMotion", payload=payload)
        )

        active_event["start_event_sent"] = True

        # Cancel any previous pulse task and start a new one
        if self._motion_pulse_task and not self._motion_pulse_task.done():
            self._motion_pulse_task.cancel()
            try:
                await self._motion_pulse_task
            except asyncio.CancelledError:
                pass

        self._motion_pulse_task = asyncio.create_task(
            self._pulse_motion_events(event_id)
        )

    async def _send_motion_pulse_event(
        self,
        event_id: int,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """Send EventSmartMotion pulse message (every 2-3s during motion window)."""
        if event_id not in self._analytics_event_history:
            return

        active_event = self._analytics_event_history[event_id]

        # Rate limit pulses to every 2-3 seconds
        current_time = time.time()
        last_pulse = active_event.get("last_pulse_time")
        if last_pulse and (current_time - last_pulse) < 2.0:
            return

        motion_levels_updated_at = float(active_event.get("motion_levels_updated_at") or 0.0)
        last_pulse_levels_updated_at = float(active_event.get("last_pulse_motion_levels_updated_at") or 0.0)
        if motion_levels_updated_at <= last_pulse_levels_updated_at:
            return

        motion_levels = active_event.get("motion_levels", {self._motion_zone_id: 50})

        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self.get_uptime() * 1000),
            "clockStream": int(self.get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "unknown",
            "eventId": 18446744073709551615,  # Special sentinel: max uint64 / -1 for pulse events
            "eventType": "pulse",
            "levels": motion_levels,
        }

        self.logger.debug(
            f"Sending motion pulse event with levels: {motion_levels}"
        )

        await self.send(
            self.gen_response("EventSmartMotion", payload=payload)
        )

        active_event["last_pulse_time"] = current_time
        active_event["last_pulse_motion_levels_updated_at"] = motion_levels_updated_at

    async def _pulse_motion_events(self, event_id: int) -> None:
        """Background task that emits motion pulse events every 2-3 seconds."""
        try:
            while True:
                await asyncio.sleep(2.5)  # Emit pulse every 2.5 seconds
                if event_id not in self._analytics_event_history:
                    break
                active_event = self._analytics_event_history[event_id]
                if active_event.get("end_time") is not None:
                    break
                await self._send_motion_pulse_event(event_id)
        except asyncio.CancelledError:
            self.logger.debug(f"Motion pulse task for event {event_id} cancelled")
            raise

    async def _send_analytics_start_event(
        self,
        event_id: int,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """Wrapper for backward compatibility - calls _send_motion_start_event."""
        await self._send_motion_start_event(event_id, event_timestamp)

    async def trigger_analytics_start(
        self,
        event_timestamp: Optional[float] = None,
    ) -> None:
        if not self.motionEvents:
            self.logger.debug("Motion events disabled, ignoring trigger_analytics_start")
            return

        self._cleanup_old_analytics_events()

        current_time = time.time()

        epoch_ms = int(time.time() * 1000)
        event_id = epoch_ms * 1000 + (self._motion_event_id % 1000)
        self._motion_event_id += 1

        if self._active_analytics_event_id is not None:
            active_event = self._analytics_event_history.get(self._active_analytics_event_id)
            if active_event:
                existing_start = active_event['start_time']
                self.logger.warning(
                    f"Analytics event {self._active_analytics_event_id} already active "
                    f"(started: {current_time - existing_start:.1f}s ago). "
                    f"Ignoring duplicate start."
                )
                return

        self.logger.info(
            f"Preparing analytics event {event_id}, will send start event after {self.lingerEventStart}ms linger period"
        )

        self._analytics_event_history[event_id] = {
            "event_id": event_id,
            "start_time": current_time,
            "end_time": None,
            "event_timestamp": event_timestamp,
            "start_event_sent": False,
            "snapshot_filename": None,
            "snapshot_fov_filename": None,
            "heatmap_filename": None,
            "snapshot_crop_path": None,
            "snapshot_fov_path": None,
            "heatmap_path": None,
            "smart_detect_event_ids": [],
            # Motion-specific fields for EventSmartMotion
            "motion_event_id": None,
            "motion_heatmap_path": None,
            "motion_snapshot_path": None,
            "motion_snapshot_fov_path": None,
            "motion_raw_heatmap_npz_path": None,
            "motion_levels": {self._motion_zone_id: 50},
            "recordings_motion_level": None,
            "recordings_motion_last_fetch": 0.0,
            "motion_levels_source": "default",
            "motion_levels_updated_at": 0.0,
            "last_pulse_motion_levels_updated_at": 0.0,
            "last_pulse_time": None,
            "pulse_snapshot_counter": 0,
        }
        self._active_analytics_event_id = event_id

        linger_seconds = self.lingerEventStart / 1000.0
        self._analytics_start_task = asyncio.create_task(
            self._delayed_analytics_start(event_id, event_timestamp, linger_seconds)
        )

        if not self._motion_event_ts:
            self._motion_event_ts = current_time

    async def _delayed_analytics_start(
        self,
        event_id: int,
        event_timestamp: Optional[float],
        delay_seconds: float,
    ) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            await self._send_analytics_start_event(event_id, event_timestamp)
        except asyncio.CancelledError:
            self.logger.debug(f"Analytics start event {event_id} was cancelled during linger period")
            raise

    async def trigger_analytics_stop(
        self,
        event_timestamp: Optional[float] = None,
    ) -> None:
        if self._active_analytics_event_id is None:
            self.logger.warning(
                f"trigger_analytics_stop called but no active event found. "
                f"Event may have already ended or never started. Ignoring."
            )
            return

        event_id = self._active_analytics_event_id
        active_event = self._analytics_event_history.get(event_id)

        if not active_event:
            self.logger.warning(
                f"trigger_analytics_stop called for event {event_id} but event not found in history. Ignoring."
            )
            self._active_analytics_event_id = None
            return

        # Cancel the pulse task before processing stop
        if self._motion_pulse_task and not self._motion_pulse_task.done():
            self._motion_pulse_task.cancel()
            try:
                await self._motion_pulse_task
            except asyncio.CancelledError:
                pass
            self.logger.debug(f"Cancelled motion pulse task for event {event_id}")

        if self._analytics_start_task and not self._analytics_start_task.done():
            self._analytics_start_task.cancel()
            try:
                await self._analytics_start_task
            except asyncio.CancelledError:
                pass
            self.logger.info(
                f"Motion event {event_id} stopped before {self.lingerEventStart}ms linger period. "
                f"No start/stop events will be sent."
            )
            del self._analytics_event_history[event_id]
            self._active_analytics_event_id = None

            if not self._active_smart_events:
                self._motion_event_ts = None

            return

        snapshot_crop_path = None
        snapshot_fov_path = None
        heatmap_path = None

        smart_detect_ids = active_event.get("smart_detect_event_ids", [])
        if smart_detect_ids:
            most_recent_smart_id = smart_detect_ids[-1]

            smart_event = self._active_smart_events.get(most_recent_smart_id)
            if smart_event:
                snapshot_crop_path = smart_event.get("snapshot_crop_path")
                snapshot_fov_path = smart_event.get("snapshot_fov_path")
                heatmap_path = smart_event.get("heatmap_path")
                self.logger.info(
                    f"Using snapshots from smart detect event {most_recent_smart_id} "
                    f"for motion event {event_id}"
                )
            else:
                self.logger.warning(
                    f"Smart detect event {most_recent_smart_id} not found in history "
                    f"(may have been cleaned up after 60 minutes)"
                )

        if not snapshot_crop_path or not snapshot_fov_path or not heatmap_path:
            self.logger.info(
                f"No smart detect snapshots available for motion event {event_id}, "
                f"fetching fresh snapshots"
            )
            try:
                snapshot_crop_path, snapshot_fov_path, heatmap_path = await self.fetch_snapshots_for_event(
                    event_id, "motion"
                )
                self.logger.info(
                    f"Fetched fresh snapshots for motion event {event_id}: "
                    f"crop={snapshot_crop_path is not None}, "
                    f"fov={snapshot_fov_path is not None}, "
                    f"heatmap={heatmap_path is not None}"
                )
            except Exception as e:
                self.logger.error(f"Error fetching snapshots for motion event {event_id}: {e}")

        active_event["snapshot_crop_path"] = snapshot_crop_path
        active_event["snapshot_fov_path"] = snapshot_fov_path
        active_event["heatmap_path"] = heatmap_path
        active_event["motion_snapshot_path"] = snapshot_crop_path
        active_event["motion_snapshot_fov_path"] = snapshot_fov_path
        active_event["motion_heatmap_path"] = heatmap_path
        active_event["motion_raw_heatmap_npz_path"] = None  # Could be generated if needed

        snapshot_filename = str(snapshot_crop_path) if snapshot_crop_path else f"motionsnap_{event_id}.jpg"
        snapshot_fov_filename = str(snapshot_fov_path) if snapshot_fov_path else f"motionsnap_{event_id}_fullfov.jpg"
        heatmap_filename = str(heatmap_path) if heatmap_path else f"heatmap_{event_id}.png"
        raw_heatmap_npz = f"motion_raw_heatmap_{int(round(event_timestamp or time.time() * 1000))}.npz"

        active_event["snapshot_filename"] = snapshot_filename
        active_event["snapshot_fov_filename"] = snapshot_fov_filename
        active_event["heatmap_filename"] = heatmap_filename
        active_event["end_time"] = time.time()

        self._motion_event_id += 1
        motion_levels = active_event.get("motion_levels", {self._motion_zone_id: 0})

        payload: dict[str, Any] = {
            "clockBestMonotonic": int(self.get_uptime() * 1000),
            "clockBestWall": int(round(active_event["start_time"] * 1000)),
            "clockMonotonic": int(self.get_uptime() * 1000),
            "clockStream": int(self.get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": int(round(time.time() * 1000)),
            "edgeType": "stop",
            "eventId": self._motion_event_id,
            "eventType": "motion",
            "levels": motion_levels,
            "motionHeatmap": heatmap_filename,
            "motionHeatmapHeight": 90,
            "motionHeatmapWidth": 160,
            "motionSnapshot": snapshot_filename,
            "motionSnapshotFullFoV": snapshot_fov_filename,
            "motionSnapshotFullFoVHeight": 360,
            "motionSnapshotFullFoVWidth": 640,
            "motionSnapshotHeight": 360,
            "motionSnapshotWidth": 360,
            "motionRawHeatmapNPZ": raw_heatmap_npz,
        }

        duration = time.time() - active_event["start_time"]
        self.logger.info(
            f"Stopping motion event {event_id} (duration: {duration:.1f}s, "
            f"smart_detect_events: {len(smart_detect_ids)})"
        )
        self.logger.debug(
            f"Motion event snapshots: crop={snapshot_filename}, fov={snapshot_fov_filename}, heatmap={heatmap_filename}"
        )

        await self.send(
            self.gen_response("EventSmartMotion", payload=payload)
        )

        self._active_analytics_event_id = None

        if not self._active_smart_events:
            self._motion_event_ts = None

    def get_active_events_summary(self) -> dict[str, Any]:
        active_analytics = self._analytics_event_history.get(self._active_analytics_event_id) if self._active_analytics_event_id else None

        return {
            "analytics_event": {
                "active": self._active_analytics_event_id is not None,
                "event_id": self._active_analytics_event_id,
                "duration": time.time() - active_analytics["start_time"]
                    if active_analytics else None,
                "smart_detect_event_ids": active_analytics["smart_detect_event_ids"]
                    if active_analytics else [],
            },
            "smart_detect_events": {
                event_id: {
                    "object_type": event["object_type"].value,
                    "duration": time.time() - event["start_time"],
                    "has_descriptor": event["last_descriptor"] is not None,
                    "has_snapshot_crop": event["snapshot_crop_path"] is not None,
                    "has_snapshot_fov": event["snapshot_fov_path"] is not None,
                    "has_heatmap": event["heatmap_path"] is not None,
                }
                for event_id, event in self._active_smart_events.items()
            },
            "total_active_events": (1 if self._active_analytics_event_id else 0) + len(self._active_smart_events),
            "analytics_history_count": len(self._analytics_event_history),
        }

    async def stop_all_motion_events(self) -> None:
        # Cancel pulse task if running
        if self._motion_pulse_task and not self._motion_pulse_task.done():
            self._motion_pulse_task.cancel()
            try:
                await self._motion_pulse_task
            except asyncio.CancelledError:
                pass

        smart_event_ids = list(self._active_smart_events.keys())
        for event_id in smart_event_ids:
            event = self._active_smart_events[event_id]
            self.logger.info(
                f"Force stopping smart detect event {event_id} "
                f"({event['object_type'].value})"
            )
            try:
                await self.trigger_motion_stop(object_type=event["object_type"])
            except Exception as e:
                self.logger.error(
                    f"Error stopping smart detect event {event_id}: {e}"
                )

        if self._active_analytics_event_id is not None:
            self.logger.info("Force stopping motion event")
            try:
                await self.trigger_motion_stop()
            except Exception as e:
                self.logger.error(f"Error stopping motion event: {e}")

    def set_motion_zone_config(self, zone_config: dict[str, str]) -> None:
        """
        Configure motion zone IDs and names.
        
        Args:
            zone_config: Dict mapping zone IDs to zone names, e.g., {"1": "Front Door", "2": "Driveway"}
        """
        self._motion_zone_config = zone_config
        if zone_config:
            self._motion_zone_id = next(iter(zone_config.keys()), "1")
        self.logger.debug(f"Motion zone config updated: {self._motion_zone_config}")

    async def fetch_to_file(self, url: str, dst: Path) -> bool:
        try:
            async with aiohttp.request("GET", url) as resp:
                if resp.status != 200:
                    self.logger.error(f"Error retrieving file {resp.status}")
                    return False
                with dst.open("wb") as f:
                    f.write(await resp.read())
                    return True
        except aiohttp.ClientError:
            return False

    # Protocol implementation
    def gen_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def init_adoption(self) -> None:
        self.logger.info(
            f"Adopting with token [{self.args.token}] and mac [{self.args.mac}]"
        )

        video1_source = None
        for stream_index in ["video1", "video2", "video3"]:
            try:
                source = await self.get_stream_source(stream_index)
                if source:
                    if stream_index == "video1":
                        video1_source = source
                        width, height = self.probe_video_resolution(stream_index, source)
                        self._detected_resolutions[stream_index] = (width, height)
                    elif source != video1_source:
                        width, height = self.probe_video_resolution(stream_index, source)
                        self._detected_resolutions[stream_index] = (width, height)
                    else:
                        self.logger.debug(f"{stream_index} using video1 source as fallback, using default resolution")
            except NotImplementedError:
                self.logger.debug(f"{stream_index} not implemented, using defaults")
                break
            except Exception as e:
                if stream_index == "video1":
                    self.logger.warning(f"Could not probe {stream_index}: {e}, using defaults")
                else:
                    self.logger.debug(f"Could not probe {stream_index}, using default resolution")

        await self.send(
            self.gen_response(
                "ubnt_avclient_hello",
                payload={
                    "adoptionCode": self.args.token,
                    "connectionHost": self.args.host,
                    "connectionSecurePort": 7442,
                    "fwVersion": self.args.fw_version,
                    "hwrev": 19,
                    "idleTime": 191.96,
                    "ip": self.args.ip,
                    "mac": (
                        self.args.mac.replace(":", "")
                                    .replace("-", "")
                                    .upper()
                    ),
                    "model": self.args.model,
                    "name": self.args.name,
                    "protocolVersion": 67,
                    "rebootTimeoutSec": 30,
                    "semver": "v4.4.8",
                    "totalLoad": 0.5474,
                    "upgradeTimeoutSec": 150,
                    "uptime": int(self.get_uptime()),
                    "features": await self.get_feature_flags(),
                },
            ),
        )

    async def process_upgrade(self, msg: AVClientRequest) -> None:
        url = msg["payload"]["uri"]
        headers = {"Range": "bytes=0-100"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, ssl=False) as r:
                content = await r.content.readexactly(54)
                version = ""
                for i in range(0, 50):
                    b = content[4 + i]
                    if b != b"\x00":
                        version += chr(b)
                self.logger.debug(f"Pretending to upgrade to: {version}")
                self.args.fw_version = version

    def gen_response(
        self, name: str, response_to: int = 0, payload: Optional[dict[str, Any]] = None
    ) -> AVClientResponse:
        if not payload:
            payload = {}
        now = datetime.now(timezone.utc)
        return {
            "from": "ubnt_avclient",
            "functionName": name,
            "inResponseTo": response_to,
            "messageId": self.gen_msg_id(),
            "payload": payload,
            "responseExpected": False,
            "timeStamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "to": "UniFiVideo",
        }
    
    def get_uptime(self) -> float:
        return time.time() - self._init_time

    async def send(self, msg: AVClientRequest) -> None:
        self.logger.debug(f"Sending: {msg}")
        ws = self._session
        if ws:
            await ws.send(json.dumps(msg).encode())

    async def process(self, msg: bytes) -> bool:
        m = json.loads(msg)
        fn = m["functionName"]

        if fn == "GetRequest" and "payload" in m:
            what = m["payload"].get("what", "N/A")
            filename = m["payload"].get("filename", "N/A")
            self.logger.info(f"Processing [{fn}] message (what={what}, filename={filename})")
        else:
            self.logger.info(f"Processing [{fn}] message")
        self.logger.debug(f"Message contents: {m}")

        if (("responseExpected" not in m) or (m["responseExpected"] is False)) and (
            fn
            not in [
                "GetRequest",
                "ChangeVideoSettings",
                "UpdateFirmwareRequest",
                "Reboot",
                "ubnt_avclient_hello",
                "ContinuousMove"
            ]
        ):
            return False

        res: Optional[AVClientResponse] = None

        if fn == "ubnt_avclient_time":
            res = await self.process_time(m)
        elif fn == "ubnt_avclient_hello":
            await self.process_hello(m)
        elif fn == "ubnt_avclient_paramAgreement":
            res = await self.process_param_agreement(m)
        elif fn == "ResetIspSettings":
            res = await self.process_isp_settings(m)
        elif fn == "ChangeVideoSettings":
            res = await self.process_video_settings(m)
        elif fn == "ChangeDeviceSettings":
            res = await self.process_device_settings(m)
        elif fn == "ChangeOsdSettings":
            res = await self.process_osd_settings(m)
        elif fn == "NetworkStatus":
            res = await self.process_network_status(m)
        elif fn == "AnalyticsTest":
            res = self.gen_response("AnalyticsTest", response_to=m["messageId"])
        elif fn == "ChangeSoundLedSettings":
            res = await self.process_sound_led_settings(m)
        elif fn == "ChangeIspSettings":
            res = await self.process_change_isp_settings(m)
        elif fn == "ChangeAnalyticsSettings":
            res = await self.process_analytics_settings(m)
        elif fn == "GetRequest":
            res = await self.process_snapshot_request(m)
        elif fn == "UpdateUsernamePassword":
            res = self.gen_response(
                "UpdateUsernamePassword", response_to=m["messageId"]
            )
        elif fn == "ChangeSmartDetectSettings":
            res = self.gen_response(
                "ChangeSmartDetectSettings", response_to=m["messageId"]
            )
        elif fn == "ChangeAudioEventsSettings":
            res = self.gen_response(
                "ChangeAudioEventsSettings", response_to=m["messageId"]
            )
        elif fn == "UpdateFaceDBRequest":
            res = await self.process_update_face_db(m)
        elif fn == "ChangeTalkbackSettings":
            res = self.gen_response(
                "ChangeTalkbackSettings", response_to=m["messageId"]
            )
        elif fn == "ChangeSmartMotionSettings":
            res = await self.process_smart_motion_settings(m)
        elif fn == "SmartMotionTest":
            res = self.gen_response(
                "SmartMotionTest", response_to=m["messageId"]
            )
        elif fn == "ChangeClarityZones":
            res = self.gen_response(
                "ChangeClarityZones", response_to=m["messageId"]
            )
        elif fn == "UpdateFirmwareRequest":
            await self.process_upgrade(m)
            return True
        elif fn == "Reboot":
            return True
        elif fn == "ContinuousMove":
            res = await self.process_continuous_move(m)
        else:
            self.logger.warning(
                f"Received unhandled message type: {fn}. "
                f"Message contents: {m}"
            )
        if res is not None:
            await self.send(res)

        return False

    async def close(self):
        self.logger.info("Protect sent close, cleaning up instance")
        await self.stop_all_motion_events()
        self.close_streams()

    # Legacy API for subclasses - backwards compatibility
    async def trigger_motion_start(
        self,
        object_type: Optional[SmartDetectObjectType] = None,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
    ) -> None:
        if object_type:
            await self.trigger_smart_detect_start(object_type, custom_descriptor, event_timestamp)
        else:
            await self.trigger_analytics_start(event_timestamp)

    async def trigger_motion_update(
        self,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        object_type: Optional[SmartDetectObjectType] = None,
    ) -> None:
        target_object_type = object_type or self._motion_object_type

        if not target_object_type:
            self.logger.warning(
                "trigger_motion_update called but no object_type specified and "
                "no active smart detect event found. Ignoring."
            )
            return

        await self.trigger_smart_detect_update(target_object_type, custom_descriptor, event_timestamp)

    async def trigger_motion_stop(
        self,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        object_type: Optional[SmartDetectObjectType] = None,
    ) -> None:
        if object_type:
            await self.trigger_smart_detect_stop(object_type, custom_descriptor, event_timestamp)
        else:
            await self.trigger_analytics_stop(event_timestamp)
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import backoff
from aiomqtt import Client
from aiomqtt.exceptions import MqttError

from unifi.cams.base import SmartDetectObjectType
from unifi.cams.frigate_lib.descriptors import build_descriptor_from_frigate_msg
from unifi.cams.frigate_lib.events import FrigateEventHandlerMixin
from unifi.cams.frigate_lib.motion import (
    DetectFpsTracker,
    fetch_recordings_motion_level,
    motion_level_from_area,
)
from unifi.cams.frigate_lib.snapshots import FrigateSnapshotClient
from unifi.cams.frigate_lib.tracker_ids import TrackerIdAllocator
from unifi.cams.frigate_lib.zones import ZoneStatusTracker
from unifi.cams.rtsp import RTSPCam


class FrigateCam(FrigateEventHandlerMixin, RTSPCam):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        super().__init__(args, logger)
        self.args = args

        self.detect_fps_tracker = DetectFpsTracker(args.frigate_http_url, args.frigate_camera, logger)
        self.snapshots = FrigateSnapshotClient(args.frigate_http_url, args.frigate_camera, logger)
        self.tracker_ids = TrackerIdAllocator(logger)

        # Active smart-detect event ID used for EventSmartDetect updates.
        # This is distinct from analytics motion-window IDs.
        self._motion_smart_event_id: Optional[int] = None
        self._motion_start_time: float = 0.0

        # Store snapshot readiness per Frigate event ID
        self.event_snapshot_ready: dict[str, asyncio.Event] = {}
        # Track last update time for each Frigate track (for timeout detection)
        self.event_last_update: dict[str, float] = {}
        self.event_timeout_seconds = 600  # Timeout after 600 seconds (10 minutes) without updates

        # Frigate zone name -> Protect numeric zone ID, static per camera.
        # Protect has no concept of Frigate's zone names; you choose the
        # numeric IDs yourself and must keep them consistent with whatever
        # ChangeSmartDetectSettings zone config your emulator advertises on
        # connect (see protocol spec Section 8.2/8.3).
        self.zone_name_to_id: dict[str, int] = self._parse_zone_map(getattr(self.args, "zone_map", None))
        self.zone_status_tracker = ZoneStatusTracker(list(set(self.zone_name_to_id.values())))

        # Track which Frigate events are currently active (for lifecycle management).
        self._active_frigate_events: set[str] = set()
        # Frigate event.id -> object type, used for label-aware snapshot matching.
        self._frigate_event_object_types: dict[str, SmartDetectObjectType] = {}

        # Track which Frigate events are currently active (for lifecycle management).
        self._active_frigate_events: set[str] = set()
        # Frigate event.id -> object type, used for label-aware snapshot matching.
        self._frigate_event_object_types: dict[str, SmartDetectObjectType] = {}
        # Track position_changes for stationary object detection.
        # Format: {event_id: {\"before\": int, \"after\": int}}
        self._frigate_event_position_changes: dict[str, dict[str, int]] = {}

    @property
    def frigate_detect_fps(self) -> float:
        return self.detect_fps_tracker.fps

    @staticmethod
    def _parse_zone_map(raw: Optional[str]) -> dict[str, int]:
        """
        Parse --zone-map JSON, e.g. '{"parking": 1, "road": 2, "sidewalk": 3}'.
        Falls back to an empty map (all detections reported as zone-less) if
        not provided or malformed, rather than crashing at startup.
        """
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return {str(k): int(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}

    @classmethod
    def add_parser(cls, parser: argparse.ArgumentParser) -> None:
        super().add_parser(parser)
        parser.add_argument("--mqtt-host", required=True, help="MQTT server")
        parser.add_argument("--mqtt-port", default=1883, type=int, help="MQTT server")
        parser.add_argument("--mqtt-username", required=False)
        parser.add_argument("--mqtt-password", required=False)
        parser.add_argument("--mqtt-prefix", default="frigate", type=str, help="Topic prefix")
        parser.add_argument(
            "--frigate-camera", required=True, type=str, help="Name of camera in frigate"
        )
        parser.add_argument(
            "--camera-width", default=1920, type=int, help="Camera frame width in pixels (default: 1920)"
        )
        parser.add_argument(
            "--camera-height", default=1080, type=int, help="Camera frame height in pixels (default: 1080)"
        )
        parser.add_argument(
            "--frigate-detect-width",
            default=1280,
            type=int,
            help="Frigate detection frame width in pixels (default: 1280)",
        )
        parser.add_argument(
            "--frigate-detect-height",
            default=720,
            type=int,
            help="Frigate detection frame height in pixels (default: 720)",
        )
        parser.add_argument(
            "--frigate-http-url",
            required=False,
            type=str,
            help="Frigate HTTP API URL (e.g., http://frigate:5000). If provided, snapshots will be fetched via HTTP instead of MQTT.",
        )
        parser.add_argument(
            "--frigate-time-sync-ms",
            default=0,
            type=int,
            help="Time synchronization offset in milliseconds to apply to Frigate event timestamps (default: 0). Positive values shift timestamps backward to compensate for Frigate event delay relative to video.",
        )
        parser.add_argument(
            "--zone-map",
            required=False,
            type=str,
            default=None,
            help=(
                'JSON mapping of Frigate zone name -> Protect numeric zone ID '
                'for this camera, e.g. \'{"parking": 1, "road": 2}\'. Must match '
                'whatever zone IDs/polygons this emulated camera advertises via '
                'ChangeSmartDetectSettings on connect.'
            ),
        )
        parser.add_argument(
            "--doorbell",
            action="store_true",
            default=False,
            help="Enable doorbell event forwarding from Frigate",
        )

    async def get_feature_flags(self) -> dict[str, Any]:
        flags = await super().get_feature_flags()
        flags["doorbell"] = self.args.doorbell
        return {
            **flags,
            **{
                "mic": True,
                "smartDetect": ["person", "vehicle", "animal", "package"],
            },
        }

    # --- descriptor / zone helpers ------------------------------------------------

    def _build_descriptor(self, frigate_msg: dict[str, Any], object_type: SmartDetectObjectType, tracker_id: int) -> dict[str, Any]:
        return build_descriptor_from_frigate_msg(
            frigate_msg,
            object_type,
            tracker_id,
            frigate_detect_width=self.args.frigate_detect_width,
            frigate_detect_height=self.args.frigate_detect_height,
            frigate_time_sync_ms=self.args.frigate_time_sync_ms,
            frigate_detect_fps=self.frigate_detect_fps,
            zone_name_to_id=self.zone_name_to_id,
            logger=self.logger,
            camera_name=self.args.frigate_camera,
        )

    def _update_zone_status_for_track(
        self, tracker_id: int, zones: list[int], confidence: float, active: bool
    ) -> dict[str, dict[str, Any]]:
        """
        Feed this track's current zone membership into the shared, decaying
        per-camera zone status tracker and return the full zonesStatus dict
        to attach to the outgoing message. This is deliberately independent
        of any single track's own start/update/end lifecycle -- zonesStatus
        reflects the union of all currently-active tracks on the camera
        (protocol spec Section 7.4/8.5), not just this one.
        """
        if active:
            self.zone_status_tracker.update_track(tracker_id, zones, confidence)
        else:
            self.zone_status_tracker.remove_track(tracker_id)
        return self.zone_status_tracker.as_dict()

    def _is_motion_window_active(self) -> bool:
        return self._active_analytics_event_id is not None

    # --- motion level -------------------------------------------------------------

    async def _update_motion_levels_from_recordings(
        self, frigate_msg: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Update the active EventSmartMotion level using recordings motion when
        available, falling back to area-derived motion only if the API
        doesn't return a usable value.
        """
        if self._active_analytics_event_id is None:
            return

        active_event = self._analytics_event_history.get(self._active_analytics_event_id)
        if not active_event:
            return

        recordings_level = await fetch_recordings_motion_level(
            self.args.frigate_http_url, self.args.frigate_camera, active_event, self.logger
        )
        if recordings_level is not None:
            active_event["motion_levels"] = {str(self._motion_zone_id): recordings_level}
            active_event["motion_levels_source"] = "recordings"
            active_event["motion_levels_updated_at"] = time.time()
            return

        after_area = frigate_msg.get("after", {}).get("area", 0) if frigate_msg else 0
        if after_area:
            level = motion_level_from_area(
                after_area, self.args.frigate_detect_width, self.args.frigate_detect_height
            )
            active_event["motion_levels"] = {str(self._motion_zone_id): level}
            active_event["motion_levels_source"] = "area"
            active_event["motion_levels_updated_at"] = time.time()

    # --- snapshots ------------------------------------------------------------

    async def _fetch_and_cache_frigate_event_snapshot(self, frigate_event_id: str, smart_event_id: int) -> None:
        """
        Fetch the event-specific snapshot from Frigate's API and cache it
        per-tracker on the smart event, under
        smart_event['tracker_snapshots'][tracker_id], so the stop payload can
        assign each trackerID its own image rather than sharing one file.
        """
        tracker_id = self.tracker_ids.peek(frigate_event_id)
        if tracker_id is None:
            return

        crop_path, fov_path = await self.snapshots.fetch_event_snapshots(frigate_event_id)

        smart_event = self._active_smart_events.get(smart_event_id)
        if smart_event and (crop_path or fov_path):
            smart_event.setdefault("tracker_snapshots", {})[tracker_id] = {
                "crop": crop_path,
                "fov": fov_path,
            }
            # Also keep the shared fallback updated (most-recently-fetched tracker wins)
            if crop_path:
                smart_event["snapshot_crop_path"] = crop_path
            if fov_path:
                smart_event["snapshot_fov_path"] = fov_path
                smart_event["heatmap_path"] = fov_path
            self.logger.debug(
                f"Cached snapshots for smart event {smart_event_id} trackerID {tracker_id}: "
                f"crop={crop_path}, fov={fov_path}"
            )

    async def fetch_snapshots_for_event(
        self, event_id: int, event_type: str = "analytics"
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Return (crop, fov, heatmap) snapshot paths for an event, preferring
        anything already cached on the associated smart-detect event over a
        fresh fetch of the camera's current live view.
        """
        if event_type in ("smart_detect", "motion"):
            smart_event_id = event_id if event_type == "smart_detect" else None
            if event_type == "motion" and event_id in self._analytics_event_history:
                smart_detect_ids = self._analytics_event_history[event_id].get("smart_detect_event_ids", [])
                if smart_detect_ids:
                    smart_event_id = smart_detect_ids[-1]
            if smart_event_id is not None:
                smart_event = self._active_smart_events.get(smart_event_id)
                if smart_event:
                    crop = smart_event.get("snapshot_crop_path")
                    fov = smart_event.get("snapshot_fov_path")
                    heatmap = smart_event.get("heatmap_path") or fov
                    if crop and fov:
                        self.logger.debug(f"Using pre-cached event-specific snapshots for event {event_id}")
                        return crop, fov, heatmap

        timestamp = None
        if event_type == "analytics" and event_id in self._analytics_event_history:
            start_time = self._analytics_event_history[event_id].get("start_time")
            if start_time is not None:
                timestamp = int(start_time)

        crop_path, fov_path = await self.snapshots.fetch_latest_snapshots(timestamp)
        return crop_path, fov_path, fov_path

    # --- mqtt loop ------------------------------------------------------------

    async def run(self) -> None:
        has_connected = False

        await self.detect_fps_tracker.load()

        @backoff.on_predicate(backoff.expo, max_value=60, logger=self.logger)
        async def mqtt_connect():
            nonlocal has_connected
            try:
                async with Client(
                    self.args.mqtt_host,
                    port=self.args.mqtt_port,
                    username=self.args.mqtt_username,
                    password=self.args.mqtt_password,
                ) as client:
                    has_connected = True
                    self.logger.info(f"Connected to {self.args.mqtt_host}:{self.args.mqtt_port}")
                    await client.subscribe(f"{self.args.mqtt_prefix}/#")
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self.monitor_event_timeouts())
                        tg.create_task(self.detect_fps_tracker.refresh_periodically())
                        async for message in client.messages:
                            if message.topic.matches(f"{self.args.mqtt_prefix}/events"):
                                tg.create_task(self.handle_detection_event(message))
                            elif message.topic.matches(
                                f"{self.args.mqtt_prefix}/{self.args.frigate_camera}/+/snapshot"
                            ):
                                tg.create_task(self.handle_snapshot_event(message))
                            elif message.topic.matches(f"{self.args.mqtt_prefix}/{self.args.frigate_camera}/motion"):
                                tg.create_task(self.handle_motion_event(message))
                            elif self.args.doorbell and message.topic.matches(
                                f"{self.args.mqtt_prefix}/{self.args.frigate_camera}/doorbell"
                            ):
                                tg.create_task(self.handle_doorbell_event(message))
                            elif message.topic.matches(f"{self.args.mqtt_prefix}/reviews"):
                                self.logger.debug(f"Received Frigate review event: {message.payload.decode()}")
            except MqttError:
                if not has_connected:
                    raise

        await mqtt_connect()

    async def monitor_event_timeouts(self) -> None:
        """Monitor active Frigate object tracks and end those that haven't been updated."""
        while True:
            await asyncio.sleep(30)
            current_time = time.time()
            expired_frigate_events = []

            for frigate_event_id in list(self._active_frigate_events):
                last_update = self.event_last_update.get(frigate_event_id, self._motion_start_time)
                time_since_update = current_time - last_update

                if time_since_update > self.event_timeout_seconds:
                    expired_frigate_events.append(frigate_event_id)
                    self.logger.warning(
                        f"TRACK TIMEOUT: Frigate event {frigate_event_id} "
                        f"has not been updated for {time_since_update:.1f}s "
                        f"(timeout: {self.event_timeout_seconds}s). Force ending track."
                    )

            for frigate_event_id in expired_frigate_events:
                self._forget_track(frigate_event_id)

    def _forget_track(self, frigate_event_id: str) -> None:
        """Drop all bookkeeping for a Frigate track: zone tracker contribution,
        allocated trackerID, and the various per-event dicts/sets."""
        tracker_id = self.tracker_ids.peek(frigate_event_id)
        if tracker_id is not None:
            self.zone_status_tracker.remove_track(tracker_id)
            self.logger.debug(
                f"Removed track {frigate_event_id} (trackerID={tracker_id}) from zone status tracker"
            )

        self.tracker_ids.release(frigate_event_id)
        self._active_frigate_events.discard(frigate_event_id)
        self._frigate_event_object_types.pop(frigate_event_id, None)
        self.event_last_update.pop(frigate_event_id, None)
        self.event_snapshot_ready.pop(frigate_event_id, None)
        # Clean up position_changes tracking
        self._frigate_event_position_changes.pop(frigate_event_id, None)


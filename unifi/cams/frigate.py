import argparse
import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp
import backoff
from aiomqtt import Client, Message
from aiomqtt.exceptions import MqttError

from unifi.cams.base import SmartDetectObjectType
from unifi.cams.rtsp import RTSPCam

# --- Zone status decay tuning (see protocol spec Section 8.5) -----------------
# Zone occupancy in real UniFi Protect devices is a smoothed/decaying aggregate
# signal, not a hard per-object boolean: once an occupying object leaves a zone,
# the zone's reported "level" decays gradually over several ticks before the
# status flips from "moving" to "leave". These constants approximate that
# behavior; tune DECAY_PER_TICK against your own captured traffic if needed.
ZONE_LEVEL_CONFIDENCE_OFFSET = 10  # observed: zonesStatus.level runs ~10 below confidenceLevel
ZONE_DECAY_PER_TICK = 8
DEFAULT_DETECT_FPS = 5.0  # Frigate's own hardcoded default for detect.fps
DETECT_FPS_MIN, DETECT_FPS_MAX = 1.0, 30.0  # sanity bounds; guards against a bad/renamed config field
DETECT_FPS_REFRESH_SECONDS = 300  # re-fetch periodically in case Frigate config changes live


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


class FrigateCam(RTSPCam):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        super().__init__(args, logger)
        self.args = args
        self.frigate_detect_fps: float = DEFAULT_DETECT_FPS
        # Map Frigate event IDs to UniFi event IDs for tracking
        self.frigate_to_unifi_event_map: dict[str, int] = {}
        # Store snapshot readiness per Frigate event ID
        self.event_snapshot_ready: dict[str, asyncio.Event] = {}
        # Track last update time for each event (for timeout detection)
        self.event_last_update: dict[int, float] = {}
        self.event_timeout_seconds = 600  # Timeout after 600 seconds (10 minutes) without updates

        # Frigate zone name -> Protect numeric zone ID, static per camera.
        # Protect has no concept of Frigate's zone names; you choose the
        # numeric IDs yourself and must keep them consistent with whatever
        # ChangeSmartDetectSettings zone config your emulator advertises on
        # connect (see protocol spec Section 8.2/8.3).
        self.zone_name_to_id: dict[str, int] = self._parse_zone_map(
            getattr(self.args, "zone_map", None)
        )
        self.zone_status_tracker = ZoneStatusTracker(list(set(self.zone_name_to_id.values())))
        # tracker_id -> Frigate event_id, needed to reverse-look-up which
        # camera-numeric trackerID owns a closing Frigate event
        self.unifi_tracker_id_by_frigate_event: dict[str, int] = {}

        # Frigate's own event/tracker ID is a string (e.g. a UUID-like value)
        # and is NOT reused as the numeric trackerID directly. Real UniFi
        # Protect devices use session-scoped, monotonically-increasing
        # integer trackerIDs (observed: 716235 -> 716236 -> 716259 across one
        # session). Deriving trackerID via hash(frigate_id) % N is unsafe:
        # Python's str hash is randomized per-process (unstable across
        # restarts) and offers no uniqueness guarantee against collisions
        # between concurrently active tracks. Instead, assign a fresh integer
        # once per Frigate "new" event and hold it for that track's lifetime.
        self._next_tracker_id: int = 700000  # arbitrary session-scoped base, mirrors observed value range
        self.frigate_id_to_tracker_id: dict[str, int] = {}

    def _allocate_tracker_id(self, frigate_event_id: str) -> int:
        """Assign a fresh trackerID for a newly-started Frigate track. Call
        exactly once per Frigate event_type == "new"."""
        tracker_id = self._next_tracker_id
        self._next_tracker_id += 1
        self.frigate_id_to_tracker_id[frigate_event_id] = tracker_id
        return tracker_id

    def _get_tracker_id(self, frigate_event_id: str) -> int:
        """Look up the trackerID already allocated for an in-progress Frigate
        track (update/end events). Falls back to allocating one if missing
        (e.g. a missed "new" event) so the bridge degrades gracefully rather
        than crashing, while logging the anomaly."""
        tracker_id = self.frigate_id_to_tracker_id.get(frigate_event_id)
        if tracker_id is None:
            self.logger.warning(
                f"No trackerID allocated for Frigate event_id={frigate_event_id} "
                f"(likely missed 'new' event); allocating one now"
            )
            tracker_id = self._allocate_tracker_id(frigate_event_id)
        return tracker_id

    def _release_tracker_id(self, frigate_event_id: str) -> None:
        """Free the trackerID mapping once a Frigate track ends."""
        self.frigate_id_to_tracker_id.pop(frigate_event_id, None)

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
        parser.add_argument(
            "--mqtt-prefix", default="frigate", type=str, help="Topic prefix"
        )
        parser.add_argument(
            "--frigate-camera",
            required=True,
            type=str,
            help="Name of camera in frigate",
        )
        parser.add_argument(
            "--camera-width",
            default=1920,
            type=int,
            help="Camera frame width in pixels (default: 1920)",
        )
        parser.add_argument(
            "--camera-height",
            default=1080,
            type=int,
            help="Camera frame height in pixels (default: 1080)",
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

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            **await super().get_feature_flags(),
            **{
                "mic": True,
                "smartDetect": [
                    "person",
                    "vehicle",
                    "animal",
                    "package",
                ],
            },
        }

    @classmethod
    def label_to_object_type(cls, label: str) -> Optional[SmartDetectObjectType]:
        if label == "person":
            # Available: person, face
            return SmartDetectObjectType.PERSON
        elif label in {"vehicle", "car", "motorcycle", "school_bus", "license_plate"}:
            # Available: car, motorcycle, bicycle, boat, school_bus, license_plate
            return SmartDetectObjectType.VEHICLE
        elif label in {"cat", "dog", "horse", "rabbit", "squirrel", "goat"}:
            # Available: dog, cat, deer, horse, bird, raccoon, fox, bear, cow, squirrel, goat, rabbit, skunk, kangaroo
            return SmartDetectObjectType.ANIMAL
        elif label in {"package"}:
            # Available: package
            return SmartDetectObjectType.PACKAGE

    def _translate_zones(self, current_zones: list[str]) -> list[int]:
        """
        Frigate's `current_zones` -> Protect numeric zone IDs via the static
        per-camera map. Order preserved as given (most-recently-entered first,
        matching Frigate's own convention); unmapped zone names are dropped
        rather than raising, so a config gap degrades to "zone-less" instead
        of crashing the bridge.
        """
        translated = []
        for zname in current_zones:
            zid = self.zone_name_to_id.get(zname)
            if zid is not None:
                translated.append(zid)
            else:
                self.logger.debug(
                    f"Frigate zone '{zname}' has no entry in --zone-map for "
                    f"camera '{self.args.frigate_camera}'; omitting from zones[]"
                )
        return translated

    def build_descriptor_from_frigate_msg(
        self, frigate_msg: dict[str, Any], object_type: SmartDetectObjectType, tracker_id: int
    ) -> dict[str, Any]:
        """
        Build a UniFi Protect-compatible descriptor from Frigate event data.

        `tracker_id` must be pre-allocated via `_allocate_tracker_id`/
        `_get_tracker_id` -- see the trackerID allocator notes in __init__.
        Frigate's own event id (a string) is never used as the numeric
        trackerID directly.

        Coordinate system note (confirmed against real device captures): both
        zone polygons and descriptor bounding boxes use a normalized 0-1000
        grid, independent of any stream's actual pixel resolution -- NOT the
        low-res sub-stream's raw pixel space. The scaling below is correct.
        """
        after = frigate_msg.get("after", {})
        type = after.get("type", "unknown")

        # Extract bounding box if available
        box = after.get("box")
        if box and len(box) == 4:
            # Frigate box format: [x_min, y_min, x_max, y_max] in configured pixel dimensions
            # UniFi format: [x, y, width, height] in a 0-1000 normalized coordinate system
            frigate_width = self.args.frigate_detect_width
            frigate_height = self.args.frigate_detect_height
            UNIFI_WIDTH = 1000
            UNIFI_HEIGHT = 1000
            UNIFI_Y_OFFSET = 0  # Offset is needed if cam width/height differ from detection dimensions

            x_scale = UNIFI_WIDTH / frigate_width
            y_scale = UNIFI_HEIGHT / frigate_height

            x_min_unifi = int(box[0] * x_scale)
            y_min_unifi = int(box[1] * y_scale) + UNIFI_Y_OFFSET
            x_max_unifi = int(box[2] * x_scale)
            y_max_unifi = int(box[3] * y_scale) + UNIFI_Y_OFFSET

            x = x_min_unifi
            y = y_min_unifi
            width = x_max_unifi - x_min_unifi
            height = y_max_unifi - y_min_unifi
            coord = [x, y, width, height]
        else:
            coord = [0, 0, 1920, 1080]

        # Extract confidence score (Frigate uses 0.0-1.0, UniFi uses 0-100)
        score = after.get("top_score" if type == "end" else "score", 0.95)
        confidence_level = int(score * 100)

        stationary = after.get("stationary", False)

        # Real zone translation (previously hardcoded to [0] regardless of
        # Frigate's actual zone membership -- see protocol spec Section 8.4).
        current_zones = after.get("current_zones", [])
        zones = self._translate_zones(current_zones)

        # boxColor: real devices show "red" for objects actively occupying a
        # zone of interest and "white" for background/idle detections outside
        # any zone (protocol spec Section 3, descriptors[].boxColor).
        box_color = "red" if zones else "white"

        average_speed = after.get("average_estimated_speed", 0)
        speed = float(average_speed) if average_speed > 0 else None

        # License plate (vehicles) via Frigate's recognized_license_plate.
        license_plate_data = after.get("recognized_license_plate")
        license_plate = None
        license_plate_score = None
        if license_plate_data and isinstance(license_plate_data, list) and len(license_plate_data) >= 1:
            license_plate = license_plate_data[0]
            if len(license_plate_data) >= 2:
                license_plate_score = license_plate_data[1]

        # Face recognition (persons) via Frigate's generic sub_label facility.
        # Real Protect devices only populate name/tag when there's an actual
        # recognition match (see protocol spec Section 3, name/tag) -- they
        # are not a static branding string, so don't hardcode one here.
        sub_label_data = after.get("sub_label")
        recognized_name = None
        if (
            object_type == SmartDetectObjectType.PERSON
            and sub_label_data
            and isinstance(sub_label_data, list)
            and len(sub_label_data) >= 1
        ):
            recognized_name = sub_label_data[0]

        if object_type == SmartDetectObjectType.VEHICLE and license_plate:
            if license_plate_score is not None:
                name = f"{license_plate} ({license_plate_score:.1%})"
            else:
                name = license_plate
        elif object_type == SmartDetectObjectType.PERSON and recognized_name:
            name = recognized_name
        else:
            name = ""

        # Real devices mirror name into tag for the same matched entity
        # rather than using a static "Tagged by Frigate" label.
        tag = name

        descriptor = {
            "attributes": None,
            "boxColor": box_color,
            "confidenceLevel": confidence_level,
            "coord": coord,
            "coord3d": [-1, -1],
            "depth": None,
            "firstShownTimeMs": int(after.get("start_time", 0) * 1000) - self.args.frigate_time_sync_ms,
            # idleSinceTimeMs derivation: converts Frigate's motionless_count
            # (a frame counter) into an epoch-ms instant anchored to frame_time,
            # using the camera's actual detect.fps rather than assuming
            # 1 frame == 1 second. Recomputing this every message is safe --
            # as long as frame_time and motionless_count advance together at a
            # consistent fps, the derived instant stays pinned automatically
            # (validated against real device idleSinceTimeMs behavior, which
            # stays constant across a stationary track's lifetime).
            "idleSinceTimeMs": (
                int(after.get("frame_time", 0) * 1000)
                - int(after.get("motionless_count", 0) * (1000 / self.frigate_detect_fps))
                - self.args.frigate_time_sync_ms
            ) if after.get("motionless_count", 0) > 0 else 0,
            "lines": [],
            "loiterZones": [],
            "name": name,
            "objectType": object_type.value,
            "secondLensZones": [],
            "speed": speed,
            "stationary": stationary,
            "tag": tag,
            "trackerID": tracker_id,
            "zones": zones,
        }

        if license_plate:
            descriptor["licensePlate"] = license_plate

        self.logger.debug(
            f"Built descriptor: trackerID={tracker_id}, confidence={confidence_level}, "
            f"coord={coord}, stationary={stationary}, speed={speed}, "
            f"licensePlate={license_plate}, zones={zones}, boxColor={box_color}, name={name!r}"
        )

        return descriptor

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

    async def fetch_snapshots_for_event(
        self, event_id: int, event_type: str = "analytics"
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Fetch and cache all three snapshot types for an event from Frigate.

        Args:
            event_id: The event ID (analytics or smart detect)
            event_type: "analytics" or "smart_detect"

        Returns:
            Tuple of (crop_path, fov_path, heatmap_path) - paths to cached snapshot files
        """
        if not self.args.frigate_http_url:
            self.logger.warning("Cannot fetch snapshots: frigate_http_url not configured")
            return (None, None, None)

        frigate_event_id = None
        if event_type == "smart_detect":
            for frig_id, unifi_id in self.frigate_to_unifi_event_map.items():
                if unifi_id == event_id:
                    frigate_event_id = frig_id
                    break

        if frigate_event_id:
            base_url = f"{self.args.frigate_http_url}/api/events/{frigate_event_id}/snapshot.jpg"
            full_url = base_url
            thumbnail_url = f"{base_url}?crop=1&quality=80"

            self.logger.debug(
                f"Using Frigate event-specific snapshot URLs for event {event_id} "
                f"(Frigate: {frigate_event_id}): {base_url}"
            )
        else:
            timestamp = None
            if event_type == "analytics" and event_id in self._analytics_event_history:
                event_data = self._analytics_event_history[event_id]
                start_time = event_data.get('start_time')
                if start_time is not None:
                    timestamp = int(start_time)

            base_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"

            full_url = base_url
            if timestamp is not None:
                full_url = f"{full_url}?timestamp={timestamp}"

            thumbnail_url = f"{base_url}?height=360&quality=80"
            if timestamp is not None:
                thumbnail_url = f"{thumbnail_url}&timestamp={timestamp}"

        self.logger.debug(f"Fetching snapshots for event {event_id}: full={full_url}, thumbnail={thumbnail_url}")

        async def fetch_url(url: str, snapshot_type: str) -> Optional[Path]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            f = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            f.write(image_data)
                            f.close()
                            self.logger.debug(
                                f"Fetched {snapshot_type} snapshot for event {event_id} "
                                f"({len(image_data)} bytes) -> {f.name}"
                            )
                            return Path(f.name)
                        else:
                            error_body = await response.text()
                            self.logger.warning(
                                f"Failed to fetch {snapshot_type} snapshot: "
                                f"HTTP {response.status}, Response: {error_body}"
                            )
                            return None
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout fetching {snapshot_type} snapshot")
                return None
            except Exception as e:
                self.logger.error(f"Error fetching {snapshot_type} snapshot: {e}")
                return None

        results = await asyncio.gather(
            fetch_url(full_url, "full"),
            fetch_url(thumbnail_url, "thumbnail"),
            return_exceptions=True
        )

        snapshot_fov = results[0] if not isinstance(results[0], Exception) else None
        snapshot_crop = results[1] if not isinstance(results[1], Exception) else None

        heatmap = snapshot_fov

        self.logger.info(
            f"Fetched snapshots for event {event_id}: "
            f"crop={'✓' if snapshot_crop else '✗'}, "
            f"fov={'✓' if snapshot_fov else '✗'}, "
            f"heatmap={'✓' if heatmap else '✗'}"
        )

        return (snapshot_crop, snapshot_fov, heatmap)

    async def run(self) -> None:
        has_connected = False

        await self.load_frigate_detect_fps()

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
                    self.logger.info(
                        f"Connected to {self.args.mqtt_host}:{self.args.mqtt_port}"
                    )
                    await client.subscribe(f"{self.args.mqtt_prefix}/#")
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self.monitor_event_timeouts())
                        tg.create_task(self.refresh_frigate_detect_fps_periodically())
                        async for message in client.messages:
                            if message.topic.matches(f"{self.args.mqtt_prefix}/events"):
                                tg.create_task(self.handle_detection_event(message))
                            elif message.topic.matches(
                                f"{self.args.mqtt_prefix}/{self.args.frigate_camera}/+/snapshot"
                            ):
                                tg.create_task(self.handle_snapshot_event(message))
                            elif message.topic.matches(
                                f"{self.args.mqtt_prefix}/{self.args.frigate_camera}/motion"
                            ):
                                tg.create_task(self.handle_motion_event(message))
            except MqttError:
                if not has_connected:
                    raise

        await mqtt_connect()

    async def load_frigate_detect_fps(self) -> None:
        """
        Load detect FPS from Frigate's resolved config (/api/config, not
        /api/config/raw -- the resolved endpoint merges in Frigate's defaults
        so detect.fps is present even if never set explicitly in config.yml).
        Falls back through camera-level -> global -> hardcoded default, with a
        final sanity bound, so a bad/renamed field can't silently corrupt
        every idleSinceTimeMs calculation downstream.
        """
        frigate_url = (self.args.frigate_http_url or "http://frigate:5000").rstrip("/")
        config_url = f"{frigate_url}/api/config"

        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(config_url) as response:
                    if response.status != 200:
                        self.logger.warning(
                            f"Unable to load Frigate config from {config_url}: HTTP {response.status}. "
                            f"Keeping current detect_fps={self.frigate_detect_fps}"
                        )
                        return

                    config = await response.json()

            cam_cfg = config.get("cameras", {}).get(self.args.frigate_camera, {})
            fps = cam_cfg.get("detect", {}).get("fps")
            if not fps:
                fps = config.get("detect", {}).get("fps")
            if not fps:
                fps = DEFAULT_DETECT_FPS
                self.logger.warning(
                    f"Frigate config missing detect.fps for camera "
                    f"'{self.args.frigate_camera}' at every level; using default {fps}"
                )

            fps = float(fps)
            if not (DETECT_FPS_MIN <= fps <= DETECT_FPS_MAX):
                self.logger.warning(
                    f"Frigate detect.fps={fps} out of sane bounds "
                    f"[{DETECT_FPS_MIN}, {DETECT_FPS_MAX}]; keeping previous value "
                    f"{self.frigate_detect_fps} instead"
                )
                return

            self.frigate_detect_fps = fps
            self.logger.info(
                f"Frigate detect FPS for camera '{self.args.frigate_camera}': {self.frigate_detect_fps}"
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as e:
            self.logger.warning(
                f"Failed to fetch Frigate detect FPS from {config_url}: {e}. "
                f"Keeping current detect_fps={self.frigate_detect_fps}"
            )

    async def refresh_frigate_detect_fps_periodically(self) -> None:
        """Re-fetch detect.fps periodically in case Frigate's config is changed live."""
        while True:
            await asyncio.sleep(DETECT_FPS_REFRESH_SECONDS)
            await self.load_frigate_detect_fps()

    async def handle_motion_event(self, message: Message) -> None:
        """Handle raw motion events from Frigate (if needed)"""
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                f"Unexpectedly received non-bytes payload for motion event: {message.payload}"
            )
            return
        msg = message.payload.decode()
        if msg == "ON":
            self.logger.debug("Frigate motion event: ON")
            await self.trigger_analytics_start()
        elif msg == "OFF":
            self.logger.debug("Frigate motion event: OFF")
            await self.trigger_analytics_stop()

    async def monitor_event_timeouts(self) -> None:
        """Monitor active events and end those that haven't received updates in 600 seconds"""
        while True:
            await asyncio.sleep(30)
            current_time = time.time()
            expired_frigate_events = []

            for unifi_event_id, event_data in list(self._active_smart_events.items()):
                if event_data.get("end_time") is not None:
                    continue

                last_update = self.event_last_update.get(unifi_event_id, event_data["start_time"])
                time_since_update = current_time - last_update

                if time_since_update > self.event_timeout_seconds:
                    frigate_event_id = None
                    for fid, uid in self.frigate_to_unifi_event_map.items():
                        if uid == unifi_event_id:
                            frigate_event_id = fid
                            break

                    expired_frigate_events.append((frigate_event_id, unifi_event_id))
                    self.logger.warning(
                        f"EVENT TIMEOUT: Event {unifi_event_id} (Frigate: {frigate_event_id}, "
                        f"{event_data['object_type'].value}) has not been updated for "
                        f"{time_since_update:.1f}s (timeout: {self.event_timeout_seconds}s). Force ending event."
                    )

            for frigate_event_id, unifi_event_id in expired_frigate_events:
                event_data = self._active_smart_events.get(unifi_event_id)
                if not event_data:
                    continue

                try:
                    await self.trigger_smart_detect_stop(event_data["object_type"], event_id=unifi_event_id)
                except Exception as e:
                    self.logger.exception(f"Error ending timed out event {unifi_event_id}: {e}")
                finally:
                    if frigate_event_id and frigate_event_id in self.frigate_to_unifi_event_map:
                        del self.frigate_to_unifi_event_map[frigate_event_id]
                    if frigate_event_id and frigate_event_id in self.event_snapshot_ready:
                        del self.event_snapshot_ready[frigate_event_id]
                    if unifi_event_id in self.event_last_update:
                        del self.event_last_update[unifi_event_id]
                    # Also drop this track from the shared zone status tracker
                    # so it stops contributing occupancy to any zone (Section 8.5),
                    # and free its trackerID allocation.
                    tracker_id = self.unifi_tracker_id_by_frigate_event.pop(frigate_event_id, None)
                    if tracker_id is not None:
                        self.zone_status_tracker.remove_track(tracker_id)
                    if frigate_event_id:
                        self._release_tracker_id(frigate_event_id)

    async def handle_detection_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                f"Unexpectedly received non-bytes payload for detection event: {message.payload}"
            )
            return

        msg = message.payload.decode()
        try:
            frigate_msg = json.loads(message.payload.decode())
            event_type = frigate_msg.get("type")
            event_id = frigate_msg.get("after", {}).get("id")
            camera = frigate_msg.get("after", {}).get("camera")
            label = frigate_msg.get("after", {}).get("label")

            if camera != self.args.frigate_camera:
                return

            self.logger.debug(f"Frigate: Received: {frigate_msg} ")

            object_type = self.label_to_object_type(label)
            if not object_type:
                self.logger.warning(
                    f"MISSED EVENT: Received unsupported detection label type: {label} "
                    f"(event_id={event_id}, type={event_type})"
                )
                return

            self.logger.debug(
                f"Frigate event: type={event_type}, id={event_id}, label={label}, "
                f"active_frigate_events={list(self.frigate_to_unifi_event_map.keys())}"
            )

            if event_type == "new":
                if event_id in self.frigate_to_unifi_event_map:
                    self.logger.warning(
                        f"Received 'new' event for already active Frigate event_id={event_id}. "
                        f"This may indicate event was not properly ended. Stopping old event first."
                    )
                    old_unifi_id = self.frigate_to_unifi_event_map[event_id]
                    if old_unifi_id in self._active_smart_events:
                        old_event = self._active_smart_events[old_unifi_id]
                        await self.trigger_smart_detect_stop(old_event["object_type"], event_id=old_unifi_id)
                    del self.frigate_to_unifi_event_map[event_id]

                self.event_snapshot_ready[event_id] = asyncio.Event()

                tracker_id = self._allocate_tracker_id(event_id)
                custom_descriptor = self.build_descriptor_from_frigate_msg(
                    frigate_msg, object_type, tracker_id
                )
                frame_time_ms = int(frigate_msg.get('after', {}).get('frame_time', 0) * 1000) - self.args.frigate_time_sync_ms

                # Register this track with the shared per-camera zone tracker,
                # keyed by trackerID (matching how the real protocol scopes
                # zone occupancy per tracked object rather than per event),
                # and capture the resulting full zonesStatus dict to send
                # alongside this message.
                self.unifi_tracker_id_by_frigate_event[event_id] = tracker_id
                zones_status = self._update_zone_status_for_track(
                    tracker_id,
                    custom_descriptor["zones"],
                    custom_descriptor["confidenceLevel"],
                    active=True,
                )

                unifi_event_id = await self.trigger_smart_detect_start(
                    object_type, custom_descriptor, frame_time_ms, zonesStatus=zones_status
                )

                self.frigate_to_unifi_event_map[event_id] = unifi_event_id
                self.event_last_update[unifi_event_id] = time.time()

                self.logger.info(
                    f"Frigate: Starting {label} smart event within motion context (Frigate: {event_id}, UniFi: {unifi_event_id}). "
                    f"Total active events: {len(self.frigate_to_unifi_event_map)}"
                )

            elif event_type == "update":
                if event_id in self.frigate_to_unifi_event_map:
                    unifi_event_id = self.frigate_to_unifi_event_map[event_id]

                    if unifi_event_id not in self._active_smart_events:
                        self.logger.warning(
                            f"Frigate event {event_id} maps to UniFi event {unifi_event_id} "
                            f"but that event is not active. Skipping update."
                            f"active _active_smart_events: {list(self._active_smart_events.keys())}"
                        )
                        return

                    tracker_id = self._get_tracker_id(event_id)
                    custom_descriptor = self.build_descriptor_from_frigate_msg(
                        frigate_msg, object_type, tracker_id
                    )

                    frame_time_ms = int(frigate_msg.get('after', {}).get('frame_time', 0) * 1000) - self.args.frigate_time_sync_ms

                    # Keep this track's contribution to the shared zone
                    # tracker current on every update, independent of
                    # whether this specific event is the one closing out.
                    zones_status = self._update_zone_status_for_track(
                        tracker_id,
                        custom_descriptor["zones"],
                        custom_descriptor["confidenceLevel"],
                        active=True,
                    )

                    # Confirmed: trigger_smart_detect_update accepts zonesStatus.
                    # event_id=unifi_event_id disambiguates against other
                    # concurrently active tracks of the same object_type.
                    await self.trigger_smart_detect_update(
                        object_type, custom_descriptor, frame_time_ms,
                        zonesStatus=zones_status, event_id=unifi_event_id,
                    )

                    self.event_last_update[unifi_event_id] = time.time()

                    after_data = frigate_msg.get('after', {})
                    has_snapshot = after_data.get('has_snapshot', False)
                    if has_snapshot and self.args.frigate_http_url:
                        self.logger.debug(f"Event {event_id} has updated snapshot, fetching and caching all types...")
                        try:
                            snapshot_crop, snapshot_fov, heatmap = await self.fetch_snapshots_for_event(
                                unifi_event_id, "smart_detect"
                            )

                            event_data = self._active_smart_events[unifi_event_id]
                            if snapshot_crop:
                                event_data["snapshot_crop_path"] = snapshot_crop
                            if snapshot_fov:
                                event_data["snapshot_fov_path"] = snapshot_fov
                            if heatmap:
                                event_data["heatmap_path"] = heatmap

                            self.logger.debug(
                                f"Cached snapshots for smart event {unifi_event_id}: "
                                f"crop={snapshot_crop}, fov={snapshot_fov}, heatmap={heatmap}"
                            )
                        except Exception as e:
                            self.logger.error(f"Error fetching/caching snapshots for event {unifi_event_id}: {e}")

                    event_data = self._active_smart_events[unifi_event_id]
                    event_age = time.time() - event_data["start_time"]
                    self.logger.debug(
                        f"Sent moving update for smart event (Frigate: {event_id}, UniFi: {unifi_event_id}). "
                        f"Age: {event_age:.1f}s"
                    )
                else:
                    self.logger.warning(
                        f"MISSED EVENT: Received 'update' for unknown Frigate event_id={event_id} "
                        f"(label={label}). Likely missed 'new' event."
                    )

            elif event_type == "end":
                if event_id in self.frigate_to_unifi_event_map:
                    unifi_event_id = self.frigate_to_unifi_event_map[event_id]

                    if unifi_event_id not in self._active_smart_events:
                        self.logger.warning(
                            f"Frigate event {event_id} maps to UniFi event {unifi_event_id} "
                            f"but that event is not active. Cleaning up mapping."
                        )
                        del self.frigate_to_unifi_event_map[event_id]
                        if event_id in self.event_snapshot_ready:
                            del self.event_snapshot_ready[event_id]
                        return

                    event_data = self._active_smart_events.get(unifi_event_id)
                    if not event_data:
                        self.logger.warning(
                            f"Event data missing for UniFi event {unifi_event_id}. Cleaning up mapping."
                        )
                        del self.frigate_to_unifi_event_map[event_id]
                        if event_id in self.event_snapshot_ready:
                            del self.event_snapshot_ready[event_id]
                        return

                    tracker_id = self._get_tracker_id(event_id)
                    final_descriptor = self.build_descriptor_from_frigate_msg(
                        frigate_msg, object_type, tracker_id
                    )
                    end_time_ms = int(frigate_msg.get('after', {}).get('end_time', 0) * 1000) - self.args.frigate_time_sync_ms
                    frame_time_ms = int(frigate_msg.get('after', {}).get('frame_time', 0) * 1000) - self.args.frigate_time_sync_ms

                    event_duration = time.time() - event_data["start_time"]
                    self.logger.info(
                        f"Frigate: Ending {label} smart event within motion context (Frigate: {event_id}, UniFi: {unifi_event_id}). "
                        f"Duration: {event_duration:.1f}s"
                    )
                    self.logger.debug(
                        f"Event timestamps: end_time={end_time_ms}, frame_time={frame_time_ms}"
                    )

                    # Send a final "moving" update carrying the real last-known
                    # position/confidence BEFORE closing the track. This
                    # mirrors real device behavior (the last real detection
                    # frame arrives as a live "moving" message immediately
                    # before the "leave" summary, which itself carries empty
                    # descriptors) and avoids handing the closing zonesStatus
                    # snapshot straight from a possibly-stale prior tick.
                    zones_status_final_update = self._update_zone_status_for_track(
                        tracker_id,
                        final_descriptor["zones"],
                        final_descriptor["confidenceLevel"],
                        active=True,
                    )
                    await self.trigger_smart_detect_update(
                        object_type, final_descriptor, frame_time_ms,
                        zonesStatus=zones_status_final_update, event_id=unifi_event_id,
                    )

                    # This track is closing -- remove its contribution from
                    # the shared zone tracker (other tracks' occupancy is
                    # unaffected, per protocol spec Section 7.4) and capture
                    # the resulting post-departure zonesStatus for the stop
                    # message.
                    zones_status = self._update_zone_status_for_track(tracker_id, [], 0, active=False)

                    # Confirmed: trigger_smart_detect_stop accepts zonesStatus.
                    await self.trigger_smart_detect_stop(
                        object_type,
                        final_descriptor,
                        end_time_ms,
                        event_id=unifi_event_id,
                        frame_time_ms=frame_time_ms,
                        zonesStatus=zones_status,
                    )

                    self.unifi_tracker_id_by_frigate_event.pop(event_id, None)
                    self._release_tracker_id(event_id)

                    del self.frigate_to_unifi_event_map[event_id]
                    if event_id in self.event_snapshot_ready:
                        del self.event_snapshot_ready[event_id]
                    if unifi_event_id in self.event_last_update:
                        del self.event_last_update[unifi_event_id]

                    self.logger.info(
                        f"Frigate: Event {event_id} ended. "
                        f"Remaining active events: {len(self.frigate_to_unifi_event_map)}"
                    )
                else:
                    self.logger.warning(
                        f"MISSED EVENT: Received 'end' for unknown Frigate event_id={event_id} "
                        f"(label={label}). Likely missed 'new' event."
                    )
            else:
                self.logger.debug(
                    f"Received unhandled event type: {event_type} for event_id={event_id}"
                )

        except json.JSONDecodeError:
            self.logger.exception(f"Could not decode payload: {msg}")
        except Exception as e:
            self.logger.exception(
                f"Unexpected error handling detection event: {e}, payload: {msg}"
            )

    async def handle_snapshot_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                f"Unexpectedly received non-bytes payload for snapshot event: {message.payload}"
            )
            return

        topic_parts = message.topic.value.split("/")
        if len(topic_parts) < 4:
            self.logger.debug(f"Unexpected snapshot topic format: {message.topic.value}")
            return

        snapshot_label = topic_parts[-2]

        self.logger.debug(
            f"Received snapshot: topic={message.topic.value}, "
            f"message={message}"
        )

        matching_frigate_event_id = None
        for frigate_event_id, unifi_event_id in self.frigate_to_unifi_event_map.items():
            if unifi_event_id in self._active_smart_events:
                event_data = self._active_smart_events[unifi_event_id]
                event_label = event_data["object_type"].value
                if event_label == snapshot_label and not message.retain:
                    matching_frigate_event_id = frigate_event_id
                    break

        if matching_frigate_event_id:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(message.payload)
            f.close()
            self.logger.debug(
                f"Updating snapshot for Frigate event {matching_frigate_event_id} ({snapshot_label}) with {f.name}"
            )
            self.update_motion_snapshot(Path(f.name))
            if matching_frigate_event_id in self.event_snapshot_ready:
                self.event_snapshot_ready[matching_frigate_event_id].set()
        else:
            self.logger.debug(
                f"Discarding snapshot for label={snapshot_label} "
                f"(size={len(message.payload)}, retained={message.retain}). "
                f"No matching active event."
            )
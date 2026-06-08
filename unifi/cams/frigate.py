import argparse
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp
import backoff
from aiomqtt import Client, Message
from aiomqtt.exceptions import MqttError

from unifi.cams.base import SmartDetectObjectType
from unifi.cams.rtsp import RTSPCam


class FrigateCam(RTSPCam):
    """UniFi Protect camera emulator backed by a Frigate NVR instance.

    Bridges Frigate MQTT events → UniFi Protect smart-detect / analytics events,
    including zone sync, snapshot caching, and event lifecycle management.
    """

    # ---------------------------------------------------------------------------
    # Class-level constants and label maps
    # ---------------------------------------------------------------------------

    UNIFI_COORD_SCALE: int = 1000  # UniFi uses a 0–1000 coordinate space

    # Canonical Frigate labels for each UniFi object-type bucket.
    _VEHICLE_LABELS: frozenset[str] = frozenset(
        {"vehicle", "car", "motorcycle", "truck", "bus", "school_bus", "bicycle", "boat"}
    )
    _ANIMAL_LABELS: frozenset[str] = frozenset(
        {
            "cat", "dog", "deer", "horse", "bird", "raccoon", "fox",
            "bear", "cow", "squirrel", "goat", "rabbit", "skunk", "kangaroo",
        }
    )

    # Flat dispatch map built once at class definition time.
    _LABEL_TO_TYPE: dict[str, SmartDetectObjectType] = {
        **{"person": SmartDetectObjectType.PERSON},
        **{"face": SmartDetectObjectType.FACE},
        **{"license_plate": SmartDetectObjectType.LICENSEPLATE},
        **{"package": SmartDetectObjectType.PACKAGE},
        **{lbl: SmartDetectObjectType.VEHICLE for lbl in _VEHICLE_LABELS},
        **{lbl: SmartDetectObjectType.ANIMAL for lbl in _ANIMAL_LABELS},
    }

    # Frigate labels to expand when a UniFi "vehicle" zone object-type is received.
    _VEHICLE_ZONE_LABELS: tuple[str, ...] = (
        "car", "motorcycle", "truck", "bus",
    )
    _ANIMAL_ZONE_LABELS: tuple[str, ...] = (
        "dog", "cat", "deer", "horse", "bird", "raccoon", "fox",
        "bear", "cow", "squirrel", "goat", "rabbit", "skunk", "kangaroo",
    )

    EVENT_TIMEOUT_SECONDS: int = 600   # Force-end events silent for this long
    TIMEOUT_CHECK_INTERVAL: int = 30   # How often the monitor loop runs

    # ---------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------

    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        super().__init__(args, logger)
        self.args = args

        # Frigate event ID → UniFi event ID
        self._frigate_to_unifi: dict[str, int] = {}
        # UniFi event ID → Frigate event ID  (reverse map — O(1) lookups)
        self._unifi_to_frigate: dict[int, str] = {}

        # wall-clock time of last update, keyed by UniFi event ID
        self._event_last_update: dict[int, float] = {}

        # Zone state
        self._unifi_zones: dict[str, Any] = {}
        # Frigate zone name → UniFi zone ID (int)
        self._frigate_zone_to_unifi_id: dict[str, int] = {}

        # Shared aiohttp session (lazy-initialised, reused across all HTTP calls)
        self._http_session: Optional[aiohttp.ClientSession] = None

    # ---------------------------------------------------------------------------
    # CLI argument registration
    # ---------------------------------------------------------------------------

    @classmethod
    def add_parser(cls, parser: argparse.ArgumentParser) -> None:
        super().add_parser(parser)
        parser.add_argument("--mqtt-host", required=True, help="MQTT broker hostname")
        parser.add_argument("--mqtt-port", default=1883, type=int, help="MQTT broker port")
        parser.add_argument("--mqtt-username", required=False)
        parser.add_argument("--mqtt-password", required=False)
        parser.add_argument(
            "--mqtt-prefix", default="frigate", type=str, help="MQTT topic prefix"
        )
        parser.add_argument(
            "--frigate-camera", required=True, type=str,
            help="Name of the camera in Frigate",
        )
        parser.add_argument(
            "--camera-width", default=1920, type=int,
            help="Camera frame width in pixels (default: 1920)",
        )
        parser.add_argument(
            "--camera-height", default=1080, type=int,
            help="Camera frame height in pixels (default: 1080)",
        )
        parser.add_argument(
            "--frigate-detect-width", default=1280, type=int,
            help="Frigate detection frame width in pixels (default: 1280)",
        )
        parser.add_argument(
            "--frigate-detect-height", default=720, type=int,
            help="Frigate detection frame height in pixels (default: 720)",
        )
        parser.add_argument(
            "--frigate-http-url", required=False, type=str,
            help=(
                "Frigate HTTP API base URL (e.g. http://frigate:5000). "
                "Required for snapshot fetching and zone sync."
            ),
        )
        parser.add_argument(
            "--frigate-time-sync-ms", default=0, type=int,
            help=(
                "Timestamp offset in milliseconds applied to Frigate event times. "
                "Positive values shift timestamps backward to compensate for detection lag."
            ),
        )

    # ---------------------------------------------------------------------------
    # Feature flags
    # ---------------------------------------------------------------------------

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            **await super().get_feature_flags(),
            "mic": True,
            "smartDetect": [
                "person", "vehicle", "animal", "package", "face", "licensePlate",
            ],
        }

    # ---------------------------------------------------------------------------
    # Zone / settings handlers
    # ---------------------------------------------------------------------------

    async def process_smart_detect_settings(self, msg: dict[str, Any]) -> None:
        """Handle ChangeSmartDetectSettings from UniFi Protect.

        Persists zone definitions and pushes them to Frigate.  Always delegates
        to super() so that exclusion zones and change callbacks remain functional.
        """
        zones: dict[str, Any] = msg.get("payload", {}).get("zones", {})

        if zones:
            self._unifi_zones = zones
            self.logger.info(
                "ChangeSmartDetectSettings: received %d zone(s) from UniFi Protect: %s",
                len(zones), list(zones.keys()),
            )
            if self.args.frigate_http_url:
                await self._push_zones_to_frigate(zones)
            else:
                self.logger.warning(
                    "ChangeSmartDetectSettings: --frigate-http-url not set; "
                    "cannot sync zones to Frigate. Objects will report in zone 0 only."
                )
        else:
            self.logger.debug("ChangeSmartDetectSettings: no zones in payload — skipping zone sync")

        await super().process_smart_detect_settings(msg)

    async def process_smart_motion_settings(self, msg: dict[str, Any]) -> Any:
        """Handle ChangeSmartMotionSettings.

        Motion zones are managed directly in Frigate; we only call super() to
        keep motionEvents / lingerEventStart in sync with UniFi Protect.
        """
        return await super().process_smart_motion_settings(msg)

    # ---------------------------------------------------------------------------
    # Zone sync helpers
    # ---------------------------------------------------------------------------

    async def _push_zones_to_frigate(self, unifi_zones: dict[str, Any]) -> None:
        """Convert UniFi Protect zone definitions and push them to Frigate.

        UniFi is the sole zone source-of-truth. Frigate's config/set deep-merge
        ensures only the zones section is touched.

        Coordinate conversion:
          UniFi: flat int list [x0,y0,x1,y1,…] in 0–1000 space
          Frigate: comma-separated floats in 0.0–1.0 space
        """
        camera_tracked = await self._fetch_camera_tracked_objects()

        frigate_zones: dict[str, Any] = {}
        new_mapping: dict[str, int] = {}

        for zone_id_str, zone_data in unifi_zones.items():
            try:
                zone_id = int(zone_id_str)
            except ValueError:
                self.logger.warning(
                    "Zone sync: non-integer zone id '%s' — skipping", zone_id_str
                )
                continue

            coord = zone_data.get("coord", [])
            if len(coord) < 4 or len(coord) % 2 != 0:
                self.logger.warning(
                    "Zone sync: zone %s has invalid coord %s — skipping", zone_id_str, coord
                )
                continue

            # Convert 0-1000 → 0.0-1.0.
            # Use :.4g (not :.4f) so boundary value 1.0 stays as "1" not "1.0000";
            # Frigate uses lexicographic comparison `p > "1.0"` to detect pixel mode.
            frigate_coords = ",".join(f"{v / 1000.0:.4g}" for v in coord)

            zone_name = f"unifi_zone_{zone_id}"
            frigate_objects = self._unifi_object_types_to_frigate_labels(
                zone_data.get("objectTypes", []), camera_tracked
            )

            frigate_zones[zone_name] = {
                "coordinates": frigate_coords,
                "objects": frigate_objects,
                "inertia": 3,
                "loitering_time": 0,
            }
            new_mapping[zone_name] = zone_id
            self.logger.debug(
                "Zone sync: unifi zone %d → '%s' coords=%s objects=%s",
                zone_id, zone_name, frigate_coords, frigate_objects,
            )

        await self._save_zones_via_config_set(frigate_zones, new_mapping)

    async def _fetch_camera_tracked_objects(self) -> set[str]:
        """Return the set of labels this camera tracks in Frigate's resolved config."""
        if not self.args.frigate_http_url:
            return set()
        try:
            session = await self._get_session()
            async with session.get(f"{self.args.frigate_http_url}/api/config") as resp:
                if resp.status == 200:
                    cfg = await resp.json()
                    tracked = (
                        cfg.get("cameras", {})
                        .get(self.args.frigate_camera, {})
                        .get("objects", {})
                        .get("track", [])
                    )
                    result = set(tracked)
                    self.logger.debug(
                        "Zone sync: camera '%s' tracks: %s",
                        self.args.frigate_camera, sorted(result),
                    )
                    return result
                self.logger.warning(
                    "Zone sync: could not fetch Frigate config (HTTP %d)", resp.status
                )
        except Exception:
            self.logger.warning("Zone sync: failed to fetch Frigate config", exc_info=True)
        return set()

    def _unifi_object_types_to_frigate_labels(
        self,
        object_types: list[str],
        camera_tracked: set[str],
    ) -> list[str]:
        """Expand UniFi object-type names to Frigate label strings.

        Filters out labels the camera doesn't track (Frigate rejects unknown labels).
        """
        labels: list[str] = []
        for ot in object_types:
            if ot == "person":
                labels.append("person")
            elif ot == "vehicle":
                labels.extend(self._VEHICLE_ZONE_LABELS)
            elif ot == "animal":
                labels.extend(self._ANIMAL_ZONE_LABELS)
            elif ot == "package":
                labels.append("package")
            elif ot == "face":
                labels.append("face")
            elif ot == "licensePlate":
                labels.append("license_plate")

        # Deduplicate, preserving insertion order; filter to tracked labels.
        seen: dict[str, None] = {}
        for lbl in labels:
            if lbl not in seen and (not camera_tracked or lbl in camera_tracked):
                seen[lbl] = None
        return list(seen)

    async def _save_zones_via_config_set(
        self,
        frigate_zones: dict[str, Any],
        new_mapping: dict[str, int],
    ) -> None:
        """Push zones to Frigate via PUT /api/config/set (deep-merge)."""
        if not frigate_zones:
            self.logger.warning("Zone sync: no valid zones to push to Frigate")
            return

        payload = {
            "requires_restart": 1,
            "config_data": {
                "cameras": {
                    self.args.frigate_camera: {"zones": frigate_zones}
                }
            },
        }
        try:
            session = await self._get_session()
            async with session.put(
                f"{self.args.frigate_http_url}/api/config/set", json=payload
            ) as resp:
                if resp.status == 200:
                    self._frigate_zone_to_unifi_id = new_mapping
                    self.logger.info(
                        "Zone sync: pushed %d zone(s) to Frigate camera '%s': %s",
                        len(new_mapping), self.args.frigate_camera, list(new_mapping.keys()),
                    )
                else:
                    body = await resp.text()
                    self.logger.error(
                        "Zone sync: Frigate config API returned %d: %s", resp.status, body
                    )
        except Exception:
            self.logger.exception("Zone sync: failed to push zones to Frigate")

    # ---------------------------------------------------------------------------
    # Label / type helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def label_to_object_type(cls, label: str) -> Optional[SmartDetectObjectType]:
        """O(1) dispatch from a Frigate label string to a UniFi SmartDetectObjectType."""
        return cls._LABEL_TO_TYPE.get(label)

    # ---------------------------------------------------------------------------
    # Descriptor builder
    # ---------------------------------------------------------------------------

    def build_descriptor_from_frigate_msg(
        self,
        frigate_msg: dict[str, Any],
        object_type: SmartDetectObjectType,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Build a UniFi Protect-compatible smart-detect descriptor and zonesStatus from a Frigate event.
        
        Returns:
            Tuple of (descriptor, zonesStatus) where zonesStatus maps zone IDs to status dicts.
        """
        before = frigate_msg.get("before", {})
        after = frigate_msg.get("after", {})
        msg_type = frigate_msg.get("type")

        # --- Bounding box --------------------------------------------------
        box = after.get("box")
        if box and len(box) == 4:
            scale = self.UNIFI_COORD_SCALE
            x_scale = scale / self.args.frigate_detect_width
            y_scale = scale / self.args.frigate_detect_height

            x = int(box[0] * x_scale)
            y = int(box[1] * y_scale)
            w = int(box[2] * x_scale) - x
            h = int(box[3] * y_scale) - y
            coord = [x, y, w, h]
        else:
            coord = [0, 0, 1920, 1080]

        # --- Confidence ----------------------------------------------------
        score_key = "top_score" if msg_type == "end" else "score"
        confidence_level = int(after.get(score_key, 0.95) * 100)

        # --- Tracker ID ----------------------------------------------------
        raw_id = after.get("id", 1)
        tracker_id = hash(raw_id) % 1_000_000 if isinstance(raw_id, str) else raw_id

        # --- Zones ---------------------------------------------------------
        current_zones: list[str] = after.get("current_zones", [])
        entered_zones: list[str] = after.get("entered_zones", [])
        before_current_zones: list[str] = before.get("current_zones", [])

        # zones is an array of UniFi zone IDs (ints) that the object is currently in; 0 is always included, and enter's on the start type and leaves on the end type.
        # zones_status contains status for each zone in zones, valid values per zone include "enter", "leave", and "moving".
        # before.current_zones and before.entered_zones shows the status from the previous update, which we can use to determine if the object just entered a zone (enter) or is still in it (moving).
        # When a zone first appears in after.current_zones and not listed in before.current_zones,  we report "enter" status for that zone. 
        # When a zone appears in before.current_zones, after.current_zones we reporting "moving" status for that zone.
        # When a zone is in before.current_zones and is not listed in after.current_zones, we report "leave" status for that zone (but only on update and end events, not on new events since the object technically was never in the zone).

        zones = [0]  # Zone 0 is always included
        
        if current_zones and self._frigate_zone_to_unifi_id:
            mapped_zones = [
                self._frigate_zone_to_unifi_id[z]
                for z in current_zones
                if z in self._frigate_zone_to_unifi_id
            ]
            if mapped_zones:
                zones.extend(mapped_zones)
                self.logger.debug(
                    "Object in Frigate zone(s) %s mapped to UniFi zones: %s",
                    current_zones, mapped_zones,
                )
            else:
                self.logger.debug(
                    "Object in Frigate zone(s) %s but none map to UniFi zones. "
                    "Available mapping: %s",
                    current_zones, self._frigate_zone_to_unifi_id,
                )
        elif current_zones and (self._frigate_zone_to_unifi_id is not None and not self._frigate_zone_to_unifi_id):
            self.logger.debug(
                "Object in Frigate zone(s) %s but no zone mapping available. "
                "Zones may not have been pushed yet (--frigate-http-url configured?).",
                current_zones,
            )

        self.logger.debug("msg_type=%s before_current_zones=%s entered_zones=%s current_zones=%s",
            msg_type, before_current_zones, entered_zones, current_zones,
        )
        self.logger.debug("UnfiZones: %s", self._unifi_zones)
        # For update and end events, also include zones that were in before but not in after (to report "leave" status)
        if msg_type in ("update", "end"):
            self.logger.debug(
                "Processing left zones for %s event: before_current_zones=%s, current_zones=%s",
                msg_type, before_current_zones, current_zones,
            )
            for frigate_zone in before_current_zones:
                if frigate_zone not in current_zones:
                    # Get zone ID from mapping
                    if frigate_zone in self._frigate_zone_to_unifi_id:
                        left_zone_id = self._frigate_zone_to_unifi_id[frigate_zone]
                        if left_zone_id not in zones:
                            zones.append(left_zone_id)
                            self.logger.debug(
                                "Object left Frigate zone '%s' (UniFi zone %d)",
                                frigate_zone, left_zone_id,
                            )
                    else:
                        self.logger.debug(
                            "Frigate zone '%s' left but not in mapping. Available: %s",
                            frigate_zone, self._frigate_zone_to_unifi_id,
                        )

        # Build zonesStatus for each zone
        zones_status: dict[str, dict[str, Any]] = {}
        for zone_id in zones:
            # Determine status based on event type and zone membership
            if msg_type == "new":
                # All zones show enter status on new event (object never was in any zone before)
                status = "enter"
            elif msg_type == "end":
                # All zones show leave status on end event
                status = "leave"
            elif zone_id == 0:
                # Zone 0 (default) shows moving status during update events
                status = "moving"
            else:
                # For named zones on update events: determine if just entered, left, or moving
                # by comparing before.current_zones and after.current_zones
                frigate_zone_name = None
                for fname, uid in self._frigate_zone_to_unifi_id.items():
                    if uid == zone_id:
                        frigate_zone_name = fname
                        break
                
                if frigate_zone_name is None:
                    status = "moving"
                elif frigate_zone_name not in before_current_zones and frigate_zone_name in current_zones:
                    # Zone first appears in after.current_zones (was not in before.current_zones)
                    status = "enter"
                elif frigate_zone_name in before_current_zones and frigate_zone_name not in current_zones:
                    # Zone disappears from after.current_zones (was in before.current_zones but not now)
                    # This only applies to update events since we only add left zones for update/end,
                    # and "new" events are already handled above
                    status = "leave"
                else:
                    # Zone is in both before.current_zones and after.current_zones
                    status = "moving"
            
            # Add zone status: zone_id (as string key) → {"level": confidence_level, "status": status}
            zones_status[str(zone_id)] = {"level": confidence_level, "status": status}

        if msg_type == "end":
            self.logger.debug("End event: setting all zones to leave status")
            zones_status["2"] = {"level": confidence_level, "status": "leave"}

        # --- Speed ---------------------------------------------------------
        avg_speed = after.get("average_estimated_speed", 0)
        speed: Optional[float] = float(avg_speed) if avg_speed > 0 else None

        # --- License plate -------------------------------------------------
        plate_data = after.get("recognized_license_plate")
        license_plate: Optional[str] = None
        plate_score: Optional[float] = None
        if isinstance(plate_data, list) and plate_data:
            license_plate = plate_data[0]
            plate_score = plate_data[1] if len(plate_data) >= 2 else None

        # --- Display name --------------------------------------------------
        if object_type == SmartDetectObjectType.VEHICLE and license_plate:
            name = f"{license_plate} ({plate_score:.1%})" if plate_score is not None else license_plate
        else:
            name = ""

        descriptor: dict[str, Any] = {
            "attributes": None,
            "boxColor": "red",
            "confidenceLevel": confidence_level,
            "coord": coord,
            "coord3d": [-1, -1],
            "depth": None,
            "firstShownTimeMs": int(after.get("start_time", 0) * 1000) - self.args.frigate_time_sync_ms,
            "idleSinceTimeMs": int(after.get("motionless_count", 0) * 1000),
            "lines": [],
            "loiterZones": [],
            "name": name,
            "objectType": object_type.value,
            "secondLensZones": [],
            "speed": speed,
            "stationary": after.get("stationary", False),
            "tag": "Tagged by Frigate",
            "trackerID": tracker_id,
            "zones": zones,
        }
        if license_plate:
            descriptor["licensePlate"] = license_plate

        self.logger.debug(
            "Built descriptor: trackerID=%s confidence=%d coord=%s zones=%s zonesStatus=%s "
            "stationary=%s speed=%s licensePlate=%s",
            tracker_id, confidence_level, coord, zones, zones_status,
            descriptor["stationary"], speed, license_plate,
        )
        return descriptor, zones_status

    # ---------------------------------------------------------------------------
    # Snapshot fetching
    # ---------------------------------------------------------------------------

    async def fetch_snapshots_for_event(
        self,
        event_id: int,
        event_type: str = "analytics",
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Fetch crop, full-frame (fov), and heatmap snapshots for a UniFi event.

        Returns:
            (crop_path, fov_path, heatmap_path) — any may be None on failure.
        """
        if not self.args.frigate_http_url:
            self.logger.warning("Cannot fetch snapshots: --frigate-http-url not configured")
            return (None, None, None)

        frigate_event_id = self._unifi_to_frigate.get(event_id) if event_type == "smart_detect" else None

        if frigate_event_id:
            base = f"{self.args.frigate_http_url}/api/events/{frigate_event_id}/snapshot.jpg"
            full_url = base
            thumbnail_url = f"{base}?crop=1&quality=80"
            self.logger.debug(
                "Using event-specific snapshot URLs for event %d (Frigate: %s)",
                event_id, frigate_event_id,
            )
        else:
            base = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
            timestamp: Optional[int] = None
            if event_type == "analytics" and event_id in self._analytics_event_history:
                start = self._analytics_event_history[event_id].get("start_time")
                if start is not None:
                    timestamp = int(start)
            ts_suffix = f"?timestamp={timestamp}" if timestamp is not None else ""
            full_url = f"{base}{ts_suffix}"
            thumbnail_url = f"{base}?height=360&quality=80" + (f"&timestamp={timestamp}" if timestamp else "")

        self.logger.debug(
            "Fetching snapshots for event %d: full=%s thumbnail=%s",
            event_id, full_url, thumbnail_url,
        )

        fov, crop = await asyncio.gather(
            self._fetch_image(full_url, "full", event_id),
            self._fetch_image(thumbnail_url, "thumbnail", event_id),
        )
        heatmap = fov  # reuse full-frame as heatmap

        self.logger.info(
            "Snapshots for event %d: crop=%s fov=%s heatmap=%s",
            event_id,
            "✓" if crop else "✗",
            "✓" if fov else "✗",
            "✓" if heatmap else "✗",
        )
        return (crop, fov, heatmap)

    async def _fetch_image(
        self, url: str, label: str, event_id: int
    ) -> Optional[Path]:
        """Download a single image URL to a temp file. Returns the path or None."""
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    f = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    f.write(data)
                    f.close()
                    self.logger.debug(
                        "Fetched %s snapshot for event %d (%d bytes) → %s",
                        label, event_id, len(data), f.name,
                    )
                    return Path(f.name)
                body = await resp.text()
                self.logger.warning(
                    "Failed to fetch %s snapshot for event %d: HTTP %d — %s",
                    label, event_id, resp.status, body,
                )
        except asyncio.TimeoutError:
            self.logger.warning("Timeout fetching %s snapshot for event %d", label, event_id)
        except Exception:
            self.logger.exception("Error fetching %s snapshot for event %d", label, event_id)
        return None

    # ---------------------------------------------------------------------------
    # HTTP session management
    # ---------------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared aiohttp session, creating it if necessary."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _close_session(self) -> None:
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    # ---------------------------------------------------------------------------
    # Event-map helpers (keep both dicts in sync)
    # ---------------------------------------------------------------------------

    def _register_event(self, frigate_id: str, unifi_id: int) -> None:
        self._frigate_to_unifi[frigate_id] = unifi_id
        self._unifi_to_frigate[unifi_id] = frigate_id
        self._event_last_update[unifi_id] = time.time()

    def _deregister_event(self, frigate_id: str, unifi_id: int) -> None:
        self._frigate_to_unifi.pop(frigate_id, None)
        self._unifi_to_frigate.pop(unifi_id, None)
        self._event_last_update.pop(unifi_id, None)

    # ---------------------------------------------------------------------------
    # MQTT run loop
    # ---------------------------------------------------------------------------

    async def run(self) -> None:
        has_connected = False

        @backoff.on_predicate(backoff.expo, max_value=60, logger=self.logger)
        async def mqtt_connect() -> None:
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
                        "Connected to %s:%d", self.args.mqtt_host, self.args.mqtt_port
                    )
                    await client.subscribe(f"{self.args.mqtt_prefix}/#")
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self.monitor_event_timeouts())
                        async for message in client.messages:
                            topic = message.topic
                            prefix = self.args.mqtt_prefix
                            camera = self.args.frigate_camera
                            if topic.matches(f"{prefix}/events"):
                                tg.create_task(self.handle_detection_event(message))
                            elif topic.matches(f"{prefix}/{camera}/+/snapshot"):
                                tg.create_task(self.handle_snapshot_event(message))
                            elif topic.matches(f"{prefix}/{camera}/motion"):
                                tg.create_task(self.handle_motion_event(message))
                            elif topic.matches(f"{prefix}/reviews"):
                                tg.create_task(self.handle_review_event(message))
                            elif topic.matches(f"{prefix}/tracked_object_update"):
                                tg.create_task(self.handle_object_update(message))
            except MqttError:
                if not has_connected:
                    raise
            finally:
                await self._close_session()

        await mqtt_connect()

    # ---------------------------------------------------------------------------
    # MQTT message handlers
    # ---------------------------------------------------------------------------

    async def handle_review_event(self, message: Message) -> None:
        self.logger.debug("Frigate Message: review event: %s", message.payload) 

    async def handle_object_update(self, message: Message) -> None:
        self.logger.debug("Frigate Message: tracked object update: %s", message.payload)

    async def handle_motion_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                "Non-bytes payload for motion event: %r", message.payload
            )
            return
        payload = message.payload.decode()
        self.logger.debug("Frigate Message: motion: %s", payload)
        if payload == "ON":
            await self.trigger_analytics_start()
        elif payload == "OFF":
            await self.trigger_analytics_stop()

    async def monitor_event_timeouts(self) -> None:
        """Periodically end smart-detect events that have gone silent."""
        while True:
            await asyncio.sleep(self.TIMEOUT_CHECK_INTERVAL)
            now = time.time()
            expired: list[tuple[Optional[str], int]] = []

            for unifi_id, event_data in list(self._active_smart_events.items()):
                if event_data.get("end_time") is not None:
                    continue
                last = self._event_last_update.get(unifi_id, event_data["start_time"])
                if now - last > self.EVENT_TIMEOUT_SECONDS:
                    frigate_id = self._unifi_to_frigate.get(unifi_id)
                    expired.append((frigate_id, unifi_id))
                    self.logger.warning(
                        "EVENT TIMEOUT: UniFi event %d (Frigate: %s, type=%s) "
                        "silent for %.1fs — force-ending.",
                        unifi_id, frigate_id,
                        event_data["object_type"].value, now - last,
                    )

            for frigate_id, unifi_id in expired:
                event_data = self._active_smart_events.get(unifi_id)
                if not event_data:
                    continue
                try:
                    await self.trigger_smart_detect_stop(
                        event_data["object_type"], event_id=unifi_id
                    )
                except Exception:
                    self.logger.exception("Error force-ending timed-out event %d", unifi_id)
                finally:
                    if frigate_id:
                        self._deregister_event(frigate_id, unifi_id)
                    else:
                        self._event_last_update.pop(unifi_id, None)

    async def handle_detection_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning("Non-bytes payload for detection event: %r", message.payload)
            return

        raw = message.payload.decode()
        try:
            frigate_msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.exception("Could not decode detection event payload: %s", raw)
            return

        try:
            await self._dispatch_detection_event(frigate_msg)
        except Exception:
            self.logger.exception(
                "Unexpected error handling detection event; payload: %s", raw
            )

    async def _dispatch_detection_event(self, frigate_msg: dict[str, Any]) -> None:
        """Core logic for routing new / update / end Frigate events."""
        after = frigate_msg.get("after", {})
        event_type = frigate_msg.get("type")
        event_id: str = after.get("id", "")
        camera: str = after.get("camera", "")
        label: str = after.get("label", "")

        if camera != self.args.frigate_camera:
            return

        self.logger.debug("Frigate Message: event: %s", frigate_msg)

        object_type = self.label_to_object_type(label)
        if not object_type:
            self.logger.warning(
                "MISSED EVENT: unsupported label '%s' (event_id=%s type=%s)",
                label, event_id, event_type,
            )
            return

        self.logger.debug(
            "Frigate event: type=%s id=%s label=%s zones=%s active=%s",
            event_type, event_id, label,
            after.get("current_zones", []),
            list(self._frigate_to_unifi.keys()),
        )

        if event_type == "new":
            await self._handle_new_event(frigate_msg, event_id, label, object_type)
        elif event_type == "update":
            await self._handle_update_event(frigate_msg, event_id, label, object_type, after)
        elif event_type == "end":
            await self._handle_end_event(frigate_msg, event_id, label, object_type, after)
        else:
            self.logger.debug("Unhandled event type '%s' for event_id=%s", event_type, event_id)

    async def _handle_new_event(
        self,
        frigate_msg: dict[str, Any],
        event_id: str,
        label: str,
        object_type: SmartDetectObjectType,
    ) -> None:
        if event_id in self._frigate_to_unifi:
            old_unifi_id = self._frigate_to_unifi[event_id]
            self.logger.warning(
                "'new' event received for already-active Frigate event %s "
                "(UniFi %d). Stopping old event first.",
                event_id, old_unifi_id,
            )
            if old_unifi_id in self._active_smart_events:
                old_data = self._active_smart_events[old_unifi_id]
                await self.trigger_smart_detect_stop(old_data["object_type"], event_id=old_unifi_id)
            self._deregister_event(event_id, old_unifi_id)

        after = frigate_msg.get("after", {})
        descriptor, zonesStatus = self.build_descriptor_from_frigate_msg(frigate_msg, object_type)
        frame_time_ms = int(after.get("frame_time", 0) * 1000) - self.args.frigate_time_sync_ms

        unifi_id = await self.trigger_smart_detect_start(
            object_type,
            custom_descriptor=descriptor,
            event_timestamp=frame_time_ms,
            zones_status=zonesStatus,
        )
        self._register_event(event_id, unifi_id)

        self.logger.info(
            "Started %s smart event (Frigate: %s, UniFi: %d). Active events: %d",
            label, event_id, unifi_id, len(self._frigate_to_unifi),
        )

    async def _handle_update_event(
        self,
        frigate_msg: dict[str, Any],
        event_id: str,
        label: str,
        object_type: SmartDetectObjectType,
        after: dict[str, Any],
    ) -> None:
        unifi_id = self._frigate_to_unifi.get(event_id)
        if unifi_id is None:
            self.logger.warning(
                "MISSED EVENT: 'update' for unknown Frigate event_id=%s (label=%s). "
                "Likely missed 'new'.",
                event_id, label,
            )
            return

        if unifi_id not in self._active_smart_events:
            self.logger.warning(
                "Frigate event %s maps to UniFi event %d but that event is not active. "
                "Active events: %s",
                event_id, unifi_id, list(self._active_smart_events.keys()),
            )
            return

        descriptor, zonesStatus = self.build_descriptor_from_frigate_msg(frigate_msg, object_type)
        frame_time_ms = int(after.get("frame_time", 0) * 1000) - self.args.frigate_time_sync_ms
        await self.trigger_smart_detect_update(
            object_type,
            custom_descriptor=descriptor,
            event_timestamp=frame_time_ms,
            zones_status=zonesStatus,
        )
        self._event_last_update[unifi_id] = time.time()

        if after.get("has_snapshot") and self.args.frigate_http_url:
            await self._cache_snapshots_for_event(unifi_id)

        event_data = self._active_smart_events[unifi_id]
        self.logger.debug(
            "Update for smart event (Frigate: %s, UniFi: %d). Age: %.1fs",
            event_id, unifi_id, time.time() - event_data["start_time"],
        )

    async def _handle_end_event(
        self,
        frigate_msg: dict[str, Any],
        event_id: str,
        label: str,
        object_type: SmartDetectObjectType,
        after: dict[str, Any],
    ) -> None:
        unifi_id = self._frigate_to_unifi.get(event_id)
        if unifi_id is None:
            self.logger.warning(
                "MISSED EVENT: 'end' for unknown Frigate event_id=%s (label=%s). "
                "Likely missed 'new'.",
                event_id, label,
            )
            return

        if unifi_id not in self._active_smart_events:
            self.logger.warning(
                "Frigate event %s maps to UniFi event %d but event is not active. "
                "Cleaning up mapping.",
                event_id, unifi_id,
            )
            self._deregister_event(event_id, unifi_id)
            return

        event_data = self._active_smart_events.get(unifi_id)
        if not event_data:
            self.logger.warning(
                "Event data missing for UniFi event %d. Cleaning up.", unifi_id
            )
            self._deregister_event(event_id, unifi_id)
            return

        final_descriptor,zonesStatus = self.build_descriptor_from_frigate_msg(frigate_msg, object_type)
        end_time_ms = int(after.get("end_time", 0) * 1000) - self.args.frigate_time_sync_ms
        frame_time_ms = int(after.get("frame_time", 0) * 1000) - self.args.frigate_time_sync_ms

        self.logger.info(
            "Ending %s smart event (Frigate: %s, UniFi: %d). Duration: %.1fs",
            label, event_id, unifi_id, time.time() - event_data["start_time"],
        )
        self.logger.debug(
            "Event end timestamps: end_time=%d frame_time=%d", end_time_ms, frame_time_ms
        )

        await self.trigger_smart_detect_stop(
            object_type,
            custom_descriptor=final_descriptor,
            event_timestamp=end_time_ms,
            event_id=unifi_id,
            frame_time_ms=frame_time_ms,
            zones_status=zonesStatus,
        )
        self._deregister_event(event_id, unifi_id)

        self.logger.info(
            "Event %s ended. Remaining active events: %d",
            event_id, len(self._frigate_to_unifi),
        )

    # ---------------------------------------------------------------------------
    # Snapshot helpers
    # ---------------------------------------------------------------------------

    async def _cache_snapshots_for_event(self, unifi_id: int) -> None:
        """Fetch snapshots and store paths on the active smart-detect event."""
        try:
            crop, fov, heatmap = await self.fetch_snapshots_for_event(unifi_id, "smart_detect")
            event_data = self._active_smart_events.get(unifi_id)
            if event_data is None:
                return
            if crop:
                event_data["snapshot_crop_path"] = crop
            if fov:
                event_data["snapshot_fov_path"] = fov
            if heatmap:
                event_data["heatmap_path"] = heatmap
            self.logger.debug(
                "Cached snapshots for event %d: crop=%s fov=%s heatmap=%s",
                unifi_id, crop, fov, heatmap,
            )
        except Exception:
            self.logger.exception("Error caching snapshots for event %d", unifi_id)

    async def handle_snapshot_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                "Non-bytes payload for snapshot event: %r", message.payload
            )
            return

        parts = message.topic.value.split("/")
        if len(parts) < 4:
            self.logger.debug("Unexpected snapshot topic: %s", message.topic.value)
            return

        snapshot_label = parts[-2]
        self.logger.debug("Snapshot received: topic=%s", message.topic.value)

        # Match to an active event by object-type label, skipping retained messages.
        matching_frigate_id: Optional[str] = None
        for frigate_id, unifi_id in self._frigate_to_unifi.items():
            event_data = self._active_smart_events.get(unifi_id)
            if (
                event_data is not None
                and event_data["object_type"].value == snapshot_label
                and not message.retain
            ):
                matching_frigate_id = frigate_id
                break

        if matching_frigate_id:
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            f.write(message.payload)
            f.close()
            self.logger.debug(
                "Updating motion snapshot for Frigate event %s (%s) → %s",
                matching_frigate_id, snapshot_label, f.name,
            )
            self.update_motion_snapshot(Path(f.name))
        else:
            self.logger.debug(
                "Discarding snapshot label=%s size=%d retained=%s — no matching active event.",
                snapshot_label, len(message.payload), message.retain,
            )
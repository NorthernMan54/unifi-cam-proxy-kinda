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
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        super().__init__(args, logger)
        self.args = args
        # Map Frigate event IDs to UniFi event IDs for tracking
        self.frigate_to_unifi_event_map: dict[str, int] = {}
        # Store snapshot readiness per Frigate event ID
        self.event_snapshot_ready: dict[str, asyncio.Event] = {}
        # Track last update time for each event (for timeout detection)
        self.event_last_update: dict[int, float] = {}
        self.event_timeout_seconds = 600  # Timeout after 600 seconds (10 minutes) without updates

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

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            **await super().get_feature_flags(),
            **{
                "mic": True,
                "smartDetect": [
                    "person",
                    "vehicle",
                ],
            },
        }

    @classmethod
    def label_to_object_type(cls, label: str) -> Optional[SmartDetectObjectType]:
        if label == "person":
            return SmartDetectObjectType.PERSON
        elif label in {"vehicle", "car", "motorcycle", "bus"}:
            return SmartDetectObjectType.VEHICLE

    def build_descriptor_from_frigate_msg(
        self, frigate_msg: dict[str, Any], object_type: SmartDetectObjectType
    ) -> dict[str, Any]:
        """
        Build a UniFi Protect-compatible descriptor from Frigate event data.
        
        Frigate provides bounding boxes as pixel coordinates.
        Testing with raw pixel values to see UniFi Protect's expected format.
        """
        after = frigate_msg.get("after", {})
        type = after.get("type", "unknown")
        
        # Extract bounding box if available
        box = after.get("box")
        if box and len(box) == 4:
            # Frigate box format: [x_min, y_min, x_max, y_max] in configured pixel dimensions
            # UniFi format: [x, y, width, height] in 1000x1000 coordinate system

            # Scale factors
            frigate_width = self.args.frigate_detect_width
            frigate_height = self.args.frigate_detect_height
            UNIFI_WIDTH = 1000
            UNIFI_HEIGHT = 1000
            UNIFI_Y_OFFSET = 0  # Offset is needed if cam width/height differ from detection dimensions

            x_scale = UNIFI_WIDTH / frigate_width
            y_scale = UNIFI_HEIGHT / frigate_height

            # Transform coordinates
            x_min_unifi = int(box[0] * x_scale)
            y_min_unifi = int(box[1] * y_scale) + UNIFI_Y_OFFSET
            x_max_unifi = int(box[2] * x_scale)
            y_max_unifi = int(box[3] * y_scale) + UNIFI_Y_OFFSET

            # Convert to UniFi format
            x = x_min_unifi
            y = y_min_unifi
            width = x_max_unifi - x_min_unifi
            height = y_max_unifi - y_min_unifi
            coord = [x, y, width, height]
        else:
            # Fallback to default if no bounding box
            coord = [0, 0, 1920, 1080]
        
        # Extract confidence score (Frigate uses 0.0-1.0, UniFi uses 0-100)
        # If type is "end" use after.top_score if available, else after.score
        score = after.get("top_score" if type == "end" else "score", 0.95)
        confidence_level = int(score * 100)
        
        # Extract tracker ID if available
        tracker_id = after.get("id", 1)
        if isinstance(tracker_id, str):
            # Convert string ID to numeric hash for tracker ID
            tracker_id = hash(tracker_id) % 1000000
        
        # Check if object is stationary
        stationary = after.get("stationary", False)
        
        # Get current position and velocity if available
        current_zones = after.get("current_zones", [])
        zones = [0] if not current_zones else [0]  # Default to zone 0
        
        # Extract speed information if available (Frigate provides speeds in km/h or mph)
        average_speed = after.get("average_estimated_speed", 0)
        # Convert to appropriate units if needed
        # UniFi Protect expects positive number or null, stored as float
        speed = float(average_speed) if average_speed > 0 else None
        
        # Extract license plate information if available
        # Frigate format: ["PLATE-TEXT", confidence_score]
        license_plate_data = after.get("recognized_license_plate")
        license_plate = None
        license_plate_score = None
        if license_plate_data and isinstance(license_plate_data, list) and len(license_plate_data) >= 1:
            license_plate = license_plate_data[0]  # Extract the plate text
            if len(license_plate_data) >= 2:
                license_plate_score = license_plate_data[1]  # Extract the confidence score
        
        # Set name based on object type and available data
        # For vehicles with license plates, use the plate as the name with confidence score
        if object_type == SmartDetectObjectType.VEHICLE and license_plate:
            if license_plate_score is not None:
                name = f"{license_plate} ({license_plate_score:.1%})"
            else:
                name = license_plate
        elif object_type == SmartDetectObjectType.VEHICLE:
            name = None # "No license plate available"
        else:
            name = None # "Named by Frigate"

        descriptor = {
            "attributes": None,  # Optional and validated
            "boxColor": "red", # validated
            "confidenceLevel": confidence_level, # validated
            "coord": coord, # validated
            "coord3d": [-1,-1],  # validated Required field: no 3D coordinates available
            "depth": None,  # Optional depth information
            "firstShownTimeMs": int(frigate_msg.get('after', {}).get('start_time', 0) * 1000) - self.args.frigate_time_sync_ms,  # validated
            "idleSinceTimeMs": int(frigate_msg.get('after', {}).get('motionless_count', 0) * 1000), # validated
            "lines": [],  # validated Required field: no line crossing detection
            "loiterZones": [],  # validated Optional but included for completeness
            "name": name, # validated - License plate for vehicles, or default name
            "objectType": object_type.value, # validated
            "secondLensZones": [],  # validated Optional but included for completeness
            "speed": speed,  # Average estimated speed from Frigate
            "stationary": stationary, # validated
            "tag": "Tagged by Frigate",  # validated
            "trackerID": tracker_id, # validated
            "zones": zones, # validated
        }
        
        # Only include licensePlate if it has a value
        if license_plate:
            descriptor["licensePlate"] = license_plate
        
        self.logger.debug(
            f"Built descriptor: trackerID={tracker_id}, confidence={confidence_level}, "
            f"coord={coord}, stationary={stationary}, speed={speed}, licensePlate={license_plate}"
        )
        
        return descriptor

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
        
        # For smart detect events, try to find the corresponding Frigate event ID
        frigate_event_id = None
        if event_type == "smart_detect":
            # Look up Frigate event ID by searching the mapping
            for frig_id, unifi_id in self.frigate_to_unifi_event_map.items():
                if unifi_id == event_id:
                    frigate_event_id = frig_id
                    break
        
        # Build URLs based on whether we have a Frigate event ID
        if frigate_event_id:
            # Use event-specific snapshot endpoint for smart detect events
            # https://demo.frigate.video/api/events/:event_id/snapshot.jpg
            base_url = f"{self.args.frigate_http_url}/api/events/{frigate_event_id}/snapshot.jpg"
            full_url = base_url
            thumbnail_url = f"{base_url}?crop=1&quality=80"  # Crop gives bounding box snapshot
            
            self.logger.debug(
                f"Using Frigate event-specific snapshot URLs for event {event_id} "
                f"(Frigate: {frigate_event_id}): {base_url}"
            )
        else:
            # Fallback to latest.jpg endpoint with timestamp for analytics events
            timestamp = None
            if event_type == "analytics" and event_id in self._analytics_event_history:
                event_data = self._analytics_event_history[event_id]
                start_time = event_data.get('start_time')
                if start_time is not None:
                    timestamp = int(start_time)
            
            # Build URLs for snapshots from latest frame endpoint
            base_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
            
            # Full resolution snapshot
            full_url = base_url
            if timestamp is not None:
                full_url = f"{full_url}?timestamp={timestamp}"
            
            # Thumbnail snapshot (360p for fast UniFi Protect processing)
            thumbnail_url = f"{base_url}?height=360&quality=80"
            if timestamp is not None:
                thumbnail_url = f"{thumbnail_url}&timestamp={timestamp}"
        
        self.logger.debug(f"Fetching snapshots for event {event_id}: full={full_url}, thumbnail={thumbnail_url}")
        
        # Fetch both snapshots in parallel
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
        
        # Fetch both in parallel
        results = await asyncio.gather(
            fetch_url(full_url, "full"),
            fetch_url(thumbnail_url, "thumbnail"),
            return_exceptions=True
        )
        
        snapshot_fov = results[0] if not isinstance(results[0], Exception) else None
        snapshot_crop = results[1] if not isinstance(results[1], Exception) else None
        
        # Use full snapshot as heatmap
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
                        # Start event timeout monitor
                        tg.create_task(self.monitor_event_timeouts())
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

    async def handle_motion_event(self, message: Message) -> None:
        """Handle raw motion events from Frigate (if needed)"""
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                f"Unexpectedly received non-bytes payload for motion event: {message.payload}"
            )
            return
        msg = message.payload.decode()
        # self.logger.debug(f"Received raw motion event: {msg}")
        if msg == "ON":
            self.logger.debug("Frigate motion event: ON")
            await self.trigger_analytics_start()
        elif msg == "OFF":
            self.logger.debug("Frigate motion event: OFF")
            await self.trigger_analytics_stop()

    async def monitor_event_timeouts(self) -> None:
        """Monitor active events and end those that haven't received updates in 600 seconds"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            current_time = time.time()
            expired_frigate_events = []
            
            # Check all active smart detect events (from base class)
            for unifi_event_id, event_data in list(self._active_smart_events.items()):
                # Skip events that have already ended
                if event_data.get("end_time") is not None:
                    continue
                    
                # Get last update time, defaulting to start time if no updates yet
                last_update = self.event_last_update.get(unifi_event_id, event_data["start_time"])
                time_since_update = current_time - last_update
                
                # Only timeout if no update for over 600 seconds
                if time_since_update > self.event_timeout_seconds:
                    # Find the Frigate event ID that maps to this UniFi event ID
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
            
            # End expired events
            for frigate_event_id, unifi_event_id in expired_frigate_events:
                event_data = self._active_smart_events.get(unifi_event_id)
                if not event_data:
                    continue
                    
                try:
                    # End smart detect event (motion event will end separately)
                    await self.trigger_smart_detect_stop(event_data["object_type"], event_id=unifi_event_id)
                except Exception as e:
                    self.logger.exception(f"Error ending timed out event {unifi_event_id}: {e}")
                finally:
                    # Clean up mappings
                    if frigate_event_id and frigate_event_id in self.frigate_to_unifi_event_map:
                        del self.frigate_to_unifi_event_map[frigate_event_id]
                    if frigate_event_id and frigate_event_id in self.event_snapshot_ready:
                        del self.event_snapshot_ready[frigate_event_id]
                    if unifi_event_id in self.event_last_update:
                        del self.event_last_update[unifi_event_id]

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
                #self.logger.debug(
                #    f"Frigate: Ignoring Frigate event for different camera: {camera} "
                #    f"(expecting {self.args.frigate_camera})"
                #)
                return
            
            self.logger.debug(
                    f"Frigate: Received: {frigate_msg} "
                )

            before_snapshot_time = frigate_msg.get('before', {}).get('snapshot', {}).get('frame_time', 'N/A') if frigate_msg.get('before', {}).get('snapshot') else 'N/A'
            after_snapshot_time = frigate_msg.get('after', {}).get('snapshot', {}).get('frame_time', 'N/A') if frigate_msg.get('after', {}).get('snapshot') else 'N/A'
            
            before_data = frigate_msg.get('before', {})
            after_data = frigate_msg.get('after', {})
            
            #self.logger.debug(
            #    f"Times - before: frame={before_data.get('frame_time', 'N/A')}, snapshot_frame={before_snapshot_time}, start={before_data.get('start_time', 'N/A')}, end={before_data.get('end_time', 'N/A')} | after: frame={after_data.get('frame_time', 'N/A')}, snapshot_frame={after_snapshot_time}, start={after_data.get('start_time', 'N/A')}, end={after_data.get('end_time', 'N/A')}"
            #)

            #self.logger.debug(
            #    f"{before_data.get('frame_time', 'N/A')},{before_snapshot_time},{before_data.get('start_time', 'N/A')},{before_data.get('end_time', 'N/A')},{after_data.get('frame_time', 'N/A')},{after_snapshot_time},{after_data.get('start_time', 'N/A')},{after_data.get('end_time', 'N/A')}"
            #)

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
                # Ignore new events without snapshots
                has_snapshot = frigate_msg.get("after", {}).get("has_snapshot", True)
                if not has_snapshot:
                    self.logger.debug(
                        f"Frigate: Ignoring new event {event_id} with has_snapshot=false"
                    )
                    return
                
                if event_id in self.frigate_to_unifi_event_map:
                    self.logger.warning(
                        f"Received 'new' event for already active Frigate event_id={event_id}. "
                        f"This may indicate event was not properly ended. Stopping old event first."
                    )
                    # Stop the old event before starting new one
                    old_unifi_id = self.frigate_to_unifi_event_map[event_id]
                    if old_unifi_id in self._active_smart_events:
                        old_event = self._active_smart_events[old_unifi_id]
                        await self.trigger_smart_detect_stop(old_event["object_type"], event_id=old_unifi_id)
                    del self.frigate_to_unifi_event_map[event_id]
                
                # Create snapshot ready event for this Frigate event
                self.event_snapshot_ready[event_id] = asyncio.Event()
                
                # Send smart detect event as update to existing motion event
                # Build custom descriptor from Frigate data
                custom_descriptor = self.build_descriptor_from_frigate_msg(
                    frigate_msg, object_type
                )
                start_time_ms = int(frigate_msg.get('after', {}).get('start_time', 0) * 1000) - self.args.frigate_time_sync_ms
                frame_time_ms = int(frigate_msg.get('after', {}).get('frame_time', 0) * 1000) - self.args.frigate_time_sync_ms
                 
                unifi_event_id = await self.trigger_smart_detect_start(object_type, custom_descriptor, frame_time_ms)
                
                # Store mapping from Frigate event ID to UniFi event ID
                self.frigate_to_unifi_event_map[event_id] = unifi_event_id
                
                # Track event creation time as last update
                self.event_last_update[unifi_event_id] = time.time()
                
                self.logger.info(
                    f"Frigate: Starting {label} smart event within motion context (Frigate: {event_id}, UniFi: {unifi_event_id}). "
                    f"Total active events: {len(self.frigate_to_unifi_event_map)}"
                )
                
                self.logger.info(
                    f"Frigate: Starting {label} smart event within motion context (Frigate: {event_id}, UniFi: {unifi_event_id}). "
                    f"Total active events: {len(self.frigate_to_unifi_event_map)}"
                )

            elif event_type == "update":
                if event_id in self.frigate_to_unifi_event_map:
                    unifi_event_id = self.frigate_to_unifi_event_map[event_id]
                    
                    # Verify the UniFi event is still active
                    if unifi_event_id not in self._active_smart_events:
                        self.logger.warning(
                            f"Frigate event {event_id} maps to UniFi event {unifi_event_id} "
                            f"but that event is not active. Skipping update."
                            f"active _active_smart_events: {list(self._active_smart_events.keys())}"
                        )
                        return
                    
                    # Build updated descriptor from Frigate data
                    custom_descriptor = self.build_descriptor_from_frigate_msg(
                        frigate_msg, object_type
                    )

                    frame_time_ms = int(frigate_msg.get('after', {}).get('frame_time', 0) * 1000) - self.args.frigate_time_sync_ms
                    # Send moving update with updated bounding box
                    await self.trigger_smart_detect_update(object_type, custom_descriptor, frame_time_ms)
                    
                    # Update last update time for timeout tracking
                    self.event_last_update[unifi_event_id] = time.time()
                    
                    # Fetch and cache snapshots if available
                    has_snapshot = after_data.get('has_snapshot', False)
                    if has_snapshot and self.args.frigate_http_url:
                        self.logger.debug(f"Event {event_id} has updated snapshot, fetching and caching all types...")
                        try:
                            # Fetch snapshots using the UniFi event ID
                            snapshot_crop, snapshot_fov, heatmap = await self.fetch_snapshots_for_event(
                                unifi_event_id, "smart_detect"
                            )
                            
                            # Cache snapshots in the smart detect event data
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
                    
                    # Verify the UniFi event is still active
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
                    
                    # Build final descriptor from end event data
                    final_descriptor = self.build_descriptor_from_frigate_msg(
                        frigate_msg, object_type
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
                    
                    # End the smart detect event (motion event will end when Frigate sends motion OFF)
                    # Pass both end_time (for stop event) and frame_time (for final update)
                    await self.trigger_smart_detect_stop(
                        object_type, 
                        final_descriptor, 
                        end_time_ms, 
                        event_id=unifi_event_id,
                        frame_time_ms=frame_time_ms
                    )
                    
                    # Clean up mappings
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

        # Extract label from topic: frigate/<camera>/<label>/snapshot
        topic_parts = message.topic.value.split("/")
        if len(topic_parts) < 4:
            self.logger.debug(f"Unexpected snapshot topic format: {message.topic.value}")
            return
            
        snapshot_label = topic_parts[-2]
        
        self.logger.debug(
            f"Received snapshot: topic={message.topic.value}, "
            f"message={message}"
        )
        
        # Find matching active Frigate event by label
        # We need to match based on label since snapshots don't include event ID
        matching_frigate_event_id = None
        for frigate_event_id, unifi_event_id in self.frigate_to_unifi_event_map.items():
            # Check if this UniFi event is still active and matches the label
            if unifi_event_id in self._active_smart_events:
                event_data = self._active_smart_events[unifi_event_id]
                # Match by object type (person -> person, vehicle labels -> vehicle)
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

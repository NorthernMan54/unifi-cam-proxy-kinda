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
        score = after.get("score", 0.95)
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
        
        descriptor = {
            "trackerID": tracker_id,
            "name": object_type.value,
            "confidenceLevel": confidence_level,
            "coord": coord,
            "objectType": object_type.value,
            "zones": zones,
            "lines": [],  # Required field: no line crossing detection
            "loiterZones": [],  # Optional but included for completeness
            "stationary": stationary,
            "attributes": {},  # Optional but included for completeness
            "coord3d": [0, 0, 0],  # Required field: no 3D coordinates available
            "depth": None,  # Optional depth information
            "speed": speed,  # Average estimated speed from Frigate
        }
        
        self.logger.debug(
            f"Built descriptor: trackerID={tracker_id}, confidence={confidence_level}, "
            f"coord={coord}, stationary={stationary}, speed={speed}"
        )
        
        return descriptor

    async def fetch_snapshot_from_api(
        self, event_id: str, snapshot_type: str = "snapshot", timestamp: Optional[float] = None
    ) -> Optional[Path]:
        """
        Fetch a specific snapshot type from Frigate HTTP API.
        
        Args:
            event_id: Frigate event ID
            snapshot_type: Type of snapshot - "snapshot" (full), "thumbnail" (360p), or "clip"
            timestamp: Optional Unix timestamp for the snapshot (typically event midpoint)
        
        Returns:
            Path to temporary file with snapshot, or None if fetch failed.
        """
        if not self.args.frigate_http_url:
            return None
        
        # Frigate API endpoints for different snapshot types
        if snapshot_type == "thumbnail":
            # Use 360p height for thumbnail to ensure fast processing by UniFi Protect
            url = f"{self.args.frigate_http_url}/api/events/{event_id}/snapshot.jpg?height=360&quality=80"
        elif snapshot_type == "clip":
            url = f"{self.args.frigate_http_url}/api/events/{event_id}/clip.mp4"
        else:  # snapshot (full resolution)
            # Use full resolution for FoV snapshot
            url = f"{self.args.frigate_http_url}/api/events/{event_id}/snapshot.jpg"
        
        # Add timestamp parameter if provided (for all snapshot types)
        if timestamp is not None:
            separator = "&" if "?" in url else "?"
            # Frigate API expects integer timestamp (seconds since epoch)
            url = f"{url}{separator}timestamp={int(timestamp)}"
        
        self.logger.debug(f"Fetching {snapshot_type} for event {event_id} from {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        suffix = ".mp4" if snapshot_type == "clip" else ".jpg"
                        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        f.write(image_data)
                        f.close()
                        self.logger.info(
                            f"Successfully fetched {snapshot_type} for event {event_id} from API "
                            f"({len(image_data)} bytes) -> {f.name}"
                        )
                        return Path(f.name)
                    else:
                        # Log error response body for debugging
                        error_body = await response.text()
                        self.logger.warning(
                            f"Failed to fetch {snapshot_type} for event {event_id}: "
                            f"HTTP {response.status}, Response: {error_body}"
                        )
                        return None
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {snapshot_type} for event {event_id}")
            return None
        except Exception as e:
            self.logger.warning(f"Error fetching {snapshot_type} for event {event_id}: {e}")
            return None

    async def fetch_all_snapshots_from_api(
        self, event_id: str
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Fetch all three snapshot types from Frigate HTTP API.
        Uses timestamp from active analytics event to get snapshot at event midpoint.
        
        Args:
            event_id: Frigate event ID
        
        Returns:
            Tuple of (snapshot_full, snapshot_crop/thumbnail, heatmap)
            - snapshot_full: Full resolution snapshot (for FoV)
            - snapshot_crop: Thumbnail/cropped version (for cropped view)
            - heatmap: Use full snapshot as heatmap (Frigate doesn't provide separate heatmap)
        """
        if not self.args.frigate_http_url:
            return (None, None, None)
        
        # Calculate midpoint timestamp from active analytics event
        timestamp = None
        if self._active_analytics_event_id is not None:
            active_event = self._analytics_event_history.get(self._active_analytics_event_id)
            if active_event:
                start_time = active_event.get('start_time')
                if start_time is not None:
                    current_time = time.time()
                    # Frigate API expects integer timestamp (seconds since epoch)
                    timestamp = int((start_time + current_time) / 2)
                    self.logger.debug(
                        f"Using event midpoint timestamp for snapshots: {timestamp} "
                        f"(start={start_time}, current={current_time})"
                    )
        
        if timestamp is None:
            self.logger.debug(f"No active analytics event, fetching snapshots without timestamp")
        
        # Fetch both full snapshot and thumbnail in parallel
        results = await asyncio.gather(
            self.fetch_snapshot_from_api(event_id, "snapshot", timestamp),
            self.fetch_snapshot_from_api(event_id, "thumbnail", timestamp),
            return_exceptions=True
        )
        
        snapshot_full = results[0] if not isinstance(results[0], Exception) else None
        snapshot_crop = results[1] if not isinstance(results[1], Exception) else None
        
        # Use full snapshot as heatmap (Frigate doesn't provide separate heatmap)
        heatmap = snapshot_full
        
        self.logger.info(
            f"Fetched snapshots for event {event_id}: "
            f"full={'✓' if snapshot_full else '✗'}, "
            f"crop={'✓' if snapshot_crop else '✗'}, "
            f"heatmap={'✓' if heatmap else '✗'}"
        )
        
        return (snapshot_full, snapshot_crop, heatmap)

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

    async def grab_most_recent_event_snapshot(self) -> Optional[Path]:
        """
        Grab the most recent snapshot based on event API.
        Queries Frigate's events API to find the most recent event with a snapshot,
        then fetches the snapshot image.
        """
        if not self.args.frigate_http_url:
            self.logger.warning("Cannot grab recent event snapshot: frigate_http_url not configured")
            return None
        
        # Build the events API URL
        events_url = f"{self.args.frigate_http_url}/api/events"
        params = {
            "camera": self.args.frigate_camera,
            "limit": 1,
            "has_snapshot": 1
        }
        
        self.logger.debug(f"Fetching most recent event from {events_url} with params {params}")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch the most recent event
                async with session.get(
                    events_url, 
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status != 200:
                        self.logger.warning(
                            f"Failed to fetch recent events: HTTP {response.status}"
                        )
                        return None
                    
                    events = await response.json()
                    
                    if not events or len(events) == 0:
                        self.logger.warning("No recent events with snapshots found")
                        return None
                    
                    # Get the most recent event
                    most_recent_event = events[0]
                    event_id = most_recent_event.get("id")
                    
                    if not event_id:
                        self.logger.warning("Most recent event missing 'id' field")
                        return None
                    
                    # Log full event details
                    # self.logger.debug(f"Most recent event details: {json.dumps(most_recent_event, indent=2)}")
                    
                    # Log summary
                    start_time = most_recent_event.get("start_time")
                    end_time = most_recent_event.get("end_time")
                    event_data = most_recent_event.get("data", {})
                    
                    self.logger.info(
                        f"Event summary: id={event_id}, "
                        f"label={most_recent_event.get('label')}, "
                        f"camera={most_recent_event.get('camera')}, "
                        f"start_time={start_time}, "
                        f"end_time={end_time}, "
                        f"score={event_data.get('score')}, "
                        f"top_score={event_data.get('top_score')}, "
                        f"box={event_data.get('box')}"
                    )
                    
                    # Calculate midpoint timestamp from active analytics event if available
                    timestamp = None
                    if self._active_analytics_event_id is not None:
                        active_event = self._analytics_event_history.get(self._active_analytics_event_id)
                        if active_event:
                            event_start = active_event.get('start_time')
                            if event_start is not None:
                                current_time = time.time()
                                # Frigate API expects integer timestamp (seconds since epoch)
                                timestamp = int((event_start + current_time) / 2)
                    
                    # Now fetch the snapshot for this event
                    snapshot_path = await self.fetch_snapshot_from_api(event_id, "snapshot", timestamp)
                    
                    if snapshot_path:
                        self.logger.info(
                            f"Successfully retrieved snapshot for most recent event {event_id}"
                        )
                    
                    return snapshot_path
                    
        except asyncio.TimeoutError:
            self.logger.warning("Timeout fetching most recent event")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching most recent event snapshot: {e}")
            return None

    async def grab_most_recent_motion_snapshot(self) -> Optional[Path]:
        """
        Grab the most recent motion snapshot from Frigate's latest frame API.
        This is a simple snapshot fetch without event context, suitable for generic motion events.
        Uses the /api/<camera>/latest.jpg endpoint with timestamp from active analytics event.
        """
        if not self.args.frigate_http_url:
            self.logger.warning("Cannot grab motion snapshot: frigate_http_url not configured")
            return None
        
        # Use Frigate's latest frame endpoint
        snapshot_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
        
        # Add timestamp parameter from active analytics event if available
        if self._active_analytics_event_id is not None:
            active_event = self._analytics_event_history.get(self._active_analytics_event_id)
            if active_event:
                start_time = active_event.get('start_time')
                if start_time is not None:
                    current_time = time.time()
                    timestamp = (start_time + current_time) / 2
                    # Frigate API expects integer timestamp (seconds since epoch)
                    snapshot_url = f"{snapshot_url}?timestamp={int(timestamp)}"
                    self.logger.debug(
                        f"Using event midpoint timestamp for motion snapshot: {int(timestamp)} "
                        f"(start={start_time}, current={current_time})"
                    )
        
        self.logger.debug(f"Fetching latest motion snapshot from {snapshot_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    snapshot_url,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        f = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        f.write(image_data)
                        f.close()
                        self.logger.info(
                            f"Successfully fetched latest motion snapshot "
                            f"({len(image_data)} bytes) -> {f.name}"
                        )
                        return Path(f.name)
                    else:
                        # Log error response body for debugging
                        error_body = await response.text()
                        self.logger.warning(
                            f"Failed to fetch latest snapshot: HTTP {response.status}, "
                            f"Response: {error_body}"
                        )
                        return None
                    
        except asyncio.TimeoutError:
            self.logger.warning("Timeout fetching latest motion snapshot")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching latest motion snapshot: {e}")
            return None

    async def grab_all_motion_snapshots(self) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Grab all three snapshot types for generic motion events from Frigate's latest frame API.
        Fetches full resolution, thumbnail (360p), and heatmap snapshots using timestamp from active analytics event.
        
        Returns:
            Tuple of (snapshot_full, snapshot_crop/thumbnail, heatmap)
            - snapshot_full: Full resolution snapshot (for FoV)
            - snapshot_crop: Thumbnail/cropped version (360p for fast processing)
            - heatmap: Use full snapshot as heatmap
        """
        if not self.args.frigate_http_url:
            self.logger.warning("Cannot grab motion snapshots: frigate_http_url not configured")
            return (None, None, None)
        
        # Calculate midpoint timestamp from active analytics event
        timestamp = None
        if self._active_analytics_event_id is not None:
            active_event = self._analytics_event_history.get(self._active_analytics_event_id)
            if active_event:
                start_time = active_event.get('start_time')
                if start_time is not None:
                    current_time = time.time()
                    timestamp = int((start_time + current_time) / 2)
                    self.logger.debug(
                        f"Using event midpoint timestamp for motion snapshots: {timestamp} "
                        f"(start={start_time}, current={current_time})"
                    )
        
        # Build URLs for full and thumbnail snapshots
        base_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
        
        # Full resolution snapshot
        full_url = base_url
        if timestamp is not None:
            full_url = f"{full_url}?timestamp={timestamp}"
        
        # Thumbnail snapshot (360p for fast UniFi Protect processing)
        thumbnail_url = f"{base_url}?height=360&quality=80"
        if timestamp is not None:
            thumbnail_url = f"{thumbnail_url}&timestamp={timestamp}"
        
        self.logger.debug(f"Fetching motion snapshots: full={full_url}, thumbnail={thumbnail_url}")
        
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
                            self.logger.info(
                                f"Successfully fetched {snapshot_type} motion snapshot "
                                f"({len(image_data)} bytes) -> {f.name}"
                            )
                            return Path(f.name)
                        else:
                            # Log error response body for debugging
                            error_body = await response.text()
                            self.logger.warning(
                                f"Failed to fetch {snapshot_type} motion snapshot: "
                                f"HTTP {response.status}, Response: {error_body}"
                            )
                            return None
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout fetching {snapshot_type} motion snapshot")
                return None
            except Exception as e:
                self.logger.error(f"Error fetching {snapshot_type} motion snapshot: {e}")
                return None
        
        # Fetch both in parallel
        results = await asyncio.gather(
            fetch_url(full_url, "full"),
            fetch_url(thumbnail_url, "thumbnail"),
            return_exceptions=True
        )
        
        snapshot_full = results[0] if not isinstance(results[0], Exception) else None
        snapshot_crop = results[1] if not isinstance(results[1], Exception) else None
        
        # Use full snapshot as heatmap
        heatmap = snapshot_full
        
        self.logger.info(
            f"Fetched motion snapshots: "
            f"full={'✓' if snapshot_full else '✗'}, "
            f"crop={'✓' if snapshot_crop else '✗'}, "
            f"heatmap={'✓' if heatmap else '✗'}"
        )
        
        return (snapshot_full, snapshot_crop, heatmap)

    async def monitor_event_timeouts(self) -> None:
        """Monitor active events and end those that haven't received updates in 600 seconds"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            current_time = time.time()
            expired_frigate_events = []
            
            # Check all active smart detect events (from base class)
            for unifi_event_id, event_data in list(self._active_smart_events.items()):
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
                    await self.trigger_smart_detect_stop(event_data["object_type"])
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
                self.logger.debug(
                    f"Frigate: Ignoring Frigate event for different camera: {camera} "
                    f"(expecting {self.args.frigate_camera})"
                )
                return
            
            self.logger.debug(
                    f"Frigate: Received: {frigate_msg} "
                )

            before_snapshot_time = frigate_msg.get('before', {}).get('snapshot', {}).get('frame_time', 'N/A') if frigate_msg.get('before', {}).get('snapshot') else 'N/A'
            after_snapshot_time = frigate_msg.get('after', {}).get('snapshot', {}).get('frame_time', 'N/A') if frigate_msg.get('after', {}).get('snapshot') else 'N/A'
            
            before_data = frigate_msg.get('before', {})
            after_data = frigate_msg.get('after', {})
            
            self.logger.debug(
                f"Times - before: frame={before_data.get('frame_time', 'N/A')}, snapshot_frame={before_snapshot_time}, start={before_data.get('start_time', 'N/A')}, end={before_data.get('end_time', 'N/A')} | after: frame={after_data.get('frame_time', 'N/A')}, snapshot_frame={after_snapshot_time}, start={after_data.get('start_time', 'N/A')}, end={after_data.get('end_time', 'N/A')}"
            )

            self.logger.debug(
                f"{before_data.get('frame_time', 'N/A')},{before_snapshot_time},{before_data.get('start_time', 'N/A')},{before_data.get('end_time', 'N/A')},{after_data.get('frame_time', 'N/A')},{after_snapshot_time},{after_data.get('start_time', 'N/A')},{after_data.get('end_time', 'N/A')}"
            )

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
                    # Stop the old event before starting new one
                    old_unifi_id = self.frigate_to_unifi_event_map[event_id]
                    if old_unifi_id in self._active_smart_events:
                        old_event = self._active_smart_events[old_unifi_id]
                        await self.trigger_smart_detect_stop(old_event["object_type"])
                    del self.frigate_to_unifi_event_map[event_id]
                
                # Create snapshot ready event for this Frigate event
                self.event_snapshot_ready[event_id] = asyncio.Event()
                
                # Send smart detect event as update to existing motion event
                # Build custom descriptor from Frigate data
                custom_descriptor = self.build_descriptor_from_frigate_msg(
                    frigate_msg, object_type
                )
                start_time_ms = int(frigate_msg.get('after', {}).get('start_time', 0) * 1000) - self.args.frigate_time_sync_ms
                unifi_event_id = await self.trigger_smart_detect_start(object_type, custom_descriptor, start_time_ms)
                
                # Store mapping from Frigate event ID to UniFi event ID
                self.frigate_to_unifi_event_map[event_id] = unifi_event_id
                
                # Track event creation time as last update
                self.event_last_update[unifi_event_id] = time.time()
                
                self.logger.info(
                    f"Frigate: Starting {label} smart event within motion context (Frigate: {event_id}, UniFi: {unifi_event_id}). "
                    f"Total active events: {len(self.frigate_to_unifi_event_map)}"
                )
                
                # Fetch snapshots if available
                has_snapshot = after_data.get('has_snapshot', False)
                if has_snapshot and self.args.frigate_http_url:
                    self.logger.debug(f"Event {event_id} has snapshot, fetching all types...")
                    snapshot_full, snapshot_crop, heatmap = await self.fetch_all_snapshots_from_api(event_id)
                    if snapshot_full or snapshot_crop or heatmap:
                        self.update_motion_snapshots(
                            crop=snapshot_crop,
                            fov=snapshot_full,
                            heatmap=heatmap
                        )
                        self.logger.info(
                            f"Updated snapshots for event {event_id}: "
                            f"crop={snapshot_crop is not None}, fov={snapshot_full is not None}, "
                            f"heatmap={heatmap is not None}"
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
                    
                    # Fetch updated snapshots if available
                    has_snapshot = after_data.get('has_snapshot', False)
                    if has_snapshot and self.args.frigate_http_url:
                        self.logger.debug(f"Event {event_id} has updated snapshot, fetching all types...")
                        snapshot_full, snapshot_crop, heatmap = await self.fetch_all_snapshots_from_api(event_id)
                        if snapshot_full or snapshot_crop or heatmap:
                            self.update_motion_snapshots(
                                crop=snapshot_crop,
                                fov=snapshot_full,
                                heatmap=heatmap
                            )
                            self.logger.debug(
                                f"Updated snapshots for event {event_id}: "
                                f"crop={snapshot_crop is not None}, fov={snapshot_full is not None}, "
                                f"heatmap={heatmap is not None}"
                            )
                    
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
                    # This will be included in the "leave" event by trigger_motion_stop
                    final_descriptor = self.build_descriptor_from_frigate_msg(
                        frigate_msg, object_type
                    )
                    end_time_ms = int(frigate_msg.get('after', {}).get('end_time', 0) * 1000) - self.args.frigate_time_sync_ms
                    
                    # Fetch all snapshot types - try HTTP API first, fall back to MQTT
                    if self.args.frigate_http_url:
                        has_snapshot = after_data.get('has_snapshot', False)
                        if has_snapshot:
                            self.logger.info(f"Fetching final snapshots for event {event_id}...")
                            snapshot_full, snapshot_crop, heatmap = await self.fetch_all_snapshots_from_api(event_id)
                            if snapshot_full or snapshot_crop or heatmap:
                                self.update_motion_snapshots(
                                    crop=snapshot_crop,
                                    fov=snapshot_full,
                                    heatmap=heatmap
                                )
                                self.logger.info(
                                    f"Updated final snapshots for event {event_id}: "
                                    f"crop={snapshot_crop is not None}, fov={snapshot_full is not None}, "
                                    f"heatmap={heatmap is not None}"
                                )
                        else:
                            self.logger.warning(f"Event {event_id} ended but has_snapshot=False")
                    else:
                        # Fall back to MQTT snapshot wait
                        if event_id in self.event_snapshot_ready:
                            snapshot_ready = self.event_snapshot_ready[event_id]
                            self.logger.info(f"Frigate: Awaiting snapshot via MQTT (Frigate: {event_id})")
                            try:
                                await asyncio.wait_for(snapshot_ready.wait(), timeout=10.0)
                            except asyncio.TimeoutError:
                                self.logger.warning(
                                    f"Snapshot wait timeout for Frigate event {event_id}. Proceeding without snapshot."
                                )
                    
                    event_duration = time.time() - event_data["start_time"]
                    self.logger.info(
                        f"Frigate: Ending {label} smart event within motion context (Frigate: {event_id}, UniFi: {unifi_event_id}). "
                        f"Duration: {event_duration:.1f}s"
                    )
                    self.logger.info(
                        f"{end_time_ms}, {int(frigate_msg.get('after', {}).get('frame_time', 0) * 1000)}"
                    )
                    
                    # End the smart detect event (motion event will end when Frigate sends motion OFF)
                    await self.trigger_smart_detect_stop(object_type, final_descriptor, end_time_ms)
                    
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

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

import aiohttp
import websockets

from unifi.core import RetryableError
from unifi.cams.handlers import ProtocolHandlers, VideoStreamHandlers, SnapshotHandlers

AVClientRequest = AVClientResponse = dict[str, Any]


class SmartDetectObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"


class UnifiCamBase(ProtocolHandlers, VideoStreamHandlers, SnapshotHandlers, metaclass=ABCMeta):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        """
        Initialize the camera base with configuration and state management.
        Args:
            args: Command-line arguments containing:
                - cert: Path to SSL certificate file
                - (other camera-specific configuration options)
            logger: Logger instance for outputting diagnostic information
        Attributes:
            Message & Timing:
                _msg_id (int): Counter for WebSocket message identification
                _init_time (float): Timestamp when camera was initialized
            Streams:
                _streams (dict[str, str]): Mapping of stream names to URLs/identifiers
                _ffmpeg_handles (dict[str, subprocess.Popen]): Active FFmpeg process handles by stream name
            Snapshots:
                # Structure: Optional[Path] - filesystem path to stored snapshot image
                _motion_snapshot: Legacy cropped snapshot with bounding box
                _motion_snapshot_crop: Cropped image with bounding box overlay
                _motion_snapshot_fov: Full field-of-view image with bounding box overlay
                _motion_heatmap: Motion heatmap visualization (falls back to FoV if unavailable)
            Event Tracking:
                _motion_event_id (int): Sequential identifier for motion events
                # Structure: dict[int, dict[str, Any]] - keyed by event_id, values contain:
                #   - event_id (int): Unique event identifier
                #   - start_time (float): Event start time
                #   - end_time (Optional[float]): Event end time (None if still active)
                #   - event_timestamp (Optional[float]): Event timestamp
                #   - snapshot_filename (Optional[str]): Filename for motionSnapshot
                #   - snapshot_fov_filename (Optional[str]): Filename for motionSnapshotFullFoV
                #   - heatmap_filename (Optional[str]): Filename for motionHeatmap
                #   - smart_detect_event_ids (list[int]): Array of smart detect event IDs that occurred during this analytics event
                _analytics_event_history: History of all analytics events (kept for 1 hour after completion)
                _active_analytics_event_id: ID of current active analytics event (None if no active event)
                # Structure: dict[int, dict[str, Any]] - keyed by event_id, values contain:
                #   - event_id (int): Unique event identifier
                #   - object_type (SmartDetectObjectType): Detected object classification
                #   - timestamp (float): Event start time
                #   - descriptor (dict): Additional event metadata
                _active_smart_events: Smart detection events (active and ended, kept for 60 minutes after end)
                # Legacy fields (deprecated - use _active_* instead):
                _motion_event_ts (Optional[float]): Timestamp of last motion event
                _motion_object_type (Optional[SmartDetectObjectType]): Last detected object type
                _motion_last_descriptor (Optional[dict[str, Any]]): Last event descriptor metadata
            Video Configuration:
                # Structure: dict[str, tuple[int, int]] - keyed by stream name ("video1", "video2", "video3")
                #   Values are (width, height) tuples representing detected video resolution
                _detected_resolutions: Video resolution for each stream quality level
            Network:
                _ssl_context (ssl.SSLContext): SSL context for secure WebSocket connections
                _session (Optional[WebSocketClientProtocol]): Active WebSocket connection to UniFi Protect
        Side Effects:
            - Registers atexit handler to clean up streams on program termination
            - Creates SSL context with certificate validation disabled
        """
        self.args = args
        self.logger = logger

        self._msg_id: int = 0
        self._init_time: float = time.time()
        self._streams: dict[str, str] = {}
        
        # Snapshot storage - UniFi Protect requests three types:
        # 1. motionSnapshot - Cropped image with bounding box
        # 2. motionSnapshotFullFoV - Full size image with bounding box  
        # 3. motionHeatmap - Heatmap visualization (use full FoV as fallback)
        self._motion_snapshot: Optional[Path] = None  # Legacy field, typically the cropped version
        self._motion_snapshot_crop: Optional[Path] = None  # Cropped with bounding box
        self._motion_snapshot_fov: Optional[Path] = None  # Full field of view with bounding box
        self._motion_heatmap: Optional[Path] = None  # Heatmap (defaults to FoV if not available)
        
        # Global event ID counter - UniFi requires unique IDs across all event types
        self._motion_event_id: int = 0  # Shared counter for both analytics and smart detect events
        
        # Enhanced event tracking to support overlapping events
        # Track both generic motion (EventAnalytics) and smart detect events (EventSmartDetect)
        # Store all events (active and completed) for 1 hour to support snapshot retrieval
        self._analytics_event_history: dict[int, dict[str, Any]] = {}  # All analytics events by event_id
        self._active_analytics_event_id: Optional[int] = None  # Current active analytics event ID
        self._active_smart_events: dict[int, dict[str, Any]] = {}  # Smart detect events by event_id
        
        # Legacy compatibility (deprecated, use _active_* instead)
        self._motion_event_ts: Optional[float] = None
        self._motion_object_type: Optional[SmartDetectObjectType] = None
        self._motion_last_descriptor: Optional[dict[str, Any]] = None
        
        self._ffmpeg_handles: dict[str, subprocess.Popen] = {}
        
        # Video resolution detected from source (will be probed during init_adoption)
        # Store separate resolutions for each stream with defaults
        self._detected_resolutions: dict[str, tuple[int, int]] = {
            "video1": (2560, 1920),  # High quality default
            "video2": (1280, 704),   # Medium quality default
            "video3": (640, 360),    # Low quality default
        }

        # Set up ssl context for requests
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
        """
        Fetch and cache all three snapshot types for an event.
        Subclasses can override this to provide event-specific snapshot fetching.
        
        Args:
            event_id: The event ID (analytics or smart detect)
            event_type: "analytics" or "smart_detect"
            
        Returns:
            Tuple of (crop_path, fov_path, heatmap_path) - paths to cached snapshot files
        """
        # Default implementation: fetch current snapshot
        # Subclasses (like FrigateCam) should override this
        snapshot = await self.get_snapshot()
        return (snapshot, snapshot, snapshot)

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            "mic": True,
            "aec": [],
            "videoMode": ["default"],
            "motionDetect": ["enhanced"],
        }


    ###

    # Payload structure reference for motion events:
    # clockBestMonotonic: i.z.number(),
    # clockBestWall: i.z.number(),
    # clockMonotonic: i.z.number(),
    # clockStream: i.z.number(),
    # clockStreamRate: i.z.number(),
    # clockWall: i.z.number(),
    # edgeType: i.z.enum(["start", "stop", "unknown"]),
    # eventId: i.z.number(),
    # eventType: i.z.enum(["motion", "pulse"]),
    # levels: i.z.record(i.z.string(), i.z.number()).optional(),
    # These fields appear to be only used on a stop event, and are passed as part of the snapshot getRequest
    # motionHeatmap: i.z.string(),          - passed as filename in getRequest
    # motionHeatmapHeight: i.z.number().optional(),
    # motionHeatmapWidth: i.z.number().optional(),
    # motionRawHeatmapNPZ: i.z.string().optional(),
    # motionSnapshot: i.z.string(),         - passed as filename in getRequest
    # motionSnapshotFullFoV: i.z.string().optional(),
    # motionSnapshotFullFoVHeight: i.z.number().optional(),
    # motionSnapshotFullFoVWidth: i.z.number().optional(),
    # motionSnapshotHeight: i.z.number().optional(),
    # motionSnapshotWidth: i.z.number().optional()

    # payload structure reference for smart detect events:
    # clockWall: a.default.number(),
    # clockStream: a.default.number().optional(),
    # clockStreamRate: a.default.number().optional(),
    # displayTimeoutMSec: a.default.number(),
    # descriptors: t.smartDetectObjectDescriptorSchema.passthrough().array().default([]),
    # linesStatus: u.linesStatusesSchema.optional(),
    # zonesStatus: u.zonesStatusesSchema.optional(),
    # loiterZonesStatus: u.loiterStatusesSchema.optional(),
    # edgeType: a.default.union([a.default.nativeEnum(m), a.default.string()]).optional(),
    # objectTypes: a.default.nativeEnum(s.OBJECT_TYPES).array().optional()

    # descriptors object structure:
    # trackerID: n.trackerIdSchema,
    # name: a.default.string(),
    # confidenceLevel: a.default.number(),
    # coord: f,
    # depth: a.default.number().positive().nullable().optional(),
    # speed: a.default.number().positive().nullable().optional(),
    # objectType: s.objectTypesSchema,
    # zones: a.default.number().finite().array(),
    # lines: a.default.number().finite().array(),
    # loiterZones: a.default.number().finite().array().optional(),
    # stationary: a.default.coerce.boolean(),
    # attributes: a.default.record(a.default.unknown()).nullable().optional(),
    # coord3d: a.default.number().finite().array(),
    # faceEmbed: a.default.number().finite().array().optional(),
    # matchedId: a.default.number().optional(),
    # firstShownTimeMs: a.default.number().finite().optional(),
    # idleSinceTimeMs: a.default.number().finite().optional()
    ###


    # API for subclasses - Smart Detect Events
    def _cleanup_old_analytics_events(self) -> None:
        """
        Remove analytics events older than 1 hour from history.
        Also deletes cached snapshot files for old events.
        Called at the start of each new motion event to keep memory usage bounded.
        """
        current_time = time.time()
        one_hour_ago = current_time - 3600  # 1 hour in seconds
        
        # Find events to remove
        events_to_remove = []
        for event_id, event_data in self._analytics_event_history.items():
            # Use end_time if event is completed, otherwise use start_time
            event_time = event_data.get('end_time') or event_data.get('start_time', 0)
            if event_time < one_hour_ago:
                events_to_remove.append(event_id)
        
        # Remove old events and delete their cached snapshot files
        for event_id in events_to_remove:
            event_data = self._analytics_event_history[event_id]
            
            # Delete cached snapshot files
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
        """
        Clean up smart detect events that ended more than 60 minutes ago.
        Called lazily when new smart detect events are added.
        """
        current_time = time.time()
        sixty_minutes_ago = current_time - 3600  # 60 minutes in seconds
        
        # Find ended events to remove
        events_to_remove = []
        for event_id, event_data in self._active_smart_events.items():
            end_time = event_data.get('end_time')
            # Only clean up events that have ended and are older than 60 minutes
            if end_time and end_time < sixty_minutes_ago:
                events_to_remove.append(event_id)
        
        # Remove old events and delete their cached snapshot files
        for event_id in events_to_remove:
            event_data = self._active_smart_events[event_id]
            
            # Delete cached snapshot files
            for snapshot_key in ['snapshot_crop_path', 'snapshot_fov_path', 'heatmap_path']:
                snapshot_path = event_data.get(snapshot_key)
                if snapshot_path and isinstance(snapshot_path, Path) and snapshot_path.exists():
                    try:
                        snapshot_path.unlink()
                        self.logger.debug(f"Deleted cached snapshot: {snapshot_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cached snapshot {snapshot_path}: {e}")
            
            del self._active_smart_events[event_id]
            self.logger.debug(
                f"Cleaned up smart detect event {event_id} "
                f"(age: {(current_time - end_time) / 60:.1f} minutes)"
            )
        
        if events_to_remove:
            self.logger.info(
                f"Cleaned up {len(events_to_remove)} old smart detect events. "
                f"Remaining: {len(self._active_smart_events)}"
            )

    # API for subclasses - Smart Detect Events
    async def trigger_smart_detect_start(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
    ) -> int:
        """
        Start a smart detect event for a specific object type.
        
        Args:
            object_type: The type of object detected (person, vehicle, etc.)
            custom_descriptor: Optional descriptor data (bounding box, etc.)
            event_timestamp: Optional timestamp for the event
            
        Returns:
            The UniFi event ID for this smart detect event
        """
        current_time = time.time()
        
        # Compose a globally-unique event ID using epoch milliseconds plus a local counter.
        # Embedding time reduces collisions across restarts/instances while keeping a small
        # incrementing counter for uniqueness within the same millisecond.
        epoch_ms = int(time.time() * 1000)
        event_id = epoch_ms * 1000 + (self._motion_event_id % 1000)
        self._motion_event_id += 1
        
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
        
        # Build descriptors array
        descriptors = []
        if custom_descriptor:
            descriptors = [custom_descriptor]
        
        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "enter",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 99},
            "objectTypes": [object_type.value],
            "zonesStatus": {"1": {"score": 99}},
            "smartDetectSnapshot": "smartDetectSnapshotline209.png",
            "displayTimeoutMSec": 10000,
            "motionHeatmap": "motionHeatmapline211.png",
            "motionSnapshot": "motionSnapshotline212.png",
            "smartDetectZoneSnapshot": "smartDetectZoneSnapshotline326.png",
            "snapshot": "smartDetectSnapshotline326.png",
            "descriptors": descriptors,
        }
        
        self.logger.info(
            f"Starting smart detect event {event_id} for {object_type.value} "
            f"(active smart events: {len(self._active_smart_events)})"
        )
        
        await self.send(
            self.gen_response("EventSmartDetect", payload=payload)
        )
        
        # Clean up old smart detect events (keep for 60 minutes after end)
        self._cleanup_old_smart_events()
        
        # Track this smart detect event
        self._active_smart_events[event_id] = {
            "object_type": object_type,
            "start_time": current_time,
            "end_time": None,  # Will be set when event ends
            "event_timestamp": event_timestamp,
            "last_descriptor": custom_descriptor,
            # UniFi Protect requests three snapshot types - store cached file paths:
            "snapshot_crop_path": None,  # motionSnapshot - cropped with bounding box
            "snapshot_fov_path": None,   # motionSnapshotFullFoV - full size with bounding box
            "heatmap_path": None,        # motionHeatmap - heatmap visualization
        }
        
        # If there's an active analytics event, associate this smart detect event with it
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
        
        # Update legacy compatibility fields
        self._motion_event_ts = current_time
        self._motion_object_type = object_type
        self._motion_last_descriptor = custom_descriptor
        
        return event_id

    async def trigger_smart_detect_update(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """
        Send a smart detect update (moving) event with updated descriptor information.
        
        Args:
            object_type: The type of object to update
            custom_descriptor: Updated descriptor data (bounding box, etc.)
            event_timestamp: Optional timestamp for the event
        """
        # Find the active smart detect event with matching object type
        event_id = None
        for eid, event in self._active_smart_events.items():
            if event["object_type"] == object_type:
                event_id = eid
                break
        
        if event_id is None:
            self.logger.warning(
                f"trigger_smart_detect_update called for {object_type.value} "
                f"but no active event found. Event may have already ended or never started. Ignoring."
            )
            return
        
        active_event = self._active_smart_events[event_id]
        
        # Build descriptors array
        descriptors = []
        if custom_descriptor:
            descriptors = [custom_descriptor]
            # Update the stored descriptor for this event
            active_event["last_descriptor"] = custom_descriptor
            self._motion_last_descriptor = custom_descriptor  # Legacy compatibility
        
        payload: dict[str, Any] = {
            "clockBestMonotonic": int(self.get_uptime()),
            "clockBestWall": int(round(active_event["start_time"] * 1000)),
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "moving",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 48},
            "objectTypes": [object_type.value],
            "zonesStatus": {"1": {"score": 75}},
            "smartDetectSnapshot": "smartDetectSnapshotline399.png",
            "displayTimeoutMSec": 10000,
            "descriptors": descriptors,
            "motionHeatmap": "motionHeatmapline402.png",
            "motionSnapshot": "motionSnapshotline403.png",
            "smartDetectZoneSnapshot": "smartDetectZoneSnapshotline413.png",
            "snapshot": "smartDetectSnapshotline414.png",
        }
        
        self.logger.debug(
            f"Updating smart detect event {event_id} for {object_type.value}"
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
    ) -> None:
        """
        Stop a smart detect event for a specific object type.
        
        Args:
            object_type: The type of object to stop detecting
            custom_descriptor: Optional final descriptor data
            event_timestamp: Optional timestamp for the event
            event_id: Optional specific event ID to stop. If not provided, stops the first active event of the given object type.
        """
        # Find the active smart detect event
        target_event_id = event_id
        
        if target_event_id is None:
            # Fallback to finding by object type (legacy behavior)
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
        
        # Build descriptors array - use custom_descriptor if provided, 
        # otherwise fall back to last saved descriptor for this event
        descriptors = []
        if custom_descriptor:
            descriptors = [custom_descriptor]
        elif active_event["last_descriptor"]:
            descriptors = [active_event["last_descriptor"]]
        
        payload: dict[str, Any] = {
            "clockBestMonotonic": int(self.get_uptime()),
            "clockBestWall": int(round(active_event["start_time"] * 1000)),
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "leave",
            "eventId": target_event_id,
            "eventType": "motion",
            "levels": {"0": 49},
            "objectTypes": [object_type.value],
            "zonesStatus": {"1": {"score": 75}},
            "smartDetectSnapshot": "smartDetectSnapshotline472.jpg",
            "displayTimeoutMSec": 2000,
            "descriptors": descriptors,
            "motionHeatmap": "motionHeatmapline477.png",
            "motionSnapshot": "motionSnapshotline478.png",
            "smartDetectZoneSnapshot": "smartDetectZoneSnapshotline481.png",
            "snapshot": "smartDetectSnapshotline481.png",
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
        
        # Mark this smart detect event as ended (keep in memory for 60 minutes)
        active_event["end_time"] = time.time()
        
        # Update legacy compatibility fields
        # Check if there are any other active (not ended) smart detect events
        active_smart_events = [e for e in self._active_smart_events.values() if e.get("end_time") is None]
        if not active_smart_events:
            # No more active smart detect events
            self._motion_object_type = None
            self._motion_last_descriptor = None
            # Only clear motion_event_ts if no analytics event is active
            if self._active_analytics_event_id is None:
                self._motion_event_ts = None

    # API for subclasses - Analytics (Motion) Events
    async def trigger_analytics_start(
        self,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """
        Start a generic analytics motion event.
        
        Args:
            event_timestamp: Optional timestamp for the event
        """
        # Clean up old events before starting a new one
        self._cleanup_old_analytics_events()
        
        current_time = time.time()
        
        # Get the next available event ID and increment counter
        epoch_ms = int(time.time() * 1000)
        event_id = epoch_ms * 1000 + (self._motion_event_id % 1000)
        self._motion_event_id += 1
        
        # Check if we already have an active analytics event
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
        
        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "start",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 47},
            "motionHeatmap": "motionHeatmapline263.png",
            "motionSnapshot": "motionSnapshotline264.png",
        }
        
        self.logger.info(
            f"Starting analytics event {event_id} "
            f"(active smart events: {len(self._active_smart_events)})"
        )
        
        await self.send(
            self.gen_response("EventAnalytics", payload=payload)
        )
        
        # Track this analytics event in history
        self._analytics_event_history[event_id] = {
            "event_id": event_id,
            "start_time": current_time,
            "end_time": None,  # Will be set when event stops
            "event_timestamp": event_timestamp,
            # Snapshot filenames to be used in EventAnalytics stop payload
            "snapshot_filename": None,
            "snapshot_fov_filename": None,
            "heatmap_filename": None,
            # Cached snapshot file paths for serving GetRequest
            "snapshot_crop_path": None,
            "snapshot_fov_path": None,
            "heatmap_path": None,
            # Track smart detect events that occurred during this analytics event
            "smart_detect_event_ids": [],
        }
        self._active_analytics_event_id = event_id
        
        # Update legacy compatibility fields
        if not self._motion_event_ts:  # Only set if not already set by smart detect
            self._motion_event_ts = current_time

    async def trigger_analytics_stop(
        self,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """
        Stop the active analytics motion event.
        
        Handles snapshot caching:
        - If smart detect events occurred during this analytics event, uses the most recent
          smart detect event's cached snapshots
        - Otherwise, fetches fresh snapshots and caches them
        
        Args:
            event_timestamp: Optional timestamp for the event
        """
        # Get the event ID from the active analytics event
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
        
        # Determine which snapshots to use
        snapshot_crop_path = None
        snapshot_fov_path = None
        heatmap_path = None
        
        # Check if any smart detect events occurred during this analytics event
        smart_detect_ids = active_event.get("smart_detect_event_ids", [])
        if smart_detect_ids:
            # Use snapshots from the most recent smart detect event
            most_recent_smart_id = smart_detect_ids[-1]  # Last element is most recent
            
            # Look up in _active_smart_events (includes ended events kept for 60 min)
            smart_event = self._active_smart_events.get(most_recent_smart_id)
            if smart_event:
                snapshot_crop_path = smart_event.get("snapshot_crop_path")
                snapshot_fov_path = smart_event.get("snapshot_fov_path")
                heatmap_path = smart_event.get("heatmap_path")
                self.logger.info(
                    f"Using snapshots from smart detect event {most_recent_smart_id} "
                    f"for analytics event {event_id}"
                )
            else:
                self.logger.warning(
                    f"Smart detect event {most_recent_smart_id} not found in history "
                    f"(may have been cleaned up after 60 minutes)"
                )
        
        # If no smart detect snapshots available, fetch fresh snapshots
        if not snapshot_crop_path or not snapshot_fov_path or not heatmap_path:
            self.logger.info(
                f"No smart detect snapshots available for analytics event {event_id}, "
                f"fetching fresh snapshots"
            )
            try:
                snapshot_crop_path, snapshot_fov_path, heatmap_path = await self.fetch_snapshots_for_event(
                    event_id, "analytics"
                )
                self.logger.info(
                    f"Fetched fresh snapshots for analytics event {event_id}: "
                    f"crop={snapshot_crop_path is not None}, "
                    f"fov={snapshot_fov_path is not None}, "
                    f"heatmap={heatmap_path is not None}"
                )
            except Exception as e:
                self.logger.error(f"Error fetching snapshots for analytics event {event_id}: {e}")
                # Continue without snapshots
        
        # Store cached paths in the event history for later GetRequest handling
        active_event["snapshot_crop_path"] = snapshot_crop_path
        active_event["snapshot_fov_path"] = snapshot_fov_path
        active_event["heatmap_path"] = heatmap_path
        
        # Generate filenames using the cached file paths
        # These will be passed to UniFi Protect and then matched in GetRequest
        snapshot_filename = str(snapshot_crop_path) if snapshot_crop_path else f"snapshot_{event_id}.jpg"
        snapshot_fov_filename = str(snapshot_fov_path) if snapshot_fov_path else f"snapshot_fov_{event_id}.jpg"
        heatmap_filename = str(heatmap_path) if heatmap_path else f"heatmap_{event_id}.jpg"
        
        # Store filenames and end time in the event history
        active_event["snapshot_filename"] = snapshot_filename
        active_event["snapshot_fov_filename"] = snapshot_fov_filename
        active_event["heatmap_filename"] = heatmap_filename
        active_event["end_time"] = time.time()
        
        payload: dict[str, Any] = {
            "clockBestMonotonic": int(self.get_uptime()),
            "clockBestWall": int(round(active_event["start_time"] * 1000)),
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": int(round(time.time() * 1000)),
            "edgeType": "stop",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 49},
            "motionHeatmap": heatmap_filename,
            "motionSnapshot": snapshot_filename,
            "motionSnapshotFullFoV": snapshot_fov_filename,
        }
        
        duration = time.time() - active_event["start_time"]
        self.logger.info(
            f"Stopping analytics event {event_id} (duration: {duration:.1f}s, "
            f"smart_detect_events: {len(smart_detect_ids)})"
        )
        self.logger.debug(
            f"Analytics event snapshots: crop={snapshot_filename}, fov={snapshot_fov_filename}, heatmap={heatmap_filename}"
        )
        
        await self.send(
            self.gen_response("EventAnalytics", payload=payload)
        )
        
        # Mark as no longer active (but keep in history for snapshot retrieval)
        self._active_analytics_event_id = None
        
        # Update legacy compatibility fields
        if not self._active_smart_events:
            # No smart detect events either, fully clear state
            self._motion_event_ts = None


    def get_active_events_summary(self) -> dict[str, Any]:
        """
        Get a summary of currently active motion events.
        Useful for debugging and monitoring event state.
        """
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
        """
        Stop all active motion events (both analytics and smart detect).
        Useful during cleanup or when forcing a reset of event state.
        """
        # Stop all smart detect events
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
        
        # Stop analytics event if active
        if self._active_analytics_event_id is not None:
            self.logger.info("Force stopping analytics event")
            try:
                await self.trigger_motion_stop()
            except Exception as e:
                self.logger.error(f"Error stopping analytics event: {e}")

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
        
        # Probe video resolutions only for streams that are actually configured
        # video1 is required, video2 and video3 use their defaults if not probed
        video1_source = None
        for stream_index in ["video1", "video2", "video3"]:
            try:
                source = await self.get_stream_source(stream_index)
                # Only probe if we got a valid source
                if source:
                    # For video1, always probe
                    if stream_index == "video1":
                        video1_source = source
                        width, height = self.probe_video_resolution(stream_index, source)
                        self._detected_resolutions[stream_index] = (width, height)
                    # For video2/video3, only probe if source is different from video1 (not a fallback)
                    elif source != video1_source:
                        width, height = self.probe_video_resolution(stream_index, source)
                        self._detected_resolutions[stream_index] = (width, height)
                    else:
                        # Stream is using video1 as fallback, skip probing
                        self.logger.debug(f"{stream_index} using video1 source as fallback, using default resolution")
            except NotImplementedError:
                # If get_stream_source is not implemented, skip probing this stream
                self.logger.debug(f"{stream_index} not implemented, using defaults")
                break  # No need to try other streams if method not implemented
            except Exception as e:
                # If stream probe fails, use the default resolution for that stream
                if stream_index == "video1":
                    # video1 is required, so keep the default
                    self.logger.warning(f"Could not probe {stream_index}: {e}, using defaults")
                else:
                    # For video2/video3, silently use their default resolutions
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
                    "mac": self.args.mac,
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
                # Parse the new version string from the upgrade binary
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
        return {
            "from": "ubnt_avclient",
            "functionName": name,
            "inResponseTo": response_to,
            "messageId": self.gen_msg_id(),
            "payload": payload,
            "responseExpected": False,
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

        # Add extra details for GetRequest messages
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
            res = self.gen_response(
                "ChangeSmartMotionSettings", response_to=m["messageId"]
            )
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
        self.logger.info("Cleaning up instance")
        await self.stop_all_motion_events()
        self.close_streams()


    # Legacy API for subclasses - backwards compatibility
    async def trigger_motion_start(
        self,
        object_type: Optional[SmartDetectObjectType] = None,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """
        Start a motion event. Supports both generic motion (EventAnalytics) and 
        smart detect events (EventSmartDetect with object_type).
        
        DEPRECATED: Use trigger_analytics_start() or trigger_smart_detect_start() instead.
        
        All events use a globally unique event ID counter that increments
        for each new event regardless of type.
        """
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
        """
        Send a motion update (moving) event with updated descriptor information.
        Only applicable to SmartDetect events (not generic EventAnalytics).
        
        DEPRECATED: Use trigger_smart_detect_update() instead.
        
        Args:
            custom_descriptor: Updated descriptor data (bounding box, etc.)
            event_timestamp: Optional timestamp for the event
            object_type: Optional object type to update. If not provided, uses the
                        most recent active smart detect event (legacy behavior).
        """
        # Determine which event to update
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
        """
        Stop a motion event. Can stop either a generic motion event (EventAnalytics) 
        or a specific smart detect event (EventSmartDetect).
        
        DEPRECATED: Use trigger_analytics_stop() or trigger_smart_detect_stop() instead.
        
        Args:
            custom_descriptor: Optional final descriptor data
            event_timestamp: Optional timestamp for the event
            object_type: If provided, stops a specific smart detect event. 
                        If None, stops the generic analytics event.
        """
        if object_type:
            await self.trigger_smart_detect_stop(object_type, custom_descriptor, event_timestamp)
        else:
            await self.trigger_analytics_stop(event_timestamp)

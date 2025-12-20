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

AVClientRequest = AVClientResponse = dict[str, Any]


class SmartDetectObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"


class UnifiCamBase(metaclass=ABCMeta):
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
                # Structure: Optional[dict[str, Any]] with keys:
                #   - event_id (int): Unique event identifier
                #   - start_time (float): Event start time
                #   - event_timestamp (Optional[float]): Event timestamp
                #   - snapshot_filename (Optional[str]): Filename for motionSnapshot
                #   - snapshot_fov_filename (Optional[str]): Filename for motionSnapshotFullFoV
                #   - heatmap_filename (Optional[str]): Filename for motionHeatmap
                _active_analytics_event: Current generic motion (EventAnalytics) event state
                # Structure: dict[int, dict[str, Any]] - keyed by event_id, values contain:
                #   - event_id (int): Unique event identifier
                #   - object_type (SmartDetectObjectType): Detected object classification
                #   - timestamp (float): Event start time
                #   - descriptor (dict): Additional event metadata
                _active_smart_events: Active smart detection events supporting concurrent detections
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
        self._active_analytics_event: Optional[dict[str, Any]] = None  # Generic motion event
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

    def get_extra_ffmpeg_args(self, stream_index: str = "") -> str:
        return self.args.ffmpeg_args

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
    async def trigger_smart_detect_start(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """
        Start a smart detect event for a specific object type.
        
        Args:
            object_type: The type of object detected (person, vehicle, etc.)
            custom_descriptor: Optional descriptor data (bounding box, etc.)
            event_timestamp: Optional timestamp for the event
        """
        current_time = time.time()
        
        # Get the next available event ID and increment counter
        event_id = self._motion_event_id
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
            return
        
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
            "zonesStatus": {"0": {"score": 99}},
            "smartDetectSnapshot": "smartDetectSnapshotline209.png",
            "displayTimeoutMSec": 5000,
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
        
        # Track this smart detect event
        self._active_smart_events[event_id] = {
            "object_type": object_type,
            "start_time": current_time,
            "event_timestamp": event_timestamp,
            "last_descriptor": custom_descriptor,
            # UniFi Protect requests three snapshot types:
            "snapshot_crop": None,  # motionSnapshot - cropped with bounding box
            "snapshot_fov": None,   # motionSnapshotFullFoV - full size with bounding box
            "heatmap": None,        # motionHeatmap - heatmap visualization
        }
        
        # Update legacy compatibility fields
        self._motion_event_ts = current_time
        self._motion_object_type = object_type
        self._motion_last_descriptor = custom_descriptor

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
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "moving",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 48},
            "objectTypes": [object_type.value],
            "zonesStatus": {"0": {"score": 48}},
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
    ) -> None:
        """
        Stop a smart detect event for a specific object type.
        
        Args:
            object_type: The type of object to stop detecting
            custom_descriptor: Optional final descriptor data
            event_timestamp: Optional timestamp for the event
        """
        # Find the active smart detect event by object type
        event_id = None
        for eid, event in self._active_smart_events.items():
            if event["object_type"] == object_type:
                event_id = eid
                break
        
        if event_id is None:
            self.logger.warning(
                f"trigger_smart_detect_stop called for {object_type.value} "
                f"but no active event found. Event may have already ended or never started. Ignoring."
            )
            return
        
        active_event = self._active_smart_events[event_id]
        
        # Build descriptors array - use custom_descriptor if provided, 
        # otherwise fall back to last saved descriptor for this event
        descriptors = []
        if custom_descriptor:
            descriptors = [custom_descriptor]
        elif active_event["last_descriptor"]:
            descriptors = [active_event["last_descriptor"]]
        
        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "leave",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 49},
            "objectTypes": [object_type.value],
            "zonesStatus": {"0": {"score": 48}},
            "smartDetectSnapshot": "smartDetectSnapshotline472.jpg",
            "displayTimeoutMSec": 1000,
            "descriptors": descriptors,
            "motionHeatmap": "motionHeatmapline477.png",
            "motionSnapshot": "motionSnapshotline478.png",
            "smartDetectZoneSnapshot": "smartDetectZoneSnapshotline481.png",
            "snapshot": "smartDetectSnapshotline481.png",
        }
        
        duration = time.time() - active_event["start_time"]
        self.logger.info(
            f"Stopping smart detect event {event_id} for {object_type.value} "
            f"(duration: {duration:.1f}s, remaining smart events: "
            f"{len(self._active_smart_events) - 1})"
        )
        
        await self.send(
            self.gen_response("EventSmartDetect", payload=payload)
        )
        
        # Clean up this smart detect event
        del self._active_smart_events[event_id]
        
        # Update legacy compatibility fields
        if not self._active_smart_events:
            # No more smart detect events
            self._motion_object_type = None
            self._motion_last_descriptor = None
            # Only clear motion_event_ts if no analytics event is active
            if self._active_analytics_event is None:
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
        current_time = time.time()
        
        # Get the next available event ID and increment counter
        event_id = self._motion_event_id
        self._motion_event_id += 1
        
        # Check if we already have an active analytics event
        if self._active_analytics_event is not None:
            existing_start = self._active_analytics_event['start_time']
            self.logger.warning(
                f"Analytics event {self._active_analytics_event['event_id']} already active "
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
        
        # Track this analytics event
        self._active_analytics_event = {
            "event_id": event_id,
            "start_time": current_time,
            "event_timestamp": event_timestamp,
            # Snapshot filenames to be used in EventAnalytics stop payload
            "snapshot_filename": None,
            "snapshot_fov_filename": None,
            "heatmap_filename": None,
        }
        
        # Update legacy compatibility fields
        if not self._motion_event_ts:  # Only set if not already set by smart detect
            self._motion_event_ts = current_time

    async def trigger_analytics_stop(
        self,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """
        Stop the active analytics motion event.
        
        Args:
            event_timestamp: Optional timestamp for the event
        """
        # Get the event ID from the active analytics event
        if self._active_analytics_event is None:
            self.logger.warning(
                f"trigger_analytics_stop called but no active event found. "
                f"Event may have already ended or never started. Ignoring."
            )
            return
        
        event_id = self._active_analytics_event["event_id"]
        
        # Generate URL-based filenames using Frigate API pattern
        # UniFi expects different filename for FoV, but same base for snapshot and heatmap
        current_timestamp = int(time.time())
        snapshot_filename = f"latest.jpg?timestamp={current_timestamp}"
        snapshot_fov_filename = f"latest_fullfov.jpg?timestamp={current_timestamp}"
        heatmap_filename = f"latest.jpg?timestamp={current_timestamp}"
        
        # Store filenames in the active event for later retrieval
        self._active_analytics_event["snapshot_filename"] = snapshot_filename
        self._active_analytics_event["snapshot_fov_filename"] = snapshot_fov_filename
        self._active_analytics_event["heatmap_filename"] = heatmap_filename
        
        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self.get_uptime()),
            "clockStream": int(self.get_uptime()),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "stop",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 49},
            "motionHeatmap": heatmap_filename,
            "motionSnapshot": snapshot_filename,
            "motionSnapshotFullFoV": snapshot_fov_filename,
        }
        
        duration = time.time() - self._active_analytics_event["start_time"]
        self.logger.info(
            f"Stopping analytics event {event_id} (duration: {duration:.1f}s)"
        )
        
        await self.send(
            self.gen_response("EventAnalytics", payload=payload)
        )
        
        # Clean up analytics event
        self._active_analytics_event = None
        
        # Update legacy compatibility fields
        if not self._active_smart_events:
            # No smart detect events either, fully clear state
            self._motion_event_ts = None

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

    def update_motion_snapshot(self, path: Path) -> None:
        """
        Update motion snapshot (legacy method).
        By default, updates all three snapshot types to the same path.
        For more granular control, use update_motion_snapshots().
        """
        self._motion_snapshot = path
        self._motion_snapshot_crop = path
        self._motion_snapshot_fov = path
        self._motion_heatmap = path
    
    def update_motion_snapshots(
        self,
        crop: Optional[Path] = None,
        fov: Optional[Path] = None,
        heatmap: Optional[Path] = None,
    ) -> None:
        """
        Update specific motion snapshot types.
        
        Args:
            crop: Path to cropped snapshot with bounding box (motionSnapshot)
            fov: Path to full field-of-view snapshot with bounding box (motionSnapshotFullFoV)
            heatmap: Path to heatmap visualization (motionHeatmap)
        """
        if crop is not None:
            self._motion_snapshot_crop = crop
            self._motion_snapshot = crop  # Update legacy field
        if fov is not None:
            self._motion_snapshot_fov = fov
        if heatmap is not None:
            self._motion_heatmap = heatmap
    
    def get_active_events_summary(self) -> dict[str, Any]:
        """
        Get a summary of currently active motion events.
        Useful for debugging and monitoring event state.
        """
        return {
            "analytics_event": {
                "active": self._active_analytics_event is not None,
                "event_id": self._active_analytics_event["event_id"] if self._active_analytics_event else None,
                "duration": time.time() - self._active_analytics_event["start_time"] 
                    if self._active_analytics_event else None,
            },
            "smart_detect_events": {
                event_id: {
                    "object_type": event["object_type"].value,
                    "duration": time.time() - event["start_time"],
                    "has_descriptor": event["last_descriptor"] is not None,
                    "has_snapshot_crop": event["snapshot_crop"] is not None,
                    "has_snapshot_fov": event["snapshot_fov"] is not None,
                    "has_heatmap": event["heatmap"] is not None,
                }
                for event_id, event in self._active_smart_events.items()
            },
            "total_active_events": (1 if self._active_analytics_event else 0) + len(self._active_smart_events),
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
        if self._active_analytics_event is not None:
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

    def probe_video_resolution(self, stream_index: str, source_url: str) -> tuple[int, int]:
        """Probe video source to detect width and height using ffprobe"""
        # Get default resolution for this stream
        default_width, default_height = self._detected_resolutions[stream_index]
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json',
                '-rtsp_transport', self.args.rtsp_transport,
                source_url
            ]
            self.logger.info(f"Probing {stream_index} source: {source_url}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=15
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get('streams') and len(data['streams']) > 0:
                    width = data['streams'][0].get('width', default_width)
                    height = data['streams'][0].get('height', default_height)
                    self.logger.info(f"Detected {stream_index} resolution: {width}x{height}")
                    return width, height
                    
        except subprocess.TimeoutExpired:
            self.logger.warning(f"{stream_index} probe timed out after 15 seconds, using defaults")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Could not parse ffprobe output for {stream_index}: {e}, using defaults")
        except Exception as e:
            self.logger.warning(f"Could not probe {stream_index} source: {e}, using defaults")
        
        # Fallback to defaults for this stream
        self.logger.info(f"Using default resolution for {stream_index}: {default_width}x{default_height}")
        return default_width, default_height

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

    async def process_hello(self, msg: AVClientRequest) -> None:
        pass

    async def process_param_agreement(self, msg: AVClientRequest) -> AVClientResponse:
        return self.gen_response(
            "ubnt_avclient_paramAgreement",
            msg["messageId"],
            {
                "authToken": self.args.token,
                "features": await self.get_feature_flags(),
            },
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

    async def process_isp_settings(self, msg: AVClientRequest) -> AVClientResponse:
        payload = {
            "aeMode": "auto",
            "aeTargetPercent": 50,
            "aggressiveAntiFlicker": 0,
            "brightness": 50,
            "contrast": 50,
            "criticalTmpOfProtect": 40,
            "darkAreaCompensateLevel": 0,
            "denoise": 50,
            "enable3dnr": 1,
            "enableMicroTmpProtect": 1,
            "enablePauseMotion": 0,
            "flip": 0,
            "focusMode": "ztrig",
            "focusPosition": 0,
            "forceFilterIrSwitchEvents": 0,
            "hue": 50,
            "icrLightSensorNightThd": 0,
            "icrSensitivity": 0,
            "irLedLevel": 215,
            "irLedMode": "auto",
            "irOnStsBrightness": 0,
            "irOnStsContrast": 0,
            "irOnStsDenoise": 0,
            "irOnStsHue": 0,
            "irOnStsSaturation": 0,
            "irOnStsSharpness": 0,
            "irOnStsWdr": 0,
            "irOnValBrightness": 50,
            "irOnValContrast": 50,
            "irOnValDenoise": 50,
            "irOnValHue": 50,
            "irOnValSaturation": 50,
            "irOnValSharpness": 50,
            "irOnValWdr": 1,
            "mirror": 0,
            "queryIrLedStatus": 0,
            "saturation": 50,
            "sharpness": 50,
            "touchFocusX": 1001,
            "touchFocusY": 1001,
            "wdr": 1,
            "zoomPosition": 0,
        }
        payload.update(await self.get_video_settings())
        return self.gen_response(
            "ResetIspSettings",
            msg["messageId"],
            payload,
        )
  
    async def process_video_settings(self, msg: AVClientRequest) -> AVClientResponse:
        vid_dst = {
            "video1": ["file:///dev/null"],
            "video2": ["file:///dev/null"],
            "video3": ["file:///dev/null"],
        }

        if msg["payload"] is not None and "video" in msg["payload"]:
            for k, v in msg["payload"]["video"].items():
                if v:
                    if "avSerializer" in v:
                        vid_dst[k] = v["avSerializer"]["destinations"]
                        # Check if any destination contains /dev/null (means stop stream)
                        if any("/dev/null" in dest for dest in vid_dst[k]):
                            self.stop_video_stream(k)
                            # Remove stream from tracking when stopping
                            if k in self._streams:
                                del self._streams[k]
                        elif "parameters" in v["avSerializer"]:
                            self._streams[k] = stream = v["avSerializer"]["parameters"][
                                "streamName"
                            ]
                            try:
                                host, port = urlparse(
                                    v["avSerializer"]["destinations"][0]
                                ).netloc.split(":")
                                await self.start_video_stream(
                                    k, stream, destination=(host, int(port))
                                )
                            except ValueError:
                                pass

        return self.gen_response(
            "ChangeVideoSettings",
            msg["messageId"],
            {
                "audio": {
                    "bitRate": 32000,
                    "channels": 1,
                    "description": "audio track",
                    "enableTemporalNoiseShaping": False,
                    "enabled": True,
                    "mode": 0,
                    "quality": 0,
                    "sampleRate": 11025,
                    "type": "aac",
                    "volume": 0,
                },
                "firmwarePath": "/lib/firmware/",
                "video": {
                    "enableHrd": False,
                    "hdrMode": 0,
                    "lowDelay": False,
                    "videoMode": "default",
                    "mjpg": {
                        "avSerializer": {
                            "destinations": [
                                "file:///tmp/snap.jpeg",
                                "file:///tmp/snap_av.jpg",
                            ],
                            "parameters": {
                                "audioId": 1000,
                                "enableTimestampsOverlapAvoidance": False,
                                "suppressAudio": True,
                                "suppressVideo": False,
                                "videoId": 1001,
                            },
                            "type": "mjpg",
                        },
                        "bitRateCbrAvg": 500000,
                        "bitRateVbrMax": 500000,
                        "bitRateVbrMin": None,
                        "description": "JPEG pictures",
                        "enabled": True,
                        "fps": 5,
                        "height": 720,
                        "isCbr": False,
                        "maxFps": 5,
                        "minClientAdaptiveBitRate": 0,
                        "minMotionAdaptiveBitRate": 0,
                        "nMultiplier": None,
                        "name": "mjpg",
                        "quality": 80,
                        "sourceId": 3,
                        "streamId": 8,
                        "streamOrdinal": 3,
                        "type": "mjpg",
                        "validBitrateRangeMax": 6000000,
                        "validBitrateRangeMin": 32000,
                        "width": 1280,
                    },
                    "video1": {
                        "M": 1,
                        "N": 30,
                        "avSerializer": {
                            "destinations": vid_dst["video1"],
                            "parameters": (
                                None
                                if "video1" not in self._streams
                                else {
                                    "audioId": None,
                                    "streamName": self._streams["video1"],
                                    "suppressAudio": None,
                                    "suppressVideo": None,
                                    "videoId": None,
                                }
                            ),
                            "type": "extendedFlv",
                        },
                        "bitRateCbrAvg": 8192000,
                        "bitRateVbrMax": 2800000,
                        "bitRateVbrMin": 48000,
                        "description": "Hi quality video track",
                        "enabled": True,
                        "fps": 20,
                        "gopModel": 0,
                        "height": self._detected_resolutions["video1"][1],
                        "horizontalFlip": False,
                        "isCbr": False,
                        "maxFps": 30,
                        "minClientAdaptiveBitRate": 0,
                        "minMotionAdaptiveBitRate": 0,
                        "nMultiplier": 6,
                        "name": "video1",
                        "sourceId": 0,
                        "streamId": 1,
                        "streamOrdinal": 0,
                        "type": "h264",
                        "validBitrateRangeMax": 2800000,
                        "validBitrateRangeMin": 32000,
                        "validFpsValues": [
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            8,
                            9,
                            10,
                            12,
                            15,
                            16,
                            18,
                            20,
                            24,
                            25,
                            30,
                        ],
                        "verticalFlip": False,
                        "width": self._detected_resolutions["video1"][0],
                    },
                    "video2": {
                        "M": 1,
                        "N": 30,
                        "avSerializer": {
                            "destinations": vid_dst["video2"],
                            "parameters": (
                                None
                                if "video2" not in self._streams
                                else {
                                    "audioId": None,
                                    "streamName": self._streams["video2"],
                                    "suppressAudio": None,
                                    "suppressVideo": None,
                                    "videoId": None,
                                }
                            ),
                            "type": "extendedFlv",
                        },
                        "bitRateCbrAvg": 1024000,
                        "bitRateVbrMax": 1200000,
                        "bitRateVbrMin": 48000,
                        "currentVbrBitrate": 1200000,
                        "description": "Medium quality video track",
                        "enabled": True,
                        "fps": 10,
                        "gopModel": 0,
                        "height": self._detected_resolutions["video2"][1],
                        "horizontalFlip": False,
                        "isCbr": False,
                        "maxFps": 30,
                        "minClientAdaptiveBitRate": 0,
                        "minMotionAdaptiveBitRate": 0,
                        "nMultiplier": 6,
                        "name": "video2",
                        "sourceId": 1,
                        "streamId": 2,
                        "streamOrdinal": 1,
                        "type": "h264",
                        "validBitrateRangeMax": 1500000,
                        "validBitrateRangeMin": 32000,
                        "validFpsValues": [
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            8,
                            9,
                            10,
                            12,
                            15,
                            16,
                            18,
                            20,
                            24,
                            25,
                            30,
                        ],
                        "verticalFlip": False,
                        "width": self._detected_resolutions["video2"][0],
                    },
                    "video3": {
                        "M": 1,
                        "N": 30,
                        "avSerializer": {
                            "destinations": vid_dst["video3"],
                            "parameters": (
                                None
                                if "video3" not in self._streams
                                else {
                                    "audioId": None,
                                    "streamName": self._streams["video3"],
                                    "suppressAudio": None,
                                    "suppressVideo": None,
                                    "videoId": None,
                                }
                            ),
                            "type": "extendedFlv",
                        },
                        "bitRateCbrAvg": 300000,
                        "bitRateVbrMax": 200000,
                        "bitRateVbrMin": 48000,
                        "currentVbrBitrate": 200000,
                        "description": "Low quality video track",
                        "enabled": True,
                        "fps": 15,
                        "gopModel": 0,
                        "height": self._detected_resolutions["video3"][1],
                        "horizontalFlip": False,
                        "isCbr": False,
                        "maxFps": 30,
                        "minClientAdaptiveBitRate": 0,
                        "minMotionAdaptiveBitRate": 0,
                        "nMultiplier": 6,
                        "name": "video3",
                        "sourceId": 2,
                        "streamId": 4,
                        "streamOrdinal": 2,
                        "type": "h264",
                        "validBitrateRangeMax": 750000,
                        "validBitrateRangeMin": 32000,
                        "validFpsValues": [
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            8,
                            9,
                            10,
                            12,
                            15,
                            16,
                            18,
                            20,
                            24,
                            25,
                            30,
                        ],
                        "verticalFlip": False,
                        "width": self._detected_resolutions["video3"][0],
                    },
                    "vinFps": 30,
                },
            },
        )

    async def process_device_settings(self, msg: AVClientRequest) -> AVClientResponse:
        return self.gen_response(
            "ChangeDeviceSettings",
            msg["messageId"],
            {
                "name": self.args.name,
                "timezone": "PST8PDT,M3.2.0,M11.1.0",
            },
        )

    async def process_osd_settings(self, msg: AVClientRequest) -> AVClientResponse:
        return self.gen_response(
            "ChangeOsdSettings",
            msg["messageId"],
            {
                "_1": {
                    "enableDate": 1,
                    "enableLogo": 1,
                    "enableReportdStatsLevel": 0,
                    "enableStreamerStatsLevel": 0,
                    "tag": self.args.name,
                },
                "_2": {
                    "enableDate": 1,
                    "enableLogo": 1,
                    "enableReportdStatsLevel": 0,
                    "enableStreamerStatsLevel": 0,
                    "tag": self.args.name,
                },
                "_3": {
                    "enableDate": 1,
                    "enableLogo": 1,
                    "enableReportdStatsLevel": 0,
                    "enableStreamerStatsLevel": 0,
                    "tag": self.args.name,
                },
                "_4": {
                    "enableDate": 1,
                    "enableLogo": 1,
                    "enableReportdStatsLevel": 0,
                    "enableStreamerStatsLevel": 0,
                    "tag": self.args.name,
                },
                "enableOverlay": 1,
                "logoScale": 50,
                "overlayColorId": 0,
                "textScale": 50,
                "useCustomLogo": 0,
            },
        )

    async def process_network_status(self, msg: AVClientRequest) -> AVClientResponse:
        return self.gen_response(
            "NetworkStatus",
            msg["messageId"],
            {
                "connectionState": 2,
                "connectionStateDescription": "CONNECTED",
                "defaultInterface": "eth0",
                "dhcpLeasetime": 86400,
                "dnsServer": "8.8.8.8 4.2.2.2",
                "gateway": "192.168.103.1",
                "ipAddress": self.args.ip,
                "linkDuplex": 1,
                "linkSpeedMbps": 100,
                "mode": "dhcp",
                "networkMask": "255.255.255.0",
            },
        )

    async def process_sound_led_settings(
        self, msg: AVClientRequest
    ) -> AVClientResponse:
        return self.gen_response(
            "ChangeSoundLedSettings",
            msg["messageId"],
            {
                "ledFaceAlwaysOnWhenManaged": 1,
                "ledFaceEnabled": 1,
                "speakerEnabled": 1,
                "speakerVolume": 100,
                "systemSoundsEnabled": 1,
                "userLedBlinkPeriodMs": 0,
                "userLedColorFg": "blue",
                "userLedOnNoff": 1,
            },
        )

    async def process_change_isp_settings(
        self, msg: AVClientRequest
    ) -> AVClientResponse:
        payload = {
            "aeMode": "auto",
            "aeTargetPercent": 50,
            "aggressiveAntiFlicker": 0,
            "brightness": 50,
            "contrast": 50,
            "criticalTmpOfProtect": 40,
            "dZoomCenterX": 50,
            "dZoomCenterY": 50,
            "dZoomScale": 0,
            "dZoomStreamId": 4,
            "darkAreaCompensateLevel": 0,
            "denoise": 50,
            "enable3dnr": 1,
            "enableExternalIr": 0,
            "enableMicroTmpProtect": 1,
            "enablePauseMotion": 0,
            "flip": 0,
            "focusMode": "ztrig",
            "focusPosition": 0,
            "forceFilterIrSwitchEvents": 0,
            "hue": 50,
            "icrLightSensorNightThd": 0,
            "icrSensitivity": 0,
            "irLedLevel": 215,
            "irLedMode": "auto",
            "irOnStsBrightness": 0,
            "irOnStsContrast": 0,
            "irOnStsDenoise": 0,
            "irOnStsHue": 0,
            "irOnStsSaturation": 0,
            "irOnStsSharpness": 0,
            "irOnStsWdr": 0,
            "irOnValBrightness": 50,
            "irOnValContrast": 50,
            "irOnValDenoise": 50,
            "irOnValHue": 50,
            "irOnValSaturation": 50,
            "irOnValSharpness": 50,
            "irOnValWdr": 1,
            "lensDistortionCorrection": 1,
            "masks": None,
            "mirror": 0,
            "queryIrLedStatus": 0,
            "saturation": 50,
            "sharpness": 50,
            "touchFocusX": 1001,
            "touchFocusY": 1001,
            "wdr": 1,
            "zoomPosition": 0,
        }

        if msg["payload"]:
            await self.change_video_settings(msg["payload"])

        payload.update(await self.get_video_settings())
        return self.gen_response("ChangeIspSettings", msg["messageId"], payload)

    async def process_analytics_settings(
        self, msg: AVClientRequest
    ) -> AVClientResponse:
        return self.gen_response(
            "ChangeAnalyticsSettings", msg["messageId"], msg["payload"]
        )

    async def process_snapshot_request(
        self, msg: AVClientRequest
    ) -> Optional[AVClientResponse]:
        snapshot_type = msg["payload"]["what"]
        filename = msg["payload"].get("filename", "")
        
        self.logger.debug(f"Snapshot request: type={snapshot_type}, filename={filename}")
        
        # Check if filename contains URL parameters (indicates Frigate API URL)
        if filename and ("?" in filename or filename.startswith("latest.jpg")):
            # This is a URL-based filename, fetch from Frigate API
            if hasattr(self.args, 'frigate_http_url') and hasattr(self.args, 'frigate_camera'):
                if self.args.frigate_http_url:
                    # Determine query parameters based on snapshot type
                    # motionSnapshot: 360p thumbnail with crop
                    # motionSnapshotFullFoV: full resolution
                    # motionHeatmap: full resolution (same as FoV)
                    if snapshot_type == "motionSnapshot":
                        # Thumbnail version with height and quality parameters
                        query_params = "height=360&quality=80"
                    elif snapshot_type == "motionSnapshotFullFoV":
                        # Full resolution, no additional parameters beyond timestamp
                        query_params = ""
                    elif snapshot_type == "motionHeatmap":
                        # Heatmap uses full resolution
                        query_params = ""
                    else:
                        # Default to no extra parameters
                        query_params = ""
                    
                    # Extract timestamp from filename if present
                    timestamp_param = ""
                    if "timestamp=" in filename:
                        # Extract existing timestamp parameter
                        timestamp_param = filename.split("?", 1)[1] if "?" in filename else ""
                    
                    # Build final query string
                    if query_params and timestamp_param:
                        final_query = f"{query_params}&{timestamp_param}"
                    elif query_params:
                        final_query = query_params
                    elif timestamp_param:
                        final_query = timestamp_param
                    else:
                        final_query = ""
                    
                    # Build the full URL to Frigate
                    # Always use 'latest.jpg' as the base - Frigate API endpoint
                    # (UniFi may send latest_fullfov.jpg, latest_heatmap.jpg, etc. but Frigate only has latest.jpg)
                    if final_query:
                        snapshot_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg?{final_query}"
                    else:
                        snapshot_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
                    
                    self.logger.info(f"Fetching {snapshot_type} from Frigate: {snapshot_url}")
                    
                    try:
                        async with aiohttp.ClientSession() as session:
                            # Fetch from Frigate
                            async with session.get(snapshot_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                                if response.status == 200:
                                    image_data = await response.read()
                                    self.logger.info(f"Fetched {snapshot_type} from Frigate ({len(image_data)} bytes)")
                                    
                                    # Upload to UniFi Protect
                                    files = {"payload": image_data}
                                    files.update(msg["payload"].get("formFields", {}))
                                    
                                    try:
                                        await session.post(
                                            msg["payload"]["uri"],
                                            data=files,
                                            ssl=self._ssl_context,
                                        )
                                        self.logger.debug(f"Uploaded {snapshot_type} from Frigate URL")
                                    except aiohttp.ClientError:
                                        self.logger.exception("Failed to upload snapshot to UniFi Protect")
                                else:
                                    error_body = await response.text()
                                    self.logger.warning(
                                        f"Failed to fetch {snapshot_type} from Frigate: "
                                        f"HTTP {response.status}, Response: {error_body}"
                                    )
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout fetching {snapshot_type} from Frigate")
                    except Exception as e:
                        self.logger.error(f"Error fetching {snapshot_type} from Frigate: {e}")
                else:
                    self.logger.warning(f"URL-based filename but frigate_http_url not configured")
            else:
                self.logger.warning(f"URL-based filename but Frigate configuration not available")
        else:
            # Check if this is a regular snapshot request and Frigate is configured
            if snapshot_type == "snapshot" and hasattr(self.args, 'frigate_http_url') and hasattr(self.args, 'frigate_camera'):
                if self.args.frigate_http_url:
                    # Determine query parameters based on quality
                    quality_param = msg["payload"].get("quality", "medium")
                    
                    # Map UniFi quality levels to height parameters
                    if quality_param == "high":
                        query_params = "height=1080&quality=95"
                    elif quality_param == "medium":
                        query_params = "height=720&quality=85"
                    elif quality_param == "low":
                        query_params = "height=360&quality=70"
                    else:
                        # Default to medium quality
                        query_params = "height=720&quality=85"
                    
                    # Build the full URL to Frigate
                    snapshot_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg?{query_params}"
                    
                    self.logger.info(f"Fetching snapshot (quality={quality_param}) from Frigate: {snapshot_url}")
                    
                    try:
                        async with aiohttp.ClientSession() as session:
                            # Fetch from Frigate
                            async with session.get(snapshot_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                                if response.status == 200:
                                    image_data = await response.read()
                                    self.logger.info(f"Fetched snapshot from Frigate ({len(image_data)} bytes)")
                                    
                                    # Upload to UniFi Protect
                                    files = {"payload": image_data}
                                    files.update(msg["payload"].get("formFields", {}))
                                    
                                    try:
                                        await session.post(
                                            msg["payload"]["uri"],
                                            data=files,
                                            ssl=self._ssl_context,
                                        )
                                        self.logger.debug(f"Uploaded snapshot from Frigate")
                                    except aiohttp.ClientError:
                                        self.logger.exception("Failed to upload snapshot to UniFi Protect")
                                else:
                                    error_body = await response.text()
                                    self.logger.warning(
                                        f"Failed to fetch snapshot from Frigate: "
                                        f"HTTP {response.status}, Response: {error_body}"
                                    )
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout fetching snapshot from Frigate")
                    except Exception as e:
                        self.logger.error(f"Error fetching snapshot from Frigate: {e}")
                else:
                    # Fall back to legacy method if frigate_http_url not configured
                    path = await self.get_snapshot()
                    if path and path.exists():
                        async with aiohttp.ClientSession() as session:
                            files = {"payload": open(path, "rb")}
                            files.update(msg["payload"].get("formFields", {}))
                            try:
                                await session.post(
                                    msg["payload"]["uri"],
                                    data=files,
                                    ssl=self._ssl_context,
                                )
                                self.logger.debug(f"Uploaded snapshot from {path}")
                            except aiohttp.ClientError:
                                self.logger.exception("Failed to upload snapshot")
                    else:
                        self.logger.warning(f"Snapshot file {path} is not ready yet, skipping upload")
            else:
                # Legacy path-based snapshot handling for motion events
                # Select appropriate snapshot based on request type
                if snapshot_type == "motionSnapshot":
                    # Cropped image with bounding box
                    path = self._motion_snapshot_crop or self._motion_snapshot
                elif snapshot_type == "motionSnapshotFullFoV":
                    # Full field of view image with bounding box
                    path = self._motion_snapshot_fov or self._motion_snapshot
                elif snapshot_type == "motionHeatmap":
                    # Heatmap visualization (use FoV as fallback)
                    path = self._motion_heatmap or self._motion_snapshot_fov or self._motion_snapshot
                elif snapshot_type == "smartDetectZoneSnapshot":
                    # Smart detect zone snapshot (use crop)
                    path = self._motion_snapshot_crop or self._motion_snapshot
                else:
                    # Regular snapshot request (fallback to get_snapshot method)
                    path = await self.get_snapshot()

                if path and path.exists():
                    async with aiohttp.ClientSession() as session:
                        files = {"payload": open(path, "rb")}
                        files.update(msg["payload"].get("formFields", {}))
                        try:
                            await session.post(
                                msg["payload"]["uri"],
                                data=files,
                                ssl=self._ssl_context,
                            )
                            self.logger.debug(f"Uploaded {snapshot_type} from {path}")
                        except aiohttp.ClientError:
                            self.logger.exception("Failed to upload snapshot")
                else:
                    self.logger.warning(
                        f"Snapshot file {path} is not ready yet, skipping upload for {snapshot_type}"
                    )

        if msg["responseExpected"]:
            return self.gen_response("GetRequest", response_to=msg["messageId"])

    async def process_time(self, msg: AVClientRequest) -> AVClientResponse:
        return self.gen_response(
            "ubnt_avclient_paramAgreement",
            msg["messageId"],
            {
                "monotonicMs": self.get_uptime(),
                "wallMs": int(round(time.time() * 1000)),
                "features": {},
            },
        )

    async def process_continuous_move(self, msg: AVClientRequest) -> None:
        return

    async def process_update_face_db(self, msg: AVClientRequest) -> AVClientResponse:
        # Return empty response to indicate no face database is available
        # This prevents UniFi Protect from trying to fetch a non-existent file
        return self.gen_response(
            "UpdateFaceDBRequest",
            msg["messageId"],
            {},
        )
    
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

    def get_base_ffmpeg_args(self, stream_index: str = "") -> str:
        if self.args.ffmpeg_base_args is not None:
            return self.args.ffmpeg_base_args

        base_args = [
            "-avoid_negative_ts",
            "make_zero",
            "-fflags",
            "+genpts+discardcorrupt",
            "-use_wallclock_as_timestamps 1",
        ]

        try:
            output = subprocess.check_output(["ffmpeg", "-h", "full"])
            if b"stimeout" in output:
                base_args.append("-stimeout 15000000")
            else:
                base_args.append("-timeout 15000000")
        except subprocess.CalledProcessError:
            self.logger.exception("Could not check for ffmpeg options")

        return " ".join(base_args)

    async def start_video_stream(
        self, stream_index: str, stream_name: str, destination: tuple[str, int]
    ):
        has_spawned = stream_index in self._ffmpeg_handles
        is_dead = has_spawned and self._ffmpeg_handles[stream_index].poll() is not None

        if not has_spawned or is_dead:
            source = await self.get_stream_source(stream_index)
            cmd = (
                f"AV_LOG_FORCE_NOCOLOR=1 ffmpeg -nostdin -loglevel level+{self.args.loglevel} -y"
                f" {self.get_base_ffmpeg_args(stream_index)} -rtsp_transport"
                f' {self.args.rtsp_transport} -i "{source}"'
                f" {self.get_extra_ffmpeg_args(stream_index)} -metadata"
                f" streamName={stream_name} -f {self.args.format} - "
                f" | {sys.executable} -m unifi.clock_sync --timestamp-modifier {self.args.timestamp_modifier}"
                f" | nc"
                f" {destination[0]} {destination[1]}"
            )

            if is_dead:
                exit_code = self._ffmpeg_handles[stream_index].poll()
                self.logger.warning(f"Previous ffmpeg process for {stream_index} died with exit code {exit_code}.")

            self.logger.info(
                f"Spawning ffmpeg for {stream_index} ({stream_name}): {cmd}"
            )
            # Start process in a new process group so we can kill the entire pipeline
            import os
            self._ffmpeg_handles[stream_index] = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL, 
                shell=True,
                preexec_fn=os.setsid  # Create new process group
            )

    def stop_video_stream(self, stream_index: str):
        if stream_index in self._ffmpeg_handles:
            self.logger.info(f"Stopping stream {stream_index}")
            proc = self._ffmpeg_handles[stream_index]
            
            # Check if process is already dead
            if proc.poll() is not None:
                self.logger.debug(f"Process for {stream_index} already terminated with code {proc.poll()}")
                del self._ffmpeg_handles[stream_index]
                return
            
            try:
                # Terminate the process group to kill all processes in the pipeline
                import os
                import signal
                pgid = os.getpgid(proc.pid)
                self.logger.debug(f"Sending SIGTERM to process group {pgid} for {stream_index}")
                os.killpg(pgid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=2)
                    self.logger.debug(f"Stream {stream_index} terminated gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Stream {stream_index} did not terminate gracefully, sending SIGKILL")
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                        proc.wait(timeout=1)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
                        
            except (ProcessLookupError, PermissionError, AttributeError, OSError) as e:
                self.logger.debug(f"Error stopping {stream_index}: {e}, trying proc.kill()")
                # Fall back to killing just the parent process
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            
            # Remove from handles
            del self._ffmpeg_handles[stream_index]

    async def close(self):
        self.logger.info("Cleaning up instance")
        await self.stop_all_motion_events()
        self.close_streams()

    def close_streams(self):
        for stream in self._ffmpeg_handles:
            self.stop_video_stream(stream)

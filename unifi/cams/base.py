"""
unifi/cams/base.py — UniFi Protect camera emulator base class.

Provides the WebSocket protocol, event lifecycle management (analytics motion
events and smart-detect events), snapshot handling, and stream management that
all concrete camera implementations build on.
"""

import argparse
import asyncio
import atexit
import json
import logging
import ssl
import subprocess
import tempfile
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
import websockets

from unifi.core import RetryableError
from unifi.cams.handlers import ProtocolHandlers, VideoStreamHandlers, SnapshotHandlers

AVClientRequest = AVClientResponse = dict[str, Any]

# Module-level PIL availability check — done once, not on every image read.
try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

# How long (seconds) to retain ended events before purging them.
_EVENT_RETENTION_SECS: int = 3600  # 1 hour

# Maximum descriptor history entries kept per smart-detect event.
# Prevents unbounded growth during very long detections.
_MAX_DESCRIPTOR_HISTORY: int = 500


class SmartDetectObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PACKAGE = "package"
    FACE = "face"
    LICENSEPLATE = "licensePlate"


class UnifiCamBase(ProtocolHandlers, VideoStreamHandlers, SnapshotHandlers, metaclass=ABCMeta):
    """Base class for UniFi Protect camera emulators.

    Manages the full event lifecycle:
      - Analytics (motion) events — ``trigger_analytics_start/stop``
      - Smart-detect events — ``trigger_smart_detect_start/update/stop``

    Subclasses must implement ``get_snapshot`` and ``get_stream_source``, and
    may override ``fetch_snapshots_for_event``, zone handlers, and settings
    handlers to integrate with their specific NVR or camera backend.

    Attributes (public, configurable after construction):
        lingerEventStart (int): Milliseconds to wait before sending an
            ``EventAnalytics`` start message.  If motion stops before this
            elapses, no messages are sent at all.  Default: 1000.
        motionEvents (bool): When ``False``, ``trigger_analytics_start`` is a
            no-op.  Default: ``True``.
    """

    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger

        # --- Timing / IDs -------------------------------------------------
        self._msg_id: int = 0
        self._init_time: float = time.time()
        # Monotonically incrementing counter embedded in every event ID.
        self._event_counter: int = 0

        # --- Streams ------------------------------------------------------
        self._streams: dict[str, str] = {}
        self._ffmpeg_handles: dict[str, subprocess.Popen] = {}

        # --- Snapshots ----------------------------------------------------
        # Three snapshot types served to UniFi Protect:
        #   motionSnapshot        — crop with bounding box
        #   motionSnapshotFullFoV — full frame with bounding box
        #   motionHeatmap         — heatmap (falls back to FoV)
        self._motion_snapshot: Optional[Path] = None        # legacy alias
        self._motion_snapshot_crop: Optional[Path] = None
        self._motion_snapshot_fov: Optional[Path] = None
        self._motion_heatmap: Optional[Path] = None

        # --- Analytics event state ----------------------------------------
        # All analytics events (active + ended) kept for _EVENT_RETENTION_SECS.
        # Structure per entry — see _new_analytics_event_record().
        self._analytics_event_history: dict[int, dict[str, Any]] = {}
        self._active_analytics_event_id: Optional[int] = None
        self._analytics_start_task: Optional[asyncio.Task] = None

        # --- Smart-detect event state -------------------------------------
        # Active and recently ended smart-detect events.
        # Structure per entry — see _new_smart_event_record().
        self._active_smart_events: dict[int, dict[str, Any]] = {}

        # --- Video resolutions --------------------------------------------
        self._detected_resolutions: dict[str, tuple[int, int]] = {
            "video1": (2560, 1920),
            "video2": (1280, 704),
            "video3": (640, 360),
        }

        # --- Zone state ---------------------------------------------------
        self._privacy_zones: dict[int, dict[str, Any]] = {}
        self._exclude_zones: dict[str, dict[str, Any]] = {}
        self._clarity_zones: dict[str, dict[str, Any]] = {}

        # --- Behavioural settings (may be changed after construction) -----
        self.lingerEventStart: int = 1000   # ms
        self.motionEvents: bool = True

        # --- Legacy compatibility -----------------------------------------
        # These are kept so that existing subclasses that read them still work.
        self._motion_event_ts: Optional[float] = None
        self._motion_object_type: Optional[SmartDetectObjectType] = None
        self._motion_last_descriptor: Optional[dict[str, Any]] = None

        # --- Network ------------------------------------------------------
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        self._ssl_context.load_cert_chain(args.cert, args.cert)
        self._session: Optional[websockets.client.WebSocketClientProtocol] = None

        atexit.register(self.close_streams)

    # =========================================================================
    # CLI argument registration
    # =========================================================================

    @classmethod
    def add_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ffmpeg-args", "-f",
            default="-c:v copy -ar 32000 -ac 1 -codec:a aac -b:a 32k",
            help="Transcoding args for `ffmpeg -i <src> <args> <dst>`",
        )
        parser.add_argument(
            "--ffmpeg-base-args", "-b",
            type=str,
            help="Base args for `ffmpeg <base_args> -i <src> <args> <dst>`",
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
            default=90,
            help="Timestamp correction factor (default: 90)",
        )
        parser.add_argument(
            "--loglevel",
            default="error",
            choices=[
                "trace", "debug", "verbose", "info",
                "warning", "error", "fatal", "panic", "quiet",
            ],
            help="FFmpeg log level",
        )
        parser.add_argument(
            "--format",
            default="flv",
            help="FFmpeg output format",
        )

    # =========================================================================
    # WebSocket run loop
    # =========================================================================

    async def _run(self, ws) -> None:
        self._session = ws
        await self.init_adoption()
        while True:
            try:
                msg = await ws.recv()
            except websockets.exceptions.ConnectionClosedError:
                self.logger.info("Connection to %s was closed.", self.args.host)
                raise RetryableError()
            if msg is not None:
                force_reconnect = await self.process(msg)
                if force_reconnect:
                    self.logger.info("Reconnecting...")
                    raise RetryableError()

    async def run(self) -> None:
        return

    # =========================================================================
    # Abstract interface for subclasses
    # =========================================================================

    @abstractmethod
    async def get_snapshot(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    async def get_stream_source(self, stream_index: str) -> str:
        raise NotImplementedError

    # =========================================================================
    # Feature flags & zone accessors
    # =========================================================================

    async def get_video_settings(self) -> dict[str, Any]:
        return {}

    async def change_video_settings(self, options) -> None:
        return

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            "mic": True,
            "aec": [],
            "videoMode": ["default"],
            "motionDetect": ["enhanced"],
            "privacyMask": True,
            "privacyMasks": {"maxZones": 16, "rectangleOnly": False},
            "smartDetect": ["person", "vehicle", "animal", "package"],
            # Must be arrays (not null) to enable the respective Protect UI menus.
            "excludeZone": [],
            "clarityZones": [],
        }

    async def get_privacy_zones(self) -> list[dict[str, Any]]:
        return list(self._privacy_zones.values())

    async def change_privacy_zones(self, zones: list[dict[str, Any]]) -> None:
        pass

    async def get_exclude_zones(self) -> dict[str, dict[str, Any]]:
        return dict(self._exclude_zones)

    async def change_exclude_zones(self, zones: dict[str, dict[str, Any]]) -> None:
        pass

    async def get_clarity_zones(self) -> dict[str, dict[str, Any]]:
        return dict(self._clarity_zones)

    async def change_clarity_zones(self, zones: dict[str, dict[str, Any]]) -> None:
        pass

    # =========================================================================
    # Snapshot helpers
    # =========================================================================

    async def fetch_snapshots_for_event(
        self,
        event_id: int,
        event_type: str = "analytics",
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Fetch crop, FoV, and heatmap snapshots for an event.

        The base implementation uses ``get_snapshot`` for all three.
        Subclasses (e.g. FrigateCam) override this to pull event-specific
        images from their NVR backend.

        Returns:
            (crop_path, fov_path, heatmap_path) — any may be None.
        """
        snapshot = await self.get_snapshot()
        return (snapshot, snapshot, snapshot)

    def update_snapshot_dimensions_from_file(
        self, event_id: int, snapshot_path: Optional[Path]
    ) -> None:
        """Read the actual pixel dimensions from *snapshot_path* and store them
        on the smart-detect event record and all its descriptor history entries.

        Subclasses should call this after fetching a snapshot so that the stop
        payload carries accurate width/height values.
        """
        if event_id not in self._active_smart_events or not snapshot_path:
            return

        width, height = self._get_image_dimensions(snapshot_path)
        event = self._active_smart_events[event_id]
        event["snapshot_width"] = width
        event["snapshot_height"] = height
        for entry in event["descriptor_history"]:
            entry["snapshot_width"] = width
            entry["snapshot_height"] = height

        self.logger.debug(
            "Updated snapshot dimensions for event %d: %dx%d (%d history entries)",
            event_id, width, height, len(event["descriptor_history"]),
        )

    def _get_image_dimensions(self, image_path: Optional[Path]) -> tuple[int, int]:
        """Return (width, height) of *image_path*, or (640, 360) on any failure."""
        _FALLBACK = (640, 360)
        if not image_path or not image_path.exists():
            return _FALLBACK
        try:
            if _PIL_AVAILABLE:
                with _PILImage.open(image_path) as img:
                    return img.size  # (width, height)
            # Minimal fallback: parse PNG header without PIL.
            with image_path.open("rb") as f:
                header = f.read(24)
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                return (width, height)
            # JPEG and other formats require PIL — return fallback.
            self.logger.debug(
                "PIL unavailable and %s is not PNG; returning default dimensions",
                image_path,
            )
        except Exception:
            self.logger.debug(
                "Could not read image dimensions from %s", image_path, exc_info=True
            )
        return _FALLBACK

    def _snapshot_dimensions_from_descriptor(
        self,
        descriptor: dict[str, Any],
        snapshot_path: Optional[Path] = None,
    ) -> tuple[int, int]:
        """Best-effort (width, height) for a snapshot.

        Priority: actual image file → bounding-box calculation → (640, 360).
        """
        if snapshot_path:
            dims = self._get_image_dimensions(snapshot_path)
            if dims != (640, 360):
                return dims

        coord = descriptor.get("coord")
        if coord and len(coord) >= 4:
            try:
                x1, y1, x2, y2 = coord[:4]
                sw, sh = self._detected_resolutions.get("video3", (640, 360))
                w = max(int(abs(x2 - x1) * sw), 100)
                h = max(int(abs(y2 - y1) * sh), 100)
                return (w, h)
            except (ValueError, TypeError):
                pass

        return (640, 360)

    # =========================================================================
    # Event ID allocation
    # =========================================================================

    def _next_event_id(self) -> int:
        """Return a globally-unique event ID.

        Encodes wall-clock milliseconds in the high bits and a local counter in
        the low three digits so IDs are both time-ordered and collision-resistant
        across restarts within the same millisecond.
        """
        self._event_counter += 1
        return int(time.time() * 1000) * 1000 + (self._event_counter % 1000)

    # =========================================================================
    # Event record factories
    # =========================================================================

    @staticmethod
    def _new_analytics_event_record(
        event_id: int,
        start_time: float,
        event_timestamp: Optional[float],
    ) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "start_time": start_time,
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
        }

    @staticmethod
    def _new_smart_event_record(
        object_type: SmartDetectObjectType,
        start_time: float,
        event_timestamp: Optional[float],
        descriptor: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "object_type": object_type,
            "start_time": start_time,
            "end_time": None,
            "event_timestamp": event_timestamp,
            "last_descriptor": descriptor,
            "descriptor_history": [],
            "snapshot_crop_path": None,
            "snapshot_fov_path": None,
            "heatmap_path": None,
            "snapshot_width": None,
            "snapshot_height": None,
        }

    # =========================================================================
    # Event history housekeeping
    # =========================================================================

    def _cleanup_old_analytics_events(self) -> None:
        """Remove analytics events older than ``_EVENT_RETENTION_SECS``."""
        cutoff = time.time() - _EVENT_RETENTION_SECS
        to_remove = [
            eid for eid, ev in self._analytics_event_history.items()
            if (ev.get("end_time") or ev.get("start_time", 0)) < cutoff
        ]
        for eid in to_remove:
            ev = self._analytics_event_history.pop(eid)
            self._delete_event_snapshots(ev)
            self.logger.debug("Purged analytics event %d from history", eid)
        if to_remove:
            self.logger.info(
                "Purged %d old analytics events; %d remaining",
                len(to_remove), len(self._analytics_event_history),
            )

    def _cleanup_old_smart_events(self) -> None:
        """Remove ended smart-detect events older than ``_EVENT_RETENTION_SECS``."""
        cutoff = time.time() - _EVENT_RETENTION_SECS
        to_remove = [
            eid for eid, ev in self._active_smart_events.items()
            if ev.get("end_time") and ev["end_time"] < cutoff
        ]
        for eid in to_remove:
            ev = self._active_smart_events.pop(eid)
            self._delete_event_snapshots(ev)
            self.logger.debug("Purged smart-detect event %d from history", eid)
        if to_remove:
            self.logger.info(
                "Purged %d old smart-detect events; %d remaining",
                len(to_remove), len(self._active_smart_events),
            )

    @staticmethod
    def _delete_event_snapshots(event: dict[str, Any]) -> None:
        """Unlink cached snapshot files stored on *event*, ignoring errors."""
        for key in ("snapshot_crop_path", "snapshot_fov_path", "heatmap_path"):
            path = event.get(key)
            if isinstance(path, Path) and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    # =========================================================================
    # Smart-detect event API (primary interface for subclasses)
    # =========================================================================

    async def trigger_smart_detect_start(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zones_status: Optional[dict[str, Any]] = None,
    ) -> int:
        """Start a smart-detect event.

        Returns:
            The UniFi event ID assigned to this event.
        """
        now = time.time()
        event_id = self._next_event_id()

        if event_id in self._active_smart_events:
            # Collision (sub-millisecond); nudge and retry once.
            self._event_counter += 1
            event_id = self._next_event_id()

        self._cleanup_old_smart_events()

        record = self._new_smart_event_record(object_type, now, event_timestamp, custom_descriptor)
        self._active_smart_events[event_id] = record

        if custom_descriptor:
            record["descriptor_history"].append(
                self._descriptor_history_entry(custom_descriptor, event_timestamp)
            )

        # Associate with the active analytics event if one exists.
        if self._active_analytics_event_id is not None:
            analytics = self._analytics_event_history.get(self._active_analytics_event_id)
            if analytics:
                analytics["smart_detect_event_ids"].append(event_id)

        if not zones_status:
            zones_status = self._zones_status_from_descriptor(custom_descriptor, "enter")

        ts_wall = event_timestamp or int(round(time.time() * 1000))
        mono = int(round(self.get_uptime() * 1000))

        payload: dict[str, Any] = {
            "clockMonotonic": mono,
            "clockStream": mono,
            "clockStreamRate": 1000,
            "clockWall": ts_wall,
            "descriptors": [custom_descriptor] if custom_descriptor else [],
            "displayTimeoutMSec": 10000,
            "edgeType": "enter",
            "eventId": event_id,
            "objectTypes": [object_type.value],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zones_status,
        }
        if custom_descriptor and (lp := custom_descriptor.get("licensePlate")):
            payload["licensePlate"] = lp

        self.logger.info(
            "Starting smart-detect event %d for %s (active: %d)",
            event_id, object_type.value, len(self._active_smart_events),
        )
        await self.send(self.gen_response("EventSmartDetect", payload=payload))

        # Update legacy shims.
        self._motion_event_ts = now
        self._motion_object_type = object_type
        self._motion_last_descriptor = custom_descriptor

        return event_id

    async def trigger_smart_detect_update(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        event_id: Optional[int] = None,
        zones_status: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send a smart-detect update (``edgeType="moving"``) message.

        Args:
            object_type: Object type to update.
            custom_descriptor: Updated bounding-box / metadata descriptor.
            event_timestamp: Wall-clock timestamp (ms) to embed in the payload.
            event_id: Specific event ID.  If omitted, the *most recent* active
                event for *object_type* is used.  Prefer passing the ID
                explicitly when multiple concurrent events of the same type
                may exist.
        """
        target_id = event_id or self._find_active_event_id(object_type)
        if target_id is None:
            self.logger.warning(
                "trigger_smart_detect_update: no active %s event found — ignoring",
                object_type.value,
            )
            return

        if target_id not in self._active_smart_events:
            self.logger.warning(
                "trigger_smart_detect_update: event %d not in active events — ignoring",
                target_id,
            )
            return

        record = self._active_smart_events[target_id]

        if custom_descriptor:
            record["last_descriptor"] = custom_descriptor
            self._motion_last_descriptor = custom_descriptor
            entry = self._descriptor_history_entry(custom_descriptor, event_timestamp)
            record["descriptor_history"].append(entry)
            # Trim history to avoid unbounded growth during long detections.
            if len(record["descriptor_history"]) > _MAX_DESCRIPTOR_HISTORY:
                # Keep first (for start metadata) + most recent N-1 entries.
                record["descriptor_history"] = (
                    record["descriptor_history"][:1]
                    + record["descriptor_history"][-((_MAX_DESCRIPTOR_HISTORY - 1)):]
                )

        if not zones_status:
            zones_status = self._zones_status_from_descriptor(custom_descriptor, "moving")
        ts_wall = event_timestamp or int(round(time.time() * 1000))
        mono = int(round(self.get_uptime() * 1000))

        payload: dict[str, Any] = {
            "clockMonotonic": mono,
            "clockStream": mono,
            "clockStreamRate": 1000,
            "clockWall": ts_wall,
            "descriptors": [custom_descriptor] if custom_descriptor else [],
            "displayTimeoutMSec": 10000,
            "edgeType": "moving",
            "eventId": target_id,
            "objectTypes": [object_type.value],
            "smartDetectSnapshotFullFoV": "",
            "smartDetectSnapshotFullFoVHeight": 0,
            "smartDetectSnapshotFullFoVWidth": 0,
            "smartDetectSnapshots": [],
            "zonesStatus": zones_status,
        }
        if custom_descriptor and (lp := custom_descriptor.get("licensePlate")):
            payload["licensePlate"] = lp

        self.logger.debug(
            "Updating smart-detect event %d (%s)", target_id, object_type.value
        )
        await self.send(self.gen_response("EventSmartDetect", payload=payload))

    async def trigger_smart_detect_stop(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        event_id: Optional[int] = None,
        frame_time_ms: Optional[int] = None,
        zones_status: Optional[dict[str, Any]] = None,
    ) -> None:
        """Stop a smart-detect event.

        Args:
            object_type: Object type being stopped.
            custom_descriptor: Optional final descriptor (triggers one last
                ``trigger_smart_detect_update`` before the stop message).
            event_timestamp: Wall-clock end time (ms).
            event_id: Specific event to stop.  Prefers explicit ID over type lookup.
            frame_time_ms: Frame timestamp for the final update, if different
                from *event_timestamp*.
        """
        target_id = self._find_or_validate_smart_event(event_id, object_type)
        if target_id is None:
            return

        record = self._active_smart_events[target_id]

        # 1. Final update if a closing descriptor was supplied.
        if custom_descriptor:
            await self.trigger_smart_detect_update(
                object_type, custom_descriptor,
                frame_time_ms or event_timestamp,
                event_id=target_id,
            )

        # 2. Pick the best per-tracker descriptors from history.
        history = self._get_descriptors_for_stop(record, custom_descriptor, event_timestamp)
        best = self._best_descriptors_by_tracker(history)

        # 3. Build snapshots array.
        smart_snapshots, tracker_map = self._build_smart_detect_snapshots(best, record, object_type)

        # 4. Build and send the stop payload.
        payload = self._build_stop_payload(
            target_id, object_type, record, smart_snapshots, tracker_map, event_timestamp
        )
        await self.send(self.gen_response("EventSmartDetect", payload=payload))

        # 5. Mark event as ended.
        record["end_time"] = time.time()
        duration = record["end_time"] - record["start_time"]
        self.logger.info(
            "Stopped smart-detect event %d (%s) after %.1fs",
            target_id, object_type.value, duration,
        )

        # 6. Update legacy shims.
        if not any(e["end_time"] is None for e in self._active_smart_events.values()):
            self._motion_object_type = None
            self._motion_last_descriptor = None
            if self._active_analytics_event_id is None:
                self._motion_event_ts = None

    # =========================================================================
    # Smart-detect internals
    # =========================================================================

    def _find_active_event_id(
        self, object_type: SmartDetectObjectType
    ) -> Optional[int]:
        """Return the ID of the most recently started active event for *object_type*."""
        candidates = [
            (eid, ev["start_time"])
            for eid, ev in self._active_smart_events.items()
            if ev["object_type"] == object_type and ev["end_time"] is None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t[1])[0]

    def _find_or_validate_smart_event(
        self, event_id: Optional[int], object_type: SmartDetectObjectType
    ) -> Optional[int]:
        """Resolve and validate a smart-detect event ID for stop/update.

        Returns the resolved ID, or ``None`` if not found (with a warning logged).
        """
        target = event_id if event_id is not None else self._find_active_event_id(object_type)

        if target is None:
            self.logger.warning(
                "No active %s event found (event_id=%s) — ignoring stop",
                object_type.value, event_id,
            )
            return None

        if target not in self._active_smart_events:
            self.logger.warning(
                "Event %d is not in active events — ignoring stop", target
            )
            return None

        return target

    def _descriptor_history_entry(
        self,
        descriptor: dict[str, Any],
        event_timestamp: Optional[float],
    ) -> dict[str, Any]:
        return {
            "descriptor": descriptor,
            "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
            "monotonic": int(self.get_uptime()),
            "snapshot_width": 640,
            "snapshot_height": 360,
        }

    def _get_descriptors_for_stop(
        self,
        record: dict[str, Any],
        custom_descriptor: Optional[dict[str, Any]],
        event_timestamp: Optional[float],
    ) -> list[dict[str, Any]]:
        """Return the descriptor history to use when building the stop payload.

        Falls back gracefully through history → custom_descriptor → last_descriptor.
        """
        history = record.get("descriptor_history", [])
        if history:
            return history

        # No history — synthesize a single entry.
        fallback = custom_descriptor or record.get("last_descriptor")
        if fallback:
            w, h = self._snapshot_dimensions_from_descriptor(
                fallback, record.get("snapshot_crop_path")
            )
            return [{
                "descriptor": fallback,
                "timestamp_ms": event_timestamp or int(round(time.time() * 1000)),
                "monotonic": int(self.get_uptime()),
                "snapshot_width": w,
                "snapshot_height": h,
            }]

        return []

    @staticmethod
    def _best_descriptors_by_tracker(
        entries: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """For each tracker ID, keep the entry with the highest confidence."""
        best: dict[int, dict[str, Any]] = {}
        for entry in entries:
            desc = entry["descriptor"]
            tid = desc.get("trackerID", 1)
            if tid not in best or desc.get("confidenceLevel", 0) > best[tid]["descriptor"].get("confidenceLevel", 0):
                best[tid] = entry
        return best

    def _build_smart_detect_snapshots(
        self,
        best: dict[int, dict[str, Any]],
        record: dict[str, Any],
        object_type: SmartDetectObjectType,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build the ``smartDetectSnapshots`` array and ``trackerIDAttrMap``."""
        snapshots: list[dict[str, Any]] = []
        tracker_map: dict[str, Any] = {}

        crop_path = record.get("snapshot_crop_path")
        event_w = record.get("snapshot_width") or 640
        event_h = record.get("snapshot_height") or 360

        for tid, entry in best.items():
            desc = entry["descriptor"]
            w = entry.get("snapshot_width") or event_w
            h = entry.get("snapshot_height") or event_h
            filename = str(crop_path) if crop_path else (
                f"smartdetectsnap_zone_{tid}_{entry['timestamp_ms']}.jpg"
            )
            snapshots.append({
                "clockBestMonotonic": entry["monotonic"],
                "clockBestWall": entry["timestamp_ms"],
                "smartDetectSnapshot": filename,
                "smartDetectSnapshotHeight": h,
                "smartDetectSnapshotName": desc.get("name", ""),
                "smartDetectSnapshotType": object_type.value,
                "smartDetectSnapshotWidth": w,
                "trackerID": tid,
            })
            tracker_map[str(tid)] = {
                "objectType": object_type.value,
                "zone": [desc.get("zones") or [1]],
            }

        if not snapshots:
            # Produce a minimal valid entry so Protect doesn't reject the payload.
            now_ms = int(round(time.time() * 1000))
            fn = str(crop_path) if crop_path else f"smartdetectsnap_1_{now_ms}.jpg"
            snapshots.append({
                "clockBestMonotonic": int(self.get_uptime()),
                "clockBestWall": now_ms,
                "smartDetectSnapshot": fn,
                "smartDetectSnapshotHeight": event_h,
                "smartDetectSnapshotName": "",
                "smartDetectSnapshotType": object_type.value,
                "smartDetectSnapshotWidth": event_w,
                "trackerID": 1,
            })
            tracker_map["1"] = {"objectType": object_type.value, "zone": [1]}

        return snapshots, tracker_map

    def _build_stop_payload(
        self,
        event_id: int,
        object_type: SmartDetectObjectType,
        record: dict[str, Any],
        smart_snapshots: list[dict[str, Any]],
        tracker_map: dict[str, Any],
        event_timestamp: Optional[float],
    ) -> dict[str, Any]:
        last_desc = record.get("last_descriptor") or {}
        zones_status = self._zones_status_from_descriptor(last_desc or None, "leave")

        fov_path = record.get("snapshot_fov_path")
        fov_filename = str(fov_path) if fov_path else f"smartdetectsnap_{event_id}_fullfov.jpg"
        if fov_path:
            fov_w, fov_h = self._get_image_dimensions(fov_path)
        else:
            fov_w = record.get("snapshot_width") or 640
            fov_h = record.get("snapshot_height") or 360

        ts_wall = event_timestamp or int(round(time.time() * 1000))
        mono = int(round(self.get_uptime() * 1000))

        payload: dict[str, Any] = {
            "clockMonotonic": mono,
            "clockStream": mono,
            "clockStreamRate": 1000,
            "clockWall": ts_wall,
            "descriptors": [],
            "displayTimeoutMSec": 2000,
            "edgeType": "leave",
            "eventId": event_id,
            "objectTypes": [object_type.value],
            "smartDetectSnapshotFullFoV": fov_filename,
            "smartDetectSnapshotFullFoVHeight": fov_h,
            "smartDetectSnapshotFullFoVWidth": fov_w,
            "smartDetectSnapshots": smart_snapshots,
            "trackerIDAttrMap": tracker_map,
            "zonesStatus": zones_status,
        }
        if lp := last_desc.get("licensePlate"):
            payload["licensePlate"] = lp

        return payload

    # =========================================================================
    # Analytics (motion) event API
    # =========================================================================

    async def trigger_analytics_start(
        self,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """Begin a generic motion analytics event.

        Sending is deferred by ``lingerEventStart`` ms.  If ``trigger_analytics_stop``
        is called before the linger elapses, no messages are sent at all.
        """
        if not self.motionEvents:
            self.logger.debug("motionEvents disabled — ignoring trigger_analytics_start")
            return

        self._cleanup_old_analytics_events()
        now = time.time()

        if self._active_analytics_event_id is not None:
            existing = self._analytics_event_history.get(self._active_analytics_event_id)
            if existing:
                self.logger.warning(
                    "Analytics event %d already active (%.1fs) — ignoring duplicate start",
                    self._active_analytics_event_id, now - existing["start_time"],
                )
                return

        event_id = self._next_event_id()
        self._analytics_event_history[event_id] = self._new_analytics_event_record(
            event_id, now, event_timestamp
        )
        self._active_analytics_event_id = event_id

        linger = self.lingerEventStart / 1000.0
        self.logger.info(
            "Scheduling analytics event %d (linger: %.0fms)", event_id, self.lingerEventStart
        )

        self._analytics_start_task = asyncio.create_task(
            self._delayed_analytics_start(event_id, event_timestamp, linger)
        )

        if not self._motion_event_ts:
            self._motion_event_ts = now

    async def trigger_analytics_stop(
        self,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """Stop the active analytics event.

        If the linger period hasn't elapsed, the pending start is cancelled and
        no ``EventAnalytics`` messages are sent for this cycle.
        """
        if self._active_analytics_event_id is None:
            self.logger.warning(
                "trigger_analytics_stop: no active analytics event — ignoring"
            )
            return

        event_id = self._active_analytics_event_id
        record = self._analytics_event_history.get(event_id)

        if not record:
            self.logger.warning(
                "trigger_analytics_stop: event %d not in history — cleaning up", event_id
            )
            self._active_analytics_event_id = None
            return

        # Cancel linger task if still pending.
        if self._analytics_start_task and not self._analytics_start_task.done():
            self._analytics_start_task.cancel()
            try:
                await self._analytics_start_task
            except asyncio.CancelledError:
                pass
            self.logger.info(
                "Analytics event %d cancelled within linger period — no messages sent", event_id
            )
            del self._analytics_event_history[event_id]
            self._active_analytics_event_id = None
            if not self._active_smart_events:
                self._motion_event_ts = None
            return

        # --- Determine snapshots to include --------------------------------
        crop, fov, heatmap = await self._resolve_analytics_snapshots(event_id, record)
        record.update({
            "snapshot_crop_path": crop,
            "snapshot_fov_path": fov,
            "heatmap_path": heatmap,
            "snapshot_filename": str(crop) if crop else f"snapshot_{event_id}.jpg",
            "snapshot_fov_filename": str(fov) if fov else f"snapshot_fov_{event_id}.jpg",
            "heatmap_filename": str(heatmap) if heatmap else f"heatmap_{event_id}.jpg",
            "end_time": time.time(),
        })

        mono = int(round(self.get_uptime() * 1000))
        payload: dict[str, Any] = {
            "clockBestMonotonic": mono,
            "clockBestWall": int(round(record["start_time"] * 1000)),
            "clockMonotonic": mono,
            "clockStream": mono,
            "clockStreamRate": 1000,
            "clockWall": int(round(time.time() * 1000)),
            "edgeType": "stop",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 49},
            "motionHeatmap": record["heatmap_filename"],
            "motionSnapshot": record["snapshot_filename"],
            "motionSnapshotFullFoV": record["snapshot_fov_filename"],
        }

        duration = time.time() - record["start_time"]
        self.logger.info(
            "Stopping analytics event %d (%.1fs, smart-detects: %d)",
            event_id, duration, len(record["smart_detect_event_ids"]),
        )
        await self.send(self.gen_response("EventAnalytics", payload=payload))

        self._active_analytics_event_id = None
        if not self._active_smart_events:
            self._motion_event_ts = None

    async def _resolve_analytics_snapshots(
        self, event_id: int, record: dict[str, Any]
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Return (crop, fov, heatmap) paths for an analytics stop event.

        Prefers snapshots cached on an associated smart-detect event; falls back
        to ``fetch_snapshots_for_event``.
        """
        smart_ids = record.get("smart_detect_event_ids", [])
        if smart_ids:
            smart = self._active_smart_events.get(smart_ids[-1])
            if smart:
                crop = smart.get("snapshot_crop_path")
                fov = smart.get("snapshot_fov_path")
                heatmap = smart.get("heatmap_path")
                if crop or fov or heatmap:
                    self.logger.info(
                        "Analytics event %d using snapshots from smart-detect event %d",
                        event_id, smart_ids[-1],
                    )
                    return (crop, fov, heatmap)

        self.logger.info("Analytics event %d fetching fresh snapshots", event_id)
        try:
            return await self.fetch_snapshots_for_event(event_id, "analytics")
        except Exception:
            self.logger.exception(
                "Error fetching snapshots for analytics event %d", event_id
            )
            return (None, None, None)

    # =========================================================================
    # Analytics internals
    # =========================================================================

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
            self.logger.debug(
                "Analytics start event %d cancelled during linger", event_id
            )
            raise

    async def _send_analytics_start_event(
        self,
        event_id: int,
        event_timestamp: Optional[float],
    ) -> None:
        record = self._analytics_event_history.get(event_id)
        if not record or record.get("end_time") is not None:
            self.logger.debug(
                "Analytics event %d ended before linger elapsed — not sending start",
                event_id,
            )
            return

        mono = int(round(self.get_uptime() * 1000))
        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": mono,
            "clockStream": mono,
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "start",
            "eventId": event_id,
            "eventType": "motion",
            "levels": {"0": 47},
            "motionHeatmap": "motionHeatmapline101.png",
            "motionSnapshot": "motionSnapshotline102.png",
        }

        self.logger.info(
            "Sending analytics start for event %d (after %.0fms linger)",
            event_id, self.lingerEventStart,
        )
        await self.send(self.gen_response("EventAnalytics", payload=payload))
        record["start_event_sent"] = True

    # =========================================================================
    # Utility helpers
    # =========================================================================

    def build_zones_status_from_descriptor(
        self,
        descriptor: dict[str, Any],
        edge_type: str = "moving",
    ) -> dict[str, dict[str, Any]]:
        """Build a ``zonesStatus`` dict from a descriptor's zones and confidence."""
        confidence = descriptor.get("confidenceLevel", 75)
        zones = descriptor.get("zones", [0])
        return {str(z): {"level": confidence, "status": edge_type} for z in zones}

    def _zones_status_from_descriptor(
        self,
        descriptor: Optional[dict[str, Any]],
        edge_type: str,
    ) -> dict[str, dict[str, Any]]:
        if descriptor:
            return self.build_zones_status_from_descriptor(descriptor, edge_type)
        level = {"enter": 60, "moving": 75, "leave": 75}.get(edge_type, 75)
        return {"0": {"level": level, "status": edge_type}}

    def get_uptime(self) -> float:
        return time.time() - self._init_time

    def get_active_events_summary(self) -> dict[str, Any]:
        """Diagnostic snapshot of current event state."""
        analytics = self._analytics_event_history.get(self._active_analytics_event_id)
        return {
            "analytics_event": {
                "active": self._active_analytics_event_id is not None,
                "event_id": self._active_analytics_event_id,
                "duration": (time.time() - analytics["start_time"]) if analytics else None,
                "smart_detect_ids": analytics["smart_detect_event_ids"] if analytics else [],
            },
            "smart_detect_events": {
                eid: {
                    "object_type": ev["object_type"].value,
                    "duration": time.time() - ev["start_time"],
                    "active": ev["end_time"] is None,
                    "has_crop": ev["snapshot_crop_path"] is not None,
                    "has_fov": ev["snapshot_fov_path"] is not None,
                    "has_heatmap": ev["heatmap_path"] is not None,
                }
                for eid, ev in self._active_smart_events.items()
            },
            "analytics_history_count": len(self._analytics_event_history),
        }

    async def stop_all_motion_events(self) -> None:
        """Force-stop all active smart-detect and analytics events."""
        for event_id, ev in list(self._active_smart_events.items()):
            if ev["end_time"] is not None:
                continue
            self.logger.info(
                "Force-stopping smart-detect event %d (%s)",
                event_id, ev["object_type"].value,
            )
            try:
                await self.trigger_smart_detect_stop(
                    ev["object_type"], event_id=event_id
                )
            except Exception:
                self.logger.exception(
                    "Error force-stopping smart-detect event %d", event_id
                )

        if self._active_analytics_event_id is not None:
            self.logger.info("Force-stopping analytics event")
            try:
                await self.trigger_analytics_stop()
            except Exception:
                self.logger.exception("Error force-stopping analytics event")

    # =========================================================================
    # Debug replay
    # =========================================================================

    async def replay_json_file(self, path: str = "./protect_EventSmartDetect.json") -> None:
        """Replay events from a JSON file; useful for development testing.

        Waits 15 seconds after startup, then sends each event with a 2-second gap.
        """
        elapsed = time.time() - self._init_time
        delay = max(15.0 - elapsed, 0.0)
        if delay > 0:
            self.logger.info("replay_json_file: waiting %.1fs before starting", delay)
            await asyncio.sleep(delay)

        try:
            events: list = json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            self.logger.error("replay_json_file: could not load %s: %s", path, exc)
            return

        self.logger.info("replay_json_file: replaying %d events from %s", len(events), path)
        mono = int(round(self.get_uptime() * 1000))

        for i, obj in enumerate(events):
            now_wall = int(round(time.time() * 1000))
            payload = obj.setdefault("payload", {})
            payload.update(clockWall=now_wall, clockMonotonic=mono, clockStream=mono)
            self.logger.debug("replay_json_file: sending %d/%d", i + 1, len(events))
            await self.send(self.gen_response("EventSmartDetect", payload=payload))
            await asyncio.sleep(2.0)

        self.logger.info("replay_json_file: complete")

    # =========================================================================
    # Protocol implementation
    # =========================================================================

    def gen_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def gen_response(
        self,
        name: str,
        response_to: int = 0,
        payload: Optional[dict[str, Any]] = None,
    ) -> AVClientResponse:
        now = datetime.now(timezone.utc)
        return {
            "from": "ubnt_avclient",
            "functionName": name,
            "inResponseTo": response_to,
            "messageId": self.gen_msg_id(),
            "payload": payload or {},
            "responseExpected": False,
            "to": "UniFiVideo",
            "timeStamp": (
                now.strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{now.microsecond // 1000:03d}+00:00"
            ),
        }

    async def send(self, msg: AVClientRequest) -> None:
        self.logger.debug("Sending: %s", msg)
        if self._session:
            await self._session.send(json.dumps(msg).encode())

    async def fetch_to_file(self, url: str, dst: Path) -> bool:
        try:
            async with aiohttp.request("GET", url) as resp:
                if resp.status != 200:
                    self.logger.error("Error retrieving file: HTTP %d", resp.status)
                    return False
                with dst.open("wb") as f:
                    f.write(await resp.read())
                return True
        except aiohttp.ClientError:
            return False

    async def init_adoption(self) -> None:
        self.logger.info(
            "Adopting with token [%s] and mac [%s]", self.args.token, self.args.mac
        )

        video1_source: Optional[str] = None
        for stream_index in ("video1", "video2", "video3"):
            try:
                source = await self.get_stream_source(stream_index)
                if not source:
                    continue
                if stream_index == "video1":
                    video1_source = source
                    w, h = self.probe_video_resolution(stream_index, source)
                    self._detected_resolutions[stream_index] = (w, h)
                elif source != video1_source:
                    w, h = self.probe_video_resolution(stream_index, source)
                    self._detected_resolutions[stream_index] = (w, h)
                else:
                    self.logger.debug(
                        "%s using video1 source as fallback — skipping probe", stream_index
                    )
            except NotImplementedError:
                self.logger.debug("%s not implemented — using default resolution", stream_index)
                break
            except Exception:
                self.logger.warning(
                    "Could not probe %s — using default resolution", stream_index, exc_info=True
                )

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
                    "type": "camera",
                    "upgradeTimeoutSec": 150,
                    "uptime": int(self.get_uptime()),
                    "features": await self.get_feature_flags(),
                },
            )
        )

    async def process_upgrade(self, msg: AVClientRequest) -> None:
        url = msg["payload"]["uri"]
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Range": "bytes=0-100"}, ssl=False) as r:
                content = await r.content.readexactly(54)
                version = "".join(chr(content[4 + i]) for i in range(50) if content[4 + i] != 0)
                self.logger.debug("Pretending to upgrade to: %s", version)
                self.args.fw_version = version

    async def process(self, msg: bytes) -> bool:
        m = json.loads(msg)
        fn = m["functionName"]

        if fn == "GetRequest" and "payload" in m:
            self.logger.info(
                "Processing [%s] (what=%s, filename=%s)",
                fn,
                m["payload"].get("what", "N/A"),
                m["payload"].get("filename", "N/A"),
            )
        else:
            self.logger.info("Processing [%s]", fn)
        self.logger.debug("Message contents: %s", m)

        # Messages that don't need a response are short-circuited early.
        _no_response_unless_listed = (
            "responseExpected" not in m or m["responseExpected"] is False
        )
        _always_handle = {
            "GetRequest", "ChangeVideoSettings", "UpdateFirmwareRequest",
            "Reboot", "ubnt_avclient_hello", "ContinuousMove",
        }
        if _no_response_unless_listed and fn not in _always_handle:
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
            res = self.gen_response("UpdateUsernamePassword", response_to=m["messageId"])
        elif fn == "ChangeSmartDetectSettings":
            await self.process_smart_detect_settings(m)
            res = self.gen_response("ChangeSmartDetectSettings", response_to=m["messageId"])
        elif fn == "ChangeAudioEventsSettings":
            res = self.gen_response("ChangeAudioEventsSettings", response_to=m["messageId"])
        elif fn == "UpdateFaceDBRequest":
            res = await self.process_update_face_db(m)
        elif fn == "ChangeTalkbackSettings":
            res = self.gen_response("ChangeTalkbackSettings", response_to=m["messageId"])
        elif fn == "ChangeSmartMotionSettings":
            res = await self.process_smart_motion_settings(m)
        elif fn == "SmartMotionTest":
            res = self.gen_response("SmartMotionTest", response_to=m["messageId"])
        elif fn == "ChangeClarityZones":
            res = await self.process_clarity_zones(m)
        elif fn == "ChangePrivacyZones":
            res = await self.process_privacy_zones(m)
        elif fn == "UpdateFirmwareRequest":
            await self.process_upgrade(m)
            return True
        elif fn == "Reboot":
            return True
        elif fn == "ContinuousMove":
            res = await self.process_continuous_move(m)
        else:
            self.logger.warning("Unhandled message type: %s — contents: %s", fn, m)

        if res is not None:
            await self.send(res)
        return False

    async def close(self) -> None:
        self.logger.info("Cleaning up instance")
        await self.stop_all_motion_events()
        self.close_streams()

    # =========================================================================
    # Deprecated legacy API — kept for backwards compatibility with subclasses
    # =========================================================================

    async def trigger_motion_start(
        self,
        object_type: Optional[SmartDetectObjectType] = None,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
    ) -> None:
        """Deprecated. Use ``trigger_analytics_start`` / ``trigger_smart_detect_start``."""
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
        """Deprecated. Use ``trigger_smart_detect_update``."""
        target = object_type or self._motion_object_type
        if not target:
            self.logger.warning(
                "trigger_motion_update: no object_type and no active event — ignoring"
            )
            return
        await self.trigger_smart_detect_update(target, custom_descriptor, event_timestamp)

    async def trigger_motion_stop(
        self,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        object_type: Optional[SmartDetectObjectType] = None,
    ) -> None:
        """Deprecated. Use ``trigger_analytics_stop`` / ``trigger_smart_detect_stop``."""
        if object_type:
            await self.trigger_smart_detect_stop(object_type, custom_descriptor, event_timestamp)
        else:
            await self.trigger_analytics_stop(event_timestamp)
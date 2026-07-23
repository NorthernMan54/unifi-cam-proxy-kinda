"""
Owns the analytics/motion (EventSmartMotion) lifecycle: the linger-delayed
start, the periodic pulse heartbeat, and the stop event that pulls together
snapshots from whichever smart-detect events occurred during the window.

Extracted in refactor Phase 3. This is the most stateful of the two
managers -- it runs two live asyncio background tasks (the linger-delay
start task and the pulse loop) whose cancellation ordering matters:
the pulse task must be cancelled before a stop payload is built, and the
linger task must be cancelled+awaited if stop races the start.

Cross-manager coupling: when trigger_analytics_stop needs snapshots, it
prefers whatever the most recent linked smart-detect event captured. Rather
than reaching into SmartDetectEventManager's internal dict, it goes through
`get_smart_event_snapshots`, injected at construction time.
"""
import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from unifi.cams.types import AVClientRequest, AVClientResponse

SendFn = Callable[[dict[str, Any]], Awaitable[None]]
GenResponseFn = Callable[..., dict[str, Any]]
GetUptimeFn = Callable[[], float]
FetchSnapshotsFn = Callable[[int, str], Awaitable[tuple]]
GetSmartSnapshotsFn = Callable[[int], tuple]


class SmartMotionEventManager:
    def __init__(
        self,
        logger: logging.Logger,
        send: SendFn,
        gen_response: GenResponseFn,
        get_uptime: GetUptimeFn,
        fetch_snapshots_for_event: FetchSnapshotsFn,
        get_smart_event_snapshots: Optional[GetSmartSnapshotsFn] = None,
        linger_event_start_ms: int = 1000,
    ) -> None:
        self.logger = logger
        self._send = send
        self._gen_response = gen_response
        self._get_uptime = get_uptime
        self._fetch_snapshots_for_event = fetch_snapshots_for_event
        self._get_smart_event_snapshots = get_smart_event_snapshots

        self.lingerEventStart: int = linger_event_start_ms
        self.motionEvents: bool = True

        self._motion_event_id_counter: int = 0
        self.event_history: dict[int, dict[str, Any]] = {}
        self.active_event_id: Optional[int] = None

        self._analytics_start_task: Optional[asyncio.Task] = None
        self._motion_pulse_task: Optional[asyncio.Task] = None

        self._motion_zone_id: str = "1"
        self._motion_zone_config: dict[str, str] = {"1": "Default"}

        self.last_event_ts: Optional[float] = None

    def set_motion_zone_config(self, zone_config: dict[str, str]) -> None:
        """Configure motion zone IDs and names, e.g. {"1": "Front Door"}."""
        self._motion_zone_config = zone_config
        if zone_config:
            self._motion_zone_id = next(iter(zone_config.keys()), "1")
        self.logger.debug(f"Motion zone config updated: {self._motion_zone_config}")

    def link_smart_detect(self, event_id: int) -> None:
        """
        Called by SmartDetectEventManager when a smart-detect event starts,
        so it can be associated with whichever analytics event is active.
        """
        if self.active_event_id is None:
            return
        active_event = self.event_history.get(self.active_event_id)
        if active_event:
            active_event["smart_detect_event_ids"].append(event_id)
            self.logger.debug(
                f"Associated smart detect event {event_id} with analytics event "
                f"{self.active_event_id}. Total smart detects for this analytics "
                f"event: {len(active_event['smart_detect_event_ids'])}"
            )

    # -- cleanup ---------------------------------------------------------

    def _cleanup_old_events(self) -> None:
        current_time = time.time()
        one_hour_ago = current_time - 3600

        to_remove = []
        for event_id, event_data in self.event_history.items():
            event_time = event_data.get("end_time") or event_data.get("start_time", 0)
            if event_time < one_hour_ago:
                to_remove.append(event_id)

        for event_id in to_remove:
            event_data = self.event_history[event_id]
            for key in ["snapshot_crop_path", "snapshot_fov_path", "heatmap_path"]:
                path = event_data.get(key)
                if path and path.exists():
                    try:
                        path.unlink()
                        self.logger.debug(f"Deleted cached snapshot: {path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cached snapshot {path}: {e}")
            del self.event_history[event_id]

        if to_remove:
            self.logger.info(
                f"Cleaned up {len(to_remove)} old analytics events. "
                f"Remaining in history: {len(self.event_history)}"
            )

    # -- start / linger ---------------------------------------------------

    async def trigger_start(self, event_timestamp: Optional[float] = None) -> None:
        if not self.motionEvents:
            self.logger.debug("Motion events disabled, ignoring trigger_start")
            return

        self._cleanup_old_events()
        current_time = time.time()

        epoch_ms = int(time.time() * 1000)
        event_id = epoch_ms * 1000 + (self._motion_event_id_counter % 1000)
        self._motion_event_id_counter += 1

        if self.active_event_id is not None:
            active_event = self.event_history.get(self.active_event_id)
            if active_event:
                existing_start = active_event["start_time"]
                self.logger.warning(
                    f"Analytics event {self.active_event_id} already active "
                    f"(started: {current_time - existing_start:.1f}s ago). "
                    f"Ignoring duplicate start."
                )
                return

        self.logger.info(
            f"Preparing analytics event {event_id}, will send start event after "
            f"{self.lingerEventStart}ms linger period"
        )

        self.event_history[event_id] = {
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
            "motion_event_id": None,
            "motion_heatmap_path": None,
            "motion_snapshot_path": None,
            "motion_snapshot_fov_path": None,
            "motion_raw_heatmap_npz_path": None,
            "motion_levels": {self._motion_zone_id: 30},
            "recordings_motion_level": None,
            "recordings_motion_last_fetch": 0.0,
            "motion_levels_source": "default",
            "motion_levels_updated_at": 0.0,
            "last_pulse_motion_levels_updated_at": 0.0,
            "last_pulse_time": None,
            "pulse_snapshot_counter": 0,
        }
        self.active_event_id = event_id

        linger_seconds = self.lingerEventStart / 1000.0
        self._analytics_start_task = asyncio.create_task(
            self._delayed_start(event_id, event_timestamp, linger_seconds)
        )

        if not self.last_event_ts:
            self.last_event_ts = current_time

    async def _delayed_start(
        self, event_id: int, event_timestamp: Optional[float], delay_seconds: float
    ) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            await self._send_motion_start_event(event_id, event_timestamp)
        except asyncio.CancelledError:
            self.logger.debug(f"Analytics start event {event_id} was cancelled during linger period")
            raise

    async def _send_motion_start_event(
        self, event_id: int, event_timestamp: Optional[float] = None
    ) -> None:
        if event_id not in self.event_history:
            self.logger.debug(
                f"Motion event {event_id} was stopped before linger period elapsed. "
                f"Not sending start event."
            )
            return

        active_event = self.event_history[event_id]
        if active_event.get("end_time") is not None:
            self.logger.debug(f"Motion event {event_id} already ended. Not sending start event.")
            return

        self._motion_event_id_counter += 1
        active_event["motion_event_id"] = self._motion_event_id_counter

        motion_levels = active_event.get("motion_levels", {self._motion_zone_id: 30})

        motion_heatmap_filename = f"heatmap_{self._motion_event_id_counter:08d}.png"
        motion_snapshot_filename = f"motionsnap_{self._motion_event_id_counter:08d}.jpg"
        motion_snapshot_fov_filename = f"motionsnap_{self._motion_event_id_counter:08d}_fullfov.jpg"
        motion_raw_heatmap_npz = f"motion_raw_heatmap_{int(round(event_timestamp or time.time() * 1000))}.npz"

        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self._get_uptime() * 1000),
            "clockStream": int(self._get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "start",
            "eventId": self._motion_event_id_counter,
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

        pulse_running = self._motion_pulse_task is not None and not self._motion_pulse_task.done()
        self.logger.info(
            f"Sending motion start event {self._motion_event_id_counter} (analytics {event_id}) "
            f"with levels: {motion_levels}, pulse task running: {pulse_running}"
        )

        await self._send(self._gen_response("EventSmartMotion", payload=payload))

        active_event["start_event_sent"] = True

        if self._motion_pulse_task and not self._motion_pulse_task.done():
            self._motion_pulse_task.cancel()
            try:
                await self._motion_pulse_task
            except asyncio.CancelledError:
                pass

        self._motion_pulse_task = asyncio.create_task(self._pulse_loop(event_id))

    # -- pulse -------------------------------------------------------------

    async def _send_motion_pulse_event(
        self, event_id: int, event_timestamp: Optional[float] = None
    ) -> None:
        if event_id not in self.event_history:
            return

        active_event = self.event_history[event_id]

        current_time = time.time()
        last_pulse = active_event.get("last_pulse_time")
        if last_pulse and (current_time - last_pulse) < 2.0:
            return

        motion_levels_updated_at = float(active_event.get("motion_levels_updated_at") or 0.0)
        last_pulse_updated_at = float(active_event.get("last_pulse_motion_levels_updated_at") or 0.0)
        if motion_levels_updated_at <= last_pulse_updated_at:
            return

        motion_levels = active_event.get("motion_levels", {self._motion_zone_id: 30})

        payload: dict[str, Any] = {
            "clockBestMonotonic": 0,
            "clockBestWall": 0,
            "clockMonotonic": int(self._get_uptime() * 1000),
            "clockStream": int(self._get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": event_timestamp or int(round(time.time() * 1000)),
            "edgeType": "unknown",
            "eventId": 18446744073709551615,  # sentinel: max uint64 / -1 for pulse events
            "eventType": "pulse",
            "levels": motion_levels,
            "motionHeatmap":"",
            "motionHeatmapHeight":0,
            "motionHeatmapWidth":0,
            "motionRawHeatmapNPZ":"",
            "motionSnapshot":"",
            "motionSnapshotFullFoV":"",
            "motionSnapshotFullFoVHeight":0,
            "motionSnapshotFullFoVWidth":0,
            "motionSnapshotHeight":0,
            "motionSnapshotWidth":0
        }

        self.logger.debug(f"Sending motion pulse event with levels: {motion_levels}")
        await self._send(self._gen_response("EventSmartMotion", payload=payload))

        active_event["last_pulse_time"] = current_time
        active_event["last_pulse_motion_levels_updated_at"] = motion_levels_updated_at

    async def _pulse_loop(self, event_id: int) -> None:
        try:
            while True:
                await asyncio.sleep(2.5)
                if event_id not in self.event_history:
                    break
                active_event = self.event_history[event_id]
                if active_event.get("end_time") is not None:
                    break
                await self._send_motion_pulse_event(event_id)
        except asyncio.CancelledError:
            self.logger.debug(f"Motion pulse task for event {event_id} cancelled")
            raise

    # -- stop ---------------------------------------------------------------

    async def trigger_stop(self, event_timestamp: Optional[float] = None) -> None:
        if self.active_event_id is None:
            self.logger.warning(
                "trigger_stop called but no active event found. "
                "Event may have already ended or never started. Ignoring."
            )
            return

        event_id = self.active_event_id
        active_event = self.event_history.get(event_id)

        if not active_event:
            self.logger.warning(
                f"trigger_stop called for event {event_id} but event not found in history. Ignoring."
            )
            self.active_event_id = None
            return

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
                f"Motion event {event_id} stopped before {self.lingerEventStart}ms "
                f"linger period. No start/stop events will be sent."
            )
            del self.event_history[event_id]
            self.active_event_id = None
            return

        snapshot_crop_path = None
        snapshot_fov_path = None
        heatmap_path = None

        smart_detect_ids = active_event.get("smart_detect_event_ids", [])
        if smart_detect_ids and self._get_smart_event_snapshots:
            most_recent_smart_id = smart_detect_ids[-1]
            snapshot_crop_path, snapshot_fov_path, heatmap_path = self._get_smart_event_snapshots(
                most_recent_smart_id
            )
            if snapshot_crop_path or snapshot_fov_path or heatmap_path:
                self.logger.info(
                    f"Using snapshots from smart detect event {most_recent_smart_id} "
                    f"for motion event {event_id}"
                )
            else:
                self.logger.warning(
                    f"Smart detect event {most_recent_smart_id} not found "
                    f"(may have been cleaned up after 60 minutes)"
                )

        if not snapshot_crop_path or not snapshot_fov_path or not heatmap_path:
            self.logger.info(
                f"No smart detect snapshots available for motion event {event_id}, "
                f"fetching fresh snapshots"
            )
            try:
                snapshot_crop_path, snapshot_fov_path, heatmap_path = await self._fetch_snapshots_for_event(
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
        active_event["motion_raw_heatmap_npz_path"] = None

        snapshot_filename = str(snapshot_crop_path) if snapshot_crop_path else f"motionsnap_{event_id}.jpg"
        snapshot_fov_filename = (
            str(snapshot_fov_path) if snapshot_fov_path else f"motionsnap_{event_id}_fullfov.jpg"
        )
        heatmap_filename = str(heatmap_path) if heatmap_path else f"heatmap_{event_id}.png"
        raw_heatmap_npz = f"motion_raw_heatmap_{int(round(event_timestamp or time.time() * 1000))}.npz"

        active_event["snapshot_filename"] = snapshot_filename
        active_event["snapshot_fov_filename"] = snapshot_fov_filename
        active_event["heatmap_filename"] = heatmap_filename
        active_event["end_time"] = time.time()

        self._motion_event_id_counter += 1
        motion_levels = active_event.get("motion_levels", {self._motion_zone_id: 0})

        payload: dict[str, Any] = {
            "clockBestMonotonic": int(self._get_uptime() * 1000),
            "clockBestWall": int(round(active_event["start_time"] * 1000)),
            "clockMonotonic": int(self._get_uptime() * 1000),
            "clockStream": int(self._get_uptime() * 1000),
            "clockStreamRate": 1000,
            "clockWall": int(round(time.time() * 1000)),
            "edgeType": "stop",
            "eventId": self._motion_event_id_counter,
            "eventType": "motion",
        #    "levels": motion_levels,
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
            f"Motion event snapshots: crop={snapshot_filename}, "
            f"fov={snapshot_fov_filename}, heatmap={heatmap_filename}"
        )

        await self._send(self._gen_response("EventSmartMotion", payload=payload))

        self.active_event_id = None

    # -- queries used by SnapshotHandlers / summaries -------------------

    def has_active_event(self) -> bool:
        return self.active_event_id is not None

    def find_snapshot_by_path(self, snapshot_type: str, filename: str, original_filename: str) -> Optional[Any]:
        """Look up a cached snapshot path matching a GetRequest's filename."""
        key_by_type = {
            "motionSnapshot": "snapshot_crop_path",
            "motionSnapshotFullFoV": "snapshot_fov_path",
            "motionHeatmap": "heatmap_path",
        }
        key = key_by_type.get(snapshot_type)
        if not key:
            return None

        for event_data in self.event_history.values():
            cached_path = event_data.get(key)
            if cached_path and cached_path.exists():
                if (filename and str(cached_path) == filename) or (
                    original_filename and str(cached_path) == original_filename
                ):
                    return cached_path
        return None

    def summary(self) -> dict[str, Any]:
        active_event = self.event_history.get(self.active_event_id) if self.active_event_id else None
        return {
            "active": self.active_event_id is not None,
            "event_id": self.active_event_id,
            "duration": time.time() - active_event["start_time"] if active_event else None,
            "smart_detect_event_ids": active_event["smart_detect_event_ids"] if active_event else [],
        }

    async def force_stop(self) -> None:
        if self.active_event_id is not None:
            self.logger.info("Force stopping motion event")
            try:
                await self.trigger_stop()
            except Exception as e:
                self.logger.error(f"Error stopping motion event: {e}")

    async def process_smart_motion_settings(
        self, msg: AVClientRequest
    ) -> AVClientResponse:
        """Process smart motion settings change request and update lingerEventStart and motionEvents."""
        payload = msg.get("payload", {})
        
        # Update motion event enable/disable flag
        if "enable" in payload:
            self.motionEvents = payload["enable"]
            self.logger.info(
                f"Motion events {'enabled' if self.motionEvents else 'disabled'} from ChangeSmartMotionSettings"
            )

        if self.motionEvents:
            await self.force_stop()
            await asyncio.sleep(0.1)
            await self.force_stop()
        
        if "lingerEventStartMSec" in payload:
            self.lingerEventStart = payload["lingerEventStartMSec"]
            self.logger.info(
                f"Updated lingerEventStart to {self.lingerEventStart}ms from ChangeSmartMotionSettings"
            )

        return self._gen_response(
            "ChangeSmartMotionSettings", msg["messageId"], payload
        )
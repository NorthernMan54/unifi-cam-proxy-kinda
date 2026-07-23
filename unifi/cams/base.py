import argparse
import asyncio
import atexit
import json
import logging
import ssl
import subprocess
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

import aiohttp
import websockets

from unifi.core import RetryableError
from unifi.cams.handlers import ProtocolHandlers, VideoStreamHandlers
from unifi.cams.handlers.snapshot_handlers import SnapshotHandlers
from unifi.cams.managers.motion_analytics_manager import MotionAnalyticsManager
from unifi.cams.managers.smart_detect_manager import SmartDetectEventManager, SmartDetectObjectType

AVClientRequest = AVClientResponse = dict[str, Any]

# Re-exported so existing call sites doing
# `from unifi.cams.base import SmartDetectObjectType` keep working --
# the enum's canonical home is now smart_detect_manager.py.
__all__ = ["UnifiCamBase", "SmartDetectObjectType", "AVClientRequest", "AVClientResponse"]


class UnifiCamBase(
    ProtocolHandlers,
    VideoStreamHandlers,
    SnapshotHandlers,
    metaclass=ABCMeta,
):
    """
    Composition root for UniFi Protect camera emulation.

    Owns the websocket connection/protocol plumbing, snapshot caching, and
    ffmpeg stream management directly. Smart-detect and motion/analytics
    event *lifecycle* is delegated to SmartDetectEventManager and
    MotionAnalyticsManager (see those modules) -- this class just wires them
    together and exposes the same public trigger_* methods subclasses
    (e.g. FrigateCam) already call, so no caller-side changes are required.
    """

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

        self._ffmpeg_handles: dict[str, subprocess.Popen] = {}

        self._detected_resolutions: dict[str, tuple[int, int]] = {
            "video1": (2560, 1920),
            "video2": (1280, 704),
            "video3": (640, 360),
        }

        self.motionEvents: bool = True

        # -- event lifecycle managers ----------------------------------
        # MotionAnalyticsManager needs to be able to pull cached snapshots
        # from whichever smart-detect event most recently occurred during
        # its window, so it's constructed with a getter into the smart
        # events manager rather than a direct reference to its internals.
        self._motion_analytics = MotionAnalyticsManager(
            logger=self.logger,
            send=self.send,
            gen_response=self.gen_response,
            get_uptime=self.get_uptime,
            fetch_snapshots_for_event=self.fetch_snapshots_for_event,
            get_smart_event_snapshots=lambda event_id: self._smart_events.get_snapshot_paths(event_id),
        )
        self._smart_events = SmartDetectEventManager(
            logger=self.logger,
            send=self.send,
            gen_response=self.gen_response,
            get_uptime=self.get_uptime,
            detected_resolutions=self._detected_resolutions,
            on_event_started=self._motion_analytics.link_smart_detect,
        )
        # motionEvents flag is read by MotionAnalyticsManager.trigger_start;
        # keep the two in sync since subclasses toggle self.motionEvents.
        self._motion_analytics.motionEvents = self.motionEvents

        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        self._ssl_context.load_cert_chain(args.cert, args.cert)
        self._session: Optional[websockets.client.WebSocketClientProtocol] = None
        atexit.register(self.close_streams)

    @property
    def _active_analytics_event_id(self) -> Optional[int]:
        """Backward-compatible property for subclasses accessing this directly."""
        return self._motion_analytics.active_event_id

    @_active_analytics_event_id.setter
    def _active_analytics_event_id(self, value: Optional[int]) -> None:
        self._motion_analytics.active_event_id = value

    @property
    def _analytics_event_history(self) -> dict:
        """Backward-compatible property for subclasses accessing this directly."""
        return self._motion_analytics.event_history

    @property
    def _active_smart_events(self) -> dict:
        """Backward-compatible property for subclasses accessing this directly."""
        return self._smart_events.active_events

    @property
    def _motion_zone_id(self) -> str:
        """Backward-compatible property for subclasses accessing this directly."""
        return self._motion_analytics._motion_zone_id

    @_motion_zone_id.setter
    def _motion_zone_id(self, value: str) -> None:
        self._motion_analytics._motion_zone_id = value

    @property
    def lingerEventStart(self) -> int:
        return self._motion_analytics.lingerEventStart

    @lingerEventStart.setter
    def lingerEventStart(self, value: int) -> None:
        self._motion_analytics.lingerEventStart = value

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

    async def send_mcu_event_message(self, payload: dict[str, Any]) -> None:
        """
        Send MCUEventMessage to UniFi Protect (used for doorbell ring events).
        """
        await self.send(self.gen_response("MCUEventMessage", payload=payload))

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
        event = self._smart_events.active_events.get(event_id)
        if not event:
            return

        if snapshot_path:
            from unifi.cams.handlers.snapshot_utils import get_image_dimensions
            width, height = get_image_dimensions(snapshot_path, self.logger)

            event["snapshot_width"] = width
            event["snapshot_height"] = height

            descriptor_history = event["descriptor_history"]
            for desc_entry in descriptor_history:
                desc_entry["snapshot_width"] = width
                desc_entry["snapshot_height"] = height

            self.logger.debug(
                f"Updated snapshot dimensions for event {event_id}: {width}x{height} "
                f"from file {snapshot_path} (updated {len(descriptor_history)} descriptor entries)"
            )

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            "mic": True,
            "aec": [],
            "videoMode": ["default"],
            "motionDetect": ["enhanced"],
            "hotplug": {"extender": {"attached": False}},
        }

    # -- Smart Detect Events: thin delegation to SmartDetectEventManager ---
    # Public signatures are unchanged from the pre-refactor UnifiCamBase so
    # existing subclasses (FrigateCam) don't need to change any call sites.

    async def trigger_smart_detect_start(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zonesStatus: Optional[dict[str, Any]] = None,
    ) -> int:
        return await self._smart_events.trigger_start(
            object_type, custom_descriptor, event_timestamp, zonesStatus
        )

    async def trigger_smart_detect_update(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zonesStatus: Optional[dict[str, Any]] = None,
        event_id: Optional[int] = None,
    ) -> None:
        await self._smart_events.trigger_update(
            object_type, custom_descriptor, event_timestamp, zonesStatus, event_id
        )

    async def trigger_smart_detect_stationary(
        self,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        zonesStatus: Optional[dict[str, Any]] = None,
        event_id: Optional[int] = None,
    ) -> None:
        await self._smart_events.trigger_stationary(
            custom_descriptor, event_timestamp, zonesStatus, event_id
        )

    async def trigger_smart_detect_stop(
        self,
        object_type: SmartDetectObjectType,
        custom_descriptor: Optional[dict[str, Any]] = None,
        event_timestamp: Optional[float] = None,
        event_id: Optional[int] = None,
        frame_time_ms: Optional[int] = None,
        zonesStatus: Optional[dict[str, Any]] = None,
    ) -> None:
        await self._smart_events.trigger_stop(
            object_type, custom_descriptor, event_timestamp, event_id, frame_time_ms, zonesStatus
        )

    def attach_tracker_snapshot(
        self, event_id: int, tracker_id: int, crop: Optional[Path] = None, fov: Optional[Path] = None
    ) -> None:
        """Public entry point for FrigateCam to record a per-tracker snapshot,
        replacing direct access to the manager's internal event dict."""
        self._smart_events.attach_tracker_snapshot(event_id, tracker_id, crop, fov)

    # -- Motion / Analytics Events: thin delegation ------------------------

    async def trigger_analytics_start(self, event_timestamp: Optional[float] = None) -> None:
        await self._motion_analytics.trigger_start(event_timestamp)

    async def trigger_analytics_stop(self, event_timestamp: Optional[float] = None) -> None:
        await self._motion_analytics.trigger_stop(event_timestamp)

    def set_motion_zone_config(self, zone_config: dict[str, str]) -> None:
        self._motion_analytics.set_motion_zone_config(zone_config)

    def get_active_events_summary(self) -> dict[str, Any]:
        smart_summary = self._smart_events.summary()
        return {
            "analytics_event": self._motion_analytics.summary(),
            "smart_detect_events": smart_summary,
            "total_active_events": (
                (1 if self._motion_analytics.has_active_event() else 0) + len(smart_summary)
            ),
            "analytics_history_count": len(self._motion_analytics.event_history),
        }

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

    # -- Protocol implementation --------------------------------------------

    def gen_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def init_adoption(self) -> None:
        self.logger.info(f"Adopting with token [{self.args.token}] and mac [{self.args.mac}]")

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
                        self.logger.debug(
                            f"{stream_index} using video1 source as fallback, using default resolution"
                        )
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
                    "mac": (self.args.mac.replace(":", "").replace("-", "").upper()),
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

    # Dispatch table for process(): functionName -> handler. Handlers that
    # need special-cased control flow (no response payload, or an early
    # return without going through the responseExpected gate below) are
    # NOT in this table -- see the explicit branches in process().
    def _dispatch_table(self) -> dict[str, Any]:
        return {
            "ubnt_avclient_time": self.process_time,
            "ubnt_avclient_paramAgreement": self.process_param_agreement,
            "ResetIspSettings": self.process_isp_settings,
            "ChangeVideoSettings": self.process_video_settings,
            "ChangeDeviceSettings": self.process_device_settings,
            "ChangeOsdSettings": self.process_osd_settings,
            "NetworkStatus": self.process_network_status,
            "ChangeSoundLedSettings": self.process_sound_led_settings,
            "ChangeIspSettings": self.process_change_isp_settings,
            "ChangeAnalyticsSettings": self.process_analytics_settings,
            "GetRequest": self.process_snapshot_request,
            "UpdateFaceDBRequest": self.process_update_face_db,
            "ChangeSmartMotionSettings": self.process_smart_motion_settings,
            "ContinuousMove": self.process_continuous_move,
        }

    # functionName -> canned ack response (no processing logic required).
    def _ack_only_functions(self) -> set[str]:
        return {
            "AnalyticsTest",
            "UpdateUsernamePassword",
            "ChangeSmartDetectSettings",
            "ChangeAudioEventsSettings",
            "ChangeTalkbackSettings",
            "SmartMotionTest",
            "ChangeClarityZones",
        }

    # functions handled specially before the responseExpected gate --
    # these either return early without a normal response, or process
    # regardless of responseExpected.
    _EARLY_RETURN_FUNCTIONS = frozenset({
        "GetRequest",
        "ChangeVideoSettings",
        "UpdateFirmwareRequest",
        "Reboot",
        "ubnt_avclient_hello",
        "ContinuousMove",
    })

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
            fn not in self._EARLY_RETURN_FUNCTIONS
        ):
            return False

        if fn == "ubnt_avclient_hello":
            await self.process_hello(m)
            return False
        elif fn == "UpdateFirmwareRequest":
            await self.process_upgrade(m)
            return True
        elif fn == "Reboot":
            return True

        res: Optional[AVClientResponse] = None

        if fn in self._ack_only_functions():
            res = self.gen_response(fn, response_to=m["messageId"])
        else:
            handler = self._dispatch_table().get(fn)
            if handler:
                res = await handler(m)
            else:
                self.logger.warning(f"Received unhandled message type: {fn}. Message contents: {m}")

        if res is not None:
            await self.send(res)

        return False

    async def close(self):
        self.logger.info("Protect sent close, cleaning up instance")
        # Force-stop all motion/analytics events directly (replaces stop_all_motion_events())
        if self._smart_events.active_events:
            self.logger.info(f"Force stopping {len(self._smart_events.active_events)} smart detect events")
            for event_id in list(self._smart_events.active_events.keys()):
                event = self._smart_events.active_events[event_id]
                try:
                    await self._smart_events.trigger_stop(object_type=event["object_type"])
                except Exception as e:
                    self.logger.error(f"Error stopping smart detect event {event_id}: {e}")
        
        if self._motion_analytics.active_event_id is not None:
            self.logger.info("Force stopping motion event")
            try:
                await self._motion_analytics.trigger_stop()
            except Exception as e:
                self.logger.error(f"Error stopping motion event: {e}")
        
        self.close_streams()

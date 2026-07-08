"""
Simple protocol message handlers for UniFi Protect camera protocol.

This module contains straightforward message handlers that primarily return
static or simple responses. More complex handlers remain in base.py.
"""

from typing import Any, Optional, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from unifi.cams.base import UnifiCamBase, AVClientRequest, AVClientResponse


class ProtocolHandlers:
    """Mixin class providing simple protocol message handlers."""
    
    async def process_hello(self: "UnifiCamBase", msg: "AVClientRequest") -> None:
        """Process hello message from UniFi Protect (no-op)."""
        pass

    async def process_param_agreement(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process parameter agreement request."""
        return self.gen_response(
            "ubnt_avclient_paramAgreement",
            msg["messageId"],
            {
                "authToken": self.args.token,
                "features": await self.get_feature_flags(),
            },
        )

    async def process_device_settings(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process device settings change request."""
        return self.gen_response(
            "ChangeDeviceSettings",
            msg["messageId"],
            {
                "name": self.args.name,
                "timezone": "PST8PDT,M3.2.0,M11.1.0",
            },
        )

    async def process_osd_settings(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process on-screen display settings change request."""
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

    async def process_network_status(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process network status request."""
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
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process sound and LED settings change request."""
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

    async def process_analytics_settings(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process analytics settings change request."""
        return self.gen_response(
            "ChangeAnalyticsSettings", msg["messageId"], msg["payload"]
        )

    async def process_time(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process time synchronization request."""
        import time

        return self.gen_response(
            "ubnt_avclient_paramAgreement",
            msg["messageId"],
            {
                # FIX: get_uptime() returns fractional SECONDS. "monotonicMs"
                # (and its sibling "wallMs", already correctly in ms via
                # time.time()*1000) both indicate this field is expected in
                # milliseconds -- same units bug as clockMonotonic/clockStream
                # in base.py's trigger_smart_detect_* methods, fixed there too.
                "monotonicMs": int(self.get_uptime() * 1000),
                "wallMs": int(round(time.time() * 1000)),
                "features": {},
            },
        )

    async def process_continuous_move(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> None:
        """Process continuous move request for PTZ cameras (no-op for fixed cameras)."""
        return

    async def process_update_face_db(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """
        Process face database update request.
        
        Returns empty response to indicate no face database is available.
        This prevents UniFi Protect from trying to fetch a non-existent file.
        """
        return self.gen_response(
            "UpdateFaceDBRequest",
            msg["messageId"],
            {},
        )

    async def process_isp_settings(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process ISP (Image Signal Processor) settings request."""
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

    async def process_change_isp_settings(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process ISP settings change request."""
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

    async def process_video_settings(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process video settings change request."""
        # FIX: previously `vid_dst` was rebuilt fresh every call, defaulting
        # every stream to /dev/null unless *this specific* incoming message
        # happened to configure it. Since self._streams (stream names)
        # already persists across calls but destinations did not, a stream
        # started in an earlier message would get reported back with a
        # contradictory response on a later, unrelated ChangeVideoSettings
        # call: a real streamName paired with destinations: ["file:///dev/null"],
        # even though nothing was actually stopped. Persist destinations the
        # same way stream names are persisted.
        if not hasattr(self, "_stream_destinations"):
            self._stream_destinations: dict[str, list[str]] = {}

        vid_dst = {
            "video1": self._stream_destinations.get("video1", ["file:///dev/null"]),
            "video2": self._stream_destinations.get("video2", ["file:///dev/null"]),
            "video3": self._stream_destinations.get("video3", ["file:///dev/null"]),
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
                            # Also clear persisted destination so a future
                            # unrelated ChangeVideoSettings call doesn't keep
                            # echoing a stopped stream's old real destination.
                            self._stream_destinations[k] = vid_dst[k]
                        elif "parameters" in v["avSerializer"]:
                            # FIX: previously this branch started a new
                            # ffmpeg process for key k on every
                            # ChangeVideoSettings that (re)configures it,
                            # with no guard against k already having a live
                            # stream. If Protect's controller re-sends
                            # ChangeVideoSettings to try to recover/restart a
                            # stalled stream (rather than only sending it
                            # once at initial setup), this would spawn a
                            # SECOND ffmpeg process reading the same RTSP
                            # source concurrently with the first, silently
                            # orphaning the old process -- a likely cause of
                            # "Timed out waiting for stream to open"
                            # symptoms if the upstream RTSP source doesn't
                            # support multiple simultaneous clients on one
                            # path. Stop any existing stream on this key
                            # first so a reconfigure/recovery attempt tears
                            # down and replaces cleanly instead of piling up.
                            if k in self._streams or k in self._ffmpeg_handles:
                                self.logger.info(
                                    f"Restarting stream {k}: stopping existing process "
                                    f"before starting new one (possible controller-initiated recovery)"
                                )
                                self.stop_video_stream(k)

                            self._streams[k] = stream = v["avSerializer"]["parameters"][
                                "streamName"
                            ]
                            # Persist the real destination alongside the
                            # stream name so later, unrelated
                            # ChangeVideoSettings calls report it consistently
                            # instead of falling back to /dev/null.
                            self._stream_destinations[k] = vid_dst[k]
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

    async def process_smart_motion_settings(
        self: "UnifiCamBase", msg: "AVClientRequest"
    ) -> "AVClientResponse":
        """Process smart motion settings change request and update lingerEventStart and motionEvents."""
        payload = msg.get("payload", {})
        
        # Update motion event enable/disable flag
        if "enable" in payload:
            self.motionEvents = payload["enable"]
            self.logger.info(
                f"Motion events {'enabled' if self.motionEvents else 'disabled'} from ChangeSmartMotionSettings"
            )
        
        # Update lingerEventStart if provided in the message
        if "lingerEventStartMSec" in payload:
            self.lingerEventStart = payload["lingerEventStartMSec"]
            self.logger.info(
                f"Updated lingerEventStart to {self.lingerEventStart}ms from ChangeSmartMotionSettings"
            )
        
        # Log other settings for debugging
        if "lingerEventStopMSec" in payload:
            self.logger.debug(f"Received lingerEventStopMSec: {payload['lingerEventStopMSec']}ms")
        if "eventMaxDurationMSec" in payload:
            self.logger.debug(f"Received eventMaxDurationMSec: {payload['eventMaxDurationMSec']}ms")
        
        return self.gen_response(
            "ChangeSmartMotionSettings", msg["messageId"], payload
        )
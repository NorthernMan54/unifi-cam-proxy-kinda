"""
Pairing / hello handshake tests that exercise live code.

Full interaction from the Driveway (UVC G4 Dome) capture:

  1. Proxy SENDS  ubnt_avclient_hello  (via init_adoption)
  2. Proxy RECEIVES server ubnt_avclient_hello reply (via process)
     → process_hello (no-op) → no outbound reply
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import assert_message_equal
from unifi.cams.base import UnifiCamBase

FIXTURES = Path(__file__).parent / "fixtures" / "hello_handshake"
CERT_PATH = Path("/tmp/test-client.pem")


def _load(name: str) -> dict:
    with (FIXTURES / f"{name}.json").open() as f:
        return json.load(f)


def _make_args() -> argparse.Namespace:
    """Minimal Namespace sufficient to construct UnifiCamBase for process() tests."""
    return argparse.Namespace(
        cert=str(CERT_PATH),
        token="Y6aU73WEsQuehmnlg7PKMbu1JZybzOK5",
        host="192.168.1.1",
        mac="FF:FF:FF:FF:FF:BB",
        ip="192.168.1.42",
        name="Driveway",
        model="UVC G4 Dome",
        fw_version="5.3.95",
        snapshot_url=None,
        source=None,
        video1="rtsp://example/stream",
        video2=None,
        video3=None,
        ffmpeg_args="-c:v copy",
        ffmpeg_base_args=None,
        rtsp_transport="tcp",
        timestamp_modifier=90,
        loglevel="error",
        doorbell=True,
        zone_map='{"roads": 1, "parking": 5, "plants": 6, "front_door": 7}',
    )


class _TestCam(UnifiCamBase):
    """Minimal concrete camera for exercising init_adoption/process without RTSP/ffmpeg."""

    async def get_snapshot(self) -> Path:
        return Path("/tmp/nonexistent.jpg")

    async def get_stream_source(self, stream_index: str) -> str:
        return "rtsp://example/stream"

    def probe_video_resolution(self, stream_index: str, source_url: str) -> tuple[int, int]:
        # Avoid real ffmpeg probing in unit tests.
        return self._detected_resolutions.get(stream_index, (2560, 1920))

    def get_uptime(self) -> float:
        # Pin to the Driveway capture value so structural compare works.
        return 3.0

    async def get_feature_flags(self) -> dict[str, Any]:
        # Match the Driveway capture feature set (doorbell + smartDetect).
        flags = await super().get_feature_flags()
        flags["doorbell"] = True
        flags["smartDetect"] = ["person", "vehicle", "animal", "package"]
        return flags

    async def start_video_stream(self, stream_index: str, destination: str, stream_name: str = "") -> None:
        # Avoid spawning ffmpeg/nc during unit tests.
        self.logger.debug(
            f"stub start_video_stream({stream_index!r}, {destination!r}, {stream_name!r})"
        )


@pytest.fixture
def cam() -> _TestCam:
    logger = logging.getLogger("test_pairing")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    instance = _TestCam(_make_args(), logger)
    instance._sent: list[dict[str, Any]] = []

    async def _capture_send(msg: dict) -> None:
        instance.logger.debug(f"Sending: {msg}")
        instance._sent.append(msg)

    instance.send = _capture_send  # type: ignore[method-assign]
    return instance


@pytest.mark.asyncio
async def test_init_adoption_sends_client_hello(cam: _TestCam):
    """
    Live path: init_adoption() must emit the outbound ubnt_avclient_hello
    that matches the Driveway capture (modulo nondeterministic fields).
    """
    await cam.init_adoption()

    assert len(cam._sent) == 1, f"expected exactly one outbound message, got {len(cam._sent)}"
    sent = cam._sent[0]

    assert sent["functionName"] == "ubnt_avclient_hello"
    assert sent["from"] == "ubnt_avclient"
    assert sent["to"] == "UniFiVideo"
    assert sent["responseExpected"] is False
    assert sent["inResponseTo"] == 0

    payload = sent["payload"]
    assert payload["adoptionCode"] == cam.args.token
    assert payload["connectionHost"] == cam.args.host
    assert payload["connectionSecurePort"] == 7442
    assert payload["fwVersion"] == cam.args.fw_version
    assert payload["ip"] == cam.args.ip
    assert payload["mac"] == "FFFFFFFFFFBB"
    assert payload["model"] == "UVC G4 Dome"
    assert payload["name"] == "Driveway"
    assert payload["protocolVersion"] == 67
    assert payload["features"]["doorbell"] is True
    assert payload["features"]["smartDetect"] == [
        "person",
        "vehicle",
        "animal",
        "package",
    ]

    # Full structural match against the capture (ignores messageId/timeStamp/uptime).
    assert_message_equal(
        sent,
        _load("client_hello"),
        msg="init_adoption outbound hello vs Driveway capture",
    )


@pytest.mark.asyncio
async def test_server_hello_reply_produces_silence(cam: _TestCam):
    """
    Live path: after the controller replies with ubnt_avclient_hello,
    process() must invoke process_hello and send nothing back.
    """
    server_hello = _load("server_hello_reply")
    raw = json.dumps(server_hello).encode()

    called = {"count": 0}
    original = cam.process_hello

    async def _spy(msg: dict) -> None:
        called["count"] += 1
        await original(msg)

    cam.process_hello = _spy  # type: ignore[method-assign]

    result = await cam.process(raw)

    assert called["count"] == 1, "process_hello was not invoked by process()"
    assert result is False
    assert cam._sent == [], f"expected silence, but process() sent: {cam._sent}"


@pytest.mark.asyncio
async def test_full_hello_handshake(cam: _TestCam):
    """
    End-to-end interaction matching the Driveway log:

      SEND  ubnt_avclient_hello  (client, messageId=1)
      RECV  ubnt_avclient_hello  (server, inResponseTo=1) → silence
    """
    # 1. Proxy initiates adoption → outbound client hello
    await cam.init_adoption()
    assert len(cam._sent) == 1
    client_hello = cam._sent[0]
    assert client_hello["functionName"] == "ubnt_avclient_hello"
    assert client_hello["from"] == "ubnt_avclient"
    client_msg_id = client_hello["messageId"]

    # 2. Controller replies (use capture, but wire inResponseTo to the live id)
    server_hello = _load("server_hello_reply")
    server_hello = dict(server_hello)
    server_hello["inResponseTo"] = client_msg_id

    result = await cam.process(json.dumps(server_hello).encode())

    # Still only the original client hello outbound — no reply to the server hello
    assert result is False
    assert len(cam._sent) == 1, (
        f"expected no additional outbound messages after server hello, "
        f"but queue is now: {cam._sent}"
    )
    assert_message_equal(
        cam._sent[0],
        _load("client_hello"),
        msg="client hello still matches capture after full handshake",
    )


@pytest.mark.asyncio
async def test_param_agreement_reply(cam: _TestCam):
    """
    Live path: controller sends ubnt_avclient_paramAgreement (responseExpected
    true) → process() must reply with authToken + features matching the
    Driveway capture.
    """
    request = _load("param_agreement_request")
    raw = json.dumps(request).encode()

    result = await cam.process(raw)

    assert result is False
    assert len(cam._sent) == 1, f"expected one reply, got {len(cam._sent)}: {cam._sent}"
    sent = cam._sent[0]

    assert sent["functionName"] == "ubnt_avclient_paramAgreement"
    assert sent["from"] == "ubnt_avclient"
    assert sent["to"] == "UniFiVideo"
    assert sent["responseExpected"] is False
    assert sent["inResponseTo"] == request["messageId"]

    payload = sent["payload"]
    assert payload["authToken"] == cam.args.token
    assert payload["features"]["doorbell"] is True
    assert payload["features"]["smartDetect"] == [
        "person",
        "vehicle",
        "animal",
        "package",
    ]

    assert_message_equal(
        sent,
        _load("param_agreement_reply"),
        msg="paramAgreement reply vs Driveway capture",
    )


@pytest.mark.asyncio
async def test_hello_then_param_agreement_sequence(cam: _TestCam):
    """
    Pairing sequence so far (Driveway log order):

      SEND  ubnt_avclient_hello
      RECV  ubnt_avclient_hello (server) → silence
      RECV  ubnt_avclient_paramAgreement → SEND reply
    """
    # 1. Client hello
    await cam.init_adoption()
    assert len(cam._sent) == 1
    assert cam._sent[0]["functionName"] == "ubnt_avclient_hello"
    client_msg_id = cam._sent[0]["messageId"]

    # 2. Server hello → silence
    server_hello = dict(_load("server_hello_reply"))
    server_hello["inResponseTo"] = client_msg_id
    await cam.process(json.dumps(server_hello).encode())
    assert len(cam._sent) == 1

    # 3. Param agreement → one reply
    request = _load("param_agreement_request")
    await cam.process(json.dumps(request).encode())
    assert len(cam._sent) == 2
    assert cam._sent[1]["functionName"] == "ubnt_avclient_paramAgreement"
    assert cam._sent[1]["inResponseTo"] == request["messageId"]

    assert_message_equal(
        cam._sent[0],
        _load("client_hello"),
        msg="client hello still matches after sequence",
    )
    assert_message_equal(
        cam._sent[1],
        _load("param_agreement_reply"),
        msg="paramAgreement reply matches after sequence",
    )


# Ordered cascade steps after hello + paramAgreement (Driveway log).
# expect_reply: None = silence; str = fixture name for full payload match;
# True = must reply with same functionName (payload not fully pinned).
_CASCADE_AFTER_PARAM = [
    ("stop_service", None),
    ("network_status", "network_status_reply"),
    ("change_device_settings_timezone", "change_device_settings_timezone_reply"),
    ("change_device_settings_name", None),
    ("change_osd_settings", "change_osd_settings_reply"),
    ("change_sound_led_1", None),
    ("change_brightness", None),
    ("change_sound_led_2", None),
    ("change_talkback", "change_talkback_reply"),
    ("change_video_audio", True),  # large stateful video payload
    ("audio_agent_tuning", None),
    ("change_isp", True),  # large ISP payload; shape asserted loosely
    ("change_video_fps", True),
    ("change_smart_motion", "change_smart_motion_reply"),
    ("enable_logging", None),
    ("change_device_analytics", None),
    ("smart_motion_test", "smart_motion_test_reply"),
    ("change_video_empty", True),
    ("change_clarity_zones", "change_clarity_zones_reply"),
    ("send_weather", None),
    ("change_audio_events", "change_audio_events_reply"),
    ("change_smart_detect", "change_smart_detect_reply"),
    ("change_interface", None),
    ("update_face_db", "update_face_db_reply"),
]


@pytest.mark.asyncio
async def test_pairing_config_cascade(cam: _TestCam):
    """
    Feed the remaining pairing cascade in Driveway log order.

    For each message:
      - responseExpected false (and not a video stream start) → silence
      - responseExpected true → one outbound with matching functionName
        and inResponseTo; stable replies also match captured payload.
    """
    outbound_before = 0

    for fixture_name, expect in _CASCADE_AFTER_PARAM:
        req = _load(fixture_name)
        before = len(cam._sent)
        await cam.process(json.dumps(req).encode())
        after = len(cam._sent)
        new_msgs = cam._sent[before:after]

        if expect is None:
            assert new_msgs == [], (
                f"{fixture_name}: expected silence, got {new_msgs}"
            )
        else:
            assert len(new_msgs) == 1, (
                f"{fixture_name}: expected 1 reply, got {len(new_msgs)}: {new_msgs}"
            )
            sent = new_msgs[0]
            assert sent["functionName"] == req["functionName"]
            assert sent["inResponseTo"] == req["messageId"]
            assert sent["from"] == "ubnt_avclient"
            assert sent["responseExpected"] is False
            if isinstance(expect, str):
                assert_message_equal(
                    sent,
                    _load(expect),
                    msg=f"{fixture_name} reply vs capture",
                )


@pytest.mark.asyncio
async def test_network_status_reply_shape(cam: _TestCam):
    """Pin NetworkStatus reply fields (static, high-signal)."""
    req = _load("network_status")
    await cam.process(json.dumps(req).encode())
    assert len(cam._sent) == 1
    assert_message_equal(
        cam._sent[0],
        _load("network_status_reply"),
        msg="NetworkStatus reply",
    )


@pytest.mark.asyncio
async def test_change_smart_detect_settings_enables_and_echoes(cam: _TestCam):
    """ChangeSmartDetectSettings must echo payload and enable smart detect."""
    req = _load("change_smart_detect")
    await cam.process(json.dumps(req).encode())
    assert len(cam._sent) == 1
    sent = cam._sent[0]
    assert sent["functionName"] == "ChangeSmartDetectSettings"
    assert sent["inResponseTo"] == req["messageId"]
    assert_message_equal(
        sent,
        _load("change_smart_detect_reply"),
        msg="ChangeSmartDetectSettings echo",
    )
    # Zone IDs from the request should be visible to the smart-detect path.
    zones = sent["payload"]["zones"]
    assert set(zones.keys()) == {"1", "5", "6", "7"}

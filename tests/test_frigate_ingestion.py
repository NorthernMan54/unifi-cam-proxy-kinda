"""
Frigate-injection test harness.

Complements test_pairing.py (which exercises the Protect -> proxy direction)
by exercising the Frigate -> proxy -> Protect direction: feed a real,
captured Frigate MQTT message into live FrigateCam code and assert on
exactly what gets sent to Protect via `self.send(...)`.

Unlike test_pairing.py, this harness constructs `args` via the real
argparse chain (`FrigateCam.add_parser` + top-level host/cert/mac/etc.
args from unifi.main.parse_args' parser) rather than a hand-built
Namespace, so it can't silently drift out of sync with the CLI as new
arguments are added.

Fixtures live in tests/fixtures/frigate_events/ and are verbatim captures
from real Frigate MQTT traffic (see TEST_CASE_CREATION_PLAN.md).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import assert_message_equal
from unifi.cams.frigate import FrigateCam

FIXTURES = Path(__file__).parent / "fixtures" / "frigate_events"


def _load(name: str) -> dict:
    with (FIXTURES / f"{name}.json").open() as f:
        return json.load(f)


class FakeMqttTopic:
    """Duck-typed stand-in for aiomqtt.Topic — only `.value` is read by
    FrigateEventHandlerMixin (handle_snapshot_event)."""

    def __init__(self, value: str) -> None:
        self.value = value


class FakeMqttMessage:
    """Duck-typed stand-in for aiomqtt.Message.

    FrigateEventHandlerMixin only ever reads `.payload` (bytes) and, for
    snapshot events, `.topic.value` — it never touches qos/retain/mid or
    any other paho-internal field. Constructing a real aiomqtt.Message
    requires a live paho MQTTMessage instance; this stub avoids that
    dependency entirely and keeps the harness fast and offline.
    """

    def __init__(self, payload: bytes, topic: str = "frigate/driveway/events") -> None:
        self.payload = payload
        self.topic = FakeMqttTopic(topic)


def _make_frigate_args(
    cert_path: Path,
    zone_map: str = '{"roads": 1, "parking": 5, "plants": 6, "front_door": 7}',
    **overrides: Any,
) -> argparse.Namespace:
    """Build args via the real argparse chain (top-level common args +
    FrigateCam.add_parser), so this can't drift from the actual CLI.
    `overrides` are appended as extra CLI tokens, e.g. doorbell=True ->
    '--doorbell'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", "-H", required=True)
    parser.add_argument("--nvr-username", required=False)
    parser.add_argument("--nvr-password", required=False)
    parser.add_argument("--cert", "-c", required=True)
    parser.add_argument("--token", "-t", required=False, default=None)
    parser.add_argument("--mac", "-m", default="AABBCCDDEEFF")
    parser.add_argument("--ip", "-i", default="192.168.1.10")
    parser.add_argument("--name", "-n", default="unifi-cam-proxy")
    parser.add_argument("--model", default="UVC G4 Dome")
    parser.add_argument("--fw-version", "-f", default="5.3.95")
    parser.add_argument("--verbose", "-v", action="store_true")
    sp = parser.add_subparsers(dest="impl", required=True)
    frigate_sp = sp.add_parser("frigate")
    FrigateCam.add_parser(frigate_sp)

    argv = [
        "--host", "192.168.1.1",
        "--cert", str(cert_path),
        "--mac", "FF:FF:FF:FF:FF:BB",
        "--ip", "192.168.1.42",
        "--name", "Driveway",
        "--model", "UVC G4 Dome",
        "--fw-version", "5.3.95",
        "frigate",
        "--mqtt-host", "192.168.1.3",
        "--frigate-camera", "driveway",
        "--video1", "rtsp://192.168.1.38:8554/driveway_src",
        "--zone-map", zone_map,
        # Setting snapshot-url skips RTSPCam.__init__'s real-ffmpeg
        # snapshot-monitor spawn entirely (it's gated on `not
        # args.snapshot_url`) -- without this, constructing a FrigateCam
        # outside a running event loop raises RuntimeError, and even
        # inside one it would shell out to real ffmpeg.
        "--snapshot-url", "http://example.invalid/snapshot.jpg",
    ]
    if overrides.get("doorbell"):
        argv.append("--doorbell")

    return parser.parse_args(argv)


class _TestFrigateCam(FrigateCam):
    """Minimal concrete FrigateCam for exercising Frigate-origin handlers
    without RTSP/ffmpeg/network side effects.

    `send` MUST be overridden here as a real method, not monkey-patched
    onto the instance after construction: SmartDetectEventManager and
    SmartMotionEventManager capture `send=self.send` as a bound-method
    reference inside FrigateCam.__init__, before any post-construction
    patch could take effect. Patching `instance.send` afterward silently
    fails to intercept anything routed through those managers -- the
    real send() runs instead, logs "Sending: ...", and drops the
    message on the floor since there's no live connection in a test.
    """

    async def send(self, msg: dict[str, Any]) -> None:
        self.logger.debug(f"[test] Captured send: {msg}")
        self._sent.append(msg)

    async def get_snapshot(self) -> Path:
        return Path("/tmp/nonexistent.jpg")

    async def get_stream_source(self, stream_index: str) -> str:
        return "rtsp://example/stream"

    def probe_video_resolution(self, stream_index: str, source_url: str) -> tuple[int, int]:
        return self._detected_resolutions.get(stream_index, (2560, 1920))

    def get_uptime(self) -> float:
        return 3.0

    async def start_video_stream(self, stream_index: str, destination: str, stream_name: str = "") -> None:
        self.logger.debug(
            f"stub start_video_stream({stream_index!r}, {destination!r}, {stream_name!r})"
        )


@pytest.fixture(scope="session")
def _test_cert(tmp_path_factory) -> Path:
    """Generate a throwaway self-signed cert once per test session (same
    recipe as ./createUnfiCert), rather than depending on a fixed /tmp
    path existing beforehand."""
    cert_dir = tmp_path_factory.mktemp("certs")
    key = cert_dir / "private.key"
    csr = cert_dir / "server.csr"
    pub = cert_dir / "public.key"
    pem = cert_dir / "test-client.pem"

    subprocess.run(
        ["openssl", "ecparam", "-out", str(key), "-name", "prime256v1", "-genkey", "-noout"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [
            "openssl", "req", "-new", "-sha256", "-key", str(key), "-out", str(csr),
            "-subj", "/C=TW/L=Taipei/O=Ubiquiti Networks Inc./OU=devint/CN=camera.ubnt.dev",
        ],
        check=True, capture_output=True,
    )
    subprocess.run(
        [
            "openssl", "x509", "-req", "-sha256", "-days", "36500",
            "-in", str(csr), "-signkey", str(key), "-out", str(pub),
        ],
        check=True, capture_output=True,
    )
    pem.write_bytes(key.read_bytes() + pub.read_bytes())
    return pem


@pytest.fixture
def frigate_cam(_test_cert: Path) -> _TestFrigateCam:
    logger = logging.getLogger("test_frigate_ingestion")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    args = _make_frigate_args(_test_cert)
    instance = _TestFrigateCam(args, logger)
    instance._sent: list[dict[str, Any]] = []
    return instance


@pytest.fixture
def frigate_cam_4zone(_test_cert: Path) -> _TestFrigateCam:
    """Separate camera instance matching the --zone-map in effect for the
    2026-07-23 capture (new_stationary_car.json), which used a 4-zone
    config where 'parking' -> 2, not the 7-zone config ('parking' -> 5)
    used elsewhere. Only 'parking' is confirmed by that capture (via
    current_zones); zone_1/zone_3/zone_4 are unconfirmed placeholder
    names -- only the zone *IDs* matter for zonesStatus output, not
    their names, so this reproduces the real zone-ID set {1,2,3,4}
    faithfully without claiming to know what 1/3/4 were actually called.
    """
    logger = logging.getLogger("test_frigate_ingestion")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    args = _make_frigate_args(
        _test_cert,
        zone_map='{"zone_1": 1, "parking": 2, "zone_3": 3, "zone_4": 4}',
    )
    instance = _TestFrigateCam(args, logger)
    instance._sent: list[dict[str, Any]] = []
    return instance

@pytest.mark.skip(reason="Describe why you are skipping this test ")
@pytest.mark.asyncio
async def test_stationary_object_missed_new_event(frigate_cam: _TestFrigateCam):
    """
    Real Driveway capture: a Frigate 'update' event for a stationary car
    arrives with no prior tracked 'new' event and no active motion
    window. This exercises the recovery path in _handle_update_event
    (TrackerIdAllocator.get() falling back to allocate()) and must
    produce exactly one EventSmartDetect with edgeType='none'.
    """
    frigate_msg = _load("stationary_missed_new_event")
    message = FakeMqttMessage(json.dumps(frigate_msg).encode())

    await frigate_cam.handle_detection_event(message)

    assert len(frigate_cam._sent) == 1, (
        f"expected exactly one outbound message, got {len(frigate_cam._sent)}: "
        f"{frigate_cam._sent}"
    )
    sent = frigate_cam._sent[0]

    assert sent["functionName"] == "EventSmartDetect"
    assert sent["from"] == "ubnt_avclient"
    assert sent["to"] == "UniFiVideo"
    assert sent["responseExpected"] is False

    payload = sent["payload"]
    assert payload["edgeType"] == "none"
    assert len(payload["descriptors"]) == 1

    descriptor = payload["descriptors"][0]
    # TrackerIdAllocator starts every fresh instance at DEFAULT_BASE_TRACKER_ID
    # (700000), so this is deterministic for a freshly-constructed cam --
    # not a coincidence pinned from the capture.
    assert descriptor["trackerID"] == 700000
    assert descriptor["objectType"] == "vehicle"
    assert descriptor["stationary"] is True
    assert descriptor["confidenceLevel"] == 88  # round(0.88623046875 * 100)
    assert descriptor["licensePlate"] == "DEPM-782"
    # "parking" -> 5 per the --zone-map used to build frigate_cam
    assert descriptor["zones"] == [5]

    zones_status = payload["zonesStatus"]
    assert zones_status["5"]["status"] == "enter"


@pytest.mark.asyncio
async def test_new_stationary_object(frigate_cam_4zone: _TestFrigateCam):
    """
    Real Driveway capture (2026-07-23): a Frigate 'new' event for a car
    that is already stationary (position_changes=0) on first detection --
    no prior 'new' event, no motion window involved. Exercises
    TrackerIdAllocator.allocate() (fresh allocation, not the .get()
    fallback the missed-new-event test exercises) and the 'new'-event
    branch of _handle_new_event that also calls
    trigger_smart_detect_stationary() when position_changes == 0.

    NOTE: the archived log line for this capture shows an older version
    of the descriptor shape (positionChangesBefore/After, depth, speed
    keys; no licensePlate key). Current code's descriptor shape has
    since changed (confirmed separately by the missed-new-event test):
    it adds licensePlate and does not emit positionChangesBefore/After/
    depth/speed. This test asserts what CURRENT code actually produces,
    not the archived shape -- it's characterizing today's behavior, not
    replaying history verbatim.
    """
    frigate_msg = _load("new_stationary_car")
    message = FakeMqttMessage(json.dumps(frigate_msg).encode())
    event_id = frigate_msg["after"]["id"]

    assert event_id not in frigate_cam_4zone._active_frigate_events

    await frigate_cam_4zone.handle_detection_event(message)

    assert len(frigate_cam_4zone._sent) == 1, (
        f"expected exactly one outbound message, got {len(frigate_cam_4zone._sent)}: "
        f"{frigate_cam_4zone._sent}"
    )
    sent = frigate_cam_4zone._sent[0]

    assert sent["functionName"] == "EventSmartDetect"
    assert sent["from"] == "ubnt_avclient"
    assert sent["to"] == "UniFiVideo"
    assert sent["responseExpected"] is False

    payload = sent["payload"]
    assert payload["edgeType"] == "none"
    assert len(payload["descriptors"]) == 1

    descriptor = payload["descriptors"][0]
    # Fresh instance -> TrackerIdAllocator.allocate() starts at 700000,
    # same as the missed-new-event test, but via the direct allocate()
    # path rather than the get()-fallback recovery path.
    assert descriptor["trackerID"] == 700000
    assert descriptor["objectType"] == "vehicle"
    assert descriptor["stationary"] is True
    assert descriptor["confidenceLevel"] == 78  # int(0.7880859375 * 100), truncated not rounded
    assert descriptor["zones"] == [2]  # 'parking' -> 2 per this capture's zone-map

    # Confirmed bookkeeping side effects mirroring the missed-new-event test.
    assert event_id in frigate_cam_4zone._active_frigate_events
    assert frigate_cam_4zone.tracker_ids.peek(event_id) == 700000
    # 'new' events with position_changes==0 go straight to
    # trigger_smart_detect_stationary(), never touching
    # _motion_smart_event_id -- only the position_changes>0 branch does.
    assert frigate_cam_4zone._motion_smart_event_id is None

    zones_status = payload["zonesStatus"]
    assert set(zones_status.keys()) == {"1", "2", "3", "4"}
    assert zones_status["2"]["status"] == "enter"

@pytest.mark.skip(reason="Describe why you are skipping this test ")
@pytest.mark.asyncio
async def test_stationary_object_recovers_tracking_state(frigate_cam: _TestFrigateCam):
    """
    Same capture, but asserting on internal recovery bookkeeping rather
    than just the outbound message -- this is the state the log lines
    ("Stationary object detected without motion window", "Recovered
    unknown Frigate update as active object") describe.
    """
    frigate_msg = _load("stationary_missed_new_event")
    event_id = frigate_msg["after"]["id"]
    message = FakeMqttMessage(json.dumps(frigate_msg).encode())

    assert event_id not in frigate_cam._active_frigate_events

    await frigate_cam.handle_detection_event(message)

    assert event_id in frigate_cam._active_frigate_events
    assert frigate_cam.tracker_ids.peek(event_id) == 700000
    # No motion window was ever started for this recovery path.
    assert frigate_cam._motion_smart_event_id is None

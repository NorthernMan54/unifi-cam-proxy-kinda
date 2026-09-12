import unittest
from typing import get_origin

from unifi.cameras import FrigateCam, RTSPCam, UnifiCamBase
from unifi.core import Core
from unifi.events import SmartDetectEvent, SmartDetectObjectType, SmartMotionEvent
from unifi.protocol import AVClientRequest, AVClientResponse
from unifi.transport import WebSocketTransport


class ArchitectureCompatibilityTests(unittest.TestCase):
    def test_legacy_core_name_points_to_transport(self) -> None:
        self.assertIs(Core, WebSocketTransport)

    def test_camera_namespace_exports_implementations(self) -> None:
        self.assertTrue(issubclass(RTSPCam, UnifiCamBase))
        self.assertTrue(issubclass(FrigateCam, RTSPCam))

    def test_protocol_and_event_models_are_importable(self) -> None:
        self.assertIs(get_origin(AVClientRequest), dict)
        self.assertIs(get_origin(AVClientResponse), dict)
        self.assertEqual(SmartDetectObjectType.PERSON.value, "person")
        self.assertTrue(SmartDetectEvent)
        self.assertTrue(SmartMotionEvent)

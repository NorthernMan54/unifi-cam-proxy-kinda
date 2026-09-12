import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from unifi.cams.rtsp import RTSPCam


class RTSPCameraRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = object.__new__(RTSPCam)
        self.camera.args = SimpleNamespace(loglevel="error", rtsp_transport="tcp")
        self.camera.logger = logging.getLogger("test.rtsp")
        self.camera.snapshot_dir = "/tmp/test-snapshots"
        self.camera.stream_source = {"video1": "rtsp://camera/stream"}

    @patch("unifi.cams.rtsp.subprocess.Popen")
    def test_restarts_exited_snapshot_process(self, popen: MagicMock) -> None:
        exited_process = MagicMock()
        exited_process.poll.return_value = 1
        self.camera.snapshot_stream = exited_process

        self.camera._start_snapshot_stream()

        popen.assert_called_once()
        self.assertIs(self.camera.snapshot_stream, popen.return_value)


class RTSPMotionAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_http_handlers_use_analytics_lifecycle(self) -> None:
        camera = object.__new__(RTSPCam)
        camera.logger = logging.getLogger("test.rtsp.http")
        camera.trigger_analytics_start = AsyncMock()
        camera.trigger_analytics_stop = AsyncMock()

        start_response = await camera._http_start_motion(MagicMock())
        stop_response = await camera._http_stop_motion(MagicMock())

        camera.trigger_analytics_start.assert_awaited_once_with()
        camera.trigger_analytics_stop.assert_awaited_once_with()
        self.assertEqual(start_response.text, "ok")
        self.assertEqual(stop_response.text, "ok")

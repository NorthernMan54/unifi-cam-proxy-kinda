"""Application-level camera orchestration."""

from typing import Any

from unifi.transport.websocket import WebSocketTransport


async def run_camera(args: Any, camera: Any, logger: Any) -> None:
    """Run a configured camera using the default Protect transport."""
    await WebSocketTransport(args, camera, logger).run()


__all__ = ["run_camera"]

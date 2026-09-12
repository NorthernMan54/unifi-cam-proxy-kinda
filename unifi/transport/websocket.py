"""UniFi Protect WebSocket connection lifecycle."""

import asyncio
import ssl
from typing import Any

import backoff
import websockets


class RetryableError(Exception):
    """Signal that the current camera session should be re-established."""


class WebSocketTransport:
    """Own the Protect WebSocket and the tasks attached to one session."""

    def __init__(self, args: Any, camera: Any, logger: Any) -> None:
        self.host = args.host
        self.token = args.token
        self.mac = args.mac
        self.logger = logger
        self.cam = camera

        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        self.ssl_context.load_cert_chain(args.cert, args.cert)

    async def run(self) -> None:
        uri = f"wss://{self.host}:7442/camera/1.0/ws?token={self.token}"
        headers = {
            "camera-mac": self.mac.replace(":", "").replace("-", "").upper(),
            "camera-model": "0xa573",
        }

        @backoff.on_predicate(
            backoff.expo,
            lambda retryable: retryable,
            factor=2,
            jitter=None,
            max_value=10,
            logger=self.logger,
        )
        async def connect() -> bool:
            self.logger.info("Creating ws connection to %s:7442", self.host)
            try:
                async with websockets.connect(
                    uri,
                    extra_headers=headers,
                    ssl=self.ssl_context,
                    subprotocols=["secure_transfer"],
                ) as ws:
                    tasks = {
                        asyncio.create_task(self.cam._run(ws)),
                        asyncio.create_task(self.cam.run()),
                    }
                    try:
                        await asyncio.gather(*tasks)
                    except RetryableError:
                        return True
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except websockets.exceptions.InvalidStatusCode as error:
                if error.status_code == 403:
                    self.logger.error(
                        "The adoption token is invalid. Generate a new one and retry."
                    )
                elif error.status_code == 429:
                    return True
                raise
            except (asyncio.TimeoutError, ConnectionRefusedError) as error:
                self.logger.warning("Connection to %s failed: %s", self.host, error)
                return True
            finally:
                await self.cam.close()
            return False

        await connect()


# Historical name retained while callers migrate to WebSocketTransport.
Core = WebSocketTransport

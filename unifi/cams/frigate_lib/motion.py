"""
Motion-level helpers for the Frigate <-> UniFi Protect bridge:
  - loading/refreshing Frigate's configured detect.fps for a camera
  - deriving an EventSmartMotion level (0-100) from a bounding-box area
  - fetching a more accurate motion level from Frigate's recordings API
"""
import asyncio
import logging
import time
from typing import Any, Optional

import aiohttp

DEFAULT_DETECT_FPS = 5.0  # Frigate's own hardcoded default for detect.fps
DETECT_FPS_MIN, DETECT_FPS_MAX = 1.0, 30.0  # sanity bounds; guards against a bad/renamed config field
DETECT_FPS_REFRESH_SECONDS = 300  # re-fetch periodically in case Frigate config changes live
RECORDINGS_MOTION_CACHE_SECONDS = 3.0


class DetectFpsTracker:
    """Tracks Frigate's detect.fps for one camera, refreshed periodically."""

    def __init__(self, frigate_http_url: Optional[str], camera: str, logger: logging.Logger) -> None:
        self.frigate_http_url = (frigate_http_url or "http://frigate:5000").rstrip("/")
        self.camera = camera
        self.logger = logger
        self.fps: float = DEFAULT_DETECT_FPS

    async def load(self) -> None:
        """
        Load detect FPS from Frigate's resolved config (/api/config, not
        /api/config/raw -- the resolved endpoint merges in Frigate's defaults
        so detect.fps is present even if never set explicitly in config.yml).
        Falls back through camera-level -> global -> hardcoded default, with a
        final sanity bound, so a bad/renamed field can't silently corrupt
        every idleSinceTimeMs calculation downstream.
        """
        config_url = f"{self.frigate_http_url}/api/config"
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(config_url) as response:
                    if response.status != 200:
                        self.logger.warning(
                            f"Unable to load Frigate config from {config_url}: HTTP {response.status}. "
                            f"Keeping current detect_fps={self.fps}"
                        )
                        return
                    config = await response.json()

            cam_cfg = config.get("cameras", {}).get(self.camera, {})
            fps = cam_cfg.get("detect", {}).get("fps")
            if not fps:
                fps = config.get("detect", {}).get("fps")
            if not fps:
                fps = DEFAULT_DETECT_FPS
                self.logger.warning(
                    f"Frigate config missing detect.fps for camera '{self.camera}' "
                    f"at every level; using default {fps}"
                )

            fps = float(fps)
            if not (DETECT_FPS_MIN <= fps <= DETECT_FPS_MAX):
                self.logger.warning(
                    f"Frigate detect.fps={fps} out of sane bounds "
                    f"[{DETECT_FPS_MIN}, {DETECT_FPS_MAX}]; keeping previous value {self.fps} instead"
                )
                return

            self.fps = fps
            self.logger.info(f"Frigate detect FPS for camera '{self.camera}': {self.fps}")
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as e:
            self.logger.warning(
                f"Failed to fetch Frigate detect FPS from {config_url}: {e}. "
                f"Keeping current detect_fps={self.fps}"
            )

    async def refresh_periodically(self) -> None:
        """Re-fetch detect.fps periodically in case Frigate's config is changed live."""
        while True:
            await asyncio.sleep(DETECT_FPS_REFRESH_SECONDS)
            await self.load()


def motion_level_from_area(area: int, detect_width: int, detect_height: int) -> int:
    """
    Convert a Frigate bounding-box area (pixels in the detection frame) to an
    EventSmartMotion level (0-100).

    Level = area / (detect_width * detect_height) * 100, clamped to [0, 100].
    This gives a natural 0-100 scale where a box filling the whole frame is
    100 and an empty frame is 0.
    """
    total = detect_width * detect_height
    if total <= 0:
        return 50
    return min(100, max(0, int(area * 100 / total)))


async def fetch_recordings_motion_level(
    frigate_http_url: Optional[str],
    camera: str,
    event_data: dict[str, Any],
    logger: logging.Logger,
) -> Optional[int]:
    """
    Fetch the most recent motion percentage from Frigate's recordings API for
    the analytics event described by `event_data` (mutated in place for
    caching: `recordings_motion_last_fetch`, `recordings_motion_level`).

    The recordings endpoint already reports motion on a 0-100 scale, so it is
    a better source for EventSmartMotion levels than re-deriving a value from
    bounding-box area when it is available.
    """
    if not frigate_http_url:
        return None

    current_time = time.time()
    last_fetch = float(event_data.get("recordings_motion_last_fetch") or 0.0)
    cached_motion = event_data.get("recordings_motion_level")
    if last_fetch and current_time - last_fetch < RECORDINGS_MOTION_CACHE_SECONDS:
        return cached_motion if isinstance(cached_motion, int) else None

    start_time = float(event_data.get("start_time") or current_time)
    after = max(0, int(start_time) - 5)
    before = max(after + 1, int(current_time))
    recordings_url = f"{frigate_http_url.rstrip('/')}/api/{camera}/recordings"

    try:
        timeout = aiohttp.ClientTimeout(total=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                recordings_url, params={"after": after, "before": before}
            ) as response:
                event_data["recordings_motion_last_fetch"] = current_time
                if response.status != 200:
                    logger.debug(f"Frigate recordings motion lookup failed: HTTP {response.status}")
                    return None
                payload = await response.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as e:
        event_data["recordings_motion_last_fetch"] = current_time
        logger.debug(f"Frigate recordings motion lookup failed: {e}")
        return None

    if not isinstance(payload, list) or not payload:
        return None

    latest_recording = max(
        payload, key=lambda item: float(item.get("end_time") or item.get("start_time") or 0.0)
    )

    try:
        motion_level = int(float(latest_recording.get("motion") or 0))
    except (TypeError, ValueError):
        return None

    motion_level = min(100, max(0, motion_level))
    event_data["recordings_motion_level"] = motion_level
    event_data["recordings_motion_last_fetch"] = current_time
    return motion_level

"""
Snapshot fetching for the Frigate <-> UniFi Protect bridge.

This consolidates what were previously two near-duplicate implementations in
FrigateCam (`_fetch_and_cache_frigate_event_snapshot` and
`fetch_snapshots_for_event`), each with their own `fetch_url` closure that
downloaded a URL to a NamedTemporaryFile. `fetch_snapshots_for_event` also had
a dead branch keyed on a local `frigate_event_id` variable that was always set
to `None` immediately before being checked -- that branch could never execute
and has been dropped.
"""
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import aiohttp

DEFAULT_FETCH_TIMEOUT = 5.0


async def download_to_tempfile(
    url: str,
    logger: logging.Logger,
    *,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
    suffix: str = ".jpg",
) -> Optional[Path]:
    """Download `url` to a temp file, returning its Path, or None on failure."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                data = await response.read()
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.close()
    logger.debug(f"Fetched {url} ({len(data)} bytes) -> {f.name}")
    return Path(f.name)


class FrigateSnapshotClient:
    """Builds Frigate snapshot URLs and fetches them, for one camera."""

    def __init__(self, frigate_http_url: Optional[str], camera: str, logger: logging.Logger) -> None:
        self.frigate_http_url = frigate_http_url.rstrip("/") if frigate_http_url else None
        self.camera = camera
        self.logger = logger

    @property
    def enabled(self) -> bool:
        return bool(self.frigate_http_url)

    def event_snapshot_urls(self, frigate_event_id: str) -> tuple[str, str]:
        """URLs for a specific Frigate event's snapshot: (full, cropped-thumbnail)."""
        base_url = f"{self.frigate_http_url}/api/events/{frigate_event_id}/snapshot.jpg"
        return base_url, f"{base_url}?crop=1&quality=80"

    def latest_snapshot_urls(self, timestamp: Optional[int] = None) -> tuple[str, str]:
        """URLs for the camera's current live snapshot: (full, thumbnail)."""
        base_url = f"{self.frigate_http_url}/api/{self.camera}/latest.jpg"
        full_url = f"{base_url}?timestamp={timestamp}" if timestamp is not None else base_url
        thumbnail_url = f"{base_url}?height=360&quality=80"
        if timestamp is not None:
            thumbnail_url = f"{thumbnail_url}&timestamp={timestamp}"
        return full_url, thumbnail_url

    async def fetch_event_snapshots(self, frigate_event_id: str) -> tuple[Optional[Path], Optional[Path]]:
        """Fetch (crop, full) snapshots for a specific Frigate event."""
        if not self.enabled:
            return None, None
        full_url, crop_url = self.event_snapshot_urls(frigate_event_id)
        self.logger.debug(
            f"Fetching event-specific snapshots for Frigate event {frigate_event_id}: {full_url}"
        )
        fov_path, crop_path = await asyncio.gather(
            download_to_tempfile(full_url, self.logger),
            download_to_tempfile(crop_url, self.logger),
        )
        return crop_path, fov_path

    async def fetch_latest_snapshots(
        self, timestamp: Optional[int] = None
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Fetch (crop/thumbnail, full) snapshots of the camera's current view."""
        if not self.enabled:
            self.logger.warning("Cannot fetch snapshots: frigate_http_url not configured")
            return None, None
        full_url, thumbnail_url = self.latest_snapshot_urls(timestamp)
        self.logger.debug(f"Fetching snapshots: full={full_url}, thumbnail={thumbnail_url}")
        fov_path, crop_path = await asyncio.gather(
            download_to_tempfile(full_url, self.logger),
            download_to_tempfile(thumbnail_url, self.logger),
        )
        self.logger.info(
            f"Fetched snapshots: crop={'✓' if crop_path else '✗'}, fov={'✓' if fov_path else '✗'}"
        )
        return crop_path, fov_path

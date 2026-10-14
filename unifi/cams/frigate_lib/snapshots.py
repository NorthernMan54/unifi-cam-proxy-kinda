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
import atexit
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import aiohttp

DEFAULT_FETCH_TIMEOUT = 5.0

# Snapshot temp files are referenced from event dicts that are retained for up
# to an hour after the event ends, and ownership is shared between several
# managers, so rather than tracking each file we keep them all in one
# directory and prune anything older than this.
SNAPSHOT_MAX_AGE = 2 * 3600
SNAPSHOT_PRUNE_INTERVAL = 60.0

_snapshot_dir: Optional[Path] = None
_last_prune = 0.0


def snapshot_dir() -> Path:
    """Per-process directory holding snapshot temp files; removed at exit."""
    global _snapshot_dir
    if _snapshot_dir is None or not _snapshot_dir.is_dir():
        _snapshot_dir = Path(tempfile.mkdtemp(prefix="unifi-cam-proxy-snapshots-"))
        atexit.register(shutil.rmtree, _snapshot_dir, ignore_errors=True)
    return _snapshot_dir


def prune_snapshot_dir(max_age: float = SNAPSHOT_MAX_AGE, now: Optional[float] = None) -> int:
    """Delete snapshot temp files older than `max_age` seconds. Returns count removed."""
    cutoff = (now if now is not None else time.time()) - max_age
    removed = 0
    for path in snapshot_dir().iterdir():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            pass
    return removed


def write_snapshot_tempfile(data: bytes, suffix: str = ".jpg") -> Path:
    """Write `data` to a new file in the snapshot dir, pruning stale files periodically."""
    global _last_prune
    now = time.time()
    if now - _last_prune >= SNAPSHOT_PRUNE_INTERVAL:
        _last_prune = now
        prune_snapshot_dir(now=now)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=snapshot_dir()) as f:
        f.write(data)
    return Path(f.name)


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

    path = write_snapshot_tempfile(data, suffix)
    logger.debug(f"Fetched {url} ({len(data)} bytes) -> {path}")
    return path


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
        return base_url, f"{base_url}?crop=1&bbox=1&quality=80"

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

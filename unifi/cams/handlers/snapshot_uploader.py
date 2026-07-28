"""
Handles pushing snapshot bytes to UniFi Protect, whether the bytes come
from a file already on disk or need to be fetched from a URL (e.g. Frigate's
latest.jpg) first.

Extracted out of SnapshotHandlers so the "get bytes from somewhere, POST to
Protect" concern has no dependency on smart-detect/motion-analytics event
state -- it's just an I/O helper.
"""
import asyncio
import logging
import ssl
from pathlib import Path
from typing import Any, Optional

import aiohttp


class SnapshotUploader:
    def __init__(self, ssl_context: ssl.SSLContext, logger: logging.Logger) -> None:
        self._ssl_context = ssl_context
        self.logger = logger

    async def fetch_and_upload(
        self,
        snapshot_url: str,
        upload_uri: str,
        form_fields: dict[str, Any],
        snapshot_type: str,
    ) -> bool:
        """Fetch snapshot bytes from a URL, then upload them to Protect."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    snapshot_url, timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        self.logger.warning(
                            f"Failed to fetch {snapshot_type}: HTTP "
                            f"{response.status}, {error_body} from {snapshot_url}"
                        )
                        return False

                    image_data = await response.read()
                    self.logger.info(
                        f"Fetched {snapshot_type} from Frigate ({len(image_data)} bytes)"
                    )

                    files = {"payload": image_data}
                    files.update(form_fields)

                    await session.post(upload_uri, data=files, ssl=self._ssl_context)
                    self.logger.debug(f"Uploaded {snapshot_type}")
                    return True

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {snapshot_type} from {snapshot_url}")
            return False
        except aiohttp.ClientError:
            self.logger.exception(
                f"Failed to fetch/upload {snapshot_type} from {snapshot_url}"
            )
            return False

    async def upload_file(
        self,
        file_path: Optional[Path],
        upload_uri: str,
        form_fields: dict[str, Any],
        snapshot_type: str,
    ) -> bool:
        """Upload a file already on disk to Protect."""
        if not file_path or not file_path.exists():
            self.logger.warning(f"Snapshot file {file_path} not ready for {snapshot_type}")
            return False

        try:
            # NOTE: original implementation opened this file with a bare
            # open() call and never closed it -- a file-descriptor leak on
            # every snapshot request. Fixed here with a context manager.
            async with aiohttp.ClientSession() as session:
                with open(file_path, "rb") as fh:
                    files = {"payload": fh}
                    files.update(form_fields)
                    await session.post(upload_uri, data=files, ssl=self._ssl_context)
            self.logger.debug(f"Uploaded {snapshot_type} from {file_path}")
            return True
        except aiohttp.ClientError:
            self.logger.exception(f"Failed to upload {snapshot_type}")
            return False

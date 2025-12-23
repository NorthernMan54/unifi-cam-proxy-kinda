import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import aiohttp


class SnapshotHandlers:
    """Mixin class providing snapshot management functionality"""

    def update_motion_snapshot(self, path: Path) -> None:
        """
        Update motion snapshot (legacy method).
        By default, updates all three snapshot types to the same path.
        For more granular control, use update_motion_snapshots().
        """
        self._motion_snapshot = path
        self._motion_snapshot_crop = path
        self._motion_snapshot_fov = path
        self._motion_heatmap = path
    
    def update_motion_snapshots(
        self,
        crop: Optional[Path] = None,
        fov: Optional[Path] = None,
        heatmap: Optional[Path] = None,
    ) -> None:
        """
        Update specific motion snapshot types.
        
        Args:
            crop: Path to cropped snapshot with bounding box (motionSnapshot)
            fov: Path to full field-of-view snapshot with bounding box (motionSnapshotFullFoV)
            heatmap: Path to heatmap visualization (motionHeatmap)
        """
        if crop is not None:
            self._motion_snapshot_crop = crop
            self._motion_snapshot = crop  # Update legacy field
        if fov is not None:
            self._motion_snapshot_fov = fov
        if heatmap is not None:
            self._motion_heatmap = heatmap

    async def process_snapshot_request(
        self, msg: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Process a snapshot request from UniFi Protect.
        
        Handles cached snapshot files (from motion events) and fallback to Frigate API.
        """
        snapshot_type = msg["payload"]["what"]
        filename = msg["payload"].get("filename", "")
        
        self.logger.debug(f"Snapshot request: type={snapshot_type}, filename={filename}")
        
        # Try to find cached snapshot file from event history
        cached_path = self._find_cached_snapshot(filename, snapshot_type)
        
        if cached_path and cached_path.exists():
            # Serve cached snapshot
            self.logger.info(f"Serving cached {snapshot_type} from {cached_path}")
            await self._upload_file_to_protect(
                cached_path,
                msg["payload"]["uri"],
                msg["payload"].get("formFields", {}),
                snapshot_type
            )
        else:
            # Fallback to Frigate API or regular snapshot
            use_frigate = (
                hasattr(self.args, 'frigate_http_url') and 
                self.args.frigate_http_url
            )
            
            if use_frigate:
                # Fetch from Frigate latest.jpg endpoint
                snapshot_url = self._build_frigate_fallback_url(snapshot_type)
                self.logger.info(f"Fetching {snapshot_type} from Frigate (no cached): {snapshot_url}")
                await self._fetch_and_upload_snapshot(
                    snapshot_url,
                    msg["payload"]["uri"],
                    msg["payload"].get("formFields", {}),
                    snapshot_type
                )
            else:
                # Use regular snapshot method
                await self._process_motion_event_snapshot(msg, snapshot_type)
        
        if msg["responseExpected"]:
            return self.gen_response("GetRequest", response_to=msg["messageId"])

    def _find_cached_snapshot(self, filename: str, snapshot_type: str) -> Optional[Path]:
        """
        Find cached snapshot file from event history.
        
        Args:
            filename: Filename from GetRequest (is actually a file path in our implementation)
            snapshot_type: Type of snapshot requested
            
        Returns:
            Path to cached snapshot file, or None if not found
        """
        # UniFi Protect modifies filenames for FoV snapshots by appending _fullfov before extension
        # e.g., /tmp/tmpvgwqfkeo.jpg -> /tmp/tmpvgwqfkeo_fullfov.jpg
        # Strip this suffix to find the original cached file
        original_filename = filename
        if filename and snapshot_type == "motionSnapshotFullFoV" and "_fullfov" in filename:
            # Remove _fullfov suffix to get original filename
            original_filename = filename.replace("_fullfov", "")
            self.logger.debug(
                f"UniFi modified FoV filename: {filename} -> looking for original: {original_filename}"
            )
        
        # The filename is actually the cached file path that we set in trigger_analytics_stop
        if original_filename and "/" in original_filename:
            # It's a file path - try to use it directly
            path = Path(original_filename)
            if path.exists():
                self.logger.debug(f"Found cached snapshot at original path: {path}")
                return path
        
        # Fallback: search through event history for matching snapshots
        # This handles edge cases where filename doesn't match expected format
        for event_data in self._analytics_event_history.values():
            # Match based on snapshot type
            if snapshot_type == "motionSnapshot":
                cached_path = event_data.get("snapshot_crop_path")
            elif snapshot_type == "motionSnapshotFullFoV":
                cached_path = event_data.get("snapshot_fov_path")
            elif snapshot_type == "motionHeatmap":
                cached_path = event_data.get("heatmap_path")
            else:
                continue
            
            if cached_path and isinstance(cached_path, Path) and cached_path.exists():
                # Check if this matches the requested filename (original or modified)
                if (filename and str(cached_path) == filename) or \
                   (original_filename and str(cached_path) == original_filename):
                    return cached_path
        
        return None

    def _build_frigate_fallback_url(self, snapshot_type: str) -> str:
        """
        Build Frigate fallback URL for latest snapshot.
        
        Args:
            snapshot_type: Type of snapshot requested
            
        Returns:
            URL to Frigate's latest.jpg endpoint
        """
        base_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
        
        # Add quality params for thumbnails
        if snapshot_type == "motionSnapshot":
            return f"{base_url}?height=360&quality=80"
        
        return base_url


    async def _fetch_and_upload_snapshot(
        self, 
        snapshot_url: str, 
        upload_uri: str, 
        form_fields: dict[str, Any],
        snapshot_type: str
    ) -> bool:
        """
        Fetch snapshot from URL and upload to UniFi Protect.
        
        Args:
            snapshot_url: URL to fetch snapshot from
            upload_uri: UniFi Protect upload endpoint
            form_fields: Additional form fields for upload
            snapshot_type: Type of snapshot for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(snapshot_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        self.logger.warning(
                            f"Failed to fetch {snapshot_type}: HTTP {response.status}, {error_body}"
                        )
                        return False
                    
                    image_data = await response.read()
                    self.logger.info(f"Fetched {snapshot_type} from Frigate ({len(image_data)} bytes)")
                    
                    # Upload to UniFi Protect
                    files = {"payload": image_data}
                    files.update(form_fields)
                    
                    await session.post(upload_uri, data=files, ssl=self._ssl_context)
                    self.logger.debug(f"Uploaded {snapshot_type}")
                    return True
                    
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {snapshot_type}")
            return False
        except aiohttp.ClientError:
            self.logger.exception(f"Failed to fetch/upload {snapshot_type}")
            return False

    async def _upload_file_to_protect(
        self, 
        file_path: Optional[Path], 
        upload_uri: str, 
        form_fields: dict[str, Any], 
        snapshot_type: str
    ) -> bool:
        """
        Upload a file from disk to UniFi Protect.
        
        Args:
            file_path: Path to snapshot file
            upload_uri: UniFi Protect upload endpoint
            form_fields: Additional form fields for upload
            snapshot_type: Type of snapshot for logging
            
        Returns:
            True if successful, False otherwise
        """
        if not file_path or not file_path.exists():
            self.logger.warning(f"Snapshot file {file_path} not ready for {snapshot_type}")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                files = {"payload": open(file_path, "rb")}
                files.update(form_fields)
                await session.post(upload_uri, data=files, ssl=self._ssl_context)
                self.logger.debug(f"Uploaded {snapshot_type} from {file_path}")
                return True
        except aiohttp.ClientError:
            self.logger.exception(f"Failed to upload {snapshot_type}")
            return False


    async def _process_motion_event_snapshot(
        self, msg: dict[str, Any], snapshot_type: str
    ) -> None:
        """
        Process motion event snapshot (crop, FoV, heatmap).
        
        Args:
            msg: Message from UniFi Protect
            snapshot_type: Type of snapshot requested
        """
        # Select appropriate snapshot based on request type
        if snapshot_type == "motionSnapshot":
            # Cropped image with bounding box
            path = self._motion_snapshot_crop or self._motion_snapshot
        elif snapshot_type == "motionSnapshotFullFoV":
            # Full field of view image with bounding box
            path = self._motion_snapshot_fov or self._motion_snapshot
        elif snapshot_type == "motionHeatmap":
            # Heatmap visualization (use FoV as fallback)
            path = self._motion_heatmap or self._motion_snapshot_fov or self._motion_snapshot
        elif snapshot_type == "smartDetectZoneSnapshot":
            # Smart detect zone snapshot (use crop)
            path = self._motion_snapshot_crop or self._motion_snapshot
        else:
            # Regular snapshot request (fallback to get_snapshot method)
            path = await self.get_snapshot()

        await self._upload_file_to_protect(
            path,
            msg["payload"]["uri"],
            msg["payload"].get("formFields", {}),
            snapshot_type
        )

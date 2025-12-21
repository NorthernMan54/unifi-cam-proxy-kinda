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
        
        Handles three types of snapshot requests:
        1. URL-based filenames (Frigate API integration)
        2. Regular snapshots with quality parameters
        3. Motion event snapshots (crop, FoV, heatmap)
        """
        snapshot_type = msg["payload"]["what"]
        filename = msg["payload"].get("filename", "")
        
        self.logger.debug(f"Snapshot request: type={snapshot_type}, filename={filename}")
        
        # Check if filename contains URL parameters (indicates Frigate API URL)
        if filename and ("?" in filename or filename.startswith("latest.jpg")):
            await self._process_frigate_url_snapshot(msg, snapshot_type, filename)
        # Check if this is a regular snapshot request and Frigate is configured
        elif snapshot_type == "snapshot" and hasattr(self.args, 'frigate_http_url') and hasattr(self.args, 'frigate_camera'):
            if self.args.frigate_http_url:
                await self._process_frigate_regular_snapshot(msg)
            else:
                # Fall back to legacy method if frigate_http_url not configured
                await self._process_legacy_snapshot(msg)
        else:
            # Legacy path-based snapshot handling for motion events
            await self._process_motion_event_snapshot(msg, snapshot_type)

        if msg["responseExpected"]:
            return self.gen_response("GetRequest", response_to=msg["messageId"])

    async def _process_frigate_url_snapshot(
        self, msg: dict[str, Any], snapshot_type: str, filename: str
    ) -> None:
        """Process URL-based snapshot request (Frigate API integration)"""
        if not (hasattr(self.args, 'frigate_http_url') and hasattr(self.args, 'frigate_camera')):
            self.logger.warning(f"URL-based filename but Frigate configuration not available")
            return
            
        if not self.args.frigate_http_url:
            self.logger.warning(f"URL-based filename but frigate_http_url not configured")
            return
        
        # Determine query parameters based on snapshot type
        # motionSnapshot: 360p thumbnail with crop
        # motionSnapshotFullFoV: full resolution
        # motionHeatmap: full resolution (same as FoV)
        if snapshot_type == "motionSnapshot":
            # Thumbnail version with height and quality parameters
            query_params = "height=360&quality=80"
        elif snapshot_type == "motionSnapshotFullFoV":
            # Full resolution, no additional parameters beyond timestamp
            query_params = ""
        elif snapshot_type == "motionHeatmap":
            # Heatmap uses full resolution
            query_params = ""
        else:
            # Default to no extra parameters
            query_params = ""
        
        # Extract timestamp from filename if present
        timestamp_param = ""
        if "timestamp=" in filename:
            # Extract existing timestamp parameter
            timestamp_param = filename.split("?", 1)[1] if "?" in filename else ""
        
        # Build final query string
        if query_params and timestamp_param:
            final_query = f"{query_params}&{timestamp_param}"
        elif query_params:
            final_query = query_params
        elif timestamp_param:
            final_query = timestamp_param
        else:
            final_query = ""
        
        # Build the full URL to Frigate
        # Always use 'latest.jpg' as the base - Frigate API endpoint
        # (UniFi may send latest_fullfov.jpg, latest_heatmap.jpg, etc. but Frigate only has latest.jpg)
        if final_query:
            snapshot_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg?{final_query}"
        else:
            snapshot_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
        
        self.logger.info(f"Fetching {snapshot_type} from Frigate: {snapshot_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch from Frigate
                async with session.get(snapshot_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        self.logger.info(f"Fetched {snapshot_type} from Frigate ({len(image_data)} bytes)")
                        
                        # Upload to UniFi Protect
                        files = {"payload": image_data}
                        files.update(msg["payload"].get("formFields", {}))
                        
                        try:
                            await session.post(
                                msg["payload"]["uri"],
                                data=files,
                                ssl=self._ssl_context,
                            )
                            self.logger.debug(f"Uploaded {snapshot_type} from Frigate URL")
                        except aiohttp.ClientError:
                            self.logger.exception("Failed to upload snapshot to UniFi Protect")
                    else:
                        error_body = await response.text()
                        self.logger.warning(
                            f"Failed to fetch {snapshot_type} from Frigate: "
                            f"HTTP {response.status}, Response: {error_body}"
                        )
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {snapshot_type} from Frigate")
        except Exception as e:
            self.logger.error(f"Error fetching {snapshot_type} from Frigate: {e}")

    async def _process_frigate_regular_snapshot(self, msg: dict[str, Any]) -> None:
        """Process regular snapshot request with Frigate integration"""
        # Determine query parameters based on quality
        quality_param = msg["payload"].get("quality", "medium")
        
        # Map UniFi quality levels to height parameters
        if quality_param == "high":
            query_params = "height=1080&quality=95"
        elif quality_param == "medium":
            query_params = "height=720&quality=85"
        elif quality_param == "low":
            query_params = "height=360&quality=70"
        else:
            # Default to medium quality
            query_params = "height=720&quality=85"
        
        # Build the full URL to Frigate
        snapshot_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg?{query_params}"
        
        self.logger.info(f"Fetching snapshot (quality={quality_param}) from Frigate: {snapshot_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch from Frigate
                async with session.get(snapshot_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        self.logger.info(f"Fetched snapshot from Frigate ({len(image_data)} bytes)")
                        
                        # Upload to UniFi Protect
                        files = {"payload": image_data}
                        files.update(msg["payload"].get("formFields", {}))
                        
                        try:
                            await session.post(
                                msg["payload"]["uri"],
                                data=files,
                                ssl=self._ssl_context,
                            )
                            self.logger.debug(f"Uploaded snapshot from Frigate")
                        except aiohttp.ClientError:
                            self.logger.exception("Failed to upload snapshot to UniFi Protect")
                    else:
                        error_body = await response.text()
                        self.logger.warning(
                            f"Failed to fetch snapshot from Frigate: "
                            f"HTTP {response.status}, Response: {error_body}"
                        )
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching snapshot from Frigate")
        except Exception as e:
            self.logger.error(f"Error fetching snapshot from Frigate: {e}")

    async def _process_legacy_snapshot(self, msg: dict[str, Any]) -> None:
        """Process snapshot using legacy get_snapshot method"""
        path = await self.get_snapshot()
        if path and path.exists():
            async with aiohttp.ClientSession() as session:
                files = {"payload": open(path, "rb")}
                files.update(msg["payload"].get("formFields", {}))
                try:
                    await session.post(
                        msg["payload"]["uri"],
                        data=files,
                        ssl=self._ssl_context,
                    )
                    self.logger.debug(f"Uploaded snapshot from {path}")
                except aiohttp.ClientError:
                    self.logger.exception("Failed to upload snapshot")
        else:
            self.logger.warning(f"Snapshot file {path} is not ready yet, skipping upload")

    async def _process_motion_event_snapshot(
        self, msg: dict[str, Any], snapshot_type: str
    ) -> None:
        """Process motion event snapshot (crop, FoV, heatmap)"""
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

        if path and path.exists():
            async with aiohttp.ClientSession() as session:
                files = {"payload": open(path, "rb")}
                files.update(msg["payload"].get("formFields", {}))
                try:
                    await session.post(
                        msg["payload"]["uri"],
                        data=files,
                        ssl=self._ssl_context,
                    )
                    self.logger.debug(f"Uploaded {snapshot_type} from {path}")
                except aiohttp.ClientError:
                    self.logger.exception("Failed to upload snapshot")
        else:
            self.logger.warning(
                f"Snapshot file {path} is not ready yet, skipping upload for {snapshot_type}"
            )

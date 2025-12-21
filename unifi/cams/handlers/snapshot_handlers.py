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
        
        Handles URL-based snapshots (Frigate), regular snapshots, and motion event snapshots.
        """
        snapshot_type = msg["payload"]["what"]
        filename = msg["payload"].get("filename", "")
        
        self.logger.debug(f"Snapshot request: type={snapshot_type}, filename={filename}")
        
        # Use Frigate if configured and either URL-based or regular snapshot
        use_frigate = (
            hasattr(self.args, 'frigate_http_url') and 
            self.args.frigate_http_url and
            (filename and "?" in filename or snapshot_type == "snapshot")
        )
        
        if use_frigate:
            await self._process_frigate_snapshot(msg, snapshot_type, filename)
        else:
            await self._process_motion_event_snapshot(msg, snapshot_type)
        
        if msg["responseExpected"]:
            return self.gen_response("GetRequest", response_to=msg["messageId"])

    def _build_frigate_url(self, snapshot_type: str, filename: str) -> str:
        """
        Build Frigate snapshot URL with appropriate parameters.
        
        Args:
            snapshot_type: Type of snapshot (motionSnapshot, motionSnapshotFullFoV, etc.)
            filename: Filename from UniFi request (may contain event_id or timestamp)
            
        Returns:
            Complete URL to fetch snapshot from Frigate
        """
        base_url = f"{self.args.frigate_http_url}/api/{self.args.frigate_camera}/latest.jpg"
        
        params = []
        
        # Add quality params for thumbnails
        if snapshot_type == "motionSnapshot":
            params.extend(["height=360", "quality=80"])
        
        # Extract and convert event_id to timestamp
        if "event_id=" in filename:
            event_id_str = filename.split("event_id=", 1)[1].split("&", 1)[0]
            try:
                event_id = int(event_id_str)
                event_data = self._analytics_event_history.get(event_id)
                if event_data and event_data.get("start_time"):
                    params.append(f"timestamp={int(event_data['start_time'])}")
                    self.logger.debug(
                        f"Converted event_id {event_id} to timestamp {int(event_data['start_time'])} "
                        f"from analytics event history"
                    )
                else:
                    self.logger.warning(
                        f"Event ID {event_id} not found in analytics event history "
                        f"(history size: {len(self._analytics_event_history)})"
                    )
            except ValueError:
                self.logger.warning(f"Invalid event_id in filename: {event_id_str}")
        elif "timestamp=" in filename:
            # Preserve existing timestamp parameter
            timestamp_part = filename.split("?", 1)[1] if "?" in filename else ""
            if timestamp_part:
                params.append(timestamp_part)
        
        return f"{base_url}?{('&'.join(params))}" if params else base_url

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

    async def _process_frigate_snapshot(
        self, msg: dict[str, Any], snapshot_type: str, filename: str = ""
    ) -> None:
        """
        Process snapshot request using Frigate API.
        
        Args:
            msg: Message from UniFi Protect
            snapshot_type: Type of snapshot requested
            filename: Filename from request (may contain event_id or timestamp)
        """
        snapshot_url = self._build_frigate_url(snapshot_type, filename)
        self.logger.info(f"Fetching {snapshot_type} from Frigate: {snapshot_url}")
        
        await self._fetch_and_upload_snapshot(
            snapshot_url,
            msg["payload"]["uri"],
            msg["payload"].get("formFields", {}),
            snapshot_type
        )

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

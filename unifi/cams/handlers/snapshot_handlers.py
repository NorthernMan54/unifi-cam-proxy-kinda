"""
Mixin providing snapshot management functionality: serving cached snapshots
back to UniFi Protect on GetRequest, or falling back to a fresh Frigate/
camera snapshot when nothing is cached.

Refactored (Phase 2.5) to no longer reach directly into
SmartDetectEventManager / MotionAnalyticsManager internals -- it now asks
each manager `find_snapshot_by_path(...)` instead of walking their event
dicts by hand. Upload I/O (fetch-from-url-and-post, or read-file-and-post)
is delegated to SnapshotUploader, which also fixes a file-descriptor leak
that existed in the previous inline implementation.
"""
import logging
from pathlib import Path
from typing import Any, Optional

from unifi.cams.handlers.snapshot_uploader import SnapshotUploader
from unifi.cams.handlers.snapshot_utils import build_frigate_fallback_url, get_image_dimensions


class SnapshotHandlers:
    # Note: this is a plain mixin with no __init__ of its own -- UnifiCamBase
    # doesn't call super().__init__() through its MRO, so state here is
    # lazily created on first use rather than in a constructor.

    @property
    def _uploader(self) -> SnapshotUploader:
        if getattr(self, "_snapshot_uploader", None) is None:
            self._snapshot_uploader = SnapshotUploader(self._ssl_context, self.logger)
        return self._snapshot_uploader

    def update_motion_snapshot(self, path: Path) -> None:
        """
        Legacy method: updates all three snapshot types to the same path.
        For granular control, use update_motion_snapshots() instead.
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
        """Update specific motion snapshot types (crop, full-FoV, heatmap)."""
        if crop is not None:
            self._motion_snapshot_crop = crop
            self._motion_snapshot = crop  # keep legacy field in sync
        if fov is not None:
            self._motion_snapshot_fov = fov
        if heatmap is not None:
            self._motion_heatmap = heatmap

    async def process_snapshot_request(self, msg: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process a GetRequest snapshot request from UniFi Protect."""
        snapshot_type = msg["payload"]["what"]
        filename = msg["payload"].get("filename", "")

        self.logger.debug(f"Snapshot request: type={snapshot_type}, filename={filename}")

        cached_path = self._find_cached_snapshot(filename, snapshot_type)
        snapshot_path = None

        if cached_path and cached_path.exists():
            snapshot_path = cached_path
            self.logger.info(f"Serving cached {snapshot_type} from {cached_path}")
            await self._uploader.upload_file(
                cached_path,
                msg["payload"]["uri"],
                msg["payload"].get("formFields", {}),
                snapshot_type,
            )
        else:
            frigate_http_url = getattr(self.args, "frigate_http_url", None)

            if frigate_http_url:
                snapshot_url = build_frigate_fallback_url(
                    frigate_http_url, self.args.frigate_camera, snapshot_type
                )
                self.logger.info(f"Fetching {snapshot_type} from Frigate (no cached): {snapshot_url}")
                await self._uploader.fetch_and_upload(
                    snapshot_url,
                    msg["payload"]["uri"],
                    msg["payload"].get("formFields", {}),
                    snapshot_type,
                )
            else:
                snapshot_path = await self._process_motion_event_snapshot(msg, snapshot_type)

        if msg["responseExpected"]:
            width, height = (640, 360)
            if snapshot_path:
                width, height = get_image_dimensions(snapshot_path, self.logger)

            return self.gen_response(
                "GetRequest",
                response_to=msg["messageId"],
                payload={"height": height, "width": width},
            )

    def _find_cached_snapshot(self, filename: str, snapshot_type: str) -> Optional[Path]:
        """
        Find a cached snapshot file matching a GetRequest's filename.

        UniFi Protect appends "_fullfov" to the filename it requests for
        full-field-of-view snapshots (e.g. /tmp/x.jpg -> /tmp/x_fullfov.jpg),
        so we strip that back off to find the original cached path first.
        Falls back to asking each event manager whether it's tracking a
        snapshot matching either the original or modified filename.
        """
        original_filename = filename
        if filename and snapshot_type == "motionSnapshotFullFoV" and "_fullfov" in filename:
            original_filename = filename.replace("_fullfov", "")
            self.logger.debug(
                f"UniFi modified FoV filename: {filename} -> looking for original: {original_filename}"
            )

        if original_filename and "/" in original_filename:
            path = Path(original_filename)
            if path.exists():
                self.logger.debug(f"Found cached snapshot at original path: {path}")
                return path

        # Delegate the dict-shape knowledge to whichever manager owns it --
        # this mixin no longer knows what an analytics/smart-detect event
        # dict looks like internally.
        found = self._motion_analytics.find_snapshot_by_path(snapshot_type, filename, original_filename)
        if found:
            return found

        return self._smart_events.find_snapshot_by_path(snapshot_type, filename, original_filename)

    async def _process_motion_event_snapshot(
        self, msg: dict[str, Any], snapshot_type: str
    ) -> Optional[Path]:
        """Fallback path when nothing is cached and there's no Frigate URL configured."""
        if snapshot_type == "motionSnapshot":
            path = self._motion_snapshot_crop or self._motion_snapshot
        elif snapshot_type == "motionSnapshotFullFoV":
            path = self._motion_snapshot_fov or self._motion_snapshot
        elif snapshot_type == "motionHeatmap":
            path = self._motion_heatmap or self._motion_snapshot_fov or self._motion_snapshot
        elif snapshot_type == "smartDetectZoneSnapshot":
            path = self._motion_snapshot_crop or self._motion_snapshot
        else:
            path = await self.get_snapshot()

        await self._uploader.upload_file(
            path,
            msg["payload"]["uri"],
            msg["payload"].get("formFields", {}),
            snapshot_type,
        )

        return path

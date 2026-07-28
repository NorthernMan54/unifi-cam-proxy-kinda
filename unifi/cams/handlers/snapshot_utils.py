"""
Pure, stateless helpers for snapshot image handling.

Extracted from UnifiCamBase (refactor Phase 1). None of these functions touch
instance state directly -- callers pass in whatever they need (logger,
detected stream resolutions, etc). This makes them trivially unit-testable
without spinning up a full camera instance.
"""
import logging
from pathlib import Path
from typing import Any, Optional

DEFAULT_DIMENSIONS = (640, 360)


def get_image_dimensions(
    image_path: Optional[Path],
    logger: Optional[logging.Logger] = None,
) -> tuple[int, int]:
    """
    Read the pixel dimensions of an image file on disk.

    Falls back to DEFAULT_DIMENSIONS (640x360) if the file is missing,
    unreadable, or of a format we can't parse without PIL.
    """
    if not image_path or not image_path.exists():
        return DEFAULT_DIMENSIONS

    try:
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except ImportError:
            with image_path.open("rb") as f:
                header = f.read(24)
                if header[:8] == b"\x89PNG\r\n\x1a\n":
                    f.seek(16)
                    width = int.from_bytes(f.read(4), byteorder="big")
                    height = int.from_bytes(f.read(4), byteorder="big")
                    return (width, height)
                elif header[:2] == b"\xff\xd8":
                    if logger:
                        logger.debug(
                            "JPEG format detected but dimensions parsing "
                            "not implemented without PIL"
                        )
                    return DEFAULT_DIMENSIONS
                else:
                    if logger:
                        logger.debug(f"Unknown image format for {image_path}")
                    return DEFAULT_DIMENSIONS
    except Exception as e:
        if logger:
            logger.debug(
                f"Could not read image dimensions from {image_path}: {e}"
            )
        return DEFAULT_DIMENSIONS


def calculate_snapshot_dimensions(
    descriptor: dict[str, Any],
    detected_resolutions: dict[str, tuple[int, int]],
    snapshot_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[int, int]:
    """
    Determine snapshot dimensions, preferring an actual file on disk,
    falling back to a bounding-box estimate derived from the descriptor's
    normalized coordinates against the detect-stream resolution.
    """
    if snapshot_path:
        width, height = get_image_dimensions(snapshot_path, logger)
        if (width, height) != DEFAULT_DIMENSIONS:
            return (width, height)

    coord = descriptor.get("coord")
    if coord and len(coord) >= 4:
        try:
            x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]
            stream_width, stream_height = detected_resolutions.get(
                "video3", DEFAULT_DIMENSIONS
            )
            bbox_width = abs(x2 - x1) * stream_width
            bbox_height = abs(y2 - y1) * stream_height
            return (max(int(bbox_width), 100), max(int(bbox_height), 100))
        except (ValueError, TypeError, IndexError) as e:
            if logger:
                logger.debug(f"Could not parse coord from descriptor: {e}")

    return DEFAULT_DIMENSIONS


def build_frigate_fallback_url(
    frigate_http_url: str,
    frigate_camera: str,
    snapshot_type: str,
) -> str:
    """Build the Frigate `latest.jpg` fallback URL for a snapshot request."""
    base_url = f"{frigate_http_url}/api/{frigate_camera}/latest.jpg".lower()
    if snapshot_type == "motionSnapshot":
        return f"{base_url}?height=360&quality=80"
    return base_url

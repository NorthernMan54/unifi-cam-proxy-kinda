"""
Translation of a raw Frigate MQTT event message into a UniFi Protect
descriptor dict.
"""
import logging
from typing import Any, Optional

from unifi.cams.base import SmartDetectObjectType

UNIFI_GRID_SIZE = 1000  # both zone polygons and descriptor boxes use a normalized 0-1000 grid

_VEHICLE_LABELS = {"vehicle", "car", "motorcycle", "school_bus", "license_plate"}
_ANIMAL_LABELS = {
    "animal", "cat", "dog", "horse", "rabbit", "squirrel", "goat", "deer",
    "bird", "raccoon", "fox", "bear", "cow", "skunk", "kangaroo",
}


def label_to_object_type(label: str) -> Optional[SmartDetectObjectType]:
    if label == "person":
        # Available: person, face
        return SmartDetectObjectType.PERSON
    elif label in _VEHICLE_LABELS:
        # Available: car, motorcycle, bicycle, boat, school_bus, license_plate
        return SmartDetectObjectType.VEHICLE
    elif label in _ANIMAL_LABELS:
        # Available: animal, dog, cat, deer, horse, bird, raccoon, fox, bear,
        # cow, squirrel, goat, rabbit, skunk, kangaroo
        return SmartDetectObjectType.ANIMAL
    elif label == "package":
        return SmartDetectObjectType.PACKAGE
    return None


def translate_zones(
    current_zones: list[str],
    zone_name_to_id: dict[str, int],
    logger: logging.Logger,
    camera_name: str,
) -> list[int]:
    """
    Frigate's `current_zones` -> Protect numeric zone IDs via the static
    per-camera map. Order preserved as given (most-recently-entered first,
    matching Frigate's own convention); unmapped zone names are dropped
    rather than raising, so a config gap degrades to "zone-less" instead of
    crashing the bridge.
    """
    translated = []
    for zname in current_zones:
        zid = zone_name_to_id.get(zname)
        if zid is not None:
            translated.append(zid)
        else:
            logger.debug(
                f"Frigate zone '{zname}' has no entry in --zone-map for "
                f"camera '{camera_name}'; omitting from zones[]"
            )
    return translated


def build_descriptor_from_frigate_msg(
    frigate_msg: dict[str, Any],
    object_type: SmartDetectObjectType,
    tracker_id: int,
    *,
    frigate_detect_width: int,
    frigate_detect_height: int,
    frigate_time_sync_ms: int,
    frigate_detect_fps: float,
    zone_name_to_id: dict[str, int],
    logger: logging.Logger,
    camera_name: str,
) -> dict[str, Any]:
    """
    Build a UniFi Protect-compatible descriptor from Frigate event data.

    `tracker_id` must be pre-allocated via `TrackerIdAllocator` -- Frigate's
    own event id (a string) is never used as the numeric trackerID directly.

    Coordinate system note (confirmed against real device captures): both
    zone polygons and descriptor bounding boxes use a normalized 0-1000 grid,
    independent of any stream's actual pixel resolution -- NOT the low-res
    sub-stream's raw pixel space. The scaling below is correct.
    """
    after = frigate_msg.get("after", {})
    event_type = after.get("type", "unknown")

    box = after.get("box")
    if box and len(box) == 4:
        # Frigate box format: [x_min, y_min, x_max, y_max] in configured pixel dimensions
        # UniFi format: [x, y, width, height] in a 0-1000 normalized coordinate system
        x_scale = UNIFI_GRID_SIZE / frigate_detect_width
        y_scale = UNIFI_GRID_SIZE / frigate_detect_height

        x_min_unifi = int(box[0] * x_scale)
        y_min_unifi = int(box[1] * y_scale)
        x_max_unifi = int(box[2] * x_scale)
        y_max_unifi = int(box[3] * y_scale)

        coord = [x_min_unifi, y_min_unifi, x_max_unifi - x_min_unifi, y_max_unifi - y_min_unifi]
    else:
        coord = [0, 0, 1920, 1080]

    # Extract confidence score (Frigate uses 0.0-1.0, UniFi uses 0-100)
    score = after.get("top_score" if event_type == "end" else "score", 0.95)
    confidence_level = int(score * 100)

    stationary = after.get("stationary", False)

    # Extract position_changes for stationary object detection
    before = frigate_msg.get("before", {})
    position_changes_before = before.get("position_changes", 0) if before else 0
    position_changes_after = after.get("position_changes", 0)

    current_zones = after.get("current_zones", [])
    zones = translate_zones(current_zones, zone_name_to_id, logger, camera_name)

    # boxColor: real devices show "red" for objects actively occupying a zone
    # of interest and "white" for background/idle detections outside any zone
    # (protocol spec Section 3, descriptors[].boxColor).
    box_color = "red" if zones else "white"

    average_speed = after.get("average_estimated_speed", 0)
    speed = float(average_speed) if average_speed > 0 else None

    # License plate (vehicles) via Frigate's recognized_license_plate.
    license_plate_data = after.get("recognized_license_plate")
    license_plate = None
    license_plate_score = None
    if license_plate_data and isinstance(license_plate_data, list) and len(license_plate_data) >= 1:
        license_plate = license_plate_data[0]
        if len(license_plate_data) >= 2:
            license_plate_score = license_plate_data[1]

    # Face recognition (persons) via Frigate's generic sub_label facility.
    # Real Protect devices only populate name/tag when there's an actual
    # recognition match (see protocol spec Section 3, name/tag) -- they are
    # not a static branding string, so don't hardcode one here.
    sub_label_data = after.get("sub_label")
    recognized_name = None
    if (
        object_type == SmartDetectObjectType.PERSON
        and sub_label_data
        and isinstance(sub_label_data, list)
        and len(sub_label_data) >= 1
    ):
        recognized_name = sub_label_data[0]

    if object_type == SmartDetectObjectType.VEHICLE and license_plate:
        name = f"{license_plate} ({license_plate_score:.1%})" if license_plate_score is not None else license_plate
    elif object_type == SmartDetectObjectType.PERSON and recognized_name:
        name = recognized_name
    else:
        name = ""

    # Real devices mirror name into tag for the same matched entity rather
    # than using a static "Tagged by Frigate" label.
    tag = name

    # idleSinceTimeMs derivation: converts Frigate's motionless_count (a frame
    # counter) into an epoch-ms instant anchored to frame_time, using the
    # camera's actual detect.fps rather than assuming 1 frame == 1 second.
    # Recomputing this every message is safe -- as long as frame_time and
    # motionless_count advance together at a consistent fps, the derived
    # instant stays pinned automatically (validated against real device
    # idleSinceTimeMs behavior, which stays constant across a stationary
    # track's lifetime).
    motionless_count = after.get("motionless_count", 0)
    idle_since_ms = 0
    if motionless_count > 0:
        idle_since_ms = (
            int(after.get("frame_time", 0) * 1000)
            - int(motionless_count * (1000 / frigate_detect_fps))
            - frigate_time_sync_ms
        )

    descriptor = {
        "attributes": None,
        "boxColor": box_color,
        "confidenceLevel": confidence_level,
        "coord": coord,
        "coord3d": [-1, -1],
        "firstShownTimeMs": int(after.get("start_time", 0) * 1000) - frigate_time_sync_ms,
        "idleSinceTimeMs": idle_since_ms,
        "lines": [],
        "loiterZones": [],
        "name": name,
        "objectType": object_type.value,
        "secondLensZones": [],
        "stationary": stationary,
        "tag": tag,
        "trackerID": tracker_id,
        "zones": zones,
    }

    if license_plate:
        descriptor["licensePlate"] = license_plate

    logger.debug(
        f"Built descriptor: trackerID={tracker_id}, confidence={confidence_level}, "
        f"coord={coord}, stationary={stationary}, speed={speed}, "
        f"licensePlate={license_plate}, zones={zones}, boxColor={box_color}, name={name!r}"
    )

    return descriptor


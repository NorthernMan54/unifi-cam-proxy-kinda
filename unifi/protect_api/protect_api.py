# smart_detect.py

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar, Union


class SmartDetectObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PACKAGE = "package"
    FACE = "face"
    LICENSEPLATE = "licensePlate"


class SmartDetectEdgeType(Enum):
    ENTER = "enter"
    MOVING = "moving"
    LEAVE = "leave"


class ProtectFunctionName(Enum):
    EVENT_SMART_DETECT = "EventSmartDetect"
    # add more as discovered


@dataclass
class SmartDetectSnapshot:
    clockBestMonotonic: int
    clockBestWall: int
    smartDetectSnapshot: str
    smartDetectSnapshotHeight: int
    smartDetectSnapshotName: str
    smartDetectSnapshotType: str
    smartDetectSnapshotWidth: int
    trackerID: int                  


@dataclass
class SmartDetectDescriptor:
    confidenceLevel: int
    coord: list[int]                # [x, y, width, height]
    coord3d: list[int]              # [-1, -1] if unavailable
    firstShownTimeMs: int
    idleSinceTimeMs: int
    objectType: SmartDetectObjectType
    stationary: bool
    trackerID: int                          # unique per camera, incremented for each new object, never reused, and never reset on camera reboot; corresponds to trackerID in SmartDetectSnapshot, unique per object and consistent across events for that object until it leaves the scene and is removed from the tracker map; not globally unique and not guaranteed to be unique across cameras
    zones: list[int]
    name: str = ''
    tag: str = ''
    boxColor: str = 'red'
    attributes: dict | None = None
    depth: float | None = None
    speed: float | None = None
    lines: list = field(default_factory=list)
    loiterZones: list = field(default_factory=list)
    secondLensZones: list = field(default_factory=list)


@dataclass
class TrackerAttr:
    objectType: SmartDetectObjectType
    zone: list[list[int]]


@dataclass
class ZoneStatus:
    level: int
    status: SmartDetectEdgeType


@dataclass
class SmartDetectPayload:
    # required fields
    clockMonotonic: int                                         # Time in ms since camera boot
    clockStream: int                                            # Time in ms since stream start
    clockStreamRate: int                                        # Multiplier to convert clockStream to wall-clock ms
    clockWall: int                                              # Wall-clock timestamp in ms since epoch
    edgeType: SmartDetectEdgeType
    eventId: int                                                # UniFi event ID, incremented for each EventSmartDetect
    objectTypes: list[SmartDetectObjectType]
    zonesStatus: dict[str, ZoneStatus]
    descriptors: list[SmartDetectDescriptor]                    # [] on LEAVE, populated on ENTER/MOVING
    displayTimeoutMSec: int                                     # always supplied
    # LEAVE only (defaulted)
    smartDetectSnapshotFullFoV: str = ""                        # Only populated on LEAVE
    smartDetectSnapshotFullFoVHeight: int = 0                   # 0 if not provided; only used on LEAVE
    smartDetectSnapshotFullFoVWidth: int = 0                    # 0 if not provided; only used on LEAVE
    smartDetectSnapshots: list[SmartDetectSnapshot] = field(default_factory=list)   # Only populated on LEAVE
    trackerIDAttrMap: dict[str, TrackerAttr] = field(default_factory=dict)          # Only present on LEAVE


ProtectPayload = Union[
    SmartDetectPayload,
    # MotionPayload,        # add as discovered
    # CameraUpdatePayload,  # add as discovered
]

T = TypeVar('T', bound=ProtectPayload)


@dataclass
class ProtectResponseMessage(Generic[T]):
    functionName: ProtectFunctionName
    inResponseTo: int               # 0 for events, or the messageId of the request this is responding to
    messageId: int
    payload: T
    responseExpected: bool
    to: str = 'UniFiVideo'          # In a response, always 'UniFiVideo'; in a request, usually 'ubnt_avclient'
    timeStamp: str = ''             # ISO 8601 timestamp, e.g. "2024-06-30T12:34:56.789Z"
    from_: str = 'ubnt_avclient'


@dataclass
class EventSmartDetect(ProtectResponseMessage[SmartDetectPayload]):
    pass
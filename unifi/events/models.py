"""Typed state models for future manager migrations."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SmartDetectObjectType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PACKAGE = "package"


@dataclass
class SmartDetectEvent:
    event_id: int
    object_type: SmartDetectObjectType
    start_time: float
    end_time: float | None = None
    descriptor_history: list[dict[str, Any]] = field(default_factory=list)
    tracker_snapshots: dict[int, dict[str, Path | None]] = field(default_factory=dict)


@dataclass
class SmartMotionEvent:
    event_id: int
    start_time: float
    end_time: float | None = None
    smart_detect_event_ids: list[int] = field(default_factory=list)
    motion_levels: dict[str, int] = field(default_factory=dict)

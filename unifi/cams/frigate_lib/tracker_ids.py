"""
Allocates session-scoped, monotonically-increasing integer trackerIDs for
Frigate object tracks.

Frigate's own event/tracker ID is a string (e.g. a UUID-like value) and is NOT
reused as the numeric trackerID directly. Real UniFi Protect devices use
session-scoped, monotonically-increasing integer trackerIDs (observed: 716235
-> 716236 -> 716259 across one session). Deriving trackerID via
hash(frigate_id) % N is unsafe: Python's str hash is randomized per-process
(unstable across restarts) and offers no uniqueness guarantee against
collisions between concurrently active tracks. Instead, a fresh integer is
assigned once per Frigate "new" event and held for that track's lifetime.
"""
import logging
from typing import Optional

DEFAULT_BASE_TRACKER_ID = 700000  # arbitrary session-scoped base, mirrors observed value range


class TrackerIdAllocator:
    def __init__(self, logger: logging.Logger, base: int = DEFAULT_BASE_TRACKER_ID) -> None:
        self.logger = logger
        self._next_tracker_id = base
        self._frigate_id_to_tracker_id: dict[str, int] = {}

    def allocate(self, frigate_event_id: str) -> int:
        """Assign a fresh trackerID for a newly-started Frigate track. Call
        exactly once per Frigate event_type == "new"."""
        tracker_id = self._next_tracker_id
        self._next_tracker_id += 1
        self._frigate_id_to_tracker_id[frigate_event_id] = tracker_id
        return tracker_id

    def get(self, frigate_event_id: str) -> int:
        """Look up the trackerID already allocated for an in-progress Frigate
        track (update/end events). Falls back to allocating one if missing
        (e.g. a missed "new" event) so the bridge degrades gracefully rather
        than crashing, while logging the anomaly."""
        tracker_id = self._frigate_id_to_tracker_id.get(frigate_event_id)
        if tracker_id is None:
            self.logger.warning(
                f"No trackerID allocated for Frigate event_id={frigate_event_id} "
                f"(likely missed 'new' event); allocating one now"
            )
            tracker_id = self.allocate(frigate_event_id)
        return tracker_id

    def peek(self, frigate_event_id: str) -> Optional[int]:
        """Look up an allocated trackerID without allocating one if missing."""
        return self._frigate_id_to_tracker_id.get(frigate_event_id)

    def release(self, frigate_event_id: str) -> None:
        """Free the trackerID mapping once a Frigate track ends."""
        self._frigate_id_to_tracker_id.pop(frigate_event_id, None)

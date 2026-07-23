"""
Frigate MQTT event handling: motion window lifecycle, per-object new/update/end
detection events, doorbell, and snapshot topic handling.

Split out of FrigateCam into a mixin purely to keep file sizes manageable --
these methods assume `self` provides everything FrigateCam.__init__ sets up
(self.args, self.logger, self.tracker_ids, self.snapshots, zone tracker
helpers, etc.) plus the RTSPCam/UnifiCamBase trigger_* methods.
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from aiomqtt import Message

from unifi.cams.base import SmartDetectObjectType
from unifi.cams.frigate_lib.descriptors import label_to_object_type
from unifi.cams.frigate_lib.zones import ZoneStatusTracker


class FrigateEventHandlerMixin:


    async def handle_motion_event(self, message: Message) -> None:
        """Handle Frigate motion events: create/destroy the motion window smart event.

        Per protocol spec Section 3: one smart event spans from motion ON to
        motion OFF. All Frigate object detections within this window emit
        EventSmartDetect updates within that single smart event context.
        """
        if not isinstance(message.payload, bytes):
            self.logger.warning(f"Unexpectedly received non-bytes payload for motion event: {message.payload}")
            return
        msg = message.payload.decode()
        if msg == "ON":
            if self._is_motion_window_active():
                self.logger.warning("Motion ON received but analytics motion window is already active. Ignoring.")
                return

            self._motion_start_time = time.time()
            self.logger.debug("Frigate motion event: ON, creating motion window smart event")

            try:
                # trigger_analytics_start() allocates _active_analytics_event_id
                # immediately and only delays *sending* EventAnalytics(start)
                # via linger. Keep smart-detect event ID unset until the first
                # object arrives, then bootstrap trigger_smart_detect_start.
                await self.trigger_analytics_start()
                self._motion_smart_event_id = None
                self.logger.info(
                    f"Motion window started immediately (analytics_event_id={self._active_analytics_event_id}); "
                    "ready for object detections"
                )
                await self._update_motion_levels_from_recordings()
            except Exception as e:
                self.logger.error(f"Failed to start motion window: {e}")
                self._motion_smart_event_id = None

        elif msg == "OFF":
            if not self._is_motion_window_active():
                self.logger.debug("Motion OFF received but no active motion window, ignoring")
                return

            self.logger.debug("Frigate motion event: OFF, closing motion window smart event")

            try:
                # Any remaining active Frigate objects should have been ended
                # by Frigate's own motion-triggered cleanup, but if not, close
                # them here.
                for frigate_event_id in list(self._active_frigate_events):
                    self.logger.warning(
                        f"Motion OFF but Frigate event {frigate_event_id} still active. Force-closing it."
                    )
                    self._forget_track(frigate_event_id)

                if self._motion_smart_event_id is not None:
                    active_smart_event = self._active_smart_events.get(self._motion_smart_event_id)
                    if active_smart_event is not None:
                        await self.trigger_smart_detect_stop(
                            active_smart_event["object_type"],
                            event_id=self._motion_smart_event_id,
                            event_timestamp=int(round(time.time() * 1000)),
                            zonesStatus=self.zone_status_tracker.as_dict(),
                        )

                await self.trigger_analytics_stop()
                motion_duration = time.time() - self._motion_start_time
                self.logger.info(
                    f"Motion window closed after {motion_duration:.1f}s, "
                    f"active objects: {len(self._active_frigate_events)}"
                )
            except Exception as e:
                self.logger.error(f"Failed to stop motion window: {e}")
            finally:
                self._motion_smart_event_id = None

    async def handle_detection_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                f"Unexpectedly received non-bytes payload for detection event: {message.payload}"
            )
            return

        msg = message.payload.decode()
        try:
            frigate_msg = json.loads(msg)
            event_type = frigate_msg.get("type")
            event_id = frigate_msg.get("after", {}).get("id")
            camera = frigate_msg.get("after", {}).get("camera")
            label = frigate_msg.get("after", {}).get("label")

            if camera != self.args.frigate_camera:
                return

            self.logger.debug(f"Frigate: Received: {frigate_msg} ")

            object_type = label_to_object_type(label)
            if not object_type:
                self.logger.warning(
                    f"MISSED EVENT: Received unsupported detection label type: {label} "
                    f"(event_id={event_id}, type={event_type})"
                )
                return

            self.logger.debug(
                f"Frigate event: type={event_type}, id={event_id}, label={label}, "
                f"active_frigate_events={list(self._active_frigate_events)}"
            )

            if event_type == "new":
                await self._handle_new_event(frigate_msg, event_id, object_type, label)
            elif event_type == "update":
                await self._handle_update_event(frigate_msg, event_id, object_type, label)
            elif event_type == "end":
                await self._handle_end_event(frigate_msg, event_id, object_type, label)
            else:
                self.logger.debug(f"Received unhandled event type: {event_type} for event_id={event_id}")

        except json.JSONDecodeError:
            self.logger.exception(f"Could not decode payload: {msg}")
        except Exception as e:
            self.logger.exception(f"Unexpected error handling detection event: {e}, payload: {msg}")

    async def _handle_new_event(
        self, frigate_msg: dict[str, Any], event_id: str, object_type: SmartDetectObjectType, label: str
    ) -> None:
        if event_id in self._active_frigate_events:
            self.logger.warning(
                f"Received 'new' event for already active Frigate event_id={event_id}. Ignoring duplicate."
            )
            return

        if not self._is_motion_window_active():
            self.logger.warning(
                f"Received Frigate 'new' event {event_id} but no active motion window. "
                f"Buffering object detection until motion starts."
            )
            # In production, Frigate should not send objects outside a motion
            # window, but handle gracefully by waiting for motion to trigger.
            return

        self.event_snapshot_ready[event_id] = asyncio.Event()

        tracker_id = self.tracker_ids.allocate(event_id)
        descriptor = self._build_descriptor(frigate_msg, object_type, tracker_id)
        frame_time_ms = int(frigate_msg["after"].get("frame_time", 0) * 1000) - self.args.frigate_time_sync_ms

        # Extract position_changes for stationary object detection
        after = frigate_msg.get("after", {})
        position_changes = after.get("position_changes", 0)

        # Initialize position_changes tracking
        self._frigate_event_position_changes[event_id] = {
            "before": 0,  # "new" event has no "before"
            "after": position_changes
        }

        zones_status = self._update_zone_status_for_track(
            tracker_id, descriptor["zones"], descriptor["confidenceLevel"], active=True
        )

        # Determine edgeType based on position_changes (protocol spec Section 7)
        if position_changes == 0:
            # Stationary object: emit edgeType="none"
            self.logger.debug(
                f"Frigate event {event_id}: position_changes={position_changes}, "
                f"stationary=True → edgeType=none, calling trigger_smart_detect_stationary()"
            )
            await self.trigger_smart_detect_stationary(
                custom_descriptor=descriptor,
                event_timestamp=frame_time_ms,
                zonesStatus=zones_status,
                event_id=self._motion_smart_event_id,
            )
        else:
            # Moving object: emit edgeType="enter"
            self.logger.debug(
                f"Frigate event {event_id}: position_changes={position_changes}, "
                f"stationary=False → edgeType=enter"
            )
            if self._motion_smart_event_id is None:
                self._motion_smart_event_id = await self.trigger_smart_detect_start(
                    object_type, descriptor, frame_time_ms, zonesStatus=zones_status
                )
            else:
                await self.trigger_smart_detect_update(
                    object_type, descriptor, frame_time_ms, zonesStatus=zones_status, event_id=self._motion_smart_event_id
                )

        self._active_frigate_events.add(event_id)
        self._frigate_event_object_types[event_id] = object_type
        self.event_last_update[event_id] = time.time()

        await self._update_motion_levels_from_recordings(frigate_msg)

        self.logger.info(
            f"Frigate: Object entering zone (Frigate: {event_id}, trackerID: {tracker_id}, {label}). "
            f"Motion smart-detect event: {self._motion_smart_event_id}. "
            f"Total active objects: {len(self._active_frigate_events)}"
        )

    async def _handle_update_event(
        self, frigate_msg: dict[str, Any], event_id: str, object_type: SmartDetectObjectType, label: str
    ) -> None:
        if event_id not in self._active_frigate_events:
            # "new" event was missed - this could be an update for a known event
            # or a late-joining object. Handle gracefully.
            if not self._is_motion_window_active():
                # Check if this is a stationary object (position_changes=0)
                # If so, we can still track it even without motion window
                after = frigate_msg.get("after", {})
                position_changes_after = after.get("position_changes", 0)
                
                if position_changes_after == 0:
                    # Stationary object: create tracker and send stationary message
                    # even without active motion window
                    self.logger.info(
                        f"Stationary object detected without motion window: "
                        f"event_id={event_id}, label={label} -> creating tracker"
                    )
                else:
                    # Moving object: require active motion window
                    self.logger.warning(
                        f"MISSED EVENT: Received 'update' for unknown Frigate event_id={event_id} "
                        f"(label={label}) with no active motion window; dropping update."
                    )
                    return

            # Recovery path for out-of-order/missed MQTT delivery: if a
            # Frigate update arrives before we saw its "new" event, adopt it
            # as a late-joining object in the current motion window instead
            # of dropping it.
            self.event_snapshot_ready[event_id] = asyncio.Event()
            self._active_frigate_events.add(event_id)
            self._frigate_event_object_types[event_id] = object_type
            self.logger.info(
                f"Recovered unknown Frigate update as active object "
                f"(event_id={event_id}, label={label}) in motion window {self._motion_smart_event_id}."
            )

            # Initialize position_changes tracking for this recovered event
            if event_id not in self._frigate_event_position_changes:
                self._frigate_event_position_changes[event_id] = {
                    "before": 0,  # Assume stationary before we saw the update
                    "after": 0 # TODO: Fix this logic, as we don't need to store postition changes.  Frigate messages have before and after which is the delta tracking.
                }
            self.logger.debug(
                f"Initialized position_changes tracking for recovered event {event_id}"
            )

        tracker_id = self.tracker_ids.get(event_id)
        descriptor = self._build_descriptor(frigate_msg, object_type, tracker_id)
        frame_time_ms = int(frigate_msg["after"].get("frame_time", 0) * 1000) - self.args.frigate_time_sync_ms

        # Extract position_changes for stationary detection and transition detection
        after = frigate_msg.get("after", {})
        before = frigate_msg.get("before", {})
        position_changes_before = before.get("position_changes", 0) if before else 0
        position_changes_after = after.get("position_changes", 0)

        # Update position_changes tracking
        if event_id in self._frigate_event_position_changes:
            self._frigate_event_position_changes[event_id]["before"] = position_changes_before
            self._frigate_event_position_changes[event_id]["after"] = position_changes_after
        else:
            self._frigate_event_position_changes[event_id] = {
                "before": position_changes_before,
                "after": position_changes_after
            }

        zones_status = self._update_zone_status_for_track(
            tracker_id, descriptor["zones"], descriptor["confidenceLevel"], active=True
        )

        # Update position_changes tracking
        if event_id in self._frigate_event_position_changes:
            self._frigate_event_position_changes[event_id]["before"] = position_changes_before
            self._frigate_event_position_changes[event_id]["after"] = position_changes_after
        else:
            # "new" event was missed - initialize tracking
            self._frigate_event_position_changes[event_id] = {
                "before": 0,  # Assume stationary before we saw the update
                "after": position_changes_after
            }
            self.logger.debug(
                f"Initialized position_changes tracking for event {event_id} (missed 'new' event)"
            )

        # Determine edgeType based on position_changes transitions (protocol spec Section 7)
        if position_changes_after == 0:
            # Stationary object: emit edgeType="none"
            self.logger.debug(
                f"Frigate event {event_id}: position_changes_before={position_changes_before}, "
                f"position_changes_after={position_changes_after} → edgeType=none, calling trigger_smart_detect_stationary()"
            )
            await self.trigger_smart_detect_stationary(
                custom_descriptor=descriptor,
                event_timestamp=frame_time_ms,
                zonesStatus=zones_status,
                event_id=self._motion_smart_event_id,
            )
        elif position_changes_before == 0 and position_changes_after > 0:
            # First movement detected (object was stationary, now moving): emit edgeType="enter"
            self.logger.debug(
                f"Frigate event {event_id}: position_changes_before={position_changes_before}, "
                f"position_changes_after={position_changes_after} → FIRST MOVEMENT detected, edgeType=enter"
            )
            if self._motion_smart_event_id is None:
                self._motion_smart_event_id = await self.trigger_smart_detect_start(
                    object_type, descriptor, frame_time_ms, zonesStatus=zones_status
                )
            else:
                await self.trigger_smart_detect_update(
                    object_type, descriptor, frame_time_ms, zonesStatus=zones_status, event_id=self._motion_smart_event_id
                )
        else:
            # Already moving: emit via update (edgeType="moving" is implicit)
            self.logger.debug(
                f"Frigate event {event_id}: position_changes_before={position_changes_before}, "
                f"position_changes_after={position_changes_after} → already moving, calling trigger_smart_detect_update()"
            )
            await self.trigger_smart_detect_update(
                object_type, descriptor, frame_time_ms, zonesStatus=zones_status, event_id=self._motion_smart_event_id
            )

        self.event_last_update[event_id] = time.time()
        self._frigate_event_object_types[event_id] = object_type

        await self._update_motion_levels_from_recordings(frigate_msg)

        # Eagerly fetch and cache snapshot when Frigate has one ready. Uses
        # the event-specific Frigate snapshot URL (with native crop) rather
        # than the live latest.jpg.
        after_data = frigate_msg.get("after", {})
        if after_data.get("has_snapshot") and self._motion_smart_event_id is not None:
            smart_event = self._active_smart_events.get(self._motion_smart_event_id)
            if smart_event and not smart_event.get("snapshot_crop_path"):
                self.logger.debug(f"Event {event_id} has updated snapshot, fetching and caching all types...")
                asyncio.ensure_future(
                    self._fetch_and_cache_frigate_event_snapshot(event_id, self._motion_smart_event_id)
                )

    async def _handle_end_event(
        self, frigate_msg: dict[str, Any], event_id: str, object_type: SmartDetectObjectType, label: str
    ) -> None:
        if event_id not in self._active_frigate_events:
            self.logger.warning(
                f"MISSED EVENT: Received 'end' for unknown Frigate event_id={event_id} "
                f"(label={label}). Likely missed 'new' event."
            )
            return

        if self._motion_smart_event_id is None:
            self.logger.warning(f"Frigate object end {event_id} but no active motion window. Cleaning up track locally.")
            self._forget_track(event_id)
            return

        tracker_id = self.tracker_ids.get(event_id)
        final_descriptor = self._build_descriptor(frigate_msg, object_type, tracker_id)
        end_time_ms = int(frigate_msg["after"].get("end_time", 0) * 1000) - self.args.frigate_time_sync_ms
        frame_time_ms = int(frigate_msg["after"].get("frame_time", 0) * 1000) - self.args.frigate_time_sync_ms

        # Extract position_changes for stationary detection
        after = frigate_msg.get("after", {})
        before = frigate_msg.get("before", {})
        position_changes_before = before.get("position_changes", 0) if before else 0
        position_changes_after = after.get("position_changes", 0)

        # Update position_changes tracking
        if event_id in self._frigate_event_position_changes:
            self._frigate_event_position_changes[event_id]["before"] = position_changes_before
            self._frigate_event_position_changes[event_id]["after"] = position_changes_after
        else:
            # "new" event was missed - initialize tracking
            self._frigate_event_position_changes[event_id] = {
                "before": 0,  # Assume stationary before we saw the update
                "after": position_changes_after
            }
            self.logger.debug(
                f"Initialized position_changes tracking for event {event_id} (missed 'new' event)"
            )

        await self._update_motion_levels_from_recordings(frigate_msg)

        track_duration = time.time() - self.event_last_update.get(event_id, self._motion_start_time)
        self.logger.info(
            f"Frigate: Object leaving (Frigate: {event_id}, trackerID: {tracker_id}, {label}). "
            f"Duration: {track_duration:.1f}s"
        )

        # Determine edgeType based on position_changes (protocol spec Section 7)
        if position_changes_after == 0:
            # Stationary object leaving: emit edgeType="none"
            self.logger.debug(
                f"Frigate event {event_id}: position_changes_after={position_changes_after} → edgeType=none, "
                f"calling trigger_smart_detect_stationary()"
            )
            zones_status = self._update_zone_status_for_track(
                tracker_id, final_descriptor["zones"], final_descriptor["confidenceLevel"], active=True
            )
            await self.trigger_smart_detect_stationary(
                custom_descriptor=final_descriptor,
                event_timestamp=end_time_ms,
                zonesStatus=zones_status,
                event_id=self._motion_smart_event_id,
            )
        else:
            # Moving object leaving: emit edgeType="leave"
            self.logger.debug(
                f"Frigate event {event_id}: position_changes_after={position_changes_after} → edgeType=leave"
            )
            zones_status_final_update = self._update_zone_status_for_track(
                tracker_id, final_descriptor["zones"], final_descriptor["confidenceLevel"], active=True
            )
            await self.trigger_smart_detect_update(
                object_type, final_descriptor, frame_time_ms,
                zonesStatus=zones_status_final_update, event_id=self._motion_smart_event_id,
            )

            # This track is closing -- remove its contribution from the shared
            # zone tracker (other tracks' occupancy is unaffected, per protocol
            # spec Section 3) and capture the resulting post-departure
            # zonesStatus, overridden so any zone it still occupied reports as
            # "leave".
            zones_status = self._update_zone_status_for_track(tracker_id, [], 0, active=False)
            ZoneStatusTracker.mark_all_departed(zones_status)

            await self.trigger_smart_detect_update(
                object_type, final_descriptor, end_time_ms, zonesStatus=zones_status, event_id=self._motion_smart_event_id
            )

        self._forget_track(event_id)

        self.logger.info(
            f"Frigate: Object {event_id} ended. Remaining active objects: {len(self._active_frigate_events)}"
        )

    async def handle_doorbell_event(self, message: Message) -> None:
        """Handle Frigate doorbell ring events.

        Sends MCUEventMessage for doorbell ring events, as per the protocol spec.
        """
        if not isinstance(message.payload, bytes):
            self.logger.warning(f"Unexpectedly received non-bytes payload for doorbell event: {message.payload}")
            return

        try:
            msg = message.payload.decode()
            json.loads(msg)  # validate it's well-formed, even though we don't use the payload
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.warning(f"Could not decode doorbell event: {e}, payload: {message.payload}")
            return

        self.logger.debug(f"Received Frigate doorbell event: {msg[:200]}")

        try:
            await self.send_mcu_event_message({"eventType": "EventRingButtonPressed"})
            self.logger.info("Forwarded Frigate doorbell ring to UniFi Protect")
        except Exception as e:
            self.logger.error(f"Failed to forward doorbell ring to UniFi Protect: {e}")

    async def handle_snapshot_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(f"Unexpectedly received non-bytes payload for snapshot event: {message.payload}")
            return

        topic_parts = message.topic.value.split("/")
        if len(topic_parts) < 4:
            self.logger.debug(f"Unexpected snapshot topic format: {message.topic.value}")
            return

        snapshot_label = topic_parts[-2]
        self.logger.debug(
            f"Received snapshot: topic={message.topic.value}, label={snapshot_label}, size={len(message.payload)} bytes"
        )

        # In this architecture, all active objects within a motion window
        # share the same smart event. We just need to cache the snapshot for
        # whichever Frigate event matches the label.
        matching_frigate_event_id = None
        snapshot_object_type = label_to_object_type(snapshot_label)
        for frigate_event_id in self._active_frigate_events:
            # Snapshots come labeled by object type (person, car, etc), but do
            # not include a Frigate event ID. Match by active event object
            # type first to avoid assigning person snapshots to vehicle tracks.
            if snapshot_object_type is None:
                matching_frigate_event_id = frigate_event_id
                break
            if self._frigate_event_object_types.get(frigate_event_id) == snapshot_object_type:
                matching_frigate_event_id = frigate_event_id
                break

        if matching_frigate_event_id and self._motion_smart_event_id is not None:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(message.payload)
            f.close()
            self.logger.debug(
                f"Cached snapshot for Frigate event {matching_frigate_event_id} "
                f"({snapshot_label}): {f.name} ({len(message.payload)} bytes)"
            )
            self.update_motion_snapshot(Path(f.name))
            if matching_frigate_event_id in self.event_snapshot_ready:
                self.event_snapshot_ready[matching_frigate_event_id].set()
        else:
            self.logger.debug(
                f"Discarding snapshot for label={snapshot_label} "
                f"(size={len(message.payload)}, retained={message.retain}). "
                f"No active motion window or matching event."
            )

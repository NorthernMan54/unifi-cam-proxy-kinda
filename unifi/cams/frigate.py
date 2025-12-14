import argparse
import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

import backoff
from aiomqtt import Client, Message
from aiomqtt.exceptions import MqttError

from unifi.cams.base import SmartDetectObjectType
from unifi.cams.rtsp import RTSPCam


class FrigateCam(RTSPCam):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        super().__init__(args, logger)
        self.args = args
        self.event_id: Optional[str] = None
        self.event_label: Optional[str] = None
        self.event_snapshot_ready = None
        # Track multiple concurrent events
        self.active_events: dict[str, dict[str, Any]] = {}
        self.event_timeout_seconds = 60

    @classmethod
    def add_parser(cls, parser: argparse.ArgumentParser) -> None:
        super().add_parser(parser)
        parser.add_argument("--mqtt-host", required=True, help="MQTT server")
        parser.add_argument("--mqtt-port", default=1883, type=int, help="MQTT server")
        parser.add_argument("--mqtt-username", required=False)
        parser.add_argument("--mqtt-password", required=False)
        parser.add_argument(
            "--mqtt-prefix", default="frigate", type=str, help="Topic prefix"
        )
        parser.add_argument(
            "--frigate-camera",
            required=True,
            type=str,
            help="Name of camera in frigate",
        )
        parser.add_argument(
            "--camera-width",
            default=1920,
            type=int,
            help="Camera frame width in pixels (default: 1920)",
        )
        parser.add_argument(
            "--camera-height",
            default=1080,
            type=int,
            help="Camera frame height in pixels (default: 1080)",
        )
        parser.add_argument(
            "--frigate-detect-width",
            default=1280,
            type=int,
            help="Frigate detection frame width in pixels (default: 1280)",
        )
        parser.add_argument(
            "--frigate-detect-height",
            default=720,
            type=int,
            help="Frigate detection frame height in pixels (default: 720)",
        )

    async def get_feature_flags(self) -> dict[str, Any]:
        return {
            **await super().get_feature_flags(),
            **{
                "mic": True,
                "smartDetect": [
                    "person",
                    "vehicle",
                ],
            },
        }

    @classmethod
    def label_to_object_type(cls, label: str) -> Optional[SmartDetectObjectType]:
        if label == "person":
            return SmartDetectObjectType.PERSON
        elif label in {"vehicle", "car", "motorcycle", "bus"}:
            return SmartDetectObjectType.VEHICLE

    def build_descriptor_from_frigate_msg(
        self, frigate_msg: dict[str, Any], object_type: SmartDetectObjectType
    ) -> dict[str, Any]:
        """
        Build a UniFi Protect-compatible descriptor from Frigate event data.
        
        Frigate provides bounding boxes as pixel coordinates.
        Testing with raw pixel values to see UniFi Protect's expected format.
        """
        after = frigate_msg.get("after", {})
        
        # Extract bounding box if available
        box = after.get("box")
        if box and len(box) == 4:
            # Frigate box format: [x_min, y_min, x_max, y_max] in configured pixel dimensions
            # UniFi format: [x, y, width, height] in 1000x1000 coordinate system

            # Scale factors
            frigate_width = self.args.frigate_detect_width
            frigate_height = self.args.frigate_detect_height
            UNIFI_WIDTH = 1000
            UNIFI_HEIGHT = 1000
            UNIFI_Y_OFFSET = 0  # Offset is needed if cam width/height differ from detection dimensions

            x_scale = UNIFI_WIDTH / frigate_width
            y_scale = UNIFI_HEIGHT / frigate_height

            # Transform coordinates
            x_min_unifi = int(box[0] * x_scale)
            y_min_unifi = int(box[1] * y_scale) + UNIFI_Y_OFFSET
            x_max_unifi = int(box[2] * x_scale)
            y_max_unifi = int(box[3] * y_scale) + UNIFI_Y_OFFSET

            # Convert to UniFi format
            x = x_min_unifi
            y = y_min_unifi
            width = x_max_unifi - x_min_unifi
            height = y_max_unifi - y_min_unifi
            coord = [x, y, width, height]
        else:
            # Fallback to default if no bounding box
            coord = [0, 0, 1920, 1080]
        
        # Extract confidence score (Frigate uses 0.0-1.0, UniFi uses 0-100)
        score = after.get("score", 0.95)
        confidence_level = int(score * 100)
        
        # Extract tracker ID if available
        tracker_id = after.get("id", 1)
        if isinstance(tracker_id, str):
            # Convert string ID to numeric hash for tracker ID
            tracker_id = hash(tracker_id) % 1000000
        
        # Check if object is stationary
        stationary = after.get("stationary", False)
        
        # Get current position and velocity if available
        current_zones = after.get("current_zones", [])
        zones = [0] if not current_zones else [0]  # Default to zone 0
        
        descriptor = {
            "trackerID": tracker_id,
            "name": object_type.value,
            "confidenceLevel": confidence_level,
            "coord": coord,
            "objectType": object_type.value,
            "zones": zones,
            "lines": [],  # Required field: no line crossing detection
            "loiterZones": [],  # Optional but included for completeness
            "stationary": stationary,
            "attributes": {},  # Optional but included for completeness
            "coord3d": [0, 0, 0],  # Required field: no 3D coordinates available
            "depth": None,  # Optional depth information
            "speed": None,  # Optional speed information
        }
        
        self.logger.debug(
            f"Built descriptor: trackerID={tracker_id}, confidence={confidence_level}, "
            f"coord={coord}, stationary={stationary}"
        )
        
        return descriptor

    async def run(self) -> None:
        has_connected = False

        @backoff.on_predicate(backoff.expo, max_value=60, logger=self.logger)
        async def mqtt_connect():
            nonlocal has_connected
            try:
                async with Client(
                    self.args.mqtt_host,
                    port=self.args.mqtt_port,
                    username=self.args.mqtt_username,
                    password=self.args.mqtt_password,
                ) as client:
                    has_connected = True
                    self.logger.info(
                        f"Connected to {self.args.mqtt_host}:{self.args.mqtt_port}"
                    )
                    await client.subscribe(f"{self.args.mqtt_prefix}/#")
                    async with asyncio.TaskGroup() as tg:
                        # Start event timeout monitor
                        tg.create_task(self.monitor_event_timeouts())
                        async for message in client.messages:
                            if message.topic.matches(f"{self.args.mqtt_prefix}/events"):
                                tg.create_task(self.handle_detection_event(message))
                            elif message.topic.matches(
                                f"{self.args.mqtt_prefix}/{self.args.frigate_camera}/+/snapshot"
                            ):
                                tg.create_task(self.handle_snapshot_event(message))
            except MqttError:
                if not has_connected:
                    raise

        await mqtt_connect()

    async def monitor_event_timeouts(self) -> None:
        """Monitor active events and end those that haven't received updates in 60 seconds"""
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            current_time = asyncio.get_event_loop().time()
            expired_events = []
            
            for event_id, event_data in self.active_events.items():
                time_since_update = current_time - event_data["last_update"]
                if time_since_update > self.event_timeout_seconds:
                    expired_events.append(event_id)
                    self.logger.warning(
                        f"EVENT TIMEOUT: Event {event_id} ({event_data['label']}) has not "
                        f"received updates for {time_since_update:.1f}s. Force ending event."
                    )
            
            # End expired events
            for event_id in expired_events:
                event_data = self.active_events[event_id]
                try:
                    # Set the old event_id context for trigger_motion_stop
                    self.event_id = event_id
                    self.event_label = event_data["label"]
                    await self.trigger_motion_stop()
                except Exception as e:
                    self.logger.exception(f"Error ending timed out event {event_id}: {e}")
                finally:
                    del self.active_events[event_id]
                    self.event_id = None
                    self.event_label = None

    async def handle_detection_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                f"Unexpectedly received non-bytes payload for detection event: {message.payload}"
            )
            return

        msg = message.payload.decode()
        try:
            frigate_msg = json.loads(message.payload.decode())
            event_type = frigate_msg.get("type")
            event_id = frigate_msg.get("after", {}).get("id")
            camera = frigate_msg.get("after", {}).get("camera")
            label = frigate_msg.get("after", {}).get("label")
            
            if camera != self.args.frigate_camera:
                self.logger.debug(
                    f"Frigate: Ignoring Frigate event for different camera: {camera} "
                    f"(expecting {self.args.frigate_camera})"
                )
                return
            
            self.logger.debug(
                    f"Frigate: Received: {frigate_msg} "
                )
            
            object_type = self.label_to_object_type(label)
            if not object_type:
                self.logger.warning(
                    f"MISSED EVENT: Received unsupported detection label type: {label} "
                    f"(event_id={event_id}, type={event_type})"
                )
                return

            self.logger.debug(
                f"Frigate event: type={event_type}, id={event_id}, label={label}, "
                f"active_events={list(self.active_events.keys())}"
            )
            
            current_time = asyncio.get_event_loop().time()

            if event_type == "new":
                if event_id in self.active_events:
                    self.logger.warning(
                        f"Received 'new' event for already active event_id={event_id}. "
                        f"This may indicate event was not properly ended. Re-initializing."
                    )
                
                # Start new event
                self.active_events[event_id] = {
                    "label": label,
                    "object_type": object_type,
                    "snapshot_ready": asyncio.Event(),
                    "last_update": current_time,
                    "started_at": current_time,
                }
                
                # Set context for motion start
                self.event_id = event_id
                self.event_label = label
                self.event_snapshot_ready = self.active_events[event_id]["snapshot_ready"]
                
                self.logger.info(
                    f"Frigate: Starting {label} motion event (id: {event_id}). "
                    f"Total active events: {len(self.active_events)}"
                )
                
                # Build custom descriptor from Frigate data
                custom_descriptor = self.build_descriptor_from_frigate_msg(
                    frigate_msg, object_type
                )
                await self.trigger_motion_start(object_type, custom_descriptor)
                
            elif event_type == "update":
                if event_id in self.active_events:
                    # Update last_update timestamp
                    self.active_events[event_id]["last_update"] = current_time
                    
                    # Set context for motion update
                    self.event_id = event_id
                    self.event_label = label
                    
                    # Build updated descriptor from Frigate data
                    custom_descriptor = self.build_descriptor_from_frigate_msg(
                        frigate_msg, object_type
                    )
                    
                    # Send moving update with updated bounding box
                    await self.trigger_motion_update(custom_descriptor)
                    
                    self.logger.debug(
                        f"Sent moving update for event {event_id}. "
                        f"Age: {current_time - self.active_events[event_id]['started_at']:.1f}s"
                    )
                else:
                    self.logger.warning(
                        f"MISSED EVENT: Received 'update' for unknown event_id={event_id} "
                        f"(label={label}). Likely missed 'new' event."
                    )
                    
            elif event_type == "end":
                if event_id in self.active_events:
                    event_data = self.active_events[event_id]
                    snapshot_ready = event_data["snapshot_ready"]
                    
                    # Set context for motion stop
                    self.event_id = event_id
                    self.event_label = event_data["label"]
                    self.event_snapshot_ready = snapshot_ready
                    
                    # Build final descriptor from end event data
                    # This will be included in the "leave" event by trigger_motion_stop
                    final_descriptor = self.build_descriptor_from_frigate_msg(
                        frigate_msg, object_type
                    )
                    # Save it so trigger_motion_stop can use it
                    self._motion_last_descriptor = final_descriptor
                    
                    if snapshot_ready:
                        # Wait for the best snapshot to be ready before ending the motion event
                        self.logger.info(f"Frigate: Awaiting snapshot (id: {event_id})")
                        try:
                            await asyncio.wait_for(snapshot_ready.wait(), timeout=10.0)
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                f"Snapshot wait timeout for event {event_id}. Proceeding without snapshot."
                            )
                        
                        self.logger.info(
                            f"Frigate: Ending {event_data['label']} motion event (id: {event_id}). "
                            f"Duration: {current_time - event_data['started_at']:.1f}s"
                        )
                        await self.trigger_motion_stop()
                    else:
                        self.logger.warning(
                            f"MISSED EVENT: Received end event (id={event_id}) but "
                            f"snapshot_ready is not set. Event state may be corrupted."
                        )
                        await self.trigger_motion_stop()
                    
                    # Clean up event
                    del self.active_events[event_id]
                    self.event_id = None
                    self.event_label = None
                    self.event_snapshot_ready = None
                    
                    self.logger.info(
                        f"Frigate: Event {event_id} ended. "
                        f"Remaining active events: {len(self.active_events)}"
                    )
                else:
                    self.logger.warning(
                        f"MISSED EVENT: Received 'end' for unknown event_id={event_id} "
                        f"(label={label}). Likely missed 'new' event."
                    )
            else:
                self.logger.debug(
                    f"Received unhandled event type: {event_type} for event_id={event_id}"
                )
                
        except json.JSONDecodeError:
            self.logger.exception(f"Could not decode payload: {msg}")
        except Exception as e:
            self.logger.exception(
                f"Unexpected error handling detection event: {e}, payload: {msg}"
            )

    async def handle_snapshot_event(self, message: Message) -> None:
        if not isinstance(message.payload, bytes):
            self.logger.warning(
                f"Unexpectedly received non-bytes payload for snapshot event: {message.payload}"
            )
            return

        # Extract label from topic: frigate/<camera>/<label>/snapshot
        topic_parts = message.topic.value.split("/")
        if len(topic_parts) < 4:
            self.logger.debug(f"Unexpected snapshot topic format: {message.topic.value}")
            return
            
        snapshot_label = topic_parts[-2]
        
        # Find matching active event by label
        matching_event_id = None
        for event_id, event_data in self.active_events.items():
            if event_data["label"] == snapshot_label and not message.retain:
                matching_event_id = event_id
                break
        
        if matching_event_id:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(message.payload)
            f.close()
            self.logger.debug(
                f"Updating snapshot for event {matching_event_id} ({snapshot_label}) with {f.name}"
            )
            self.update_motion_snapshot(Path(f.name))
            self.active_events[matching_event_id]["snapshot_ready"].set()
        else:
            self.logger.debug(
                f"Discarding snapshot for label={snapshot_label} "
                f"(size={len(message.payload)}, retained={message.retain}). "
                f"No matching active event."
            )

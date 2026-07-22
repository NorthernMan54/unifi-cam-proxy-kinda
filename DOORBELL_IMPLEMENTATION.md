# Doorbell Implementation

## Overview

The `"doorbell": true` flag is implemented to indicate when the connected UniFi Protect device is a doorbell camera. This flag is set in the `features` object sent during the `ubnt_avclient_hello` handshake.

---

## Implementation Details

### 1. Feature Flag Detection

**Location**: `unifi/cams/handlers/protocol_handlers.py`

The `process_hello` method now detects the `doorbell` flag in the hello message payload:

```python
async def process_hello(self: "UnifiCamBase", msg: "AVClientRequest") -> None:
    """Process hello message from UniFi Protect.
    
    Detects if the connected device is a doorbell by checking the features object.
    """
    # Check if the message contains features with doorbell flag
    payload = msg.get("payload", {})
    features = payload.get("features", {})
    
    # Set the doorbell flag if this is a doorbell device
    if features.get("doorbell", False):
        self._is_doorbell = True
        self.logger.info("Connected to doorbell device")
    else:
        self._is_doorbell = False
        self.logger.info("Connected to standard camera device")
```

### 2. State Tracking

**Location**: `unifi/cams/base.py`

Added `_is_doorbell` attribute to track the device type:

```python
class UnifiCamBase(ProtocolHandlers, VideoStreamHandlers, SnapshotHandlers, metaclass=ABCMeta):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger

        # Doorbell support - set by subclasses based on device type
        self._is_doorbell: bool = False
        self._msg_id: int = 0
```

### 3. Feature Flag Reporting

**Location**: `unifi/cams/base.py`

The `get_feature_flags` method now includes the doorbell flag:

```python
async def get_feature_flags(self) -> dict[str, Any]:
    return {
        "mic": True,
        "aec": [],
        "videoMode": ["default"],
        "motionDetect": ["enhanced"],
        "hotplug":{"extender":{"attached":False}},
        "doorbell": getattr(self, "_is_doorbell", False),
    }
```

---

## Protocol Specification Reference

According to the [EventSmartDetect Protocol Spec](eventsmartdetect_protocol_spec.md):

### MCUEventMessage — Doorbell Event Format

```json
{
  "from": "ubnt_avclient",
  "functionName": "MCUEventMessage",
  "inResponseTo": 0,
  "messageId": <incrementing counter>,
  "payload": {
    "event": {
      "type": "ring",
      "timestamp": <epoch_ms>,
      "duration": <duration_ms>,
      "chimeType": "default",
      "motionDetected": true,
      "smartDetect": {
        "objects": [...]
      }
    },
    "doorbell": true
  },
  "responseExpected": false,
  "timeStamp": "<iso_timestamp>",
  "to": "UniFiVideo"
}
```

### ubnt_avclient_hello — Doorbell Feature Detection

```json
{
  "from": "ubnt_avclient",
  "functionName": "ubnt_avclient_hello",
  "payload": {
    "features": {
      "doorbell": true,
      "smartDetect": ["person", "vehicle", "animal", "lineCrossing", "faceEnhancedByAiKey", "lprEnhancedByAiKey", "alrmSmoke", "alrmCmonx", "alrmBabyCry", "alrmSpeak"],
      "motionDetect": ["enhanced"],
      "mic": true,
      "speaker": true,
      "welcomeLed": true,
      "ringVolume": 1,
      "talkback": {...},
      "videoCodecs": ["h264", "h265", "mjpg"],
      "streamEncryptable": true
    },
    "model": "UVC Doorbell Lite",
    "name": "Wasaga Doorbell",
    "protocolVersion": 67,
    "uptime": 569595
  }
}
```

---

## Doorbell vs. Camera Protocol Differences

| Feature | Doorbell | Standard Camera |
|---|---|---|
| `ubnt_avclient_hello.features.doorbell` | `true` | `false` or absent |
| Primary event function | `MCUEventMessage` | `EventSmartDetect` |
| Audio features | `mic: true`, `speaker: true` | May vary |
| Talkback config | Configured for doorbell audio | Standard video audio |
| Ring indicators | `welcomeLed: true`, `ringVolume` | N/A |
| Smart detect scope | Ring events + motion | Motion + object tracking |
| Chime control | `chimeControl` feature | N/A |

---

## Current Implementation Status

### ✅ Implemented
- Detection of `doorbell` flag in `ubnt_avclient_hello` message
- Setting `_is_doorbell` flag on the camera instance
- Including `doorbell` flag in `get_feature_flags()` response

### 🔄 In Progress
- Frigate doorbell event forwarding (via MQTT)
- Doorbell-specific protocol handlers (`MCUEventMessage`)

### 📋 To Do
- Implement `MCUEventMessage` handler for doorbell ring events
- Add doorbell-specific feature flags (e.g., `chimeControl`, `welcomeLed`)
- Handle doorbell audio streams (talkback)

---

## Testing

To test doorbell support:

1. **Connect to a doorbell camera** that reports `features.doorbell: true`
2. **Verify the log** shows: "Connected to doorbell device"
3. **Check the features response** includes `"doorbell": true`
4. **Test doorbell events** (ring events) are properly forwarded

---

## Future Enhancements

1. **MCUEventMessage Handler**: Implement handler for doorbell ring events
2. **Audio Support**: Handle two-way talkback for doorbells
3. **Chime Control**: Implement chime control feature
4. **Welcome LED**: Support for welcome LED indicator
5. **Ring Volume**: Configurable ring notification volume
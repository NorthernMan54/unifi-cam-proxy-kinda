# Doorbell Support Plan

## Overview

This project will add support for doorbell events from Frigate and remove support for Dahua, Hikvision, Reolink, and Tapo cameras. The goal is to simplify the codebase and focus on Frigate integration for doorbell events.

**Date**: July 21, 2026

---

## Current State Analysis

### Architecture Overview

The project uses a plugin-based architecture:
- Base class: `UnifiCamBase` in `unifi/cams/base.py`
- Camera implementations inherit from `UnifiCamBase` and implement specific methods
- Camera types are registered in `unifi/main.py` in the `CAMS` dictionary

### Current Camera Implementations

| Camera Type | File | Status |
|-------------|------|--------|
| Dahua | `unifi/cams/dahua.py` | TO BE REMOVED |
| Hikvision | `unifi/cams/hikvision.py` | TO BE REMOVED |
| Reolink | `unifi/cams/reolink.py` | TO BE REMOVED |
| Reolink NVR | `unifi/cams/reolink_nvr.py` | TO BE REMOVED |
| Tapo | `unifi/cams/tapo.py` | TO BE REMOVED |
| Frigate | `unifi/cams/frigate.py` | KEEP |
| RTSP | `unifi/cams/rtsp.py` | KEEP (fallback for non-Frigate cameras) |

### Dependencies to Remove

From `requirements.txt`:
- `amcrest==1.9.8` (Dahua uses amcrest library)
- `hikvisionapi==0.3.2` (Hikvision-specific)
- `reolinkapi==0.1.5` (Reolink-specific)
- `pytapo==3.3.21` (Tapo-specific)

---

## Implementation Plan

### Phase 1: Remove Camera Implementations

#### 1.1 Delete Camera Files

Delete the following files:
- `unifi/cams/dahua.py`
- `unifi/cams/hikvision.py`
- `unifi/cams/reolink.py`
- `unifi/cams/reolink_nvr.py`
- `unifi/cams/tapo.py`

#### 1.2 Update Main.py

**File**: `unifi/main.py`

Changes:
1. Remove imports for removed camera classes:
   ```python
   from unifi.cams import (
       DahuaCam,
       HikvisionCam,
       Reolink,
       ReolinkNVRCam,
       RTSPCam,
       TapoCam,
   )
   ```
   Keep only:
   ```python
   from unifi.cams import (
       RTSPCam,
       FrigateCam,
   )
   ```

2. Update the `CAMS` dictionary:
   ```python
   CAMS = {
       "frigate": FrigateCam,
       "rtsp": RTSPCam,
   }
   ```

3. Update argument parser to remove choices for removed cameras:
   ```python
   model = "UVC G3",
   choices=[
       "UVC",
       "UVC AI 360",
       "UVC AI Bullet",
       "UVC AI THETA",
       "UVC AI DSLR",
       "UVC Pro",
       "UVC Dome",
       "UVC Micro",
       "UVC G3",
       "UVC G3 Battery",
       "UVC G3 Dome",
       "UVC G3 Mini",
       "UVC G3 Instant",
       "UVC G3 Pro",
       "UVC G3 Flex",
       "UVC G4 Bullet",
       "UVC G4 Pro",
       "UVC G4 PTZ",
       "UVC G4 Doorbell",
       "UVC G4 Doorbell Pro",
       "UVC G4 Doorbell Pro PoE",
       "UVC G4 Dome",
       "UVC G4 Instant",
       "UVC G5 Bullet",
       "UVC G5 Dome",
       "UVC G5 Flex",
       "UVC G5 Pro",
       "AFi VC",
       "Vision Pro",
   ],
   ```
   Keep only:
   ```python
   model = "UVC G3",
   choices=[
       "UVC",
       "UVC AI 360",
       "UVC AI Bullet",
       "UVC AI THETA",
       "UVC AI DSLR",
       "UVC Pro",
       "UVC Dome",
       "UVC Micro",
       "UVC G3",
       "UVC G3 Battery",
       "UVC G3 Dome",
       "UVC G3 Mini",
       "UVC G3 Instant",
       "UVC G3 Pro",
       "UVC G3 Flex",
       "UVC G4 Bullet",
       "UVC G4 Pro",
       "UVC G4 PTZ",
       "UVC G4 Doorbell",
       "UVC G4 Doorbell Pro",
       "UVC G4 Doorbell Pro PoE",
       "UVC G4 Dome",
       "UVC G4 Instant",
       "UVC G5 Bullet",
       "UVC G5 Dome",
       "UVC G5 Flex",
       "UVC G5 Pro",
       "AFi VC",
       "Vision Pro",
   ],
   ```

### Phase 2: Update Requirements

#### 2.1 Update requirements.txt

Remove the following lines:
```
amcrest==1.9.8
hikvisionapi==0.3.2
reolinkapi==0.1.5
pytapo==3.3.21
```

Keep:
```
aiohttp==3.9.5
paho-mqtt==2.1.0
aiomqtt==2.1.0
backoff==2.2.1
coloredlogs==15.0.1
flvlib3@https://github.com/zkonge/flvlib3/archive/master.zip
packaging==24.1
Pillow>=10.0.0
uiprotect==1.2.1
websockets==12.0
xmltodict==0.13.0
httpx~=0.27.0
yarl~=1.9.4
```

### Phase 3: Update Documentation

#### 3.1 Remove Camera Documentation

Delete the following documentation files:
- `docs/docs/configuration/dahua.md`
- `docs/docs/configuration/hikvision.md`
- `docs/docs/configuration/reolink.md`
- `docs/docs/configuration/reolink_nvr.md`
- `docs/docs/configuration/tapo.md`

#### 3.2 Update README.md

Update the "Things that work" section to reflect the new focus on Frigate integration.

### Phase 4: Add Doorbell Support

#### 4.1 Frigate Doorbell Event Handling

Frigate already supports doorbell cameras. The integration needs to:

1. **Configure Frigate for Doorbell Events**:
   - Ensure Frigate is configured to publish doorbell events
   - Set up appropriate MQTT topics for doorbell events

2. **Subscribe to Doorbell Events**:
   - Add MQTT subscription to Frigate doorbell topics
   - Parse doorbell event messages

3. **Forward Events to UniFi Protect**:
   - Convert Frigate doorbell events to UniFi Protect format
   - Send events through the WebSocket connection

#### 4.2 Required MQTT Topics

Frigate typically publishes to:
- `frigate/{camera_name}/analytics/doorbell` - Doorbell motion events
- `frigate/{camera_name}/analytics/motion` - Regular motion events
- `frigate/{camera_name}/events` - Event metadata

#### 4.3 Implementation in FrigateCam

**File**: `unifi/cams/frigate.py`

Add methods to:
1. Subscribe to doorbell MQTT topics
2. Parse doorbell event messages
3. Forward doorbell events to UniFi Protect

```python
# Example structure to add to FrigateCam class

async def subscribe_to_doorbell_events(self):
    """Subscribe to Frigate doorbell events via MQTT."""
    await self.mqtt_client.subscribe(f"frigate/{self.frigate_camera}/analytics/doorbell")

async def handle_doorbell_event(self, message):
    """Handle incoming doorbell event from Frigate."""
    # Parse doorbell event
    # Forward to UniFi Protect via WebSocket
    await self.send_doorbell_event(message)

async def send_doorbell_event(self, frigate_event):
    """Send doorbell event to UniFi Protect."""
    # Convert Frigate event to UniFi Protect format
    # Send via WebSocket
    event = self._frigate_event_to_unifi_event(frigate_event)
    await self.send(event)
```

#### 4.4 Update FrigateCam Arguments

Add new arguments to `unifi/cams/frigate.py`:
```python
parser.add_argument(
    "--frigate-doorbell-topic",
    default="frigate/{camera_name}/analytics/doorbell",
    help="MQTT topic prefix for doorbell events"
)
parser.add_argument(
    "--frigate-mqtt-host",
    default="localhost",
    help="Frigate MQTT server hostname"
)
```

### Phase 5: Testing

#### 5.1 Unit Tests

Create tests for:
- Doorbell event parsing
- Event format conversion
- MQTT subscription and publishing

#### 5.2 Integration Tests

Test with:
- Real Frigate doorbell camera
- UniFi Protect NVR
- Verify events appear in UniFi Protect UI

---

## Risk Assessment

### Low Risk
- Removing camera implementations is straightforward
- Dependencies are well-documented and easy to remove
- Documentation is organized and easy to update

### Medium Risk
- Doorbell event forwarding requires understanding Frigate's MQTT protocol
- Event format conversion needs careful implementation

### Mitigation Strategies
1. Test extensively with real doorbell events
2. Log all event conversions for debugging
3. Provide clear error messages if events fail to forward

---

## Timeline

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| Phase 1: Remove Implementations | 2 hours | None |
| Phase 2: Update Requirements | 30 minutes | Phase 1 |
| Phase 3: Update Documentation | 1 hour | Phase 1 |
| Phase 4: Add Doorbell Support | 4 hours | Phase 3 |
| Phase 5: Testing | 4 hours | Phase 4 |
| **Total** | **~13.5 hours** | |

---

## Rollback Plan

If issues arise during implementation:

1. **Code Rollback**: Use git to revert changes
   ```bash
   git checkout HEAD~1
   ```

2. **Dependency Rollback**: Reinstall removed packages
   ```bash
   pip install -r requirements.txt.bak
   ```

3. **Documentation Rollback**: Restore documentation from version control

---

## Success Criteria

1. ✅ No references to Dahua, Hikvision, Reolink, or Tapo in code
2. ✅ Clean installation without unnecessary dependencies
3. ✅ Frigate doorbell events successfully forwarded to UniFi Protect
4. ✅ Documentation accurately reflects supported features
5. ✅ No regressions in existing functionality

---

## Notes

- The RTSPCam implementation is kept as a fallback for non-Frigate cameras
- Doorbell support is specifically for Frigate-integrated doorbell cameras
- The change simplifies the codebase and improves maintainability
# Changes Applied: Doorbell Support & Camera Removal

**Date**: July 21, 2026

This document summarizes all changes applied to add doorbell support from Frigate and remove support for Dahua, Hikvision, Reolink, and Tapo cameras.

---

## Summary of Changes

### ✅ Phase 1: Remove Camera Implementations

**Files Deleted:**
- `unifi/cams/dahua.py`
- `unifi/cams/hikvision.py`
- `unifi/cams/reolink.py`
- `unifi/cams/reolink_nvr.py`
- `unifi/cams/tapo.py`

**Rationale:** These camera implementations are no longer supported and their code has been removed to simplify the codebase.

---

### ✅ Phase 2: Update Requirements

**File Modified:** `requirements.txt`

**Packages Removed:**
- `amcrest==1.9.8` (used by Dahua implementation)
- `hikvisionapi==0.3.2` (used by Hikvision implementation)
- `reolinkapi==0.1.5` (used by Reolink implementation)
- `pytapo==3.3.21` (used by Tapo implementation)

**Remaining Packages:**
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

---

### ✅ Phase 3: Update Documentation

**Files Deleted:**
- `docs/docs/configuration/dahua.md`
- `docs/docs/configuration/hikvision.md`
- `docs/docs/configuration/reolink.md`
- `docs/docs/configuration/reolink_nvr.md`
- `docs/docs/configuration/tapo.md`

**File Modified:** `README.md`

**Changes:**
- Added "Doorbell events from Frigate" to the "Things that work" section
- Removed mentions of removed camera types (Dahua, Hikvision, Reolink, Tapo)

---

### ✅ Phase 4: Add Doorbell Support to FrigateCam

**File Modified:** `unifi/cams/frigate.py`

**Changes:**

1. **Added Arguments:**
   ```python
   parser.add_argument(
       "--frigate-doorbell-topic",
       default=None,
       type=str,
       help="MQTT topic for doorbell events (e.g., 'frigate/{camera_name}/analytics/doorbell')",
   )
   parser.add_argument(
       "--frigate-doorbell-enabled",
       action="store_true",
       default=False,
       help="Enable doorbell event forwarding from Frigate",
   )
   ```

2. **Updated MQTT Message Handling:**
   Added subscription to doorbell events:
   ```python
   elif self.args.frigate_doorbell_enabled and message.topic.matches(
       f"{self.args.frigate_doorbell_topic}"
   ):
       tg.create_task(self.handle_doorbell_event(message))
   ```

3. **Added Doorbell Event Handler:**
   New method `handle_doorbell_event()` that:
   - Parses incoming doorbell event messages from Frigate
   - Converts Frigate doorbell events to UniFi Protect format
   - Forwards events via the existing smart detect infrastructure

---

### ✅ Phase 5: Update Main.py

**File Modified:** `unifi/main.py`

**Changes:**

1. **Updated Imports:**
   ```python
   from unifi.cams import (
       FrigateCam,
       RTSPCam,
   )
   ```

2. **Updated CAMS Dictionary:**
   ```python
   CAMS = {
       "frigate": FrigateCam,
       "rtsp": RTSPCam,
   }
   ```

**Note:** The model choices for camera hardware (UVC G3, UVC G4 Doorbell, etc.) were preserved as they are still valid for UniFi Protect devices.

---

## Verification

All changes have been verified:
- ✅ No Python syntax errors in modified files
- ✅ No import errors in main.py
- ✅ All camera implementation files removed
- ✅ All camera documentation files removed
- ✅ Requirements.txt updated correctly
- ✅ README.md updated with new features

---

## Usage

### Using Frigate with Doorbell Events

```bash
unifi-cam-proxy --mac '{unique MAC}' -H {NVR IP} -c /client.pem -t {token} \
    frigate \
    -s {rtsp source} \
    --frigate-camera {Name of camera} \
    --frigate-mqtt-host {mqtt host} \
    --frigate-doorbell-enabled \
    --frigate-doorbell-topic "frigate/{camera_name}/analytics/doorbell"
```

### Using RTSP Camera (Fallback)

```bash
unifi-cam-proxy --mac '{unique MAC}' -H {NVR IP} -c /client.pem -t {token} \
    rtsp \
    -s {rtsp source}
```

---

## Next Steps

1. **Test with Real Doorbell Events:**
   - Configure Frigate with a doorbell camera
   - Enable doorbell detection in Frigate
   - Verify events appear in UniFi Protect

2. **Configure Frigate:**
   - Set up MQTT integration in Frigate
   - Configure doorbell detection zones
   - Ensure doorbell events are published to the correct topic

3. **Monitor and Debug:**
   - Check logs for any doorbell event forwarding issues
   - Verify event timestamps are correct
   - Confirm events appear in UniFi Protect UI

---

## Rollback Instructions

If issues arise, restore from git:

```bash
# Restore removed files
git checkout HEAD~1 unifi/cams/dahua.py unifi/cams/hikvision.py unifi/cams/reolink.py unifi/cams/reolink_nvr.py unifi/cams/tapo.py

# Restore documentation
git checkout HEAD~1 docs/docs/configuration/dahua.md docs/docs/configuration/hikvision.md docs/docs/configuration/reolink.md docs/docs/configuration/reolink_nvr.md docs/docs/configuration/tapo.md

# Restore requirements
cp requirements.txt.bak requirements.txt

# Restore main.py
git checkout HEAD~1 unifi/main.py
```

---

## Notes

- The RTSPCam implementation is kept as a fallback for non-Frigate cameras
- Doorbell support is specifically for Frigate-integrated doorbell cameras
- The change simplifies the codebase and improves maintainability
- All existing motion detection and smart detection features continue to work
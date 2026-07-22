# Implementation Summary: Doorbell Support & Camera Removal

## Executive Summary

This document provides a concise summary of the changes needed to add doorbell support from Frigate and remove support for Dahua, Hikvision, Reolink, and Tapo cameras.

---

## Files to Modify

### 1. `unifi/main.py`

**Changes Required:**
- Remove imports for `DahuaCam`, `HikvisionCam`, `Reolink`, `ReolinkNVRCam`, `TapoCam`
- Update `CAMS` dictionary to only include `FrigateCam` and `RTSPCam`
- Update argument parser choices (keep all existing model choices)

**Before:**
```python
from unifi.cams import (
    DahuaCam,
    FrigateCam,
    HikvisionCam,
    Reolink,
    ReolinkNVRCam,
    RTSPCam,
    TapoCam,
)

CAMS = {
    "amcrest": DahuaCam,
    "dahua": DahuaCam,
    "frigate": FrigateCam,
    "hikvision": HikvisionCam,
    "lorex": DahuaCam,
    "reolink": Reolink,
    "reolink_nvr": ReolinkNVRCam,
    "rtsp": RTSPCam,
    "tapo": TapoCam,
}
```

**After:**
```python
from unifi.cams import (
    FrigateCam,
    RTSPCam,
)

CAMS = {
    "frigate": FrigateCam,
    "rtsp": RTSPCam,
}
```

---

### 2. `unifi/cams/frigate.py`

**Changes Required:**
- Add MQTT subscription to doorbell events
- Add doorbell event handling methods
- Update argument parser with doorbell-specific options

**New Methods to Add:**
```python
async def subscribe_to_doorbell_events(self):
    """Subscribe to Frigate doorbell events via MQTT."""
    pass

async def handle_doorbell_event(self, message):
    """Handle incoming doorbell event from Frigate."""
    pass

async def send_doorbell_event(self, frigate_event):
    """Send doorbell event to UniFi Protect."""
    pass
```

**New Arguments to Add:**
```python
parser.add_argument(
    "--frigate-mqtt-host",
    default="localhost",
    help="Frigate MQTT server hostname"
)
parser.add_argument(
    "--frigate-doorbell-topic",
    default="frigate/{camera_name}/analytics/doorbell",
    help="MQTT topic prefix for doorbell events"
)
```

---

### 3. `requirements.txt`

**Lines to Remove:**
```
amcrest==1.9.8
hikvisionapi==0.3.2
reolinkapi==0.1.5
pytapo==3.3.21
```

**Lines to Keep:**
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

### 4. `README.md`

**Changes Required:**
- Update "Things that work" section to mention Frigate doorbell support
- Remove mentions of Dahua, Hikvision, Reolink, and Tapo

**Before:**
```
Things that work:

* Live streaming
* Full-time recording
* Motion detection with certain cameras
* Smart Detections using [Frigate](https://github.com/blakeblackshear/frigate)
```

**After:**
```
Things that work:

* Live streaming
* Full-time recording
* Motion detection with certain cameras
* Doorbell events from Frigate
* Smart Detections using [Frigate](https://github.com/blakeblackshear/frigate)
```

---

### 5. Documentation Files (to Delete)

Delete these files:
- `docs/docs/configuration/dahua.md`
- `docs/docs/configuration/hikvision.md`
- `docs/docs/configuration/reolink.md`
- `docs/docs/configuration/reolink_nvr.md`
- `docs/docs/configuration/tapo.md`

---

### 6. Python Files (to Delete)

Delete these files:
- `unifi/cams/dahua.py`
- `unifi/cams/hikvision.py`
- `unifi/cams/reolink.py`
- `unifi/cams/reolink_nvr.py`
- `unifi/cams/tapo.py`

---

## Implementation Steps

### Step 1: Remove Camera Implementations

```bash
# Delete camera implementation files
rm unifi/cams/dahua.py
rm unifi/cams/hikvision.py
rm unifi/cams/reolink.py
rm unifi/cams/reolink_nvr.py
rm unifi/cams/tapo.py
```

### Step 2: Update Main.py

1. Open `unifi/main.py`
2. Update imports (remove DahuaCam, HikvisionCam, Reolink, ReolinkNVRCam, TapoCam)
3. Update CAMS dictionary
4. No changes needed to argument parser choices (keep all existing model choices)

### Step 3: Update Requirements

```bash
# Backup current requirements
cp requirements.txt requirements.txt.bak

# Remove unnecessary packages
sed -i '/^amcrest==1.9.8$/d' requirements.txt
sed -i '/^hikvisionapi==0.3.2$/d' requirements.txt
sed -i '/^reolinkapi==0.1.5$/d' requirements.txt
sed -i '/^pytapo==3.3.21$/d' requirements.txt
```

### Step 4: Update FrigateCam

1. Open `unifi/cams/frigate.py`
2. Add MQTT client initialization
3. Add doorbell event handling methods
4. Add doorbell-specific arguments to argument parser
5. Integrate doorbell event handling into the main event loop

### Step 5: Update README.md

Open `README.md` and update the "Things that work" section.

### Step 6: Delete Documentation

```bash
# Delete camera-specific documentation
rm docs/docs/configuration/dahua.md
rm docs/docs/configuration/hikvision.md
rm docs/docs/configuration/reolink.md
rm docs/docs/configuration/reolink_nvr.md
rm docs/docs/configuration/tapo.md
```

---

## Testing Checklist

After implementation, verify:

- [ ] Project installs without errors
- [ ] No references to removed cameras in code
- [ ] Frigate camera can be selected
- [ ] RTSP camera can still be selected (fallback)
- [ ] Doorbell events are received from Frigate
- [ ] Doorbell events appear in UniFi Protect
- [ ] Regular motion events still work
- [ ] Smart detection still works

---

## Post-Implementation Verification

1. **Check for remaining references:**
   ```bash
   grep -r "dahua\|hikvision\|reolink\|tapo" --include="*.py" .
   ```

2. **Test installation:**
   ```bash
   pip install -e .
   ```

3. **Test with Frigate:**
   ```bash
   unifi-cam-proxy --mac '{unique MAC}' -H {NVR IP} -c /client.pem -t {token} \
       frigate \
       -s {rtsp source} \
       --frigate-camera {Name} \
       --frigate-mqtt-host {mqtt host}
   ```

---

## Rollback Instructions

If issues occur, restore from backup:

```bash
# Restore requirements
cp requirements.txt.bak requirements.txt

# Restore git state
git checkout HEAD~1
```
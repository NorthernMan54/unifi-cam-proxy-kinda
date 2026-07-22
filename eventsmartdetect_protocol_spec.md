# EventSmartDetect Protocol Notes — Frigate → Protect Bridge Spec

Derived from a captured `DEVICE_TO_BACKEND` log for camera `F4E2C60D4B4C` (2026-07-01/02). This confirms and refines the wire format your `UnifiCamBase`/`FrigateCam` `trigger_smart_detect_*` path needs to emit.

## 1. Log line / session framing (outside the JSON)

```
<iso_timestamp>  DEVICE_TO_BACKEND  <session_id>  <byte_len>  <json>
```

- `session_id` = `<camera_MAC>-<connection_epoch_ms>` (e.g. `F4E2C60D4B4C-1782948427777`). This is the WebSocket session key, stable for the life of one `ds` connection. Your emulator should mint a fresh one per (re)connect, formatted identically (uppercase MAC, no colons, dash, epoch ms).
- `byte_len` is just the serialized JSON length — cosmetic, not required for the emulator to reproduce.

## 2. Message envelope

```json
{
  "from": "ubnt_avclient",
  "functionName": "EventSmartDetect",
  "inResponseTo": 0,
  "messageId": 81411216,
  "payload": { ... },
  "responseExpected": false,
  "timeStamp": "2026-07-01T23:27:19.948+00:00",
  "to": "UniFiVideo"
}
```

- `messageId`: monotonically increasing per-connection counter across **all** message types (not just smart-detect) — increment a single counter in the emulator, don't scope it per-function.
- `timeStamp` is set fractionally *after* `payload.clockWall` (tens of ms later) — i.e. it's stamped at send time, not detection time. Fine to set at emit time.
- `responseExpected: false`, `inResponseTo: 0` for all observed smart-detect pushes — this is fire-and-forget telemetry, not a request/response pair.

## 3. Architectural layers: Motion events vs. object tracking vs. EventSmartDetect

There are three distinct scopes to understand:

**1. Motion event (Unifi smart event context window)**
- Triggered by motion detector `motion: ON` MQTT message
- Spans from motion start until motion stop (`motion: OFF`)
- A single motion window may contain multiple simultaneous object detections
- Unifi API: `trigger_smart_detect_start()` (enter motion) → `trigger_smart_detect_stop()` (exit motion)

**2. Frigate object tracking (per-object lifecycle)**
- Each Frigate MQTT event message corresponds to a single object being tracked by Frigate's detector
- Frigate's `event.id` = unique identifier for this object instance; analogous to Unifi's `trackerID`
- Frigate event lifecycle: `type: "new"` → `type: "update"` (0+ messages) → `type: "end"`
- Multiple Frigate object detections may occur within a single motion window

**3. EventSmartDetect messages (per-object protocol messages)**
- Sent per tracked object as it enters, moves, and exits zones
- Sent *within* an active motion window (between `trigger_smart_detect_start()` and `trigger_smart_detect_stop()`)
- Each message carries: `edgeType` (`enter`/`moving`/`leave`), object descriptor (`trackerID`, `zones`, confidence), and aggregated `zonesStatus`
- Mapping: Frigate `event.id` (string) → Unifi `trackerID` (int, allocated per object for the bridge session)

**Correct flow example:**
```
MQTT motion: ON
  └─ trigger_smart_detect_start()
     └─ Motion window begins, eventId counter starts
        
        MQTT frigate event type=new (vehicle detected in zone 1)
          └─ Allocate trackerID=700000 for this Frigate event.id
             └─ Send EventSmartDetect: edgeType="enter", trackerID=700000, zones=[1]
        
        MQTT frigate event type=update (vehicle moves to zone 2)
          └─ Send EventSmartDetect: edgeType="moving", trackerID=700000, zones=[2]
        
        MQTT frigate event type=end (vehicle leaves)
          └─ Send EventSmartDetect: edgeType="leave", trackerID=700000, zones=[], zonesStatus forced to leave
             └─ Include smartDetectSnapshots[], trackerIDAttrMap
        
        (Meanwhile, another object may have entered/updated/left within the same motion window)

MQTT motion: OFF
  └─ trigger_smart_detect_stop()
     └─ Motion window ends, all remaining trackerIDs cleaned up
```

**Key constraint:** Frigate's `event.id` is the object-level grouping key; do NOT treat it as a motion event. One motion window can contain many Frigate events (objects).

## 4. Payload fields

| Field | Notes |
|---|---|
| `clockMonotonic` | Camera uptime clock, ms, monotonic since boot. Must stay internally consistent — don't reset per event. |
| `clockStream` | Stream-relative clock, ms, base rate given by `clockStreamRate`. Runs ~20-21s behind `clockMonotonic` in this capture (i.e. offset by stream start delay) — keep a fixed `clockMonotonic - clockStream` offset per stream/session. |
| `clockStreamRate` | Always `1000` here (ms ticks, not 90kHz PTS) — simpler than the FLV path's 90k-based timestamps. |
| `clockWall` | Epoch ms wall clock. This is what you'll derive directly from Frigate's event `frame_time`. |
| `eventId` | Single incrementing counter, **shared across all tracked objects**, incremented once per emitted message (not per track). Persists for camera lifetime. |
| `edgeType` | State of *zone occupancy* for this message: `enter`, `moving`, `leave`, or `none` (no zone transition, object idle/stationary). |
| `objectTypes` | Top-level array — **correction from a second capture**: this is not a leak/bug as first assumed. It stays populated (e.g. `["person"]`) for the *entire* life of a track — `enter` through every `moving` message — and only clears to `[]` on the terminal `leave` (when `descriptors` also empties). Model it as "current active object types across all live tracks on this camera," cleared only when the last track closes. |
| `displayTimeoutMSec` | UI staleness hint, ~300-360ms in this capture, jitters per message. Not safety-critical to model precisely; a value in that band is fine. |
| `descriptors[]` | Per-tracked-object array, see below. |
| `zonesStatus` | Dict keyed by zone ID string (`"1"`,`"2"`,`"3"`) → `{level, status}`. Every zone configured on the camera appears every message, not just the active one. |
| `smartDetectSnapshotFullFoV` + dims | Only populated on the terminal `leave` message for a track. Filename pattern: `smartdetectsnap_zone_<something>_fullfov.jpg` (seen literal `00000000` placeholder in some, and a tracker-based name in others — treat as an internal reference id, not required to be globally meaningful, just resolvable by your snapshot server). |
| `smartDetectSnapshots[]` | Also only on terminal `leave`. One entry per trackerID that closed, carrying the **best-confidence frame**. **Update from a second capture**: `confidenceLevel`, `coord`, and `framingRect` are not always present — one session included them, another omitted all three keys entirely rather than sending empty/zero values. Treat these three as optional; only `clockBestMonotonic`, `clockBestWall`, `smartDetectSnapshot`, `smartDetectSnapshotHeight`/`Width`, `smartDetectSnapshotName`, `smartDetectSnapshotType`, `trackerID` are reliably present. Your emitter can always include the full set (Protect's parser tolerates extra fields), but don't build validation logic that requires `confidenceLevel`/`coord`/`framingRect` on this specific sub-object. |
| `smartDetectSnapshotFullFoV` filename | Confirmed pattern: `smartdetectsnap_zone_<8-digit zero-padded counter>_fullfov.jpg`. The counter does **not** match `eventId` or `messageId` in the observed session (fullFoV counter was `00000001` while `eventId` was already at `18`) — it appears to be an independent per-camera snapshot-save counter. Safe to implement as its own monotonic counter, incremented once per `leave` event, zero-padded to 8 digits. |
| `smartDetectSnapshot` (per-tracker) filename | Confirmed formula: `smartdetectsnap_zone_<trackerID><clockBestWall>.jpg` — i.e. the trackerID and the best-frame epoch ms wall clock concatenated directly with no separator (e.g. trackerID `2` + clockBestWall `1766543821684` → `smartdetectsnap_zone_21766543821684.jpg`). |
| `trackerIDAttrMap` | Also only on terminal `leave`. Summarizes the whole track: `{trackerID: {objectType, zone: [zones visited, most-recent-first]}}`. |

### `descriptors[]` object fields

| Field | Notes |
|---|---|
| `trackerID` | Persistent per-track integer ID, stable across all messages for one continuous track. Assign once per Frigate `event.id` (map Frigate's string ID → an incrementing int). |
| `objectType` | `person` / `animal` (and presumably `vehicle`, `package`, etc. — not seen here but expected). Maps directly to your existing `_frigate_to_unifi`/`_LABEL_TO_TYPE` dicts. |
| `coord` | `[x, y, w, h]` in the **640×360 sub-stream** pixel space (matches your Hikvision `_video3` low-res stream) — scale Frigate's detection box (which is against the full-res `_src` frame) down accordingly. |
| `boxColor` | `"red"` for the actively-tracked, in-zone object; `"white"` for background/idle detections not currently of interest (the stationary "animal" blobs in this capture, confidence 50-67%, never cross a zone). Treat as a UI hint: red when `edgeType != none` for that track, white otherwise. |
| `confidenceLevel` | 0-100 int, matches Frigate's `score * 100`. |
| `firstShownTimeMs` / `idleSinceTimeMs` | Epoch ms. For actively moving tracks these stay pinned to track start (`firstShownTimeMs` constant across all messages in a track) while `idleSinceTimeMs: 0` signals "not idle." For a stationary/idle object both fields equal the same fixed timestamp and don't advance — model this as: `idleSinceTimeMs = firstShownTimeMs` once `stationary=true`, else `0`. |
| `stationary` | bool — Frigate exposes a similar concept via `event.stationary`; pass through directly. |
| `zones` | Array of zone IDs (ints) the object currently overlaps, ordered most-recent-first when in multiple (`[2,1]` while transitioning). Single zone → single-element array. Empty when not in any zone. |
| `attributes`, `lines`, `loiterZones`, `secondLensZones`, `coord3d` | Not exercised in either capture — safe to emit as `null`/`[]`/`[-1,-1]` defaults. |
| `name`, `tag` | **Update from a second capture**: not always empty. A session with a matched face returned a populated (redacted-in-log) value in both `name` and `tag` on every descriptor for that trackerID, for the life of the track. This is Protect's facial-recognition "known person" label, populated by the camera when it matches its local face DB — not something the wire protocol invents on its own. **Relevant to your bridge**: your Frigate stats show `face_recognition_speed` is active, meaning Frigate is already doing face recognition. If Frigate's face-recognition match returns a name for a `person` event, you can populate `name`/`tag` here with that matched name to get the same "recognized person" labeling behavior in Protect's UI; otherwise leave both as `""`. |

## 5. Observed message lifecycle for one track

1. **enter** — first zone crossing. `objectTypes` populated, `edgeType: "enter"`, that zone's `zonesStatus[..].status: "enter"`.
2. **moving** — repeated every ~250-500ms while the object is tracked and changing zones/position. `objectTypes: []`, box color stays `"red"`.
3. (object may cross into/out of multiple zones — messages show 2 zones simultaneously non-empty during a transition frame, e.g. `zones:[2,1]`)
4. **leave** — terminal message for the track. This stop message explicitly carries the post-departure zone state in `zonesStatus`, with the active zone reporting `status: "leave"` (e.g. `"zonesStatus": {"2": {"level": 68, "status": "leave"}}`). Other configured zones remain present and may be `"none"`. This same terminal message also carries the enriched summary payload (`smartDetectSnapshotFullFoV`, `smartDetectSnapshots`, `trackerIDAttrMap`). This is the message Protect's UI/timeline actually keys its event thumbnail off of.

Separately, low-confidence stationary "animal" blobs cycle through their own independent `enter`→`none`→`leave` messages roughly every 5 minutes even with zero real motion — this is background noise-floor detection, not something you need to reproduce faithfully; Frigate's static "false positive"/stationary filtering already suppresses most of this on the source side.

## 6. Implications for your `FrigateCam` bridge

Given your existing dual-dict label mapping and dataclass hierarchy in `unifi/protect_api/smart_detect.py`, the additions needed:

1. **Session/message counters**: one shared `messageId` counter and one shared `eventId` counter per emulated camera connection (not per Frigate event).
2. **TrackerID allocator**: map Frigate `event.id` → synthetic incrementing int, retained for the lifetime of the Frigate event.
3. **Zone ID mapping**: Frigate zone *names* → Protect's numeric zone IDs as configured per-camera in Protect (this has to be a static config map per camera, since Protect assigns zone numbers at zone-creation time in its own UI).
4. **Clock offset tracking**: maintain `clockMonotonic` as real elapsed ms since emulated-stream start; derive `clockStream` via a fixed per-session offset; derive `clockWall` directly from Frigate's `frame_time * 1000`.
5. **State machine per Frigate event**: on Frigate `event.type == "new"` → emit `enter`; on `update` with zone change → emit `moving`; on `end` → emit `leave` with populated `smartDetectSnapshots`/`trackerIDAttrMap`, plus generate/serve the two snapshot JPEGs (crop + full FoV) from Frigate's stored snapshot for that event.
6. **`zonesStatus` completeness**: every message must include *all* configured zones for the camera, not just the changed one — build this from the static zone-name→ID map plus current per-zone occupancy state.
7. **boxColor derivation**: `"red"` while `edgeType` for the track is active (in zone), `"white"` for out-of-zone/background detections you still want to surface (optional — could just omit non-zone detections entirely, which is simpler and matches Frigate's zone-scoped semantics better).

This is enough to drive `trigger_smart_detect_enter`/`_moving`/`_leave` (or equivalent) calls in your existing `UnifiCamBase` API from a Frigate MQTT event stream with correct field population.

## 7. Stationary Tracker Behavior — `edgeType: "none"`

**Confirmed from real device capture (2026-07-01/02, trackers 716235/716236, camera F4E2C60D4B4C).**

### `edgeType: "none"` — stationary background heartbeat

When an object is tracked by the camera but is **stationary and not occupying any configured zone**, the device emits periodic updates with `edgeType: "none"` roughly every 5 minutes instead of `"moving"`:

```json
{
  "edgeType": "none",
  "objectTypes": [],
  "displayTimeoutMSec": 308,
  "descriptors": [
    { "trackerID": 716236, "stationary": true, "zones": [], "boxColor": "white", ... }
  ],
  "zonesStatus": { "1": {"level": 0, "status": "none"}, "2": {"level": 0, "status": "none"}, "3": {"level": 0, "status": "none"} }
}
```

**Rules for `edgeType: "none"` messages:**
- `objectTypes: []` — stationary background objects are NOT reported as active detections
- All `zonesStatus` entries: `level: 0`, `status: "none"` — no zone transitions
- `displayTimeoutMSec: ~300` (much lower than active `"moving"` messages at ~10000)
- `descriptors` still carries the stationary tracker with its current position and `stationary: true`
- Does NOT trigger zone enter/exit transitions in Protect's UI

**Implementation mapping (Frigate → bridge):**

- When Frigate `type: "new"` has `after.position_changes = 0` → emit `edgeType: "none"` stationary object
- When Frigate `type: "update"` has `after.position_changes = 0` → emit `edgeType: "none"` stationary object
- When Frigate `type: "end"` has `after.position_changes = 0` → emit `edgeType: "none"` stationary object

- When Frigate `type: "update"` has `before.position_changes = 0 and after.position_changes > 0` → emit `edgeType: "enter"` (first appearance)

- Stationary objects always have `zones: []` (not in any configured zone) in the observed captures
  
### Stationary bystanders in `leave` messages

When an active (non-stationary) tracker departs its zone, the resulting `leave` message may still have stationary background trackers visible. These are handled with a strict payload split:

| Payload field | Active (departing) tracker | Stationary bystander tracker |
|---|---|---|
| `descriptors[]` | ❌ empty (already in final update) | ✅ included (carry-along visibility ping) |
| `smartDetectSnapshots[]` | ✅ included with best-confidence snapshot | ❌ excluded |
| `trackerIDAttrMap` | ✅ included | ❌ excluded |

**From the captured leave message (eventId 36203, 2026-07-02T00:02:50):**
- `descriptors`: carried stationary animal 716236 (still visible, background noise)
- `smartDetectSnapshots`: only person 716259 (the departing active tracker)
- `trackerIDAttrMap`: only `"716259": {"objectType": "person", "zone": [2,3,1]}`

**Implementation rule:** In `trigger_smart_detect_stop`, use the LATEST known state of each tracker to decide which bucket it goes into. A tracker whose most recent descriptor has `stationary: true` is a bystander — include it in `descriptors` only, not in `smartDetectSnapshots`/`trackerIDAttrMap`.

### Multiple trackers batched in one message

Real device messages frequently carry multiple trackers in a single `descriptors` array within one message (e.g. both 716235 and 716236 in the 23:32 message). Our bridge emits one message per Frigate object update due to MQTT event granularity — this is a minor shape difference and does not affect correctness in Protect's UI.

## 8. EventSmartMotion — Motion Window Heartbeat Protocol

`EventSmartMotion` runs concurrently with `EventSmartDetect`, representing the raw motion detector signal (independent of object classification). It uses a separate, independent `eventId` counter.

### Message types

| `edgeType` | `eventType` | When sent |
|---|---|---|
| `start` | `motion` | Motion window opens (after `lingerEventStartMSec` delay) |
| `unknown` | `pulse` | Heartbeat every ~2-3 seconds while motion window is open |
| `stop` | `motion` | Motion window closes |

### `levels` field — Motion Intensity (0–100)

The `levels` field is a dict keyed by zone ID string → integer 0–100:

```json
"levels": {"1": 75}
```

**Recommended implementation: prefer Frigate recordings motion, fall back to bounding-box area**

Prefer the motion percentage returned by Frigate's recordings API for the active camera/event window. It is already normalized to a 0-100 scale and matches the semantics of EventSmartMotion better than raw geometry. If the recordings API is unavailable or returns no usable segment, fall back to the most recent Frigate event's `after.area` (bounding box area in pixels within the Frigate detection frame):

```
level = clamp(int(after.area / (frigate_detect_width * frigate_detect_height) * 100), 0, 100)
```

Where `frigate_detect_width` × `frigate_detect_height` is the Frigate detection resolution (default 1280×720). This gives a natural 0–100 scale proportional to how much of the frame the detected object occupies.

When recordings motion is available, use that value directly as the zone level. Keep the area-derived value only as a fallback so EventSmartMotion still emits a stable level when the API cannot be queried.

**Observed behavior:**
- Levels fluctuate between pulses as the bounding box changes size (e.g. 75 → 82 → 100 → 50)
- The level on the `stop` message reflects the final state
- `levels` keys match the motion zone IDs from `ChangeSmartMotionSettings` (typically `{"1": N}` for a single full-frame zone)
- Pulse events always use `eventId: 18446744073709551615` (max uint64 / sentinel), **not** the motion eventId counter

### `clockBestMonotonic` / `clockBestWall`

- `start` and `pulse`: both are `0` (not yet known)
- `stop`: set to the monotonic/wall time of the **first motion frame** (i.e. when the motion window opened), not the stop time

### Snapshot fields

- `start`: filename stubs are populated (`motionHeatmap`, `motionSnapshot`, `motionSnapshotFullFoV`, `motionRawHeatmapNPZ`) with size fields set; actual data uploaded on `stop` GetRequest
- `pulse`: all snapshot fields are empty strings / zero sizes
- `stop`: populated with actual filenames; Protect immediately issues `GetRequest` for each file

## 9. Doorbell-Specific Protocol

### MCUEventMessage — Doorbell Event Format

Doorbell events (ring/chime) are sent via the `MCUEventMessage` function name. The log shows this is a distinct message type from the object tracking protocols above.

**Example from `ds-extracted-doorbell.log`:**
```json
{
    "from":"ubnt_avclient",
    "functionName":"MCUEventMessage",
    "inResponseTo":0,
    "messageId":85427379,
    "payload":{
        "eventType":"EventRingButtonPressed"
        },
    "responseExpected":false,
    "timeStamp":"2026-07-21T13:54:06.440+00:00",
    "to":"UniFiVideo"
}
```

**Key characteristics:**
- `functionName`: `MCUEventMessage` (distinct from `EventSmartDetect` and `EventSmartMotion`)
- `payload.event.type`: `"ring"` for doorbell ring events
- `payload.doorbell`: `true` flag indicating this is a doorbell device
- `payload.event.smartDetect`: may contain object tracking data if smart detection is active
- `messageId`: monotonically increasing counter shared across all message types on the connection

### ubnt_avclient_hello — Doorbell Feature Detection

The `ubnt_avclient_hello` message includes a `features` object that indicates whether the device is a doorbell. This is how your emulator should detect doorbell vs. camera devices.

**Doorbell detection from `ubnt_avclient_hello`:**
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
      "doorAccessConfig": false,
      "chimeControl": false,
      "welcomeLed": true,
      "ringVolume": 1,
      "audioCodecs": ["aac", "opus"],
      "audioStyle": ["nature", "noiseReduced"],
      "talkback": {
        "typeFmt": "aac",
        "typeIn": "serverudp",
        "bindAddr": "0.0.0.0",
        "bindPort": 7004
      },
      "videoCodecs": ["h264", "h265", "mjpg"],
      "downScaleLevels": ["2K+", "2K", "HD"],
      "streamEncryptable": true
    },
    "model": "UVC Doorbell Lite",
    "name": "Wasaga Doorbell",
    "protocolVersion": 67,
    "uptime": 569595
  }
}
```

**Doorbell-specific features observed:**
- `doorbell: true` — primary indicator of doorbell device type
- `mic: true` / `speaker: true` — audio capabilities for two-way talkback
- `welcomeLed: true` — LED indicator for doorbell rings
- `talkback` settings — configured for doorbell audio communication
- `ringVolume` — ring notification volume level
- `smartDetect` — includes person/vehicle/animal detection for motion events
- `chimeControl` — may be used to control ring chime behavior

**Implementation guidance:**
- When `features.doorbell` is `true`, route the connection to doorbell-specific handlers
- Doorbell events should be processed via `MCUEventMessage` rather than `EventSmartDetect`
- Ring events may include `smartDetect` object data if motion was detected with object classification
- Audio stream settings differ from standard cameras (talkback, ring volume, chime control)

### Doorbell Event Flow

```
Device connects
  └─ ubnt_avclient_hello (features.doorbell: true)
     └─ Identify as doorbell device
     
Ring occurs
  └─ MCUEventMessage (functionName: "MCUEventMessage")
     └─ payload.event.type: "ring"
        └─ May include smartDetect object data
           └─ payload.doorbell: true
     
Motion detection (optional)
  └─ EventSmartDetect (if person/vehicle detected)
     └─ Edge types: enter, moving, leave
        └─ Triggers zone-based smart detect events
```

### Doorbell vs. Camera Protocol Differences

| Feature | Doorbell | Standard Camera |
|---|---|---|
| `ubnt_avclient_hello.features.doorbell` | `true` | `false` or absent |
| Primary event function | `MCUEventMessage` | `EventSmartDetect` |
| Audio features | `mic: true`, `speaker: true` | May vary |
| Talkback config | Configured for doorbell audio | Standard video audio |
| Ring indicators | `welcomeLed: true`, `ringVolume` | N/A |
| Smart detect scope | Ring events + motion | Motion + object tracking |
| Chime control | `chimeControl` feature | N/A |

This completes the doorbell protocol specification. Doorbell devices follow the same session framing and message envelope structure as standard cameras, but use `MCUEventMessage` for ring events and report `features.doorbell: true` in the hello message.



TODO: Need to update support for stationary object - after.position_changes s/b > 0 for an active object, after.position_changes
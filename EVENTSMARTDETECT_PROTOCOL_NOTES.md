# EventSmartDetect Protocol Notes — Frigate → Protect Bridge Spec

Derived from captured `DEVICE_TO_BACKEND` logs for camera `F4E2C60D4B4C` (2026-07-01/02, 2026-07-21). This confirms and refines the wire format your `UnifiCamBase`/`FrigateCam` `trigger_smart_detect_*` path needs to emit.

> **Revision note (2026-07-21 capture):** This revision adds a previously-undocumented parallel "loiter" tracking subsystem and the `warmup` edgeType, and corrects the earlier claim about `objectTypes` staying populated for a track's full lifetime. See Section 11.

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

- `messageId`: monotonically increasing per-connection counter across **all** message types (not just smart-detect) — increment a single counter in the emulator, don't scope it per-function. **Confirmed (2026-07-21 capture) this counter is also shared across the loiter and zone subsystems described in Section 11** — they are not independently numbered.
- `timeStamp` is set fractionally *after* `payload.clockWall` (tens of ms later) — i.e. it's stamped at send time, not detection time. Fine to set at emit time. **Exception:** the terminal `leave` message can be held significantly longer before transmission — see Section 11.6.
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
- Each message carries: `edgeType` (`enter`/`moving`/`leave`/`warmup` — see Section 11), object descriptor (`trackerID`, `zones`, confidence), and aggregated `zonesStatus`
- Mapping: Frigate `event.id` (string) → Unifi `trackerID` (int, allocated per object for the bridge session)
- **Update (Section 11):** a single tracked object can concurrently drive *two* independent EventSmartDetect state machines — a "zone" one (`zonesStatus`, this section's edgeTypes) and a "loiter" one (`loiterZonesStatus`, edgeType `warmup` plus presumably a loiter-triggered state not yet captured). Treat these as sibling emitters keyed by the same trackerID, not a single combined state machine.

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
| `eventId` | Single incrementing counter, **shared across all tracked objects, and (confirmed 2026-07-21) shared across the loiter and zone subsystems too**, incremented once per emitted message (not per track). Persists for camera lifetime. |
| `edgeType` | State of *zone occupancy* for this message: `enter`, `moving`, `leave`, `none` (no zone transition, object idle/stationary — zone subsystem), or `warmup` (loiter subsystem; object being monitored for loitering but hasn't crossed the loiter threshold — see Section 11). |
| `objectTypes` | Top-level array — **corrected in this revision.** Earlier capture suggested this stays populated (e.g. `["person"]`) for the entire life of a track. The 2026-07-21 capture shows this is only true *within zone-subsystem messages* (`enter`/`moving`): it is `[]` on the terminal zone `leave`, **and it is also `[]` on the sibling loiter-subsystem `warmup` message for the identical trackerID on the same tick.** Model it per-message, scoped to the subsystem/edgeType that emitted it, not as a single object-level flag. |
| `displayTimeoutMSec` | UI staleness hint, ~300-360ms in this capture, jitters per message. Not safety-critical to model precisely; a value in that band is fine. |
| `descriptors[]` | Per-tracked-object array, see below. |
| `zonesStatus` | Dict keyed by *zone* ID string (`"1"`,`"2"`,`"3"` in these captures) → `{level, status}`. Every configured zone appears every message, not just the active one. Belongs to the zone subsystem. |
| `loiterZonesStatus` | **New in this revision.** Dict keyed by *loiter-zone* ID string (`"5"`,`"7"`,`"9"` in this capture — a distinct ID namespace from `zonesStatus`) → `{level, loiterTriggerTime, startStreakTime, status}`. Structurally parallel to `zonesStatus` but carries two extra timing fields, both observed as `0` in this capture (loiter threshold never tripped). Only appears on loiter-subsystem messages (`edgeType: "warmup"` or loiter `leave`), never alongside `zonesStatus` in the same message in this capture. |
| `smartDetectSnapshotFullFoV` + dims | Only populated on the terminal `leave` message for a track. Filename pattern differs by subsystem — see Section 11.4. |
| `smartDetectSnapshots[]` | Also only on terminal `leave`. One entry per trackerID that closed, carrying the **best-confidence frame**. `confidenceLevel`, `coord`, and `framingRect` are not always present — one session included them, another omitted all three keys entirely rather than sending empty/zero values. Treat these three as optional; only `clockBestMonotonic`, `clockBestWall`, `smartDetectSnapshot`, `smartDetectSnapshotHeight`/`Width`, `smartDetectSnapshotName`, `smartDetectSnapshotType`, `trackerID` are reliably present (confirmed again 2026-07-21, including an explicit empty-string `smartDetectSnapshotName`). Your emitter can always include the full set (Protect's parser tolerates extra fields), but don't build validation logic that requires `confidenceLevel`/`coord`/`framingRect` on this specific sub-object. |
| `smartDetectSnapshotFullFoV` filename | Confirmed pattern: `smartdetectsnap_zone_<8-digit zero-padded counter>_fullfov.jpg` for the zone subsystem, and `smartdetectsnap_loiter_<8-digit zero-padded counter>_fullfov.jpg` for the loiter subsystem (new — see Section 11.4). The counter does **not** match `eventId` or `messageId` — it appears to be an independent per-camera-per-subsystem snapshot-save counter. The zone-subsystem counter's exact increment behavior across a session is not fully pinned down by current captures (see Section 11.4) — implement as its own monotonic counter per subsystem, but don't assume strict same-session incrementing until confirmed further. |
| `smartDetectSnapshot` (per-tracker) filename | Confirmed formula: `smartdetectsnap_zone_<trackerID><clockBestWall>.jpg` — i.e. the trackerID and the best-frame epoch ms wall clock concatenated directly with no separator (e.g. trackerID `2` + clockBestWall `1766543821684` → `smartdetectsnap_zone_21766543821684.jpg`). |
| `trackerIDAttrMap` | Also only on terminal `leave`. Summarizes the whole track: `{trackerID: {objectType, zone: [zones visited, most-recent-first]}}`. |

### `descriptors[]` object fields

| Field | Notes | 
|---|---|
| `trackerID` | Persistent per-track integer ID, stable across all messages for one continuous track. Assign once per Frigate `event.id` (map Frigate's string ID → an incrementing int). Confirmed (2026-07-21) the same `trackerID` is reused across both the zone-subsystem and loiter-subsystem messages for one object. |
| `objectType` | `person` / `animal` (and presumably `vehicle`, `package`, etc. — not seen here but expected). Maps directly to your existing `_frigate_to_unifi`/`_LABEL_TO_TYPE` dicts. |
| `coord` | `[x, y, w, h]` in the **640×360 sub-stream** pixel space (matches your Hikvision `_video3` low-res stream) — scale Frigate's detection box (which is against the full-res `_src` frame) down accordingly. |
| `boxColor` | `"red"` for the actively-tracked, in-zone object; `"white"` for background/idle detections not currently of interest (the stationary "animal" blobs in this capture, confidence 50-67%, never cross a zone). Treat as a UI hint: red when `edgeType != none` for that track, white otherwise. Confirmed `"red"` held constant through an entire zone-subsystem track in the 2026-07-21 capture. |
| `confidenceLevel` | 0-100 int, matches Frigate's `score * 100`. |
| `firstShownTimeMs` / `idleSinceTimeMs` | Epoch ms. For actively moving tracks these stay pinned to track start (`firstShownTimeMs` constant across all messages in a track) while `idleSinceTimeMs: 0` signals "not idle." Confirmed constant (`1784642058431`) across ~40 messages of a single track in the 2026-07-21 capture. For a stationary/idle object both fields equal the same fixed timestamp and don't advance — model this as: `idleSinceTimeMs = firstShownTimeMs` once `stationary=true`, else `0`. |
| `stationary` | bool — Frigate exposes a similar concept via `event.stationary`; pass through directly. |
| `zones` | Array of zone IDs (ints) the object currently overlaps, ordered most-recent-first when in multiple (`[2,1]` while transitioning). Single zone → single-element array. Empty when not in any zone. **Note:** on loiter-subsystem (`warmup`) messages, `zones` was observed empty (`[]`) even while the sibling zone-subsystem message for the same tick had the object in `zones: [1]` — `zones` in a descriptor is scoped to whichever zone namespace that message's subsystem uses. |
| `attributes`, `lines`, `loiterZones`, `secondLensZones`, `coord3d` | Not exercised in either capture — safe to emit as `null`/`[]`/`[-1,-1]` defaults. |
| `name`, `tag` | Not always empty. A session with a matched face returned a populated (redacted-in-log) value in both `name` and `tag` on every descriptor for that trackerID, for the life of the track. This is Protect's facial-recognition "known person" label, populated by the camera when it matches its local face DB — not something the wire protocol invents on its own. **Relevant to your bridge**: your Frigate stats show `face_recognition_speed` is active, meaning Frigate is already doing face recognition. If Frigate's face-recognition match returns a name for a `person` event, you can populate `name`/`tag` here with that matched name to get the same "recognized person" labeling behavior in Protect's UI; otherwise leave both as `""`. |

## 5. Observed message lifecycle for one track (zone subsystem)

1. **enter** — first zone crossing. `objectTypes` populated, `edgeType: "enter"`, that zone's `zonesStatus[..].status: "enter"`.
2. **moving** — repeated every ~250-500ms while the object is tracked and changing zones/position. `objectTypes` stays populated on these messages, box color stays `"red"`. **Confirmed 2026-07-21:** each "moving" tick's `objectTypes: ["person"]` is specific to the zone-subsystem message; the sibling loiter-subsystem message for the same instant carries `objectTypes: []` (Section 11).
3. (object may cross into/out of multiple zones — messages show 2 zones simultaneously non-empty during a transition frame, e.g. `zones:[2,1]`)
4. **leave** — terminal message for the track. This stop message explicitly carries the post-departure zone state in `zonesStatus`, with the active zone reporting `status: "leave"` (e.g. `"zonesStatus": {"2": {"level": 68, "status": "leave"}}`). Other configured zones remain present and may be `"none"`. This same terminal message also carries the enriched summary payload (`smartDetectSnapshotFullFoV`, `smartDetectSnapshots`, `trackerIDAttrMap`). This is the message Protect's UI/timeline actually keys its event thumbnail off of. **Confirmed 2026-07-21:** this terminal message's wire-clock fields (`clockMonotonic`/`clockWall`) can be nearly contiguous with the prior tick, but the message's actual transmission (`timeStamp`) can lag several seconds behind — see Section 11.6.

Separately, low-confidence stationary "animal" blobs cycle through their own independent `enter`→`none`→`leave` messages roughly every 5 minutes even with zero real motion — this is background noise-floor detection, not something you need to reproduce faithfully; Frigate's static "false positive"/stationary filtering already suppresses most of this on the source side.

Running *concurrently* with the above, the loiter subsystem for the same trackerID cycles `warmup` messages on (approximately) the same tick cadence — see Section 11.

## 6. Implications for your `FrigateCam` bridge

Given your existing dual-dict label mapping and dataclass hierarchy in `unifi/protect_api/smart_detect.py`, the additions needed:

1. **Session/message counters**: one shared `messageId` counter and one shared `eventId` counter per emulated camera connection (not per Frigate event, and not per subsystem — see Section 11.2).
2. **TrackerID allocator**: map Frigate `event.id` → synthetic incrementing int, retained for the lifetime of the Frigate event. The same trackerID is reused by both the zone-subsystem and loiter-subsystem messages for that object.
3. **Zone ID mapping**: Frigate zone *names* → Protect's numeric zone IDs as configured per-camera in Protect (this has to be a static config map per camera, since Protect assigns zone numbers at zone-creation time in its own UI). **Loiter zones use a separate ID namespace from regular zones** (observed `5`/`7`/`9` vs `1`/`2`/`3`) — maintain a second static map if you intend to emit `loiterZonesStatus`.
4. **Clock offset tracking**: maintain `clockMonotonic` as real elapsed ms since emulated-stream start; derive `clockStream` via a fixed per-session offset; derive `clockWall` directly from Frigate's `frame_time * 1000`.
5. **State machine per Frigate event**: on Frigate `event.type == "new"` → emit `enter`; on `update` with zone change → emit `moving`; on `end` → emit `leave` with populated `smartDetectSnapshots`/`trackerIDAttrMap`, plus generate/serve the two snapshot JPEGs (crop + full FoV) from Frigate's stored snapshot for that event. **Whether your bridge needs to also emit a parallel loiter/`warmup` state machine is a product decision** — Frigate doesn't have a native "loiter" concept distinct from `stationary`, so this would need to be synthesized (e.g. from Frigate's stationary-duration tracking) if you want Protect's loiter-zone UI features to light up. If you skip it, Protect simply won't show loiter-specific alerts; the regular zone enter/moving/leave path is unaffected.
6. **`zonesStatus` completeness**: every message must include *all* configured zones for the camera, not just the changed one — build this from the static zone-name→ID map plus current per-zone occupancy state. The same completeness rule applies to `loiterZonesStatus` if you implement it.
7. **boxColor derivation**: `"red"` while `edgeType` for the track is active (in zone), `"white"` for out-of-zone/background detections you still want to surface (optional — could just omit non-zone detections entirely, which is simpler and matches Frigate's zone-scoped semantics better).
8. **Terminal `leave` timing**: don't assume the terminal `leave` for a track is sent immediately when Frigate emits `type: "end"`. The real device appears to buffer/delay this message by several seconds in at least one capture (Section 11.6). If your bridge emits it immediately, that's a behavioral difference from the real device, though probably harmless for Protect's UI — flagged here for awareness, not necessarily something to replicate.

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
- **Confirmed dimensions (2026-07-21 capture):** the non-fullFoV `motionSnapshotHeight`/`motionSnapshotWidth` came through as `360×360` — square, and distinct from `motionSnapshotFullFoVHeight`/`motionSnapshotFullFoVWidth` at `360×640`. Treat `motionSnapshot` as a cropped/scaled square capture, not sharing the sub-stream's aspect ratio.

### 8.1 Cross-checked against a full EventSmartDetect/EventSmartMotion session (2026-07-21)

A session capture that interleaves `EventSmartMotion` with the `EventSmartDetect` loiter/zone traffic from Section 11 confirms and refines the above:

- **`eventId` independence confirmed with concrete numbers.** While the concurrent `EventSmartDetect` messages ran an `eventId` sequence in the 114,000s (`114455`...`114531`), the two `EventSmartMotion` messages in the same window used `eventId: 4301` then `4302` — a completely separate counter, exactly as this section already claimed.
- **`messageId` sharing confirmed across function names.** The two message types interleave on one incrementing sequence: `81497636 (EventSmartDetect/loiter), 81497637 (EventSmartDetect/zone), 81497638 (EventSmartMotion), 81497639 (loiter), 81497640 (zone), 81497641 (EventSmartMotion), ...`. Don't give `EventSmartMotion` its own `messageId` counter.
- **Correction — `levels` is not always present.** Both captured `EventSmartMotion` messages in this session (`edgeType: "stop"`) omit the `levels` field entirely — no empty dict, just absent. This contradicts treating `levels` as a guaranteed field; at minimum, don't assume it's present on every `stop` message. It may be that `levels` is specifically dropped on stale/residual stop messages (see next point) rather than genuinely absent from all stops — that distinction isn't resolved by this capture.
- **These two `stop` messages are themselves stale/residual**, following the same pattern as the residual `EventSmartDetect` `leave` pairs in Section 11.5: they appear at the very start of the connection, before any real detection, and their `clockBestMonotonic`/`clockBestWall` (`576202970` / `1784641699710`) sit roughly 5.5 minutes before the messages' own `clockWall` (`1784642021235` and `1784642024968`) — consistent with them reporting a motion window that actually opened well before this log window began, not a fresh event. No `start` or `pulse` `EventSmartMotion` messages for the real ~13-second person-detection track later in this same session (13:54:19–13:54:40) appear in this capture, so nothing new is confirmed here about `start`/`pulse` timing.
- **Cross-subsystem close-out order.** When a full track/connection close-out fires, the three sibling systems flush in a fixed order sharing one `clockMonotonic`/`clockWall` tick: loiter-subsystem `leave` → zone-subsystem `leave` → `EventSmartMotion` `stop`. All three used the same underlying clock values in this capture (`clockMonotonic: 576524495` / `576528228` for the two pairs), just packaged as three separate messages on the shared `messageId` counter. If your bridge emits a synthetic connection-teardown or long-idle flush, replicate this ordering rather than emitting the three message types in arbitrary order.
- **Logging artifact, not a protocol detail:** one `DEVICE_TO_BACKEND` line in the raw capture for this session is corrupted with unrelated camera-settings JSON (video stream config) spliced into the middle of an `EventSmartMotion` payload. This is almost certainly a buffer/overlap artifact in whatever tool produced the log, not a real wire-format behavior — don't model it.

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
     └─ Edge types: enter, moving, leave (and warmup on the loiter subsystem — see Section 11)
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

Doorbell devices follow the same session framing and message envelope structure as standard cameras, but use `MCUEventMessage` for ring events and report `features.doorbell: true` in the hello message. The `EventSmartDetect` capture used for Section 11 below happens to be from this same doorbell device (`Wasaga Doorbell`), so the loiter/`warmup` subsystem is confirmed present on doorbell hardware, not just standard cameras.

## 10. EventSmartAudio — Classified Audio Event Protocol

**Observed from a four-message device capture on 2026-08-12. The capture contains two complete `alrmSpeak` events. The nine classified-audio field names are independently confirmed by the UniFi Protect OpenAPI `smartDetectAudioTypes` enum; edge behavior for classes other than `alrmSpeak` still needs a class-specific wire capture.**

`EventSmartAudio` reports classified sounds independently of `EventSmartDetect` and `EventSmartMotion`. Each audio class is represented by its own payload field, whose value is an edge state. In the observed speech events the device emitted an `enter` message when speech began and a matching `leave` message when it ended; there were no pulse or moving messages between them.

### Message envelope and example

The normal device-to-backend envelope is used. `messageId` comes from the same connection-wide sequence as other function names, and the message remains fire-and-forget (`inResponseTo: 0`, `responseExpected: false`).

```json
{
  "from": "ubnt_avclient",
  "functionName": "EventSmartAudio",
  "inResponseTo": 0,
  "messageId": 81499615,
  "payload": {
    "alrmBabyCry": "none",
    "alrmBark": "none",
    "alrmBurglar": "none",
    "alrmCarHorn": "none",
    "alrmCmonx": "none",
    "alrmGlassBreak": "none",
    "alrmSiren": "none",
    "alrmSmoke": "none",
    "alrmSpeak": "enter",
    "clockMonotonic": 1582444314,
    "clockStream": 1582420877,
    "clockStreamRate": 1000,
    "clockWall": 1786570205352,
    "eventId": 1582458314,
    "leveldB": 0,
    "levels": 0,
    "loudNoise": "none",
    "soundLoss": "none"
  },
  "responseExpected": false,
  "timeStamp": "2026-08-12T21:30:19.383+00:00",
  "to": "UniFiVideo"
}
```

### Audio edge fields and advertised capabilities

Protect's camera feature model separates object and audio capabilities:

- `smartDetectTypes`: `person`, `vehicle`, `package`, `licensePlate`, `face`, `animal`
- `smartDetectAudioTypes`: `alrmSmoke`, `alrmCmonx`, `alrmSiren`, `alrmBabyCry`, `alrmSpeak`, `alrmBark`, `alrmBurglar`, `alrmCarHorn`, `alrmGlassBreak`

This confirms that the nine `alrm*` keys in `EventSmartAudio` are Protect's supported classified-audio type identifiers, not object types. A camera's `smartDetectAudioTypes` array advertises which classifications it supports; an `EventSmartAudio` payload reports state edges using the corresponding keys.

| Field | Sound class / meaning | OpenAPI audio type | Captured wire value |
|---|---|---|---|
| `alrmBabyCry` | Baby crying | Yes | `none` |
| `alrmBark` | Dog barking | Yes | `none` |
| `alrmBurglar` | Burglar/intrusion sound | Yes | `none` |
| `alrmCarHorn` | Vehicle horn | Yes | `none` |
| `alrmCmonx` | Carbon-monoxide alarm | Yes | `none` |
| `alrmGlassBreak` | Breaking glass | Yes | `none` |
| `alrmSiren` | Siren | Yes | `none` |
| `alrmSmoke` | Smoke alarm | Yes | `none` |
| `alrmSpeak` | Speech | Yes | `enter`, then `leave` |
| `loudNoise` | Generic loud-noise threshold | No | `none` |
| `soundLoss` | Loss of audio input | No | `none` |

Every audio edge/status field was present in every captured message. The active class alone changed to `enter` or `leave`; all other fields remained `none`. Preserve this complete shape rather than emitting only the active field. The likely state vocabulary is `enter` / `leave` / `none`, although only `alrmSpeak` was observed transitioning. `loudNoise` and `soundLoss` are wire-level status fields but are not members of the OpenAPI `smartDetectAudioTypes` capability enum.

The class names align with values advertised by `ubnt_avclient_hello.features.smartDetect` (for example `alrmSmoke`, `alrmCmonx`, `alrmBabyCry`, and `alrmSpeak`). `ChangeAudioEventsSettings` is the corresponding backend-to-device configuration request; the current bridge acknowledges it but does not yet store or apply its settings.

### Timing and identifiers

| Field | Observed behavior |
|---|---|
| `clockMonotonic` | Millisecond device-uptime time for the detected audio edge. |
| `clockStream` | Millisecond stream clock. It trailed `clockMonotonic` by 23,437–23,438 ms in all four messages. |
| `clockStreamRate` | Always `1000`. |
| `clockWall` | Epoch-ms wall time corresponding to `clockMonotonic`; `clockWall - clockMonotonic` stayed constant within 1 ms across the capture. |
| `eventId` | Equal to `clockMonotonic + 14,000` in all four messages. This looks time-derived rather than like the increment-by-one `EventSmartDetect` ID, so it should not be generated from the object-event counter. |
| envelope `timeStamp` | Approximately 14.03 seconds after `clockWall`, consistent with the `eventId` offset. This indicates that the device reports an audio edge after an approximately 14-second classification/confirmation delay while retaining the edge's original clocks. |

The two captured speech intervals were approximately 31.70 seconds and 14.05 seconds when measured from the payload clocks. Both `enter` and `leave` were delivered with the same roughly 14-second delay.

`leveldB` and `levels` were numeric scalar `0` in every message (not dictionaries like `EventSmartMotion.levels`). Their useful non-zero range and precise semantics are not established by this capture; emit numeric zero unless an audio source provides confirmed measurements.

### Lifecycle and implementation mapping

```text
Audio classifier reports speech start at time T
  └─ after ~14 s, EventSmartAudio
       └─ alrmSpeak: "enter"; every other audio edge: "none"

Audio classifier reports speech end at time U
  └─ after ~14 s, EventSmartAudio
       └─ alrmSpeak: "leave"; every other audio edge: "none"
```

Recommended bridge behavior:

1. Map each supported source audio label to its `EventSmartAudio` field and keep per-class active/inactive state.
2. Emit one complete payload on a state transition: `enter` on inactive → active and `leave` on active → inactive. Set every non-transitioning class field to `none`.
3. Use the shared connection-wide `messageId`, but maintain EventSmartAudio's time-derived `eventId` behavior separately from EventSmartDetect and EventSmartMotion counters.
4. Keep the stream/monotonic offset consistent with the active camera stream and retain the original detection-edge time in `clockWall`, even if classification introduces a delivery delay.
5. Do not synthesize periodic updates: none were observed between `enter` and `leave`.

## 11. Loiter-Zone Subsystem & the `warmup` edgeType

**Confirmed from a full session capture, camera `F4E2C60D4B4C` / doorbell "Wasaga Doorbell", session `F4E2C60D4B4C-1784642017261`, 2026-07-21T13:53:48–13:54:40. This is a genuine gap in the earlier revision of this document: `warmup` is a real, frequently-emitted `edgeType` value, and it belongs to a second tracking state machine the earlier revision didn't know existed.**

### 11.1 Two sibling state machines per tracked object

The capture shows that a single tracked person (`trackerID: 1254633`) drives **two separate, concurrently-running EventSmartDetect message streams**, distinguished by which zone-status field and edgeType vocabulary they use:

| | Zone subsystem (documented pre-2026-07-21) | Loiter subsystem (new) |
|---|---|---|
| Zone-status field | `zonesStatus` | `loiterZonesStatus` |
| Observed zone IDs | `1`, `2`, `3` | `5`, `7`, `9` |
| Per-zone shape | `{level, status}` | `{level, loiterTriggerTime, startStreakTime, status}` |
| edgeTypes seen | `enter`, `moving`, `leave` | `warmup`, `leave` |
| `descriptors[].zones` while active | populated, e.g. `[1]` | always `[]` in this capture |
| `objectTypes` while active | populated, e.g. `["person"]` | always `[]` in this capture |
| Snapshot filename prefix | `smartdetectsnap_zone_...` | `smartdetectsnap_loiter_...` |

Both subsystems emit messages for the *same* `trackerID`, on the *same* tick, as two back-to-back separate EventSmartDetect envelopes.

### 11.2 Shared counters, alternating messages

`messageId` and `eventId` are shared between the two subsystems — confirmed by the strictly alternating, monotonically increasing sequence observed for one track:

```
eventId 114459  warmup   (loiter)   trackerID 1254633, zones: []
eventId 114460  enter    (zone)     trackerID 1254633, zones: [1]
eventId 114461  warmup   (loiter)   trackerID 1254633, zones: []
eventId 114462  moving   (zone)     trackerID 1254633, zones: [1]
eventId 114463  warmup   (loiter)
eventId 114464  moving   (zone)
...
```

This pattern repeats for the whole ~13-second track: one loiter message, one zone message, per detector tick (roughly every 250-650ms), sharing a single incrementing `eventId`/`messageId` sequence. Do not implement separate counters per subsystem.

### 11.3 What `warmup` means, and what wasn't observed

`warmup` on the loiter subsystem's `loiterZonesStatus["<zone>"].status` field appears to represent an object that is being evaluated for loitering (i.e., is present but hasn't remained stationary/in-place long enough to trip whatever loiter-duration threshold Protect uses) . In this capture:

- The relevant loiter zone (`"5"`) shows `status: "warmup"` on every tick while the person is being tracked.
- The other configured loiter zones (`"7"`, `"9"`) stay `status: "none"` throughout.
- `loiterTriggerTime` and `startStreakTime` are `0` in every message — the loiter threshold was never actually reached in this capture.

**Not confirmed by this capture:** what edgeType/status Protect sends once a loiter threshold *is* crossed. By analogy with the zone subsystem's `enter`, there is presumably a distinct loiter-triggered edgeType or status value, but no such transition appears in this log. Treat any such value as unconfirmed until a capture shows it — don't infer a name for it.

### 11.4 Snapshot filename counters are per-subsystem

Confirmed distinct counters:
- Loiter subsystem: `smartdetectsnap_loiter_00000000_fullfov.jpg` (unchanged at `00000000` across the whole capture — no loiter-subsystem `leave` with a new count was observed in this window).
- Zone subsystem: `smartdetectsnap_zone_00000004_fullfov.jpg` at the start of the capture (a residual `leave` from a prior track, see 11.5), then `smartdetectsnap_zone_00000000_fullfov.jpg` on the track's actual terminal `leave` at the end.

**Open question:** the zone-subsystem counter going from `00000004` to `00000000` within one connection doesn't fit a simple always-incrementing model. Possible explanations include a counter reset on reconnect, two distinct snapshot-counter pools we haven't distinguished, or an artifact of this specific capture window. Implement the counter as monotonic-per-subsystem-per-connection as a reasonable default, but don't hard-code an assumption that it never resets — this needs another capture spanning a full connection lifecycle to pin down.

### 11.5 Leading residual `leave` pairs at connection start

At the very start of this capture, before any real detection, two pairs of `leave` messages appear (four messages total, `eventId 114455-114458`), each pair consisting of one loiter-styled `leave` (`loiterZonesStatus`, placeholder `loiter_00000000` filename) and one zone-styled `leave` (`zonesStatus`, `zone_00000004` filename), both with empty `descriptors: []`. These look like stale closeout messages left over from a track that ended before this log window began, re-sent or flushed at connection start — not a new pattern to reproduce, just residual state. Bridges emulating a fresh connection likely don't need to replicate this.

**Extends to `EventSmartMotion` too (confirmed in the same capture — see Section 8.1):** immediately following each of these two residual `leave` pairs, a residual `EventSmartMotion` `stop` message is also flushed, sharing the exact same `clockMonotonic`/`clockWall` as the pair it follows but carrying a `clockBestMonotonic`/`clockBestWall` roughly 5.5 minutes older. So the full residual close-out at connection start is three messages, not two: loiter `leave` → zone `leave` → motion `stop`, all describing state from before the log window began.

### 11.6 Terminal `leave` can be transmitted well after the underlying detection ended

The real device's last "live" tick for this track (`eventId 114530`, `warmup`) has `clockMonotonic: 576575285`, sent (`timeStamp`) at `13:54:32.300Z`. The terminal zone `leave` message (`eventId 114531`) has `clockMonotonic: 576575549` — only ~264ms later by the device's own clock — but its `timeStamp` shows it wasn't actually put on the wire until `13:54:40.803Z`, roughly **8.5 seconds later**. No other EventSmartDetect messages for this trackerID appear in the gap.

This means the device (or Frigate-side pipeline you're emulating) can hold the terminal `leave` for several seconds — plausibly for snapshot encoding/upload preparation — before transmitting it, with no intervening heartbeat. Bridge implementations should not assume Frigate's `type: "end"` should map to an immediately-sent `leave`; a short deliberate delay before flushing the terminal message is consistent with real-device behavior, though the exact cause (and whether it's required for Protect's UI to behave correctly) isn't established by this capture alone.

### 11.7 Correction to Section 4's `objectTypes` claim

The earlier revision of this document stated `objectTypes` "stays populated ... for the *entire* life of a track — enter through every moving message — and only clears to `[]` on the terminal leave." This capture shows that claim is only true for zone-subsystem messages. The sibling loiter-subsystem `warmup` message for the identical trackerID, emitted on the same tick, carries `objectTypes: []` throughout. If you implement both subsystems, `objectTypes` needs to be computed per-message based on which subsystem/edgeType is being emitted, not tracked as a single flag per object.

### 11.8 Implementation guidance

1. If your bridge doesn't emit `loiterZonesStatus`/`warmup` at all, Protect's zone-based enter/moving/leave detection is unaffected — those are the well-understood, already-implemented messages. Skipping the loiter subsystem simply means Protect's loiter-zone-specific UI/alerts won't activate for your emulated cameras.
2. If you do want to implement it: maintain a second static zone-ID map (loiter zones, distinct numbering from regular zones) per camera, and drive a simple per-track state that starts at `warmup` when the object is first seen and stays there for the observed lifetime of the track (since no loiter-triggered transition was captured to model). Emit it on the same per-tick cadence as your zone-subsystem `moving` messages, sharing the same `eventId`/`messageId` counter.
3. `objectTypes` on your loiter messages should be `[]`; on your zone messages, populated per the existing Section 6 guidance.
4. Don't rush the terminal `leave` message immediately off of Frigate's `type: "end"` if you want to match observed device timing — a delay in the low single-digit seconds before flushing is consistent with this capture, though not proven necessary.

## 12. Backend Decompile Cross-Checks (`service.js`)

**Source: decompiled Protect controller backend (`service.js`), not a device wire capture.** Everything in this section describes how the *backend* parses, indexes, or gates data — it confirms shape and existence with high confidence, but not the exact conditions under which a real camera populates these fields on the wire. Treat findings here as "confirmed via backend decompile, pending device capture" — a step below the device-log-derived sections above.

### 12.1 `attributes` field — schema now confirmed (supersedes Section 4 placeholder)

Section 4 previously listed `attributes` as "not exercised in either capture — safe to emit `null`". Two independent backend code paths now confirm a concrete shape:

- `onVehicleDescriptionDetected` (module 41286) reads `t.attributes.color.val`/`.confidence` and `t.attributes.vehicleType.val`/`.confidence` directly off a descriptor-like object.
- `extractAttributeLabelSet` (module 78899) independently confirms the same two literal keys (`color`, `vehicleType`) and the same `{val, confidence}` per-key shape, iterating a closed `AttributeKey` list containing only these two members as of this build.

**Revised shape:**

```json
"attributes": {
  "color": { "val": "red", "confidence": 87 },
  "vehicleType": { "val": "sedan", "confidence": 91 }
}
```

- Vehicle description (color/type) is filed under the `LICENSE_PLATE_WILDCARD`/`LICENSE_PLATE_DESCRIPTION` event keys on the backend — i.e. color/type search indexing shares a code path with license-plate recognition, not a separate "vehicle description" category. This suggests `lprEnhancedByAiKey` (a capability flag your bridge currently does not advertise) may gate both plate text and color/type, not just plate text. Unconfirmed — no code seen yet that checks this flag server-side.
- `matchedName` exists as a sibling field in the same internal event-value object but is left `void 0` by this handler — face/plate identity matches are evidently a separate concept from vehicle description attributes, populated by a different code path.
- **Not yet confirmed:** whether Frigate can even supply vehicle color/type without a custom-trained object classification model (`type: attribute`, Frigate 0.17+) — this isn't a Frigate+ built-in label. `descriptors.py` currently always emits `attributes: None`.

### 12.2 `lines[]` / loiter — real feature surfaces, not vestigial

The `LabelPrefix` enum (module 78899) lists `zone`, `line`, and `loiterZone` as equal-tier search facets, and `onVehicleDescriptionDetected`'s zone-state helper (`getZoneEnterState`) is called identically against `metadata.zonesStatus`, `metadata.linesStatus`, and `metadata.loiterStatus`. Section 4/5's descriptor-level `lines` field (currently documented "not exercised, safe to emit `[]`") and Section 11's loiter subsystem are both confirmed as first-class backend concepts, not edge cases. No change to current emission behavior recommended yet — this raises the priority of a future capture from a camera with line-crossing configured, to see `linesStatus` populated on the wire.

### 12.3 Capability-flag gating — confirmed mechanism, one concrete gap found

Automation trigger definitions (module 86901, `activityTriggers`) gate "Line Crossing" and "Loitering" UI triggers behind `requires: { [Provides.LINE_CROSSING]: ... }` / `[Provides.LOITERING]: ...}`. Cross-referencing against the real doorbell's captured `ubnt_avclient_hello` (`DOORBELL_IMPLEMENTATION.md`):

```
"smartDetect": ["person","vehicle","animal","lineCrossing","faceEnhancedByAiKey","lprEnhancedByAiKey","alrmSmoke","alrmCmonx","alrmBabyCry","alrmSpeak"]
```

`"lineCrossing"` is present as a literal string, consistent with `Provides.LINE_CROSSING` being derived from the device's advertised `smartDetect` array. **Action item:** the bridge's current `get_feature_flags()` advertises only `["person","vehicle","animal","package"]` — missing `lineCrossing`, `faceEnhancedByAiKey`, and `lprEnhancedByAiKey` at minimum. Without these, Protect's automation UI won't offer Line Crossing / Face / LPR triggers for the emulated camera regardless of what the descriptor payloads contain.

**Open question:** the same real capture has no obviously-corresponding string for loitering capability, despite Section 11 confirming the device actively emits loiter-subsystem messages. Either loitering is signaled via a different/undiscovered feature key, or is implicit rather than opt-in. Unresolved — needs the `Provides` enum module (imported as module 27320 in module 86901) to settle.

### 12.4 Audio alarm triggers — all nine confirmed as live, user-facing automations

Module 86901's `activityTriggers` maps all nine `alrm*` classes from Section 10 (`audioAlarmSpeak`, `audioAlarmBabyCry`, `audioAlarmBark`, `audioAlarmCo`, `audioAlarmSmoke`, `audioAlarmCarHorn`, `audioAlarmGlassBreak`, `audioAlarmSiren`, `audioAlarmBurglar`) to real Alarm Manager triggers, each gated on `scope_all_smart_cameras_with_microphone`. This confirms `EventSmartAudio`'s nine classes are actively wired into user-facing automations, not incidental telemetry — raises implementation priority for `EventSmartAudio` emission, which currently has zero code coverage (see prior review).

### 12.5 `idleSinceTimeMs`/`stationary` — confirmed downstream consumer

A dedicated `idleTrackerMiddleware` (module 35395) exists in the backend's event-producer pipeline and independently contributes `PERSON_IDLE_TIME`/`VEHICLE_IDLE_TIME` keys to the `EventKeys` enum, consumed by "Object Idle Time" automation triggers (duration + timeframe, converted to ms). This is a real, architecturally first-class consumer of the `idleSinceTimeMs`/`stationary` descriptor fields `descriptors.py` already derives from Frigate's `motionless_count`. No implementation gap — this is a confirmation, not an action item.

### 12.6 Open leads for future decompile passes

- `EventKeys` is federated across at least 7 modules (88245, 92185, 14036, 77541, 8590, 96067, 35395), not defined in one place — module 88245 is the likely "core" module (also source of `NVR_DEVICE`) and the best next pull for the full smart-detect/motion/audio key taxonomy.
- `dsRecognitionsEventProducerMiddleware` (32310) and `dsUpdatesEventProducerMiddleware` (43904) are the `ds`-prefixed (device-session) stages in the backend's event-producer pipeline — most likely candidates for seeing exactly how raw `EventSmartDetect`/`EventSmartAudio`/`MCUEventMessage` payloads get parsed into Protect's internal event model, as distinct from the five other middlewares handling NVR/schedule/webhook/external-API-originated events.
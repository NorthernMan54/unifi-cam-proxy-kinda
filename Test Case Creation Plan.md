# Test Case Creation Plan — unifi-cam-proxy-kinda

**Scope:** Build a black-box regression suite around the two external interfaces (UniFi Protect WebSocket, Frigate MQTT) before any architectural rewrite or manager deduplication work begins. Internals (`smart_detect_manager.py`, `smart_motion_manager.py`, `protocol_handlers.py`) are deliberately **not** unit-tested — they're the code expected to change or be replaced, and testing them now creates throwaway work and false confidence.

**Principle:** Characterization/golden-master testing. Pin down observed external behavior from real captures first; only add assertions beyond what's been observed once new captures confirm them. Don't invent expected behavior from assumption.

---

## 1. Harness structure

```
tests/
  fixtures/
    reconnect_existing_adoption/     # Backyard G5 Bullet capture (real UniFi hardware)
    background_polling/
    capability_decline/
    steady_state_events/
      smart_detect_start/
      smart_detect_leave/
      smart_motion_start/            # GAP — no capture yet
      smart_motion_stop/
      stationary_missed_new_event/   # Driveway capture (proxy-generated)
    fresh_adoption/                  # GAP — no capture yet
    zone_mapping/
  test_pairing.py
  test_background_polling.py
  test_capability_responses.py
  test_smart_detect_events.py
  test_smart_motion_events.py
  test_zone_mapping.py
  conftest.py
```

Each fixture directory holds paired input/output captures (raw JSON message dicts), sourced from real logs, not synthesized.

---

## 2. `conftest.py` — comparator (build first; everything else depends on it)

**Responsibilities:**
- Strip nondeterministic fields before any comparison: `messageId`, `clockMonotonic`, `clockWall`, `clockStream`, `clockBestMonotonic`, `clockBestWall`, `timeStamp`, `firstShownTimeMs`, `idleSinceTimeMs`, `eventId`, `trackerID` — unless the specific test is asserting tracker-ID allocation behavior (e.g. the missed-new-event recovery path), in which case that field stays in scope for that test only.
- Compare message **sequences** by `functionName` order where order matters, not just individual payload diffs in isolation.
- Treat "no reply sent" as a first-class, assertable outcome — several fixtures depend on silence being correct (see §3).

**Do not** diff raw bytes or do exact-match JSON comparison — it will fail constantly on legitimately-variable fields and get loosened into meaninglessness.

---

## 3. Test modules

### 3.1 `test_pairing.py` — reconnect to existing adoption
**Fixture available:** Backyard G5 Bullet capture (real hardware, reconnect — `adoptionCode` empty, `uptime` ~6.2 days, so this is reconnect not fresh adoption)

- Assert `ubnt_avclient_hello` → server reply → `ubnt_avclient_paramAgreement` handshake ordering.
- Assert the full config cascade is processed in the order it arrived: `StopService` → `NetworkStatus` → `ChangeDeviceSettings` (×2) → `ChangeOsdSettings` → `ChangeSoundLedSettings` (×2) → `ChangeBrightnessSettings` → `ChangeTalkbackSettings` → `ChangeVideoSettings` → `AudioAgentChangeTuning` → `ChangeIspSettings` → `ChangeVideoSettings` → `ChangeSmartMotionSettings` → `DisableLogging` → `ChangeDeviceSettings` → `SmartMotionTest` → `ChangeVideoSettings` (×2) → `ChangeClarityZones` → `ChangeAudioEventsSettings` → `ChangeSmartDetectSettings` → `SendWeatherUpdate` → `ChangeInterfaceSettings`.
- For every message with `responseExpected: true`, assert a reply is sent. For `responseExpected: false`, assert **no** reply is sent (silence is correct, not merely untested).

**Gap — stub as skipped, not omitted:**
```python
def test_fresh_adoption_handshake():
    pytest.skip("no fixture yet — needs factory-reset capture on real hardware")
```
Keeping this visible in test output (rather than simply absent) keeps the gap from being forgotten.

### 3.2 `test_capability_responses.py` — decline paths
**Fixture available:** Backyard capture

- `UpdateFaceDBRequest` → assert reply has `statusCode: 1` and a non-empty `payload.desc`. Assert the *shape*, not the literal string ("Not Implemented" may vary by device/firmware).
- `ChangeTalkbackSettings` under the captured conditions → assert `statusCode: 1` decline shape.
- **Design intent:** the real device's `statusCode: 1` decline is the target behavior for any capability the proxy doesn't currently implement — reuse this shape rather than inventing a different error convention.

### 3.3 `test_background_polling.py`
**Fixture available:** Backyard capture

- `GetIlluminance` request → assert a reply containing an `illuminance` field of the correct type (assert type/presence, not the specific value — it's a live sensor reading).
- Purpose: guard against silent connection degradation if a controller-initiated periodic poll stops being answered. Assert a reply is produced within the test, not just that a handler function exists.

### 3.4 `test_zone_mapping.py`
**Fixture available:** Driveway capture

- Input: `ChangeSmartDetectSettings` payload carrying UniFi zone IDs `1, 5, 6, 7` with coordinate arrays and named objectTypes.
- Assert the proxy's Frigate-zone-name-to-UniFi-zone-ID map matches the logged result: `{'roads': 1, 'parking': 5, 'plants': 6, 'front_door': 7}`.
- Assert zone coordinate arrays round-trip without corruption or reordering.
- **Priority note:** this is genuine logic (not echo-back), so it's the highest bug-catching value per line in the whole suite. Build this second, right after the comparator.

### 3.5 `test_smart_detect_events.py` — steady state, both edge shapes
**Fixtures available:** both captures

- **Ongoing/stationary event** (Driveway): Frigate `update` message arriving with no prior tracked `new` event → assert output `EventSmartDetect` has `edgeType: "none"`, a populated `descriptors` array with a synthesized `trackerID`, correct `zones`, correct `licensePlate` passthrough (`DEPM-782`). Assert the `WARNING` log line for missing tracker allocation still fires — this is documented, intentional recovery behavior and must survive any future rewrite.
- **Leave/closing event** (Backyard): assert output has `edgeType: "leave"`, **empty** `descriptors`, `objectTypes: []`, and empty/zeroed snapshot fields. Keep this as its own test, not parametrized together with the ongoing-event case — averaging the two into one parametrized test risks masking the difference in shape.
- **Multi-object single event** (Backyard, two-animal detection): assert `descriptors` contains 2 independent entries with distinct `trackerID`s in a single message. This is the direct regression guard for the already-identified architectural gap (`SmartDetectEventManager` built single-object-per-event; Frigate produces N objects per window).

### 3.6 `test_smart_motion_events.py`
**Fixture available:** Backyard capture — `edgeType: "stop"` only

- `EventSmartMotion` with `edgeType: "stop"` → assert heatmap and snapshot fields are populated, `eventType: "motion"`.

**Gap — stub as skipped:**
```python
def test_smart_motion_start_edge():
    pytest.skip("no fixture yet — no captured EventSmartMotion with edgeType != 'stop'")
```

---

## 4. Fixture-capture gaps (tracked, not blocking)

| Gap | Why it matters |
|---|---|
| Fresh adoption capture (populated `adoptionCode`, near-zero `uptime`) | Only have reconnect behavior; first-adoption sequence may differ |
| `EventSmartMotion` with `edgeType` other than `"stop"` | Motion "start" shape currently unverified |
| Doorbell ring event, end-to-end | README claims support; no capture exists yet |
| Second camera model/platform capture | Unknown whether `smartDetect` capability flags or config shapes vary by hardware |

---

## 5. Build order

1. `conftest.py` comparator — nothing else is runnable without it.
2. `test_zone_mapping.py` — pure logic, no I/O, fastest to green, highest value per line.
3. `test_smart_detect_events.py` — protects the known architectural gap directly.
4. `test_pairing.py`, `test_capability_responses.py`, `test_background_polling.py` — bulk of message-shape coverage.
5. `test_smart_motion_events.py` — smallest fixture set currently available.

**Gate:** this suite must pass against the *current* implementation before any changes land in `smart_detect_manager.py`, `smart_motion_manager.py`, or `protocol_handlers.py` — including the manager-deduplication work and any future strangler-fig cutover.
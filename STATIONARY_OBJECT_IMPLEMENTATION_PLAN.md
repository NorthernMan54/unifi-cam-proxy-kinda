# Stationary Object Detection Implementation Plan

## Overview

This plan implements support for stationary object detection in the Frigate → UniFi Protect bridge. When Frigate reports objects with `position_changes = 0`, the bridge should emit `edgeType: "none"` instead of `"enter"` or `"moving"`.

## Background

### Architecture Context

**Important:** The codebase has been refactored from the original documentation references. The event handling logic is now implemented as a mixin in `unifi/cams/frigate_lib/events.py`, not directly in `unifi/cams/frigate.py`.

**File Structure:**
- `unifi/cams/frigate.py` - Main FrigateCam class (`__init__`, state management)
- `unifi/cams/frigate_lib/events.py` - Event handling mixin (`_handle_new_event`, `_handle_update_event`, `_handle_end_event`)
- `unifi/cams/frigate_lib/descriptors.py` - Descriptor building (`build_descriptor_from_frigate_msg`)

### Current Behavior

The current implementation handles Frigate events but does not check `position_changes`:
- `type: "new"` → always calls `trigger_smart_detect_start()` with `edgeType` implicitly `"enter"`
- `type: "update"` → always calls `trigger_smart_detect_update()` with `edgeType` implicitly `"moving"`
- `type: "end"` → always calls `trigger_smart_detect_stop()` with `edgeType` implicitly `"leave"`

### Required Behavior (per protocol spec Section 7)

| Frigate Event | position_changes | UniFi Edge Type | Action |
|---------------|------------------|-----------------|--------|
| `new` | `0` | `"none"` | `trigger_smart_detect_stationary()` |
| `new` | `>0` | `"enter"` | `trigger_smart_detect_start()` |
| `update` | `before=0, after>0` | `"enter"` | `trigger_smart_detect_start()` (first movement) |
| `update` | `after=0` | `"none"` | `trigger_smart_detect_stationary()` |
| `update` | `after>0` | `"moving"` | `trigger_smart_detect_update()` |
| `end` | `0` | `"none"` | `trigger_smart_detect_stationary()` |
| `end` | `>0` | `"leave"` | `trigger_smart_detect_stop()` |

- Stationary objects always have `zones: []` (not in any configured zone)

## Implementation Plan

### Phase 1: Data Structures and State Tracking

#### 1.1 Add position_changes tracking to Frigate event state

**File:** `unifi/cams/frigate.py`

**Changes:**
- Add `position_changes_before` and `position_changes_after` fields to the `_active_frigate_events` tracking
- Track these values per `event_id` to detect transitions

```python
# In __init__ or state initialization
self._frigate_event_position_changes: dict[str, dict[str, int]] = {}
# Format: {event_id: {"before": int, "after": int}}
```

**Location:** Around line 200-250 in frigate.py where `_active_frigate_events` is defined

#### 1.2 Extend build_descriptor_from_frigate_msg to extract position_changes

**File:** `unifi/cams/frigate_lib/descriptors.py`

**Changes:**
- Extract `position_changes` from both `before` and `after` objects in the Frigate message
- Store these values for later comparison

```python
# In build_descriptor_from_frigate_msg
after = frigate_msg.get("after", {})
before = frigate_msg.get("before", {})

position_changes_before = before.get("position_changes", 0) if before else 0
position_changes_after = after.get("position_changes", 0)

# Add to descriptor for debugging/logging
descriptor = {
    "trackerID": tracker_id,
    "objectType": object_type.value,
    "coord": coord,
    "confidenceLevel": confidence_level,
    "stationary": stationary,
    "zones": zones,
    "positionChangesBefore": position_changes_before,  # NEW
    "positionChangesAfter": position_changes_after,   # NEW
    # ... existing fields ...
}
```

### Phase 2: Event Type Handling Logic

#### 2.1 Modify "new" event handling

**File:** `unifi/cams/frigate_lib/events.py`

**Location:** `_handle_new_event` function (around line 120-160)

**Current logic:**
```python
if self._motion_smart_event_id is None:
    self._motion_smart_event_id = await self.trigger_smart_detect_start(
        object_type, descriptor, frame_time_ms, zonesStatus=zones_status
    )
else:
    await self.trigger_smart_detect_update(
        object_type, descriptor, frame_time_ms, zonesStatus=zones_status, event_id=self._motion_smart_event_id
    )
```

**New logic:**
```python
position_changes = after.get("position_changes", 0)

if position_changes == 0:
    # Stationary object: emit edgeType="none"
    await self.trigger_smart_detect_stationary(
        custom_descriptor=custom_descriptor,
        event_timestamp=frame_time_ms,
        zonesStatus=zones_status,
        event_id=self._motion_smart_event_id,
    )
else:
    # Moving object: emit edgeType="enter"
    if self._motion_smart_event_id is None:
        self._motion_smart_event_id = await self.trigger_smart_detect_start(...)
    else:
        await self.trigger_smart_detect_update(...)
```

#### 2.2 Modify "update" event handling

**File:** `unifi/cams/frigate_lib/events.py`

**Location:** `_handle_update_event` function (around line 165-210)

**Current logic:**
```python
if self._motion_smart_event_id is None:
    # ... missed event handling ...
    return

# ... existing logic ...
```

**New logic:**
```python
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
    self._frigate_event_position_changes[event_id] = {
        "before": position_changes_before,
        "after": position_changes_after
    }

# Determine edgeType based on position_changes transition
if position_changes_after == 0:
    # Stationary object: emit edgeType="none"
    await self.trigger_smart_detect_stationary(...)
elif position_changes_before == 0 and position_changes_after > 0:
    # First movement: emit edgeType="enter"
    if self._motion_smart_event_id is None:
        self._motion_smart_event_id = await self.trigger_smart_detect_start(...)
    else:
        await self.trigger_smart_detect_update(...)
else:
    # Already moving: emit edgeType="moving" (via update)
    await self.trigger_smart_detect_update(...)
```

#### 2.3 Modify "end" event handling

**File:** `unifi/cams/frigate_lib/events.py`

**Location:** `_handle_end_event` function (around line 215-260)

**Current logic:**
```python
# ... existing logic ...
await self.trigger_smart_detect_stop(...)
```

**New logic:**
```python
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
    self._frigate_event_position_changes[event_id] = {
        "before": position_changes_before,
        "after": position_changes_after
    }

# Determine edgeType based on position_changes
if position_changes_after == 0:
    # Stationary object leaving: emit edgeType="none"
    await self.trigger_smart_detect_stationary(...)
else:
    # Moving object leaving: emit edgeType="leave"
    await self.trigger_smart_detect_stop(...)
```

### Phase 3: Cleanup and Memory Management

#### 3.1 Clean up position_changes tracking on event removal

**Files:** `unifi/cams/frigate_lib/events.py` and `unifi/cams/frigate.py`

**Changes:**
- When removing an event from `_active_frigate_events`, also remove from `_frigate_event_position_changes`
- Location: In `_forget_track` function in events.py

```python
# In _forget_track function
self._active_frigate_events.discard(event_id)
self._frigate_event_object_types.pop(event_id, None)
self._frigate_event_position_changes.pop(event_id, None)  # NEW
```

#### 3.2 Handle missed events gracefully

**Considerations:**
- If an "update" arrives before "new", the position_changes tracking may be incomplete
- Current implementation already handles this by checking if `event_id in _active_frigate_events`
- The position_changes tracking should be initialized on first event arrival for that event_id

### Phase 4: Testing and Validation

#### 4.1 Test Cases

1. **Stationary object (new)**
   - Frigate sends `type: "new"` with `position_changes: 0`
   - Bridge emits `edgeType: "none"`
   - `zones: []` in descriptor

2. **Stationary object (update)**
   - Frigate sends `type: "update"` with `position_changes: 0`
   - Bridge emits `edgeType: "none"`

3. **Stationary object (end)**
   - Frigate sends `type: "end"` with `position_changes: 0`
   - Bridge emits `edgeType: "none"`

4. **Moving object (first appearance)**
   - Frigate sends `type: "new"` with `position_changes > 0`
   - Bridge emits `edgeType: "enter"`

5. **Moving object (zone transition)**
   - Frigate sends `type: "update"` with `position_changes > 0`
   - Bridge emits `edgeType: "moving"` (via update)

6. **Moving object (leaving)**
   - Frigate sends `type: "end"` with `position_changes > 0`
   - Bridge emits `edgeType: "leave"`

7. **First movement detection**
   - Frigate sends `type: "update"` with `before.position_changes: 0` and `after.position_changes > 0`
   - Bridge emits `edgeType: "enter"` (first appearance)

#### 4.2 Logging and Debugging

Add enhanced logging for stationary detection:

```python
self.logger.debug(
    f"Frigate event {event_id}: position_changes before={position_changes_before}, "
    f"after={position_changes_after}, stationary={stationary} "
    f"→ edgeType={edge_type}"
)
```

### Phase 5: Documentation Updates

#### 5.1 Update eventsmartdetect_protocol_spec.md

- Add a section documenting the `position_changes` field behavior
- Clarify the mapping between Frigate's `position_changes` and UniFi's `edgeType`
- Include examples of stationary vs. moving object message sequences

#### 5.2 Update frigate.py docstrings

- Document the `_frigate_event_position_changes` tracking structure
- Document the edgeType determination logic in each event handler

## Implementation Checklist

- [ ] Add `_frigate_event_position_changes` state tracking in `__init__`
- [ ] Modify `build_descriptor_from_frigate_msg` to extract position_changes
- [ ] Update "new" event handler to check position_changes
- [ ] Update "update" event handler to check position_changes transitions
- [ ] Update "end" event handler to check position_changes
- [ ] Add cleanup logic for position_changes tracking
- [ ] Add enhanced logging for stationary detection
- [ ] Update documentation
- [ ] Run tests to validate behavior
- [ ] Manual testing with Frigate MQTT events

## Risk Assessment

### Low Risk
- Adding new state tracking (`_frigate_event_position_changes`)
- Extracting additional fields from Frigate messages
- Enhanced logging

### Medium Risk
- Modifying event type handling logic
- Edge case handling (missed events, out-of-order delivery)

### Mitigation Strategies
- Keep existing logic intact for non-stationary objects
- Add defensive checks for missing data
- Comprehensive logging for debugging
- Gradual rollout with monitoring

## Dependencies

- None (this is a pure behavior change, no new dependencies)

## Related Files

- `unifi/cams/frigate.py` - Main implementation
- `unifi/cams/base.py` - Smart detect methods (no changes needed)
- `unifi/cams/frigate_lib/events.py` - Event handling mixin
- `unifi/cams/frigate_lib/descriptors.py` - Descriptor building
- `eventsmartdetect_protocol_spec.md` - Protocol specification reference

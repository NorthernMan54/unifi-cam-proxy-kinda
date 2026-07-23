# Stationary Object Detection Implementation Plan

## Overview

This plan implements support for stationary object detection in the Frigate → UniFi Protect bridge. When Frigate reports objects with `position_changes = 0`, the bridge should emit `edgeType: "none"` instead of `"enter"` or `"moving"`.

## Background

### Current Behavior
The current implementation in `unifi/cams/frigate.py` handles Frigate events but does not check `position_changes`:
- `type: "new"` → always calls `trigger_smart_detect_start()` with `edgeType` implicitly `"enter"`
- `type: "update"` → always calls `trigger_smart_detect_update()` with `edgeType` implicitly `"moving"`
- `type: "end"` → always calls `trigger_smart_detect_stop()` with `edgeType` implicitly `"leave"`

### Required Behavior (per protocol spec Section 7)
- `type: "new"` with `after.position_changes = 0` → emit `edgeType: "none"` (stationary object)
- `type: "update"` with `after.position_changes = 0` → emit `edgeType: "none"` (stationary update)
- `type: "end"` with `after.position_changes = 0` → emit `edgeType: "none"` (stationary end)
- `type: "update"` with `before.position_changes = 0` and `after.position_changes > 0` → emit `edgeType: "enter"` (first movement)
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

**File:** `unifi/cams/frigate.py`

**Changes:**
- Extract `position_changes` from both `before` and `after` objects in the Frigate message
- Store these values for later comparison

```python
# In build_descriptor_from_frigate_msg (around line 335)
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

**File:** `unifi/cam-proxy-kinda/unifi/cams/frigate.py`

**Location:** Around line 1077 (`if event_type == "new":`)

**Current logic:**
```python
if event_type == "new":
    if event_id in self._active_frigate_events:
        return
    # ... buffer check ...
    
    tracker_id = self._allocate_tracker_id(event_id)
    custom_descriptor = self.build_descriptor_from_frigate_msg(...)
    
    if self._motion_smart_event_id is None:
        await self.trigger_smart_detect_start(...)
    else:
        await self.trigger_smart_detect_update(...)
```

**New logic:**
```python
if event_type == "new":
    if event_id in self._active_frigate_events:
        return
    
    # Extract position_changes for stationary detection
    after = frigate_msg.get("after", {})
    position_changes = after.get("position_changes", 0)
    
    tracker_id = self._allocate_tracker_id(event_id)
    custom_descriptor = self.build_descriptor_from_frigate_msg(...)
    
    # Track position_changes for this event
    self._frigate_event_position_changes[event_id] = {
        "before": 0,  # No "before" for "new" events
        "after": position_changes
    }
    
    # Determine edgeType based on position_changes
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
            self._motion_smart_event_id = await self.trigger_smart_detect_start(
                object_type,
                custom_descriptor,
                frame_time_ms,
                zonesStatus=zones_status,
            )
        else:
            await self.trigger_smart_detect_update(
                object_type,
                custom_descriptor,
                frame_time_ms,
                zonesStatus=zones_status,
                event_id=self._motion_smart_event_id,
            )
```

#### 2.2 Modify "update" event handling

**File:** `unifi/cams/frigate.py`

**Location:** Around line 1139 (`elif event_type == "update":`)

**Current logic:**
```python
elif event_type == "update":
    if event_id not in self._active_frigate_events:
        # ... missed event handling ...
        return
    
    tracker_id = self._get_tracker_id(event_id)
    custom_descriptor = self.build_descriptor_from_frigate_msg(...)
    frame_time_ms = ...
    
    zones_status = self._update_zone_status_for_track(...)
    
    if custom_descriptor.get("stationary", False):
        await self.trigger_smart_detect_stationary(...)
    else:
        await self.trigger_smart_detect_update(...)
```

**New logic:**
```python
elif event_type == "update":
    if event_id not in self._active_frigate_events:
        # ... missed event handling ...
        return
    
    # Extract position_changes for stationary detection
    after = frigate_msg.get("after", {})
    before = frigate_msg.get("before", {})
    position_changes_before = before.get("position_changes", 0) if before else 0
    position_changes_after = after.get("position_changes", 0)
    
    tracker_id = self._get_tracker_id(event_id)
    custom_descriptor = self.build_descriptor_from_frigate_msg(...)
    frame_time_ms = ...
    
    # Update position_changes tracking
    if event_id in self._frigate_event_position_changes:
        self._frigate_event_position_changes[event_id]["before"] = position_changes_before
        self._frigate_event_position_changes[event_id]["after"] = position_changes_after
    else:
        self._frigate_event_position_changes[event_id] = {
            "before": position_changes_before,
            "after": position_changes_after
        }
    
    zones_status = self._update_zone_status_for_track(...)
    
    # Determine edgeType based on position_changes transition
    if position_changes_after == 0:
        # Stationary object: emit edgeType="none"
        await self.trigger_smart_detect_stationary(
            custom_descriptor=custom_descriptor,
            event_timestamp=frame_time_ms,
            zonesStatus=zones_status,
            event_id=self._motion_smart_event_id,
        )
    elif position_changes_before == 0 and position_changes_after > 0:
        # First movement: emit edgeType="enter"
        if self._motion_smart_event_id is None:
            self._motion_smart_event_id = await self.trigger_smart_detect_start(
                object_type,
                custom_descriptor,
                frame_time_ms,
                zonesStatus=zones_status,
            )
        else:
            await self.trigger_smart_detect_update(
                object_type,
                custom_descriptor,
                frame_time_ms,
                zonesStatus=zones_status,
                event_id=self._motion_smart_event_id,
            )
    else:
        # Already moving: emit edgeType="moving" (via update)
        await self.trigger_smart_detect_update(
            object_type,
            custom_descriptor,
            frame_time_ms,
            zonesStatus=zones_status,
            event_id=self._motion_smart_event_id,
        )
```

#### 2.3 Modify "end" event handling

**File:** `unifi/cams/frigate.py`

**Location:** Around line 1224 (`elif event_type == "end":`)

**Current logic:**
```python
elif event_type == "end":
    if event_id not in self._active_frigate_events:
        # ... missed event handling ...
        return
    
    tracker_id = self._get_tracker_id(event_id)
    custom_descriptor = self.build_descriptor_from_frigate_msg(...)
    frame_time_ms = ...
    
    zones_status = self._update_zone_status_for_track(...)
    
    await self.trigger_smart_detect_stop(...)
```

**New logic:**
```python
elif event_type == "end":
    if event_id not in self._active_frigate_events:
        # ... missed event handling ...
        return
    
    # Extract position_changes for stationary detection
    after = frigate_msg.get("after", {})
    before = frigate_msg.get("before", {})
    position_changes_before = before.get("position_changes", 0) if before else 0
    position_changes_after = after.get("position_changes", 0)
    
    tracker_id = self._get_tracker_id(event_id)
    custom_descriptor = self.build_descriptor_from_frigate_msg(...)
    frame_time_ms = ...
    
    # Update position_changes tracking
    if event_id in self._frigate_event_position_changes:
        self._frigate_event_position_changes[event_id]["before"] = position_changes_before
        self._frigate_event_position_changes[event_id]["after"] = position_changes_after
    else:
        self._frigate_event_position_changes[event_id] = {
            "before": position_changes_before,
            "after": position_changes_after
        }
    
    zones_status = self._update_zone_status_for_track(...)
    
    # Determine edgeType based on position_changes
    if position_changes_after == 0:
        # Stationary object leaving: emit edgeType="none"
        await self.trigger_smart_detect_stationary(
            custom_descriptor=custom_descriptor,
            event_timestamp=frame_time_ms,
            zonesStatus=zones_status,
            event_id=self._motion_smart_event_id,
        )
    else:
        # Moving object leaving: emit edgeType="leave"
        await self.trigger_smart_detect_stop(
            object_type,
            custom_descriptor,
            event_timestamp=frame_time_ms,
            event_id=event_id,
            frame_time_ms=frame_time_ms,
            zonesStatus=zones_status,
        )
```

### Phase 3: Cleanup and Memory Management

#### 3.1 Clean up position_changes tracking on event removal

**File:** `unifi/cams/frigate.py`

**Changes:**
- When removing an event from `_active_frigate_events`, also remove from `_frigate_event_position_changes`
- Location: In the cleanup logic after `trigger_smart_detect_stop()` completes

```python
# After trigger_smart_detect_stop() in the "end" handler
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
- `eventsmartdetect_protocol_spec.md` - Protocol specification reference
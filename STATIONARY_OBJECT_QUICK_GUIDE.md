# Stationary Object Detection - Quick Implementation Guide

## Summary

Implement support for Frigate's `position_changes` field to correctly emit `edgeType: "none"` for stationary objects instead of always emitting `enter`/`moving`/`leave`.

## Key Changes

### 1. Add State Tracking (frigate.py, ~line 200)

```python
self._frigate_event_position_changes: dict[str, dict[str, int]] = {}
# Format: {event_id: {"before": int, "after": int}}
```

### 2. Extract position_changes in build_descriptor_from_frigate_msg (frigate.py, ~line 335)

```python
after = frigate_msg.get("after", {})
before = frigate_msg.get("before", {})

position_changes_before = before.get("position_changes", 0) if before else 0
position_changes_after = after.get("position_changes", 0)

# Store in descriptor for debugging
descriptor["positionChangesBefore"] = position_changes_before
descriptor["positionChangesAfter"] = position_changes_after
```

### 3. Update Event Handlers

**"new" handler (~line 1077):**
```python
position_changes = after.get("position_changes", 0)

if position_changes == 0:
    # Stationary: emit edgeType="none"
    await self.trigger_smart_detect_stationary(...)
else:
    # Moving: emit edgeType="enter"
    if self._motion_smart_event_id is None:
        self._motion_smart_event_id = await self.trigger_smart_detect_start(...)
    else:
        await self.trigger_smart_detect_update(...)
```

**"update" handler (~line 1139):**
```python
position_changes_before = before.get("position_changes", 0)
position_changes_after = after.get("position_changes", 0)

if position_changes_after == 0:
    # Stationary: emit edgeType="none"
    await self.trigger_smart_detect_stationary(...)
elif position_changes_before == 0 and position_changes_after > 0:
    # First movement: emit edgeType="enter"
    if self._motion_smart_event_id is None:
        self._motion_smart_event_id = await self.trigger_smart_detect_start(...)
    else:
        await self.trigger_smart_detect_update(...)
else:
    # Already moving: emit via update
    await self.trigger_smart_detect_update(...)
```

**"end" handler (~line 1224):**
```python
position_changes_before = before.get("position_changes", 0)
position_changes_after = after.get("position_changes", 0)

if position_changes_after == 0:
    # Stationary leaving: emit edgeType="none"
    await self.trigger_smart_detect_stationary(...)
else:
    # Moving leaving: emit edgeType="leave"
    await self.trigger_smart_detect_stop(...)
```

### 4. Cleanup on Event Removal

```python
# After removing from _active_frigate_events
self._frigate_event_position_changes.pop(event_id, None)
```

## Decision Tree

```
Frigate Event Received
        │
        ▼
position_changes == 0?
        │
        ├── YES → edgeType = "none" (stationary)
        │
        └── NO → Check transition
                │
                ├── before=0, after>0 → edgeType = "enter" (first movement)
                │
                ├── after>0 → edgeType = "moving" (via update)
                │
                └── after>0 on "end" → edgeType = "leave"
```

## Stationary Object Characteristics

- `edgeType: "none"`
- `objectTypes: []` (empty, not active detection)
- `zones: []` (not in any zone)
- `displayTimeoutMSec: ~300` (lower than moving objects)
- `zonesStatus`: all entries have `level: 0`, `status: "none"`

## Testing Checklist

- [ ] Stationary object (new) → edgeType="none"
- [ ] Stationary object (update) → edgeType="none"
- [ ] Stationary object (end) → edgeType="none"
- [ ] Moving object (new) → edgeType="enter"
- [ ] Moving object (update) → edgeType="moving"
- [ ] Moving object (end) → edgeType="leave"
- [ ] First movement from stationary (update) → edgeType="enter"
- [ ] Stationary bystander in leave message → included in descriptors only

## Files to Modify

1. `unifi/cams/frigate.py` - All changes
2. `eventsmartdetect_protocol_spec.md` - Update documentation (optional)

## No Changes Needed

- `unifi/cams/base.py` - `trigger_smart_detect_stationary()` already exists
- Smart detect dataclasses - Already support all required fields
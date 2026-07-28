# Architecture Review & Improvement Plan

## Executive Summary

This document reviews the UniFi Camera Proxy architecture and identifies opportunities for improvement. The codebase is well-structured with clear separation of concerns, but there are opportunities to enhance maintainability, reliability, and developer experience.

## Current Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Point                              │
│                      main.py                                     │
│  • CLI parsing • Camera factory • Core orchestration            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Core Layer                               │
│                      core.py                                     │
│  • WebSocket connection management                               │
│  • Reconnection logic (backoff)                                  │
│  • Session lifecycle                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Camera Abstraction                          │
│                    cams/base.py                                  │
│  • Protocol handlers (mixins)                                    │
│  • Snapshot management                                           │
│  • Event lifecycle delegation                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌─────────────────────┐                  ┌─────────────────────┐
│  Event Managers     │                  │  Camera Impl        │
│  smart_detect_      │◄────────────────►│  frigate.py         │
│  smart_motion_      │                  │  rtsp.py            │
│  managers           │                  │                     │
└─────────────────────┘                  └─────────────────────┘
```

### Key Design Patterns

1. **Mixin Composition**: `UnifiCamBase` composes protocol handlers via multiple inheritance
2. **Dependency Injection**: Managers receive callbacks (`send`, `gen_response`) rather than direct references
3. **Event Lifecycle Delegation**: Public API signatures preserved, complexity moved to managers
4. **Separation of Concerns**: Frigate-specific logic isolated in `frigate_lib/`

## Issues Identified

### 1. Potential Infinite Loops

#### Location: `smart_detect_manager.py` - `_cleanup_old_events()`

```python
def _cleanup_old_events(self) -> None:
    current_time = time.time()
    sixty_minutes_ago = current_time - 3600  # 60 minutes = 3600 seconds
    
    to_remove = [...]
    # Cleanup logic...
```

**Issue**: The cleanup runs on every `trigger_*` call. If a rapid fire of events occurs, this could create a loop where:
- Event starts → cleanup runs
- Cleanup deletes event → event no longer active
- Next trigger checks active events → may trigger re-initialization

**Fix**: Add a guard to prevent cleanup on every call:
```python
self._cleanup_cooldown_until = 0.0

def _cleanup_old_events(self) -> None:
    current_time = time.time()
    if current_time < self._cleanup_cooldown_until:
        return  # Rate-limited
    
    self._cleanup_cooldown_until = current_time + 1.0  # 1 second cooldown
    # ... rest of cleanup
```

#### Location: `smart_motion_manager.py` - `_cleanup_old_events()`

Same issue exists in the motion manager.

### 2. Task Cancellation Ordering

#### Location: `smart_motion_manager.py` - `trigger_stop()`

```python
if self._motion_pulse_task and not self._motion_pulse_task.done():
    self._motion_pulse_task.cancel()
    try:
        await self._motion_pulse_task
    except asyncio.CancelledError:
        pass

if self._analytics_start_task and not self._analytics_start_task.done():
    self._analytics_start_task.cancel()
    try:
        await self._analytics_start_task
    except asyncio.CancelledError:
        pass
```

**Issue**: The pulse task must be cancelled BEFORE the start task. If start task completes first, it may try to send a start event that should have been cancelled.

**Fix**: Use a context manager or ensure strict ordering with explicit logging.

### 3. Error Handling Gaps

#### Location: `core.py` - `connect()`

```python
except asyncio.exceptions.TimeoutError:
    self.logger.info(f"Connection to {self.args.host} timed out.")
    return True  # Retryable
```

**Issue**: Timeout errors are logged as INFO but should be WARNING or ERROR.

#### Location: `frigate.py` - `monitor_event_timeouts()`

```python
if time_since_update > self.event_timeout_seconds:
    expired_frigate_events.append(frigate_event_id)
```

**Issue**: No exception handling if a Frigate event crashes mid-flight.

### 4. Type Safety Issues

#### Location: `base.py`

```python
async def get_snapshot(self) -> Path:
    raise NotImplementedError("You need to write this!")
```

**Issue**: Return type is `Path` but implementation may return `None` in some cases.

**Fix**: Use `Optional[Path]` or ensure implementations always return a valid path.

### 5. Configuration Management

#### Current State

Configuration is passed via CLI arguments only.

**Issues**:
1. No environment variable support
2. No config file support
3. Hard to use in production without editing CLI flags

**Recommendation**: Add config file support (YAML/JSON) with CLI override capability.

### 6. Health Checks & Metrics

**Missing**:
1. No health check endpoint
2. No metrics (active events, connection state, etc.)
3. No graceful shutdown handling

**Recommendation**: Add:
- `/health` endpoint
- Prometheus metrics
- SIGTERM handler

### 7. Logging Consistency

**Inconsistencies**:
1. Some errors logged as INFO
2. No structured logging
3. No log rotation configuration

**Recommendation**: Standardize log levels and add log rotation.

### 8. Testing Coverage

**Current State**: Tests directory exists but appears empty.

**Recommendation**:
1. Unit tests for managers
2. Integration tests for protocol messages
3. Mock tests for external dependencies (MQTT, RTSP)

## Improvement Plan

### Priority 1: Critical Fixes

#### 1.1 Add Rate Limiting to Cleanup (2 hours)

**File**: `smart_detect_manager.py`, `smart_motion_manager.py`

**Change**: Add cooldown to prevent cleanup on every call.

#### 1.2 Fix Task Cancellation Ordering (1 hour)

**File**: `smart_motion_manager.py`

**Change**: Ensure pulse task is cancelled before start task.

#### 1.3 Improve Error Logging (1 hour)

**Files**: `core.py`, `frigate.py`

**Change**: Upgrade error log levels appropriately.

### Priority 2: Reliability

#### 2.1 Add Graceful Shutdown (2 hours)

**File**: `core.py`, `main.py`

**Change**: Handle SIGTERM/SIGINT, cancel all tasks, close connections cleanly.

#### 2.2 Add Health Check Endpoint (2 hours)

**File**: `rtsp.py` (extend existing HTTP API)

**Change**: Add `/health` endpoint returning connection state, active events, etc.

#### 2.3 Add Metrics (3 hours)

**File**: `core.py`, `smart_detect_manager.py`, `smart_motion_manager.py`

**Change**: Expose Prometheus metrics for:
- Active event counts
- Connection state
- Message send rates
- Error rates

### Priority 3: Developer Experience

#### 3.1 Add Configuration File Support (4 hours)

**Files**: `main.py`, `core.py`

**Change**: Support YAML/JSON config files with CLI override.

#### 3.2 Add Structured Logging (2 hours)

**File**: `core.py`, `base.py`

**Change**: Use structured logging format (JSON or similar).

#### 3.3 Add Type Hints (3 hours)

**Files**: All Python files

**Change**: Add complete type hints for all functions and classes.

### Priority 4: Documentation

#### 4.1 Expand AI_HINTS_GUIDE.md (4 hours)

See detailed suggestions below.

#### 4.2 Add Architecture Diagrams (2 hours)

**Files**: `docs/`

**Change**: Create visual diagrams of the architecture.

#### 4.3 Add Troubleshooting Guide (3 hours)

**File**: `docs/TROUBLESHOOTING.md`

**Change**: Common issues and solutions.

## AI_HINTS_GUIDE.md Improvements

### Current State

The guide is good but lacks some critical information.

### Recommended Additions

#### 1. Add Concurrency Patterns Section

```markdown
## Concurrency Patterns

### Background Tasks

All background tasks (pulse loops, timeout monitors) must be:
1. Created with `asyncio.create_task()`
2. Stored in instance variables for tracking
3. Cancelled with `task.cancel()` before awaiting
4. Awaited to ensure cleanup

Example:
```python
self._pulse_task = asyncio.create_task(self._pulse_loop(event_id))

# Later, when stopping:
if self._pulse_task and not self._pulse_task.done():
    self._pulse_task.cancel()
    try:
        await self._pulse_task
    except asyncio.CancelledError:
        pass
```

### Task Groups

For multiple concurrent tasks (FrigateCam):
```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(self.monitor_event_timeouts())
    tg.create_task(self.detect_fps_tracker.refresh_periodically())
    async for message in client.messages:
        tg.create_task(self.handle_detection_event(message))
```

**Warning**: TaskGroup will raise if any task fails. Catch exceptions and decide whether to continue.
```

#### 2. Add State Machine Diagram

```markdown
## Event State Machine

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Motion Start  │────▶│   Motion Pulse  │────▶│   Motion Stop   │
│  (linger delay) │     │     (heartbeat) │     │     (cleanup)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       │                        │                        │
       │                        │                        │
       ▼                        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Smart Detect    │◀────│ Smart Detect    │────▶│ Smart Detect    │
│   (enter/move)  │     │    (enter/move) │     │    (leave)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Key Points**:
- Motion window can contain multiple Smart Detect events
- Smart Detect events are independent but linked to motion window
- Cleanup only happens on motion stop
```

#### 3. Add Common Pitfalls Section

```markdown
## Common Pitfalls

### Pitfall 1: Event ID Collisions

**Problem**: Using `time.time() * 1000` for event IDs can cause collisions if events span seconds.

**Solution**: Use a monotonic counter:
```python
self._event_id_counter = 0
event_id = int(time.time() * 1000) + (self._event_id_counter % 1000)
self._event_id_counter += 1
```

### Pitfall 2: Snapshot Path Race Conditions

**Problem**: Snapshot paths are set in multiple places (smart detect stop, motion stop).

**Solution**: Use atomic operations or ensure only one path sets the final value.

### Pitfall 3: Zone Status Completeness

**Problem**: Forgetting to include all zones in `zonesStatus`.

**Solution**: Always iterate over ALL configured zones, not just active ones:
```python
zonesStatus = {}
for zone_id in configured_zone_ids:
    zonesStatus[str(zone_id)] = {
        "level": zone_level,
        "status": zone_status  # "enter", "moving", "leave", "none"
    }
```

### Pitfall 4: Clock Desynchronization

**Problem**: `clockMonotonic` and `clockWall` must stay in sync.

**Solution**: Update both on every message, never reset `clockMonotonic`.
```

#### 4. Add Debugging Guide

```markdown
## Debugging Guide

### Enable Verbose Logging

```bash
unifi-cam-proxy --verbose
```

### Inspect Active Events

```python
# In FrigateCam, add to MQTT loop:
self.logger.debug(f"Active smart events: {self._smart_events.summary()}")
self.logger.debug(f"Active motion events: {self._smart_motion.summary()}")
```

### Check Protocol Messages

```python
# Add to base.py process():
self.logger.debug(f"Sending: {json.dumps(msg)}")
```

### Monitor MQTT Topics

```bash
mosquitto_sub -h mqtt-host -t "frigate/#" -v
```

### Check Snapshot Files

```bash
ls -la /tmp/snapshots/
```

### Debug WebSocket Connection

```python
# In core.py, add:
try:
    ws = await websockets.connect(...)
    self.logger.info(f"WebSocket connected: {ws.remote_address}")
except Exception as e:
    self.logger.error(f"WebSocket connection failed: {e}")
```
```

## Implementation Checklist

- [ ] Priority 1.1: Add rate limiting to cleanup
- [ ] Priority 1.2: Fix task cancellation ordering
- [ ] Priority 1.3: Improve error logging
- [ ] Priority 2.1: Add graceful shutdown
- [ ] Priority 2.2: Add health check endpoint
- [ ] Priority 2.3: Add metrics
- [ ] Priority 3.1: Add configuration file support
- [ ] Priority 3.2: Add structured logging
- [ ] Priority 3.3: Add type hints
- [ ] Priority 4.1: Expand AI_HINTS_GUIDE.md
- [ ] Priority 4.2: Add architecture diagrams
- [ ] Priority 4.3: Add troubleshooting guide

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Cleanup loop causes event loss | High | Low | Add rate limiting |
| Task cancellation order causes protocol errors | Medium | Medium | Fix ordering with logging |
| Graceful shutdown incomplete | Medium | Low | Add health check |
| Config file adds complexity | Low | Low | CLI override capability |

## Conclusion

The architecture is solid with good separation of concerns. The main improvements needed are:
1. **Reliability**: Prevent loops, fix task cancellation
2. **Observability**: Add health checks, metrics, better logging
3. **Developer Experience**: Config files, better documentation

These improvements will make the project more maintainable and production-ready.
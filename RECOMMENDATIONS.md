# UniFi Camera Proxy - Code Quality & Architecture Recommendations

## Executive Summary

This document provides actionable recommendations to improve the code quality, architecture, and maintainability of the UniFi Camera Proxy project. The project is a sophisticated bridge between non-Ubiquiti cameras (particularly Frigate) and the UniFi Protect ecosystem, implementing complex protocol emulation, event lifecycle management, and real-time video streaming.

---

## 1. Code Quality Issues

### 1.1 Type Safety & Type Hints

**Issue:** Inconsistent use of type hints throughout the codebase.

**Current State:**
- Some modules use full type hints (`from typing import Any, Optional`)
- Other modules use loose typing or no hints at all
- Some type annotations are missing for critical function signatures

**Recommendations:**
```python
# Before (inconsistent)
async def process(self, msg: bytes) -> bool:
    m = json.loads(msg)
    fn = m["functionName"]

# After (consistent)
async def process(self, msg: bytes, logger: Optional[logging.Logger] = None) -> bool:
    """Process incoming message from UniFi Protect."""
    if logger is None:
        logger = self.logger
    m = json.loads(msg)
    fn = m.get("functionName")
    ...
```

**Action Items:**
- [ ] Add complete type hints to all public interfaces
- [ ] Use `from __future__ import annotations` for forward references
- [ ] Add type stubs for external dependencies with incomplete typing
- [ ] Enable `--strict` mode in pyright/mypy configuration

### 1.2 Error Handling

**Issue:** Inconsistent error handling patterns.

**Current State:**
- Some functions use bare `except Exception`
- Error messages vary in format and detail level
- Some critical operations lack proper error handling

**Recommendations:**
```python
# Before
try:
    result = await fetch_recordings_motion_level(...)
except Exception:
    return None

# After
from unifi.core import RetryableError

async def fetch_recordings_motion_level(...) -> Optional[int]:
    """Fetch motion level from Frigate recordings API."""
    try:
        # ... implementation
        return motion_level
    except aiohttp.ClientError as e:
        self.logger.warning(f"Client error fetching motion level: {e}")
        return None
    except asyncio.TimeoutError as e:
        self.logger.warning(f"Timeout fetching motion level: {e}")
        return None
    except ValueError as e:
        self.logger.error(f"Invalid motion level data: {e}")
        return None
```

**Action Items:**
- [ ] Replace bare `except Exception` with specific exception types
- [ ] Add structured error logging with context
- [ ] Implement retry logic for transient failures
- [ ] Add error recovery paths for critical operations

### 1.3 Logging Consistency

**Issue:** Inconsistent logging patterns and severity levels.

**Current State:**
- Mixed use of `logger.info`, `logger.debug`, `logger.warning`
- Some operations log errors at info level
- No consistent format for log messages

**Recommendations:**
```python
# Before
self.logger.info(f"Processing message")
self.logger.info(f"Could not probe source: {e}")

# After
from logging import Logger

def log_operation(self, logger: Logger, operation: str, success: bool, details: dict) -> None:
    """Log operation results consistently."""
    if success:
        logger.info(f"Operation {operation} succeeded")
    else:
        logger.warning(f"Operation {operation} failed: {details}")

# Usage
self.log_operation(
    self.logger,
    "probe_video_resolution",
    False,
    {"error": str(e), "stream": "video1"}
)
```

**Action Items:**
- [ ] Define standard log message formats
- [ ] Add structured logging with JSON format option
- [ ] Implement log level configuration per module
- [ ] Add correlation IDs for tracing across components

---

## 2. Architecture Improvements

### 2.1 Dependency Injection

**Issue:** Tight coupling between components.

**Current State:**
- Components directly instantiate dependencies
- Limited testability without mocking
- Hard to swap implementations

**Recommendations:**
```python
# Before (tight coupling)
class FrigateCam(RTSPCam):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.detect_fps_tracker = DetectFpsTracker(...)

# After (dependency injection)
class FrigateCam(RTSPCam):
    def __init__(
        self,
        args: Args,
        logger: Logger,
        detect_fps_tracker: Optional[DetectFpsTracker] = None,
        snapshot_client: Optional[FrigateSnapshotClient] = None,
    ):
        super().__init__(args, logger)
        self.detect_fps_tracker = detect_fps_tracker or DetectFpsTracker(...)
        self.snapshot_client = snapshot_client or FrigateSnapshotClient(...)
```

**Action Items:**
- [ ] Create dependency injection container
- [ ] Define interfaces for all external dependencies
- [ ] Add factory pattern for component creation
- [ ] Implement mock dependencies for testing

### 2.2 Event Loop Management

**Issue:** Complex asyncio task management with potential race conditions.

**Current State:**
- Multiple background tasks created inline
- Task cancellation ordering is manual
- No centralized task supervision

**Recommendations:**
```python
class TaskManager:
    """Centralized task supervision for async operations."""
    
    def __init__(self):
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
    
    async def create(self, coro: Coroutine) -> asyncio.Task:
        """Create and track a new async task."""
        task = asyncio.create_task(coro)
        async with self._lock:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return task
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all tracked tasks."""
        async with self._lock:
            tasks = list(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
```

**Action Items:**
- [ ] Create centralized TaskManager class
- [ ] Replace inline task creation with TaskManager
- [ ] Add task health monitoring
- [ ] Implement graceful shutdown sequence

### 2.3 Configuration Management

**Issue:** Configuration scattered across command-line arguments and defaults.

**Current State:**
- Command-line arguments define most configuration
- Defaults hardcoded in multiple places
- No validation of configuration

**Recommendations:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CameraConfig:
    """Complete camera configuration."""
    host: str
    token: str
    mac: str
    model: str
    fw_version: str
    # ... other fields
    
    def validate(self) -> list[str]:
        """Validate configuration and return error messages."""
        errors = []
        if not self.host:
            errors.append("Host is required")
        if not self.token:
            errors.append("Token is required")
        return errors

class ConfigLoader:
    """Load and validate configuration from multiple sources."""
    
    @staticmethod
    def load(args, env: dict, defaults: dict) -> CameraConfig:
        """Load config with priority: args > env > defaults."""
        config = CameraConfig(**defaults)
        
        # Apply environment variables
        for key, value in env.items():
            if key.startswith("UNIFI_"):
                setattr(config, key.replace("UNIFI_", "").lower(), value)
        
        # Apply command-line arguments (highest priority)
        for key, value in vars(args).items():
            if value is not None:
                setattr(config, key, value)
        
        # Validate
        errors = config.validate()
        if errors:
            raise ConfigurationError(errors)
        
        return config
```

**Action Items:**
- [ ] Create unified configuration data class
- [ ] Implement configuration validation
- [ ] Support multiple config sources (CLI, env, file)
- [ ] Add configuration schema documentation

---

## 3. Testing Strategy

### 3.1 Test Coverage

**Issue:** Limited test coverage and test organization.

**Current State:**
- Minimal test files exist
- Integration tests dominate
- No unit test infrastructure

**Recommendations:**
```python
# tests/unit/test_smart_detect_manager.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from unifi.cams.managers.smart_detect_manager import SmartDetectEventManager

class TestSmartDetectEventManager:
    @pytest.fixture
    def event_manager(self, mock_send, mock_gen_response, mock_get_uptime):
        return SmartDetectEventManager(
            logger=MagicMock(),
            send=mock_send,
            gen_response=mock_gen_response,
            get_uptime=mock_get_uptime,
            detected_resolutions={"video1": (1920, 1080)}
        )
    
    async def test_trigger_start_creates_event(self, event_manager):
        event_id = await event_manager.trigger_start(SmartDetectObjectType.PERSON)
        assert event_id > 0
        assert event_id in event_manager.active_events
        assert event_manager.active_events[event_id]["object_type"] == SmartDetectObjectType.PERSON
    
    async def test_trigger_stop_cleans_up(self, event_manager):
        event_id = await event_manager.trigger_start(SmartDetectObjectType.PERSON)
        await event_manager.trigger_stop(SmartDetectObjectType.PERSON, event_id=event_id)
        assert event_id not in event_manager.active_events
```

**Action Items:**
- [ ] Create comprehensive unit test suite (target: 80% coverage)
- [ ] Add pytest fixtures for common test setup
- [ ] Implement mock WebSocket for protocol testing
- [ ] Add property-based testing for edge cases
- [ ] Create integration test suite for end-to-end flows

### 3.2 Test Infrastructure

**Issue:** No automated testing pipeline.

**Recommendations:**
- [ ] Add GitHub Actions CI/CD pipeline
- [ ] Run linters and type checkers in CI
- [ ] Add coverage reporting
- [ ] Implement snapshot testing for protocol messages
- [ ] Add performance regression tests

---

## 4. Documentation

### 4.1 Code Documentation

**Issue:** Insufficient inline documentation.

**Current State:**
- Some functions have docstrings
- Complex logic lacks explanation
- No architecture documentation

**Recommendations:**
```python
class SmartDetectEventManager:
    """
    Manages the lifecycle of EventSmartDetect messages for object detection.
    
    This manager handles:
    - Starting new detection events (edgeType=enter)
    - Updating active events with new descriptors
    - Stationary object tracking (edgeType=none)
    - Ending events (edgeType=leave)
    
    Key Invariants:
    - Event IDs are monotonically increasing epoch-based values
    - Active events are cleaned up after 60 minutes of inactivity
    - Snapshot paths are lazily fetched and cached per event
    
    Thread Safety:
    - All public methods are asyncio-safe
    - Internal state is protected by the event loop
    """
```

**Action Items:**
- [ ] Add docstrings to all public classes and methods
- [ ] Create architecture decision records (ADRs)
- [ ] Document protocol message formats
- [ ] Add troubleshooting guide

### 4.2 User Documentation

**Recommendations:**
- [ ] Add configuration examples for common setups
- [ ] Create troubleshooting guide for common issues
- [ ] Document known limitations
- [ ] Add performance tuning guide

---

## 5. Security Considerations

### 5.1 Credential Handling

**Issue:** Credentials exposed in logs and configuration.

**Recommendations:**
```python
# Before
self.logger.info(f"Using token: {self.token}")

# After
def _log_redacted_token(self, token: str) -> str:
    """Log token with masked credentials."""
    if token and len(token) > 8:
        return token[:4] + "***" + token[-4:]
    return "***"

self.logger.info(f"Using token: {_log_redacted_token(self.token)}")
```

**Action Items:**
- [ ] Never log full tokens or passwords
- [ ] Add credential masking in error messages
- [ ] Use secure storage for sensitive data
- [ ] Implement credential rotation support

### 5.2 Input Validation

**Recommendations:**
- [ ] Validate all external inputs (MQTT messages, RTSP URLs)
- [ ] Add rate limiting for API calls
- [ ] Implement request signing validation
- [ ] Add DoS protection for snapshot endpoints

---

## 6. Performance Optimizations

### 6.1 Resource Management

**Issue:** Potential resource leaks in long-running processes.

**Recommendations:**
```python
class ResourceTracker:
    """Track and manage system resources."""
    
    def __init__(self):
        self.open_files = 0
        self.processes = 0
    
    def acquire_file(self, path: Path) -> None:
        self.open_files += 1
        self.logger.debug(f"Opened file: {path} (total: {self.open_files})")
    
    def release_file(self, path: Path) -> None:
        self.open_files -= 1
        self.logger.debug(f"Released file: {path} (total: {self.open_files})")
    
    def check_limits(self, max_files: int = 1000) -> None:
        if self.open_files > max_files:
            self.logger.error(f"File descriptor limit exceeded: {self.open_files}")
```

**Action Items:**
- [ ] Implement resource monitoring
- [ ] Add periodic cleanup of temporary files
- [ ] Monitor memory usage and implement garbage collection
- [ ] Add connection pool for HTTP requests

### 6.2 Async Optimization

**Recommendations:**
- [ ] Batch MQTT message processing
- [ ] Use asyncio.gather for parallel operations
- [ ] Implement request queuing for rate-limited APIs
- [ ] Add connection pooling for network operations

---

## 7. Dependency Management

### 7.1 Dependency Updates

**Current State:**
```toml
# pyproject.toml
[project.optional-dependencies]
test = [
    "black",
    "isort",
    "flake8",
    "flake8-bugbear",
    "pre-commit",
    "pyre-check",
    "pytest",
    "wheel"
]
```

**Recommendations:**
- [ ] Add security scanning (safety, bandit)
- [ ] Implement automated dependency updates
- [ ] Add lock file generation
- [ ] Document dependency rationale

---

## 8. Project Structure

### 8.1 Module Organization

**Current Structure:**
```
unifi/
├── main.py
├── core.py
├── cams/
│   ├── base.py
│   ├── frigate.py
│   ├── rtsp.py
│   └── handlers/
│       ├── protocol_handlers.py
│       └── snapshot_handlers.py
```

**Recommended Structure:**
```
unifi/
├── __init__.py
├── main.py
├── config/
│   ├── __init__.py
│   ├── loader.py
│   └── schemas/
├── core/
│   ├── __init__.py
│   ├── retry.py
│   ├── ssl.py
│   └── task_manager.py
├── cams/
│   ├── __init__.py
│   ├── base.py
│   ├── rtsp.py
│   ├── frigate.py
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── protocol_handlers.py
│   │   ├── snapshot_handlers.py
│   │   └── video_stream_handlers.py
│   └── managers/
│       ├── __init__.py
│       ├── smart_detect_manager.py
│       └── motion_analytics_manager.py
├── frigate_lib/
│   ├── __init__.py
│   ├── descriptors.py
│   ├── events.py
│   ├── motion.py
│   ├── snapshots.py
│   ├── tracker_ids.py
│   └── zones.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── architecture.md
│   ├── protocol-spec.md
│   └── troubleshooting.md
```
Action Items:

[ ] Refactor into package structure
[ ] Add __init__.py exports
[ ] Move shared utilities to core/
[ ] Document module responsibilities

---

## 9. Implementation Priority
Phase 1: Critical (Weeks 1-2)
[ ] Add complete type hints
[ ] Fix error handling patterns
[ ] Add credential masking
[ ] Create test infrastructure
Phase 2: Important (Weeks 3-4)
[ ] Implement dependency injection
[ ] Refactor configuration management
[ ] Add comprehensive documentation
[ ] Implement CI/CD pipeline
Phase 3: Enhancements (Weeks 5-8)
[ ] Add performance monitoring
[ ] Implement resource tracking
[ ] Optimize async operations
[ ] Add advanced testing

--- 

## 10. Monitoring & Observability
Recommendations:
[ ] Add structured logging with JSON format
[ ] Implement metrics collection (Prometheus-compatible)
[ ] Add health check endpoints
[ ] Create distributed tracing support
[ ] Implement alerting for critical failures

Conclusion
The UniFi Camera Proxy is a sophisticated project with complex requirements. Implementing these recommendations will significantly improve code quality, maintain
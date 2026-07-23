"""
Type definitions for UniFi camera protocol messages.

These are simple dict-based types that don't create circular dependencies
with other modules in the cams package.
"""
from typing import Any

AVClientRequest = dict[str, Any]
AVClientResponse = dict[str, Any]
__all__ = ["AVClientRequest", "AVClientResponse"]
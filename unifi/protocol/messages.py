"""Shared protocol message types.

The wire format is still dynamic JSON. Keeping its aliases here prevents
camera implementations and event managers from depending on one another.
"""

from typing import Any

AVClientRequest = dict[str, Any]
AVClientResponse = dict[str, Any]

__all__ = ["AVClientRequest", "AVClientResponse"]

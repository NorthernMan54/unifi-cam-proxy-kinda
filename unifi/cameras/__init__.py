"""Public camera implementations.

Implementations currently live under :mod:`unifi.cams`; these exports provide
the stable namespace for incremental migration without breaking deployments.
"""

from unifi.cams import FrigateCam, RTSPCam
from unifi.cams.base import UnifiCamBase

__all__ = ["FrigateCam", "RTSPCam", "UnifiCamBase"]

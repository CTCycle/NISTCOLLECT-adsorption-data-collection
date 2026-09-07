"""Small path helpers for Core-owned runtime locations.

Configuration is loaded by the application entry point. This module therefore
contains no implicit configuration lookup and no environment-variable based
resource discovery.
"""

from __future__ import annotations

from pathlib import Path

from adsmod_common.config import AdsmodConfig


###############################################################################
def resolve_storage_root(config: AdsmodConfig) -> Path:
    """Return the configured, absolute storage root without changing it."""

    return Path(config.storage.root).expanduser().resolve()


__all__ = ["resolve_storage_root"]

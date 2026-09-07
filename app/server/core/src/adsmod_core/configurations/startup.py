from __future__ import annotations

from adsmod_common.config import AdsmodConfig

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


###############################################################################
def public_host_mode_enabled(config: AdsmodConfig) -> bool:
    """Return whether the configured host is outside the local-only boundary."""

    host = config.runtime.host.strip().lower()
    return bool(host and host not in LOOPBACK_HOSTS)


__all__ = ["LOOPBACK_HOSTS", "public_host_mode_enabled"]

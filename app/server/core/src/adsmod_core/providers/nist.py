from __future__ import annotations

from datetime import datetime, timezone

from adsmod_core.providers.public_data import (
    ProviderCapability,
    ProviderHealth,
    PublicDataProvider,
)
from adsmod_core.services.data.nist_service import NISTDataService


###############################################################################
class NISTPublicDataProvider(PublicDataProvider):
    key = "nist"
    name = "NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials"
    description = (
        "Public adsorption experiments plus guest-species and host-material records "
        "from the ADSMOD NIST integration."
    )
    homepage_url = "https://adsorption.nist.gov/"
    license_name = None
    license_url = None
    terms_url = "https://adsorption.nist.gov/"
    capabilities = (
        ProviderCapability.ADSORPTION,
        ProviderCapability.MATERIALS,
        ProviderCapability.CHEMICALS,
        ProviderCapability.REFERENCES,
    )

    # -------------------------------------------------------------------------
    def __init__(self, service: NISTDataService) -> None:
        self.service = service

    # -------------------------------------------------------------------------
    async def health(self) -> ProviderHealth:
        checked_at = datetime.now(timezone.utc)
        try:
            status = await self.service.ping_experiments_server()
        except Exception as exc:  # noqa: BLE001
            return ProviderHealth(
                status="unavailable",
                detail=f"NIST adsorption endpoint check failed: {exc}",
                checked_at=checked_at,
            )
        if bool(status.get("server_ok", False)):
            return ProviderHealth(status="available", detail=None, checked_at=checked_at)
        return ProviderHealth(
            status="unavailable",
            detail="NIST adsorption endpoint did not respond successfully.",
            checked_at=checked_at,
        )


__all__ = ["NISTPublicDataProvider"]

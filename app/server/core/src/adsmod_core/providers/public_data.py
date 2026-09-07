from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import httpx


###############################################################################
class ProviderCapability(StrEnum):
    ADSORPTION = "adsorption"
    MATERIALS = "materials"
    CHEMICALS = "chemicals"
    STRUCTURES = "structures"
    REFERENCES = "references"


###############################################################################
class ProviderError(RuntimeError):
    """Base class for public-data provider failures."""


###############################################################################
class ProviderNotFoundError(ProviderError):
    """Raised when a provider cannot resolve the requested record."""


###############################################################################
class ProviderUnavailableError(ProviderError):
    """Raised for transient or invalid remote-service responses."""


###############################################################################
class ProviderRateLimitError(ProviderUnavailableError):
    """Raised when a provider still throttles ADSMOD after retries."""


###############################################################################
@dataclass(frozen=True)
class ProviderHealth:
    status: str
    detail: str | None
    checked_at: datetime


###############################################################################
class PublicDataProvider(ABC):
    key: str
    name: str
    description: str
    homepage_url: str
    license_name: str | None
    license_url: str | None
    terms_url: str | None
    capabilities: tuple[ProviderCapability, ...]

    # -------------------------------------------------------------------------
    def source_definition(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "homepage_url": self.homepage_url,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "terms_url": self.terms_url,
            "capabilities": [capability.value for capability in self.capabilities],
        }

    # -------------------------------------------------------------------------
    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError


###############################################################################
class RetryingHttpProvider(PublicDataProvider):
    """Shared timeout, retry, and bounded-concurrency policy for public HTTP sources."""

    retry_statuses = frozenset({429, 500, 502, 503, 504})

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        parallel_requests: int,
        request_timeout_seconds: float,
        retry_attempts: int,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max(1, parallel_requests))
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self.max_attempts = max(1, int(retry_attempts))

    # -------------------------------------------------------------------------
    async def _before_attempt(self) -> None:
        """Provider-specific pacing hook called before every remote attempt."""

    # -------------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        merged_headers = {
            "Accept": "application/json",
            "User-Agent": "ADSMOD/3 public-data client",
            **(headers or {}),
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            await self._before_attempt()
            try:
                async with self._semaphore:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(self.request_timeout_seconds),
                        follow_redirects=True,
                    ) as client:
                        response = await client.request(
                            method,
                            url,
                            params=params,
                            headers=merged_headers,
                        )
                if response.status_code == 404:
                    raise ProviderNotFoundError(f"{self.name} record was not found.")
                if response.status_code in self.retry_statuses:
                    if attempt < self.max_attempts - 1:
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    if response.status_code == 429:
                        raise ProviderRateLimitError(
                            f"{self.name} rate limit was reached; try again later."
                        )
                    raise ProviderUnavailableError(
                        f"{self.name} returned HTTP {response.status_code}."
                    )
                response.raise_for_status()
                return response
            except ProviderError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.max_attempts - 1:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
            except httpx.HTTPStatusError as exc:
                raise ProviderUnavailableError(
                    f"{self.name} returned HTTP {exc.response.status_code}."
                ) from exc
        raise ProviderUnavailableError(
            f"{self.name} could not be reached after {self.max_attempts} attempts."
        ) from last_error

    # -------------------------------------------------------------------------
    async def health(self) -> ProviderHealth:
        try:
            await self._health_request()
        except ProviderError as exc:
            return ProviderHealth(
                status="unavailable",
                detail=str(exc),
                checked_at=datetime.now(timezone.utc),
            )
        return ProviderHealth(
            status="available",
            detail=None,
            checked_at=datetime.now(timezone.utc),
        )

    # -------------------------------------------------------------------------
    @abstractmethod
    async def _health_request(self) -> None:
        raise NotImplementedError


__all__ = [
    "ProviderCapability",
    "ProviderError",
    "ProviderHealth",
    "ProviderNotFoundError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
    "PublicDataProvider",
    "RetryingHttpProvider",
]

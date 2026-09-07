from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import RedirectResponse

from adsmod_common.health import HealthResponse
from adsmod_common.version import __version__

health_router = APIRouter()


###############################################################################
@health_router.get("/health/live", response_model=HealthResponse, tags=["health"])
def liveness() -> HealthResponse:
    return HealthResponse(service="backend", version=__version__, state="ready")


###############################################################################
@health_router.get("/health/ready", response_model=HealthResponse, tags=["health"])
def readiness(request: Request) -> HealthResponse:
    ready = bool(getattr(request.app.state, "ready", False))
    return HealthResponse(
        service="backend",
        version=__version__,
        state="ready" if ready else "not-ready",
        details={} if ready else {"reason": "database initialization is pending"},
    )


###############################################################################
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


###############################################################################
def register_root_routes(app: FastAPI) -> None:
    """Keep the non-API root useful without introducing another API surface."""

    app.add_api_route("/", redirect_to_docs, methods=["GET"], include_in_schema=False)


__all__ = ["health_router", "liveness", "readiness", "register_root_routes"]

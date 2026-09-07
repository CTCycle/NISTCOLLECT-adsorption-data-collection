from __future__ import annotations

from fastapi import FastAPI

from adsmod_core.http.datasets import create_dataset_router
from adsmod_core.http.fitting import create_fitting_router
from adsmod_core.http.nist import create_nist_router
from adsmod_core.http.public_data import create_public_data_router
from adsmod_core.services.container import CoreServiceContainer

###############################################################################
def register_core_routes(
    app: FastAPI,
    container: CoreServiceContainer,
    *,
    prefix: str = "/api/v1",
    include_schema: bool = True,
) -> None:
    for router_factory in (
        create_dataset_router,
        create_fitting_router,
        create_nist_router,
        create_public_data_router,
    ):
        router = router_factory(container)
        app.include_router(router, prefix=prefix, include_in_schema=include_schema)


__all__ = ["register_core_routes"]

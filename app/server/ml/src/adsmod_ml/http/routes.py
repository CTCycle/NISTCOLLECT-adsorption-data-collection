from __future__ import annotations

from fastapi import FastAPI

from adsmod_ml.http.configuration import create_configuration_router
from adsmod_ml.http.training import create_training_router
from adsmod_ml.services.container import MlServiceContainer


###############################################################################
def register_ml_routes(
    app: FastAPI,
    container: MlServiceContainer,
    *,
    prefix: str = "/api/v1",
    include_schema: bool = True,
) -> None:
    app.include_router(
        create_configuration_router(),
        prefix=prefix,
        include_in_schema=include_schema,
    )
    app.include_router(
        create_training_router(container),
        prefix=prefix,
        include_in_schema=include_schema,
    )


__all__ = ["register_ml_routes"]

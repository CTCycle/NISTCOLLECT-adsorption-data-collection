from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, get_args

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from adsmod_common.capabilities import CapabilitiesResponse, FeatureCapabilities
from adsmod_common.config import AdsmodConfig, load_config
from adsmod_common.units import UnitRegistry
from adsmod_common.version import __version__

from .common.constants import FASTAPI_DESCRIPTION, FASTAPI_TITLE
from .common.utils.logger import close_file_logging, configure_logging, logger
from .contracts.configuration import (
    DisplayUnitCapabilities,
    FittingConfigurationResponse,
    NumericBounds,
    ParameterDefaults,
)
from .contracts.fitting import FittingRequest
from .http.entrypoint import health_router, register_root_routes
from .http.routes import register_core_routes
from .persistence.paths import resolve_storage_root
from .repositories.database.initializer import prepare_database_for_startup
from .services.container import CoreServiceContainer
from .services.training_data import TrainingDataService


###############################################################################
@dataclass
class ApplicationRuntime:
    config: AdsmodConfig
    container: CoreServiceContainer
    training_data: TrainingDataService
    machine_learning_available: bool = False
    machine_learning_reason: str | None = None
    ml_container: Any | None = None


###############################################################################
def capabilities(request: Request) -> CapabilitiesResponse:
    runtime: ApplicationRuntime = request.app.state.runtime
    available = runtime.machine_learning_available
    return CapabilitiesResponse(
        version=__version__,
        features=FeatureCapabilities(
            datasets=True,
            nist=True,
            fitting=True,
            machine_learning=available,
            training=available,
            checkpoints=available,
        ),
    )


###############################################################################
def configuration(request: Request) -> FittingConfigurationResponse:
    config: AdsmodConfig = request.app.state.runtime.config
    fitting = config.application.fitting
    pressure_units = tuple(UnitRegistry.PRESSURE_TO_PA)
    uptake_units = tuple(dict.fromkeys(UnitRegistry.UPTAKE_ALIASES.values()))
    return FittingConfigurationResponse(
        supported_optimizers=tuple(
            get_args(FittingRequest.model_fields["optimizer"].annotation)
        ),
        default_optimizer="trf",
        default_max_evaluations=fitting.default_max_iterations,
        max_evaluations_bounds=NumericBounds(
            minimum=10, maximum=fitting.max_iterations_upper_bound
        ),
        weighting_options=tuple(
            get_args(FittingRequest.model_fields["weighting"].annotation)
        ),
        default_weighting="unweighted",
        display_units=DisplayUnitCapabilities(
            pressure=pressure_units + ("1", "%"),
            uptake=uptake_units,
            default_pressure="bar",
            default_uptake="mmol/g",
        ),
        parameter_defaults=ParameterDefaults(
            lower=fitting.default_parameter_min,
            upper=fitting.default_parameter_max,
            initial=fitting.default_parameter_initial,
        ),
    )


###############################################################################
def _build_system_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/system/capabilities",
        capabilities,
        methods=["GET"],
        response_model=CapabilitiesResponse,
        tags=["system"],
    )
    router.add_api_route(
        "/system/configuration",
        configuration,
        methods=["GET"],
        response_model=FittingConfigurationResponse,
        tags=["system"],
    )
    return router


###############################################################################
def _register_optional_ml(application: FastAPI, runtime: ApplicationRuntime) -> None:
    try:
        bootstrap = import_module("adsmod_ml.bootstrap")
        bootstrap.configure_environment()
        container_module = import_module("adsmod_ml.services.container")
        routes_module = import_module("adsmod_ml.http.routes")
        ml_container = container_module.MlServiceContainer(
            runtime.config,
            snapshot_access=runtime.training_data,
        )
        routes_module.register_ml_routes(application, ml_container, prefix="/api/v1")
    except Exception as exc:  # noqa: BLE001
        runtime.machine_learning_available = False
        runtime.machine_learning_reason = str(exc)
        logger.info("Optional machine learning support is unavailable: %s", exc)
        return
    runtime.ml_container = ml_container
    runtime.machine_learning_available = True
    runtime.machine_learning_reason = None
    application.state.ml_container = ml_container


###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    config: AdsmodConfig = application.state.config
    storage_root = resolve_storage_root(config)
    storage_root.mkdir(parents=True, exist_ok=True)
    configure_logging(storage_root / "logs")
    prepare_database_for_startup(
        config.application.database,
        storage_root=storage_root,
    )
    application.state.core_container.public_data_repository.ensure_sources()
    application.state.ready = True
    try:
        yield
    finally:
        application.state.ready = False
        ml_container = getattr(application.state, "ml_container", None)
        if ml_container is not None:
            ml_container.shutdown()
        application.state.core_container.shutdown()
        application.state.core_container.database.dispose()
        close_file_logging()


###############################################################################
def create_app(config: AdsmodConfig) -> FastAPI:
    container = CoreServiceContainer(config)
    runtime = ApplicationRuntime(
        config=config,
        container=container,
        training_data=TrainingDataService(container),
    )
    application = FastAPI(
        title=FASTAPI_TITLE,
        description=FASTAPI_DESCRIPTION,
        version=__version__,
        lifespan=app_lifespan,
    )
    application.state.config = config
    application.state.ready = False
    application.state.core_container = container
    application.state.runtime = runtime
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{config.runtime.host}:{config.runtime.frontend_port}",
            f"http://127.0.0.1:{config.runtime.frontend_port}",
            f"http://localhost:{config.runtime.frontend_port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health_router)
    application.include_router(_build_system_router(), prefix="/api/v1")
    register_core_routes(application, container, prefix="/api/v1")
    _register_optional_ml(application, runtime)
    register_root_routes(application)
    return application


###############################################################################
def create_app_from_path(config_path: str | Path) -> FastAPI:
    return create_app(load_config(config_path))


__all__ = ["app_lifespan", "create_app", "create_app_from_path"]

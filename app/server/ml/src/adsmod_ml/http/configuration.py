from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from adsmod_common.config import AdsmodConfig
from adsmod_ml.contracts.configuration import RuntimeDeviceCapabilities, TrainingConfigurationResponse
from adsmod_ml.contracts.training import DatasetBuildRequest, ResumeTrainingRequest, TrainingConfigRequest


###############################################################################
def _numeric_constraints(model: type[Any]) -> dict[str, dict[str, int | float]]:
    constraints: dict[str, dict[str, int | float]] = {}
    for name, field in model.model_fields.items():
        values: dict[str, int | float] = {}
        for metadata in field.metadata:
            for attribute, key in (
                ("ge", "minimum"),
                ("gt", "exclusive_minimum"),
                ("le", "maximum"),
                ("lt", "exclusive_maximum"),
            ):
                value = getattr(metadata, attribute, None)
                if value is not None:
                    values[key] = value
        if values:
            constraints[name] = values
    return constraints


###############################################################################
def _device_capabilities() -> RuntimeDeviceCapabilities:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = tuple(str(torch.cuda.get_device_name(index)) for index in range(device_count))
    return RuntimeDeviceCapabilities(
        keras_backend="torch",
        cuda_available=cuda_available,
        device_count=device_count,
        devices=devices,
    )


###############################################################################
def configuration(request: Request) -> TrainingConfigurationResponse:
    config: AdsmodConfig = request.app.state.config
    training_defaults = TrainingConfigRequest().model_dump(mode="json")
    training_defaults.update(
        {
            "use_jit": config.application.training.use_jit,
            "jit_backend": config.application.training.jit_backend,
            "use_mixed_precision": config.application.training.use_mixed_precision,
            "dataloader_workers": config.application.training.dataloader_workers,
        }
    )
    dataset_defaults = DatasetBuildRequest.model_construct(datasets=[]).model_dump(
        mode="json", exclude={"datasets"}
    )
    return TrainingConfigurationResponse(
        defaults=training_defaults,
        dataset_defaults=dataset_defaults,
        resume_defaults=ResumeTrainingRequest.model_construct(
            checkpoint_name="configuration"
        ).model_dump(mode="json", exclude={"checkpoint_name"}),
        numeric_constraints={
            **_numeric_constraints(TrainingConfigRequest),
            **{
                f"dataset_{name}": value
                for name, value in _numeric_constraints(DatasetBuildRequest).items()
            },
            "additional_epochs": _numeric_constraints(ResumeTrainingRequest)[
                "additional_epochs"
            ],
        },
        supported_models=("SCADS Series", "SCADS Atomic"),
        checkpoint_capabilities={"save": True, "resume": True, "delete": True},
        runtime=_device_capabilities(),
    )


###############################################################################
def create_configuration_router() -> APIRouter:
    router = APIRouter(prefix="/training", tags=["training"])
    router.add_api_route(
        "/configuration",
        configuration,
        methods=["GET"],
        response_model=TrainingConfigurationResponse,
    )
    return router


__all__ = ["create_configuration_router"]

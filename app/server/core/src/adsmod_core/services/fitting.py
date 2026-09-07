from __future__ import annotations

from typing import Any

import numpy as np

from adsmod_common.config import AdsmodConfig
from adsmod_core.contracts.fitting import (
    FittingRequest,
    FittingResponse,
    ModelCatalogResponse,
    PersistedRunMetricsResponse,
    PersistedRunModelResponse,
    PersistedRunParameterResponse,
    PersistedRunResponse,
    PersistedRunCurvePointResponse,
)
from adsmod_core.services.modeling.fitting import (
    MODEL_VERSION,
    FitComputation,
    FittingPipeline,
    parameter_unit,
)
from adsmod_core.common.utils.logger import logger
from adsmod_core.contracts.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from adsmod_core.repositories.datasets import DatasetRepository
from adsmod_core.repositories.fitting import FittingRepository
from adsmod_core.services.job_responses import JobResponseFactory
from adsmod_core.services.jobs import JobManager

###############################################################################
class FittingService:
    JOB_TYPE = "fitting"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        config: AdsmodConfig,
        datasets: DatasetRepository,
        results: FittingRepository,
        job_manager: JobManager | None = None,
        pipeline: FittingPipeline | None = None,
    ) -> None:
        self.config = config
        self.datasets = datasets
        self.results = results
        self.job_manager = job_manager or JobManager(logger=logger)
        self.pipeline = pipeline or FittingPipeline()

    # -------------------------------------------------------------------------
    def model_catalog(
        self,
        pressure_unit: str = "bar",
        uptake_unit: str = "mmol/g",
        dataset_id: int | None = None,
        isotherm_id: int | None = None,
    ) -> ModelCatalogResponse:
        pressure_basis = "absolute"
        molar_mass = None
        if dataset_id is not None and isotherm_id is not None:
            context = next(
                (
                    item
                    for item in self.datasets.experiments(dataset_id)
                    if item["id"] == isotherm_id
                ),
                None,
            )
            if context is None:
                raise LookupError(
                    f"Isotherm {isotherm_id} does not belong to dataset {dataset_id}."
                )
            pressure_basis = context["pressure_basis"]
            if not context["fitting_eligible"]:
                return ModelCatalogResponse(
                    pressure_unit=pressure_unit, uptake_unit=uptake_unit, models=[]
                )
        return ModelCatalogResponse(
            pressure_unit=pressure_unit,
            uptake_unit=uptake_unit,
            models=self.pipeline.catalog(
                pressure_unit=pressure_unit,
                uptake_unit=uptake_unit,
                pressure_basis=pressure_basis,
                molar_mass_g_mol=molar_mass,
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _persistence_record(
        computation: FitComputation, observation_count: int, pressure_basis: str
    ) -> dict[str, Any]:
        metrics = computation.metrics.values
        parameters: list[dict[str, Any]] = []
        if computation.parameters is not None:
            related = {
                parameter.name: computation.parameters[index]
                for index, parameter in enumerate(computation.spec.parameters)
            }
            related_value = float(related.get("n", related.get("beta", 1.0)))
            for index, parameter in enumerate(computation.spec.parameters):
                parameters.append(
                    {
                        "name": parameter.name,
                        "position": index,
                        "value_canonical": float(computation.parameters[index]),
                        "standard_error_canonical": (
                            float(computation.standard_errors[index])
                            if computation.standard_errors is not None
                            else None
                        ),
                        "ci95_low_canonical": (
                            float(computation.ci95_low[index])
                            if computation.ci95_low is not None
                            else None
                        ),
                        "ci95_high_canonical": (
                            float(computation.ci95_high[index])
                            if computation.ci95_high is not None
                            else None
                        ),
                        "unit_canonical": parameter_unit(
                            parameter,
                            pressure_unit="Pa",
                            uptake_unit="mol/kg",
                            related_value=related_value,
                        ),
                    }
                )
        return {
            "model_name": computation.spec.key,
            "model_version": MODEL_VERSION,
            "status": computation.status,
            "convergence_message": computation.message,
            "function_evaluations": computation.function_evaluations,
            "jacobian_rank": computation.jacobian_rank,
            "condition_number": computation.condition_number,
            "observation_count": observation_count,
            "parameter_count": len(computation.spec.parameters),
            **metrics,
            "predicted_observations": [float(value) for value in computation.predicted],
            "predicted_curve": [
                {
                    "pressure": float(pressure),
                    "pressure_unit": "1" if pressure_basis == "relative" else "Pa",
                    "uptake_mol_kg": float(uptake),
                }
                for pressure, uptake in zip(
                    computation.curve_pressure,
                    computation.curve_uptake,
                    strict=True,
                )
            ],
            "warnings": computation.warnings,
            "rank": computation.rank,
            "parameters": parameters,
        }

    # -------------------------------------------------------------------------
    def _run_fitting_sync(
        self, payload_dict: dict[str, Any], run_id: int
    ) -> dict[str, Any]:
        payload = FittingRequest.model_validate(payload_dict)
        try:
            series = self.datasets.fitting_series(
                payload.dataset_id, payload.isotherm_id
            )
            computations = self.pipeline.run(series, payload)
            pressure = np.asarray(series["pressure"], dtype=np.float64)
            uptake = np.asarray(series["uptake"], dtype=np.float64)
            response_results = [
                self.pipeline.to_response_result(
                    computation,
                    pressure=pressure,
                    uptake=uptake,
                    pressure_basis=series["pressure_basis"],
                    pressure_unit=payload.display_units.pressure,
                    uptake_unit=payload.display_units.uptake,
                    molar_mass_g_mol=series.get("adsorbate_molar_mass_g_mol"),
                )
                for computation in computations
            ]
            successful = [item for item in response_results if item.status != "failed"]
            run_status = (
                "completed"
                if successful and all(item.status == "success" for item in successful)
                else "warning"
                if successful
                else "failed"
            )
            best = next(
                (item.model for item in response_results if item.rank == 1), None
            )
            summary = (
                f"{len(successful)} of {len(response_results)} models fitted; "
                f"best model: {best}."
                if best
                else "No selected model produced a valid fitted result."
            )
            self.results.complete_run(
                run_id,
                status=run_status,
                message=summary,
                results=[
                    self._persistence_record(
                        item, len(pressure), series["pressure_basis"]
                    )
                    for item in computations
                ],
            )
            return FittingResponse(
                status=(
                    "success"
                    if run_status == "completed"
                    else "warning"
                    if run_status == "warning"
                    else "error"
                ),
                run_id=run_id,
                dataset_id=series["dataset_id"],
                isotherm_id=series["isotherm_id"],
                dataset_name=series["dataset_name"],
                experiment_name=series["isotherm_name"],
                adsorbent=series["adsorbent"],
                adsorbate=series["adsorbate"],
                temperature_k=series["temperature_k"],
                pressure_basis=series["pressure_basis"],
                pressure_unit=payload.display_units.pressure,
                uptake_unit=payload.display_units.uptake,
                observation_count=len(pressure),
                best_model=best,
                results=response_results,
                summary=summary,
            ).model_dump(mode="json")
        except Exception as exc:
            self.results.fail_run(run_id, str(exc))
            raise

    # -------------------------------------------------------------------------
    def start_fitting_job(self, payload: FittingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running(self.JOB_TYPE):
            raise ValueError("A fitting job is already running.")
        series = self.datasets.fitting_series(payload.dataset_id, payload.isotherm_id)
        run_id = self.results.create_run(
            isotherm_id=payload.isotherm_id,
            input_sha256=self.pipeline.input_hash(series),
            optimizer=payload.optimizer,
            max_evaluations=payload.max_evaluations,
            pressure_display_unit=payload.display_units.pressure,
            uptake_display_unit=payload.display_units.uptake,
            configuration=payload.model_dump(mode="json"),
        )
        job_id = self.job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=self._run_fitting_sync,
            args=(payload.model_dump(mode="json"), run_id),
        )
        return JobResponseFactory.start(
            job_id=job_id,
            job_type=self.JOB_TYPE,
            message=f"Fitting run {run_id} started.",
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise LookupError(f"Job {job_id} not found.")
        return JobResponseFactory.status(
            job_status=job_status,
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        return JobResponseFactory.list(
            job_statuses=self.job_manager.list_jobs(self.JOB_TYPE),
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        if not self.job_manager.cancel_job(job_id):
            raise ValueError(
                f"Job {job_id} cannot be cancelled (not found or already completed)."
            )
        return JobResponseFactory.cancelled(job_id)

    # -------------------------------------------------------------------------
    def get_persisted_run(self, run_id: int) -> PersistedRunResponse:
        run = self.results.get_run(run_id)
        metric_names = (
            "sse",
            "rmse",
            "mae",
            "r_squared",
            "adjusted_r_squared",
            "chi_square",
            "aic",
            "aicc",
            "bic",
        )
        return PersistedRunResponse(
            run_id=run.id,
            dataset_id=run.isotherm.dataset_id,
            isotherm_id=run.isotherm_id,
            optimizer=run.optimizer,
            weighting=run.configuration.get("weighting", "unweighted"),
            status_detail=run.status,
            message=run.message,
            created_at=run.created_at.isoformat() if run.created_at else None,
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            results=[
                PersistedRunModelResponse(
                    id=result.id,
                    model=result.model_name,
                    status=result.status,
                    convergence_message=result.convergence_message,
                    metrics=PersistedRunMetricsResponse(
                        **{name: getattr(result, name) for name in metric_names}
                    ),
                    predicted_observations=result.predicted_observations,
                    predicted_curve=[
                        PersistedRunCurvePointResponse(**point)
                        for point in result.predicted_curve
                    ],
                    warnings=result.warnings,
                    parameters=[
                        PersistedRunParameterResponse(
                            name=parameter.name,
                            value=parameter.value_canonical,
                            unit=parameter.unit_canonical,
                            standard_error=parameter.standard_error_canonical,
                        )
                        for parameter in result.parameters
                    ],
                )
                for result in run.results
            ],
        )

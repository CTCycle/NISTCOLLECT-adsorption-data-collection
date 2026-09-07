from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.schemas.models import (
    FitParameter,
    FitResult,
    FittingRun,
)

###############################################################################
class FittingRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def create_run(
        self,
        *,
        isotherm_id: int,
        input_sha256: str,
        optimizer: str,
        max_evaluations: int,
        pressure_display_unit: str,
        uptake_display_unit: str,
        configuration: dict[str, Any],
    ) -> int:
        with self.database.transaction() as session:
            run = FittingRun(
                isotherm_id=isotherm_id,
                input_sha256=input_sha256,
                optimizer=optimizer,
                max_evaluations=max_evaluations,
                pressure_display_unit=pressure_display_unit,
                uptake_display_unit=uptake_display_unit,
                configuration=configuration,
                status="running",
            )
            session.add(run)
            session.flush()
            return run.id

    # -------------------------------------------------------------------------
    def complete_run(
        self,
        run_id: int,
        *,
        status: str,
        message: str,
        results: list[dict[str, Any]],
    ) -> None:
        with self.database.transaction() as session:
            run = session.get(FittingRun, run_id)
            if run is None:
                raise LookupError(f"Fitting run {run_id} does not exist.")
            for result_record in results:
                record = dict(result_record)
                parameters = record.pop("parameters")
                result = FitResult(run_id=run_id, **record)
                session.add(result)
                session.flush()
                for parameter in parameters:
                    session.add(FitParameter(result_id=result.id, **parameter))
            run.status = status
            run.message = message
            run.completed_at = datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    def fail_run(self, run_id: int, message: str) -> None:
        with self.database.transaction() as session:
            run = session.get(FittingRun, run_id)
            if run is None:
                return
            run.status = "failed"
            run.message = message
            run.completed_at = datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    def get_run(self, run_id: int) -> FittingRun:
        with self.database.session_factory() as session:
            run = session.scalar(
                select(FittingRun)
                .where(FittingRun.id == run_id)
                .options(
                    selectinload(FittingRun.isotherm),
                    selectinload(FittingRun.results).selectinload(FitResult.parameters),
                )
            )
            if run is None:
                raise LookupError(f"Fitting run {run_id} does not exist.")
            return run

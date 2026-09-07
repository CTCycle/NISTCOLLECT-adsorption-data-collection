from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from adsmod_core.repositories.schemas.types import (
    JSONList,
    JSONMapping,
    UTCDateTime,
    normalize_identity,
)

###############################################################################
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

###############################################################################
class Base(DeclarativeBase):
    pass

###############################################################################
class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    tags: Mapped[list[Any]] = mapped_column(JSONList, nullable=False, default=list)
    provenance: Mapped[dict[str, Any]] = mapped_column(
        JSONMapping, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    imports: Mapped[list[DatasetImport]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    isotherms: Mapped[list[Isotherm]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    __table_args__ = (
        UniqueConstraint("normalized_name", name="uq_datasets_normalized_name"),
        CheckConstraint("source IN ('uploaded', 'nist')", name="ck_datasets_source"),
        CheckConstraint("length(normalized_name) > 0", name="ck_datasets_name"),
        Index("ix_datasets_source_created_at", "source", "created_at"),
    )

    # -------------------------------------------------------------------------
    def __init__(self, **kwargs: Any) -> None:
        if "normalized_name" not in kwargs and "name" in kwargs:
            kwargs["normalized_name"] = normalize_identity(str(kwargs["name"]))
        super().__init__(**kwargs)

###############################################################################
class DatasetImport(Base):
    __tablename__ = "dataset_imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    mapping_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    source_structure: Mapped[str] = mapped_column(String(16), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(32), nullable=False)
    column_mapping: Mapped[dict[str, Any]] = mapped_column(JSONMapping, nullable=False)
    validation_result: Mapped[dict[str, Any]] = mapped_column(
        JSONMapping, nullable=False
    )
    warnings: Mapped[list[Any]] = mapped_column(JSONList, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False
    )

    dataset: Mapped[Dataset] = relationship(back_populates="imports", lazy="raise")

    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "source_sha256", name="uq_dataset_import_source"
        ),
        CheckConstraint(
            "source_structure IN ('atomic', 'aggregated', 'mixed')",
            name="ck_dataset_import_structure",
        ),
    )

###############################################################################
class Adsorbate(Base):
    __tablename__ = "adsorbates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    inchi_key: Mapped[str | None] = mapped_column(String(27), unique=True)
    inchi: Mapped[str | None] = mapped_column(String)
    formula: Mapped[str | None] = mapped_column(String(255))
    molar_mass_g_mol: Mapped[float | None] = mapped_column(Float)
    smiles: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    components: Mapped[list[IsothermComponent]] = relationship(
        back_populates="adsorbate", lazy="raise"
    )

    __table_args__ = (Index("ix_adsorbates_normalized_name", "normalized_name"),)

    # -------------------------------------------------------------------------
    def __init__(self, **kwargs: Any) -> None:
        if "normalized_name" not in kwargs and "name" in kwargs:
            kwargs["normalized_name"] = normalize_identity(str(kwargs["name"]))
        super().__init__(**kwargs)

###############################################################################
class Adsorbent(Base):
    __tablename__ = "adsorbents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    external_identifier: Mapped[str | None] = mapped_column(String(255), unique=True)
    formula: Mapped[str | None] = mapped_column(String(255))
    molar_mass_g_mol: Mapped[float | None] = mapped_column(Float)
    smiles: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    isotherms: Mapped[list[Isotherm]] = relationship(
        back_populates="adsorbent", lazy="raise"
    )

    __table_args__ = (Index("ix_adsorbents_normalized_name", "normalized_name"),)

    # -------------------------------------------------------------------------
    def __init__(self, **kwargs: Any) -> None:
        if "normalized_name" not in kwargs and "name" in kwargs:
            kwargs["normalized_name"] = normalize_identity(str(kwargs["name"]))
        super().__init__(**kwargs)

###############################################################################
class Isotherm(Base):
    __tablename__ = "isotherms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    external_key: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    adsorbent_id: Mapped[int] = mapped_column(
        ForeignKey("adsorbents.id", ondelete="RESTRICT"), nullable=False
    )
    temperature_original: Mapped[float] = mapped_column(Float, nullable=False)
    temperature_original_unit: Mapped[str] = mapped_column(String(16), nullable=False)
    temperature_k: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_basis: Mapped[str] = mapped_column(String(16), nullable=False)
    duplicate_policy: Mapped[str] = mapped_column(
        String(16), nullable=False, default="reject"
    )
    saturation_pressure_pa: Mapped[float | None] = mapped_column(Float)
    conditions: Mapped[dict[str, Any]] = mapped_column(
        JSONMapping, nullable=False, default=dict
    )
    provenance: Mapped[dict[str, Any]] = mapped_column(
        JSONMapping, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    dataset: Mapped[Dataset] = relationship(back_populates="isotherms", lazy="raise")
    adsorbent: Mapped[Adsorbent] = relationship(
        back_populates="isotherms", lazy="raise"
    )
    components: Mapped[list[IsothermComponent]] = relationship(
        back_populates="isotherm",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    observations: Mapped[list[Observation]] = relationship(
        back_populates="isotherm",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    fitting_runs: Mapped[list[FittingRun]] = relationship(
        back_populates="isotherm",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "external_key", name="uq_isotherms_dataset_key"),
        CheckConstraint("temperature_k > 0", name="ck_isotherms_temperature"),
        CheckConstraint(
            "pressure_basis IN ('absolute', 'partial', 'relative')",
            name="ck_isotherms_pressure_basis",
        ),
        CheckConstraint(
            "duplicate_policy IN ('reject', 'keep', 'average')",
            name="ck_isotherms_duplicate_policy",
        ),
        CheckConstraint(
            "saturation_pressure_pa IS NULL OR saturation_pressure_pa > 0",
            name="ck_isotherms_saturation_pressure",
        ),
        Index("ix_isotherms_dataset_name", "dataset_id", "name"),
    )

###############################################################################
class IsothermComponent(Base):
    __tablename__ = "isotherm_components"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    isotherm_id: Mapped[int] = mapped_column(
        ForeignKey("isotherms.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    adsorbate_id: Mapped[int] = mapped_column(
        ForeignKey("adsorbates.id", ondelete="RESTRICT"), nullable=False
    )
    mole_fraction: Mapped[float | None] = mapped_column(Float)

    isotherm: Mapped[Isotherm] = relationship(back_populates="components", lazy="raise")
    adsorbate: Mapped[Adsorbate] = relationship(
        back_populates="components", lazy="raise"
    )
    observations: Mapped[list[Observation]] = relationship(
        back_populates="component", lazy="raise"
    )

    __table_args__ = (
        UniqueConstraint("isotherm_id", "position", name="uq_components_position"),
        UniqueConstraint("isotherm_id", "adsorbate_id", name="uq_components_adsorbate"),
        CheckConstraint("position >= 1", name="ck_components_position"),
        CheckConstraint(
            "mole_fraction IS NULL OR (mole_fraction >= 0 AND mole_fraction <= 1)",
            name="ck_components_fraction",
        ),
    )

###############################################################################
class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    isotherm_id: Mapped[int] = mapped_column(
        ForeignKey("isotherms.id", ondelete="CASCADE"), nullable=False
    )
    component_id: Mapped[int] = mapped_column(
        ForeignKey("isotherm_components.id", ondelete="CASCADE"), nullable=False
    )
    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_row: Mapped[int | None] = mapped_column(Integer)
    pressure_original: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_original_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    pressure_canonical: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_canonical_unit: Mapped[str] = mapped_column(String(8), nullable=False)
    uptake_original: Mapped[float] = mapped_column(Float, nullable=False)
    uptake_original_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    uptake_mol_kg: Mapped[float] = mapped_column(Float, nullable=False)
    uptake_stddev_mol_kg: Mapped[float | None] = mapped_column(Float)
    conversion_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONMapping, nullable=False, default=dict
    )
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONMapping, nullable=False, default=dict
    )

    isotherm: Mapped[Isotherm] = relationship(
        back_populates="observations", lazy="raise"
    )
    component: Mapped[IsothermComponent] = relationship(
        back_populates="observations", lazy="raise"
    )

    __table_args__ = (
        UniqueConstraint(
            "isotherm_id",
            "component_id",
            "sequence_index",
            name="uq_observations_sequence",
        ),
        CheckConstraint("sequence_index >= 0", name="ck_observations_sequence"),
        CheckConstraint("pressure_canonical >= 0", name="ck_observations_pressure"),
        CheckConstraint("uptake_mol_kg >= 0", name="ck_observations_uptake"),
        CheckConstraint(
            "pressure_canonical_unit IN ('Pa', '1')",
            name="ck_observations_pressure_unit",
        ),
        CheckConstraint(
            "uptake_stddev_mol_kg IS NULL OR uptake_stddev_mol_kg > 0",
            name="ck_observations_uncertainty",
        ),
        Index(
            "ix_observations_isotherm_component_pressure",
            "isotherm_id",
            "component_id",
            "pressure_canonical",
            "sequence_index",
        ),
    )

###############################################################################
class FittingRun(Base):
    __tablename__ = "fitting_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    isotherm_id: Mapped[int] = mapped_column(
        ForeignKey("isotherms.id", ondelete="CASCADE"), nullable=False
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    optimizer: Mapped[str] = mapped_column(String(32), nullable=False)
    max_evaluations: Mapped[int] = mapped_column(Integer, nullable=False)
    pressure_display_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    uptake_display_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONMapping, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    isotherm: Mapped[Isotherm] = relationship(
        back_populates="fitting_runs", lazy="raise"
    )
    results: Mapped[list[FitResult]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'warning', 'failed')",
            name="ck_fitting_runs_status",
        ),
        CheckConstraint("max_evaluations > 0", name="ck_fitting_runs_evaluations"),
        Index("ix_fitting_runs_isotherm_created", "isotherm_id", "created_at"),
    )

###############################################################################
class FitResult(Base):
    __tablename__ = "fit_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("fitting_runs.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    convergence_message: Mapped[str] = mapped_column(String, nullable=False, default="")
    function_evaluations: Mapped[int | None] = mapped_column(Integer)
    jacobian_rank: Mapped[int | None] = mapped_column(Integer)
    condition_number: Mapped[float | None] = mapped_column(Float)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parameter_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sse: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    mae: Mapped[float | None] = mapped_column(Float)
    r_squared: Mapped[float | None] = mapped_column(Float)
    adjusted_r_squared: Mapped[float | None] = mapped_column(Float)
    chi_square: Mapped[float | None] = mapped_column(Float)
    aic: Mapped[float | None] = mapped_column(Float)
    aicc: Mapped[float | None] = mapped_column(Float)
    bic: Mapped[float | None] = mapped_column(Float)
    predicted_observations: Mapped[list[Any]] = mapped_column(JSONList, nullable=False)
    predicted_curve: Mapped[list[Any]] = mapped_column(JSONList, nullable=False)
    warnings: Mapped[list[Any]] = mapped_column(JSONList, nullable=False, default=list)
    rank: Mapped[int | None] = mapped_column(Integer)

    run: Mapped[FittingRun] = relationship(back_populates="results", lazy="raise")
    parameters: Mapped[list[FitParameter]] = relationship(
        back_populates="result",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    __table_args__ = (
        UniqueConstraint("run_id", "model_name", name="uq_fit_results_run_model"),
        CheckConstraint(
            "status IN ('success', 'warning', 'failed')",
            name="ck_fit_results_status",
        ),
        CheckConstraint("observation_count >= 0", name="ck_fit_results_observations"),
        CheckConstraint("parameter_count > 0", name="ck_fit_results_parameters"),
        Index("ix_fit_results_run_rank", "run_id", "rank"),
    )

###############################################################################
class FitParameter(Base):
    __tablename__ = "fit_parameters"

    result_id: Mapped[int] = mapped_column(
        ForeignKey("fit_results.id", ondelete="CASCADE"), primary_key=True
    )
    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    value_canonical: Mapped[float] = mapped_column(Float, nullable=False)
    standard_error_canonical: Mapped[float | None] = mapped_column(Float)
    ci95_low_canonical: Mapped[float | None] = mapped_column(Float)
    ci95_high_canonical: Mapped[float | None] = mapped_column(Float)
    unit_canonical: Mapped[str] = mapped_column(String(128), nullable=False)

    result: Mapped[FitResult] = relationship(back_populates="parameters", lazy="raise")

    __table_args__ = (
        UniqueConstraint("result_id", "position", name="uq_fit_parameters_position"),
        CheckConstraint("position >= 0", name="ck_fit_parameters_position"),
    )

###############################################################################
class TrainingSnapshot(Base):
    __tablename__ = "training_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, nullable=False
    )
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONMapping, nullable=False, default=dict
    )
    rows: Mapped[list[TrainingSnapshotRow]] = relationship(
        back_populates="snapshot",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
        order_by="TrainingSnapshotRow.row_index",
    )

    __table_args__ = (
        CheckConstraint("row_count > 0", name="ck_training_snapshots_row_count"),
    )

###############################################################################
class TrainingSnapshotRow(Base):
    __tablename__ = "training_snapshot_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("training_snapshots.snapshot_id", ondelete="CASCADE"), nullable=False
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONMapping, nullable=False)

    snapshot: Mapped[TrainingSnapshot] = relationship(
        back_populates="rows", lazy="raise"
    )

    __table_args__ = (
        UniqueConstraint(
            "snapshot_id", "row_index", name="uq_training_snapshot_rows_index"
        ),
        CheckConstraint("row_index >= 0", name="ck_training_snapshot_rows_index"),
        Index("ix_training_snapshot_rows_snapshot_index", "snapshot_id", "row_index"),
    )


__all__ = [
    "Adsorbate",
    "Adsorbent",
    "Base",
    "Dataset",
    "DatasetImport",
    "FitParameter",
    "FitResult",
    "FittingRun",
    "Isotherm",
    "IsothermComponent",
    "Observation",
    "TrainingSnapshot",
    "TrainingSnapshotRow",
]

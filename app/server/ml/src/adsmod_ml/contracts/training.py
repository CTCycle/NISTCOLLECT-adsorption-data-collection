from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


###############################################################################
# Shared Pydantic config
STRICT_STRIPPED_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)
# Shared regex patterns
REGEX_HEX_SHA256 = r"^[A-Fa-f0-9]{64}$"
REGEX_LABEL = r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$"
REGEX_NAME = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
REGEX_LONG_NAME = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
REGEX_BACKEND = r"^[A-Za-z0-9_.-]+$"
REGEX_DATASET_NAME = r"^[A-Za-z0-9_. -]+$"

###############################################################################
class TrainingConfigRequest(BaseModel):
    model_config = STRICT_STRIPPED_CONFIG

    # Dataset settings
    sample_size: float = Field(default=1.0, ge=0.01, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.05, le=0.5)
    batch_size: int = Field(default=32, ge=1, le=256)
    shuffle_dataset: bool = True
    max_buffer_size: int = Field(default=256, ge=1, le=1_000_000)
    dataset_label: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=REGEX_LABEL,
    )
    dataset_hash: str | None = Field(
        default=None,
        pattern=REGEX_HEX_SHA256,
    )

    # Model settings
    selected_model: str = "SCADS Series"
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    num_attention_heads: int = Field(default=8, ge=1, le=16)
    num_encoders: int = Field(default=4, ge=1, le=12)
    molecular_embedding_size: int = Field(default=256, ge=64, le=1024)

    # Training settings
    epochs: int = Field(default=50, ge=1, le=500)
    dataloader_workers: int = Field(default=0, ge=0, le=64)
    prefetch_factor: int = Field(default=1, ge=1, le=32)
    pin_memory: bool = True
    use_device_GPU: bool = True
    device_ID: int = Field(default=0, ge=0, le=32)
    use_mixed_precision: bool = False
    use_jit: bool = False
    jit_backend: str = Field(
        default="inductor",
        min_length=1,
        max_length=32,
        pattern=REGEX_BACKEND,
    )

    # LR scheduler settings
    use_lr_scheduler: bool = True
    initial_lr: float = Field(default=1e-4, ge=1e-7, le=1e-2)
    target_lr: float = Field(default=1e-5, ge=1e-8, le=1e-3)
    constant_steps: int = Field(default=5, ge=0, le=50)
    decay_steps: int = Field(default=10, ge=1, le=100)

    # Callbacks
    save_checkpoints: bool = True
    checkpoints_frequency: int = Field(default=5, ge=1, le=50)
    custom_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=REGEX_NAME,
    )

###############################################################################
class ResumeTrainingRequest(BaseModel):
    model_config = STRICT_STRIPPED_CONFIG

    checkpoint_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=REGEX_LONG_NAME,
    )
    additional_epochs: int = Field(default=10, ge=1, le=100)

###############################################################################
class TrainingDatasetResponse(BaseModel):
    available: bool
    name: str | None = None
    train_samples: int | None = None
    validation_samples: int | None = None

###############################################################################
class CheckpointDetailInfo(BaseModel):
    name: str
    epochs_trained: int | None = None
    final_loss: float | None = None
    final_accuracy: float | None = None
    is_compatible: bool = True

###############################################################################
class CheckpointFullDetailsResponse(BaseModel):
    name: str
    configuration: dict[str, Any] | None = None
    metadata: TrainingMetadata | None = None
    history: dict[str, Any] | None = None

###############################################################################
class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointDetailInfo]

###############################################################################
class TrainingStartResponse(BaseModel):
    status: str
    session_id: str
    message: str
    poll_interval: float | None = None

###############################################################################
class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_epoch: int
    total_epochs: int
    progress: float
    metrics: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    poll_interval: float | None = None

###############################################################################
class DatasetSelection(BaseModel):
    model_config = STRICT_STRIPPED_CONFIG

    source: Literal["nist", "uploaded"]
    dataset_id: int | None = Field(default=None, ge=1)
    dataset_name: str = Field(
        min_length=1,
        max_length=128,
        pattern=REGEX_DATASET_NAME,
    )

###############################################################################
class DatasetBuildRequest(BaseModel):
    model_config = STRICT_STRIPPED_CONFIG

    sample_size: float = Field(default=1.0, ge=0.01, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.05, le=0.5)
    min_measurements: int = Field(default=1, ge=1, le=100)
    max_measurements: int = Field(default=30, ge=5, le=500)
    smile_sequence_size: int = Field(default=20, ge=5, le=100)
    max_pressure: float = Field(default=10000.0, ge=100.0, le=100000.0)
    max_uptake: float = Field(default=20.0, ge=1.0, le=1000.0)
    reference_checkpoint: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=REGEX_LONG_NAME,
    )
    datasets: list[DatasetSelection] = Field(default_factory=list, min_length=1)
    dataset_label: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=REGEX_LABEL,
    )

###############################################################################
class DatasetSourceInfo(BaseModel):
    source: Literal["nist", "uploaded"]
    dataset_name: str
    display_name: str
    row_count: int
    dataset_id: int | None = None

###############################################################################
class DatasetSourcesResponse(BaseModel):
    datasets: list[DatasetSourceInfo]

###############################################################################
class OperationStatusResponse(BaseModel):
    status: str
    message: str

###############################################################################
class DatasetBuildResponse(BaseModel):
    success: bool
    message: str
    total_samples: int | None = None
    train_samples: int | None = None
    validation_samples: int | None = None

###############################################################################
class DatasetInfoResponse(BaseModel):
    available: bool
    dataset_label: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=REGEX_LABEL,
    )
    created_at: str | None = None
    sample_size: float | None = None
    validation_size: float | None = None
    min_measurements: int | None = None
    max_measurements: int | None = None
    smile_sequence_size: int | None = None
    max_pressure: float | None = None
    max_uptake: float | None = None
    total_samples: int | None = None
    train_samples: int | None = None
    validation_samples: int | None = None
    smile_vocabulary_size: int | None = None
    adsorbent_vocabulary_size: int | None = None
    normalization_stats: dict[str, Any] | None = None

###############################################################################
class ProcessedDatasetInfo(BaseModel):
    dataset_label: str
    dataset_hash: str | None = Field(
        default=None,
        pattern=REGEX_HEX_SHA256,
    )
    train_samples: int
    validation_samples: int
    created_at: str | None = None

###############################################################################
class ProcessedDatasetsResponse(BaseModel):
    datasets: list[ProcessedDatasetInfo]

###############################################################################
class TrainingMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str | None = None
    sample_size: float = 1.0
    validation_size: float = 0.2
    min_measurements: int = 1
    max_measurements: int = 30
    smile_sequence_size: int = 20
    max_pressure: float = 10000.0
    max_uptake: float = 20.0
    total_samples: int = 0
    train_samples: int = 0
    validation_samples: int = 0

    # Vocabularies
    smile_vocabulary: dict[str, int] = Field(default_factory=dict)
    adsorbent_vocabulary: dict[str, int] = Field(default_factory=dict)

    # Statistics
    normalization_stats: dict[str, list[float] | float | dict[str, Any]] = Field(
        default_factory=dict
    )

    # Integrity check
    dataset_hash: str | None = Field(
        default=None,
        pattern=REGEX_HEX_SHA256,
    )

    # Computed or derived fields
    smile_vocabulary_size: int = 0
    adsorbent_vocabulary_size: int = 0

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

###############################################################################
@dataclass
class TrainingState:
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    session_id: str | None = None
    stop_requested: bool = False
    last_error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # -------------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # -------------------------------------------------------------------------
    def add_log(self, message: str) -> None:
        with self.lock:
            self.log.append(message)
            if len(self.log) > 1000:
                self.log = self.log[-1000:]

    # -------------------------------------------------------------------------
    def add_history(self, epoch_data: dict[str, Any]) -> None:
        with self.lock:
            self.history.append(epoch_data)

    # -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "is_training": self.is_training,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "progress": self.progress,
                "session_id": self.session_id,
                "stop_requested": self.stop_requested,
                "last_error": self.last_error,
                "metrics": self.metrics.copy(),
                "history": list(self.history),
                "log": list(self.log),
            }

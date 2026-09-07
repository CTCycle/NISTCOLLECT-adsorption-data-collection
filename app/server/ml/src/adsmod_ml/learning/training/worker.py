from __future__ import annotations

from collections.abc import Callable
from typing import Any
import multiprocessing
import os
import queue
import signal
import subprocess
import time

from adsmod_ml.common.utils.logger import logger

###############################################################################
class WorkerChannels:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        progress_queue: Any,
        result_queue: Any,
        stop_event: Any,
    ) -> None:
        self.progress_queue = progress_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return bool(self.stop_event.is_set())

    # -------------------------------------------------------------------------
    def drain_progress(self) -> None:
        while True:
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                return
            except Exception:
                return

    # -------------------------------------------------------------------------
    def send_message(self, payload: dict[str, Any]) -> None:
        try:
            if payload.get("type") == "training_plot":
                self.drain_progress()
            self.progress_queue.put(payload, block=False)
        except queue.Full:
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send worker message: %s", exc)

###############################################################################
class ProcessWorker:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        progress_queue_size: int = 256,
        result_queue_size: int = 8,
    ) -> None:
        self.ctx = multiprocessing.get_context("spawn")
        self.progress_queue = self.ctx.Queue(maxsize=progress_queue_size)
        self.result_queue = self.ctx.Queue(maxsize=result_queue_size)
        self.stop_event = self.ctx.Event()
        self.process: multiprocessing.Process | None = None

    # -------------------------------------------------------------------------
    def start(
        self,
        target: Callable[..., None],
        kwargs: dict[str, Any],
    ) -> None:
        if self.process is not None and self.process.is_alive():
            raise RuntimeError("Worker process is already running")
        self.process = self.ctx.Process(
            target=process_target,
            kwargs={
                "target": target,
                "kwargs": kwargs,
                "worker": self.as_child(),
            },
            daemon=False,
        )
        self.process.start()

    # -------------------------------------------------------------------------
    def stop(self) -> None:
        self.stop_event.set()

    # -------------------------------------------------------------------------
    def interrupt(self) -> None:
        self.stop_event.set()

    # -------------------------------------------------------------------------
    def is_interrupted(self) -> bool:
        return bool(self.stop_event.is_set())

    # -------------------------------------------------------------------------
    def is_alive(self) -> bool:
        return bool(self.process is not None and self.process.is_alive())

    # -------------------------------------------------------------------------
    def join(self, timeout: float | None = None) -> None:
        if self.process is None:
            return
        self.process.join(timeout=timeout)

    # -------------------------------------------------------------------------
    def terminate(self) -> None:
        if self.process is None:
            return
        self.terminate_process_tree(self.process)

    # -------------------------------------------------------------------------
    def poll(self, timeout: float = 0.25) -> dict[str, Any] | None:
        try:
            message = self.progress_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except (EOFError, OSError):
            return None
        if isinstance(message, dict):
            return message
        return None

    # -------------------------------------------------------------------------
    def drain_progress(self) -> None:
        while True:
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return

    # -------------------------------------------------------------------------
    def read_result(self) -> dict[str, Any] | None:
        try:
            payload = self.result_queue.get_nowait()
        except queue.Empty:
            return None
        except (EOFError, OSError):
            return None
        if isinstance(payload, dict):
            return payload
        return None

    # -------------------------------------------------------------------------
    def cleanup(self) -> None:
        self.progress_queue.close()
        self.result_queue.close()
        self.progress_queue.join_thread()
        self.result_queue.join_thread()

    # -------------------------------------------------------------------------
    def as_child(self) -> WorkerChannels:
        return WorkerChannels(
            progress_queue=self.progress_queue,
            result_queue=self.result_queue,
            stop_event=self.stop_event,
        )

    # -------------------------------------------------------------------------
    def terminate_process_tree(self, process: multiprocessing.Process) -> None:
        pid = process.pid
        if pid is None:
            return
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", f"taskkill /PID {pid} /T /F"],
                check=False,
                capture_output=True,
            )
            return
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            if process.is_alive():
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return

    # -------------------------------------------------------------------------
    @property
    def exitcode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.exitcode

###############################################################################
def process_target(
    target: Callable[..., None],
    kwargs: dict[str, Any],
    worker: WorkerChannels,
) -> None:
    if os.name != "nt":
        os.setsid()
    target(worker=worker, **kwargs)

"""adsmod_core background job manager for long-running operations."""

from __future__ import annotations

import inspect
import multiprocessing
import os
import queue
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable
from logging import Logger
from time import monotonic
from typing import Any

from adsmod_core.common.utils.encoding import normalize_error_text
from adsmod_core.common.utils.logger import logger as shared_logger

###############################################################################
class _JobState:

    # -------------------------------------------------------------------------
    def __init__(self, job_id: str, job_type: str, status: str) -> None:
        self.job_id = job_id
        self.job_type = job_type
        self.status = status
        self.progress = 0.0
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = monotonic()
        self.completed_at: float | None = None
        self.stop_requested = False
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }

###############################################################################
class _JobExecutionConfig:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        run_mode: str,
        process_stop_timeout_seconds: float,
        process_message_handler: Callable[[str, dict[str, Any]], None] | None,
        completion_handler: (
            Callable[[str, str, dict[str, Any] | None, str | None], None] | None
        ),
    ) -> None:
        self.run_mode = run_mode
        self.process_stop_timeout_seconds = process_stop_timeout_seconds
        self.process_message_handler = process_message_handler
        self.completion_handler = completion_handler

###############################################################################
class _ProcessJobState:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        process: multiprocessing.Process,
        stop_event: multiprocessing.Event,
        result_queue: multiprocessing.Queue,
        message_queue: multiprocessing.Queue,
    ) -> None:
        self.process = process
        self.stop_event = stop_event
        self.result_queue = result_queue
        self.message_queue = message_queue
        self.created_at = monotonic()

###############################################################################
def run_process_runner(
    result_queue: multiprocessing.Queue,
    message_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    runner: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if stop_event.is_set():
        return

    if os.name != "nt":
        try:
            os.setsid()
        except OSError:
            pass

    try:
        result = runner(*args, **kwargs)
        try:
            result_queue.put_nowait({"status": "success", "result": result})
        except Exception:
            result_queue.put({"status": "success", "result": result})
    except Exception as exc:  # noqa: BLE001
        error_msg = format_error_message(exc)
        try:
            result_queue.put_nowait({"status": "error", "error": error_msg})
        except Exception:
            result_queue.put({"status": "error", "error": error_msg})
        try:
            message_queue.put_nowait({"type": "error", "error": error_msg})
        except Exception:
            pass

###############################################################################
class JobManager:
    PROCESS_STOP_TIMEOUT_SECONDS = 10.0

    # -------------------------------------------------------------------------
    def __init__(self, *, logger: Logger | None = None) -> None:
        self._logger = logger or shared_logger
        self.jobs: dict[str, _JobState] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.processes: dict[str, _ProcessJobState] = {}
        self.job_configs: dict[str, _JobExecutionConfig] = {}
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------
    def start_job(
        self,
        job_type: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        job_id: str | None = None,
        run_mode: str = "thread",
        process_message_handler: Callable[[str, dict[str, Any]], None] | None = None,
        completion_handler: (
            Callable[[str, str, dict[str, Any] | None, str | None], None] | None
        ) = None,
        process_stop_timeout_seconds: float | None = None,
    ) -> str:
        if run_mode not in ("thread", "process"):
            raise ValueError(f"Unsupported run_mode: {run_mode}")
        if run_mode == "process" and not self.supports_argument(runner, "stop_event"):
            raise ValueError(
                "Process jobs must accept a 'stop_event' argument for cooperative cancellation."
            )

        if job_id is None:
            job_id = str(uuid.uuid4())[:8]
        state = _JobState(job_id=job_id, job_type=job_type, status="pending")
        runner_kwargs = kwargs.copy() if kwargs else {}

        if self._runner_accepts_job_id(runner):
            runner_kwargs["job_id"] = job_id

        timeout_seconds = (
            float(process_stop_timeout_seconds)
            if process_stop_timeout_seconds is not None
            else self.PROCESS_STOP_TIMEOUT_SECONDS
        )
        config = _JobExecutionConfig(
            run_mode=run_mode,
            process_stop_timeout_seconds=timeout_seconds,
            process_message_handler=process_message_handler,
            completion_handler=completion_handler,
        )

        with self.lock:
            self.jobs[job_id] = state
            self.job_configs[job_id] = config

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, runner, args, runner_kwargs),
            daemon=True,
        )

        with self.lock:
            self.threads[job_id] = thread

        state.update(status="running")
        thread.start()

        self._logger.info("Started job %s (type=%s)", job_id, job_type)
        return job_id

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return None
        return state.snapshot()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            state = self.jobs.get(job_id)
            process_state = self.processes.get(job_id)
        if state is None:
            return False
        if state.status not in ("pending", "running"):
            return False
        state.update(stop_requested=True, status="cancelled", completed_at=monotonic())
        if process_state is not None:
            process_state.stop_event.set()
        self._logger.info("Cancelled job %s", job_id)
        return True

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        with self.lock:
            for state in self.jobs.values():
                if state.status in ("pending", "running"):
                    if job_type is None or state.job_type == job_type:
                        return True
        return False

    # -------------------------------------------------------------------------
    def list_jobs(self, job_type: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            states = list(self.jobs.values())
        results = []
        for state in states:
            if job_type is None or state.job_type == job_type:
                results.append(state.snapshot())
        return results

    # -------------------------------------------------------------------------
    def should_stop(self, job_id: str) -> bool:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return True
        return state.stop_requested

    # -------------------------------------------------------------------------
    def update_progress(self, job_id: str, progress: float) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state:
            state.update(progress=min(100.0, max(0.0, progress)))

    # -------------------------------------------------------------------------
    def update_result(self, job_id: str, patch: dict[str, Any]) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return
        with state.lock:
            existing = state.result or {}
            state.result = {**existing, **patch}

    # -------------------------------------------------------------------------
    def supports_argument(
        self, runner: Callable[..., dict[str, Any]], name: str
    ) -> bool:
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return False

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return name in signature.parameters

    # -------------------------------------------------------------------------
    def _runner_accepts_job_id(self, runner: Callable[..., dict[str, Any]]) -> bool:
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return False
        for param in signature.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                return True
        return "job_id" in signature.parameters

    # -------------------------------------------------------------------------
    def build_process_kwargs(
        self,
        runner: Callable[..., dict[str, Any]],
        kwargs: dict[str, Any],
        stop_event: multiprocessing.Event,
        message_queue: multiprocessing.Queue,
    ) -> dict[str, Any]:
        updated_kwargs = dict(kwargs)
        if self.supports_argument(runner, "stop_event"):
            updated_kwargs["stop_event"] = stop_event
        if self.supports_argument(runner, "message_queue"):
            updated_kwargs["message_queue"] = message_queue
        return updated_kwargs

    # -------------------------------------------------------------------------
    def terminate_process_tree(self, process_id: int | None) -> None:
        if process_id is None:
            return

        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process_id), "/T", "/F"],
                    check=False,
                    capture_output=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to terminate process %s: %s", process_id, exc
                )
            return

        try:
            os.killpg(process_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to terminate process %s: %s", process_id, exc)
            return

        try:
            os.killpg(process_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to force kill process %s: %s", process_id, exc)

    # -------------------------------------------------------------------------
    def consume_process_messages(
        self,
        job_id: str,
        message_queue: multiprocessing.Queue,
        handler: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        while True:
            try:
                message = message_queue.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError) as exc:
                self._logger.debug(
                    "Process message queue closed for %s: %s", job_id, exc
                )
                break

            if handler is None:
                continue
            try:
                handler(job_id, message)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Process message handler failed for %s: %s", job_id, exc
                )

    # -------------------------------------------------------------------------
    def finalize_job(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
            config = self.job_configs.get(job_id)

        if state is None:
            return

        update_payload: dict[str, Any] = {
            "status": status,
            "result": result,
            "error": error,
            "completed_at": monotonic(),
        }
        if status == "completed":
            update_payload["progress"] = 100.0
        state.update(**update_payload)

        if config and config.completion_handler is not None:
            try:
                config.completion_handler(job_id, status, result, error)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Completion handler failed for %s: %s", job_id, exc
                )

    # -------------------------------------------------------------------------
    def run_process_job(
        self,
        job_id: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: _JobExecutionConfig,
    ) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return

        if state.stop_requested:
            self.finalize_job(job_id, "cancelled", None, None)
            return

        context = multiprocessing.get_context("spawn")
        stop_event = context.Event()
        result_queue: multiprocessing.Queue = context.Queue(maxsize=1)
        message_queue: multiprocessing.Queue = context.Queue()
        run_kwargs = self.build_process_kwargs(
            runner, kwargs, stop_event, message_queue
        )

        process = context.Process(
            target=run_process_runner,
            args=(result_queue, message_queue, stop_event, runner, args, run_kwargs),
        )
        process.daemon = True

        with self.lock:
            self.processes[job_id] = _ProcessJobState(
                process=process,
                stop_event=stop_event,
                result_queue=result_queue,
                message_queue=message_queue,
            )

        try:
            process.start()
        except Exception as exc:  # noqa: BLE001
            error_msg = normalize_error_text(str(exc)).split("\n")[0][:200]
            self.finalize_job(job_id, "failed", None, error_msg)
            self._logger.error("Failed to start process job %s: %s", job_id, error_msg)
            return

        stop_requested_at: float | None = None
        while True:
            process.join(timeout=0.2)
            self.consume_process_messages(
                job_id, message_queue, config.process_message_handler
            )

            if not process.is_alive():
                break

            with state.lock:
                stop_requested = state.stop_requested

            if stop_requested or stop_event.is_set():
                if not stop_event.is_set():
                    stop_event.set()
                if stop_requested_at is None:
                    stop_requested_at = monotonic()
                elif (
                    monotonic() - stop_requested_at
                    > config.process_stop_timeout_seconds
                ):
                    self._logger.warning(
                        "Forcing process shutdown for job %s after timeout", job_id
                    )
                    self.terminate_process_tree(process.pid)
                    break

        self.consume_process_messages(
            job_id, message_queue, config.process_message_handler
        )

        process.join(timeout=1.0)
        if process.is_alive():
            self.terminate_process_tree(process.pid)
            process.join(timeout=1.0)

        result: dict[str, Any] | None = None
        error: str | None = None
        try:
            payload = result_queue.get_nowait()
            if isinstance(payload, dict) and payload.get("status") == "error":
                error = payload.get("error")
            elif isinstance(payload, dict) and payload.get("status") == "success":
                result = payload.get("result")
            elif isinstance(payload, dict):
                result = payload
            else:
                result = {"result": payload}
        except queue.Empty:
            exit_code = process.exitcode
            if exit_code not in (None, 0):
                error = f"Process exited with code {exit_code}"

        with state.lock:
            stop_requested = state.stop_requested

        if stop_requested:
            self.finalize_job(job_id, "cancelled", None, None)
        elif error:
            self.finalize_job(job_id, "failed", None, error)
        else:
            final_result = (
                result
                if result is None or isinstance(result, dict)
                else {"result": result}
            )
            with state.lock:
                merged = {**(state.result or {}), **(final_result or {})}
            self.finalize_job(job_id, "completed", merged if merged else None, None)
            self._logger.info("Job %s completed successfully", job_id)

        with self.lock:
            self.processes.pop(job_id, None)
            self.job_configs.pop(job_id, None)

        result_queue.close()
        message_queue.close()
        result_queue.join_thread()
        message_queue.join_thread()

    # -------------------------------------------------------------------------
    def _run_job(
        self,
        job_id: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
            config = self.job_configs.get(job_id)
        if state is None or config is None:
            return
        if config.run_mode == "process":
            self.run_process_job(job_id, runner, args, kwargs, config)
            return

        try:
            result = runner(*args, **kwargs)
            if state.stop_requested:
                self.finalize_job(job_id, "cancelled", None, None)
            else:
                final_result = (
                    result
                    if result is None or isinstance(result, dict)
                    else {"result": result}
                )
                with state.lock:
                    merged = {**(state.result or {}), **(final_result or {})}
                self.finalize_job(job_id, "completed", merged if merged else None, None)
                self._logger.info("Job %s completed successfully", job_id)
        except Exception as exc:  # noqa: BLE001
            error_msg = format_error_message(exc)
            self.finalize_job(job_id, "failed", None, error_msg)
            self._logger.error("Job %s failed: %s", job_id, error_msg)
            self._logger.debug("Job %s error details", job_id, exc_info=True)
        finally:
            with self.lock:
                self.job_configs.pop(job_id, None)

###############################################################################
def format_error_message(exc: Exception) -> str:
    messages: list[str] = []
    for candidate in (exc, getattr(exc, "orig", None), getattr(exc, "__cause__", None)):
        if candidate is None:
            continue
        text = normalize_error_text(str(candidate)).strip()
        if text:
            messages.append(text)
    if not messages:
        return "Unknown error"

    combined = "\n".join(messages)
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    if not lines:
        return "Unknown error"

    detail_line = next(
        (line for line in lines if line.lower().startswith("detail:")), None
    )
    if detail_line:
        message = f"{lines[0]} | {detail_line}"
    else:
        message = lines[0]
    return message[:500]


__all__ = ["JobManager", "format_error_message", "run_process_runner"]

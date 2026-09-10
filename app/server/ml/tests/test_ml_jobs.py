from __future__ import annotations

import multiprocessing
import time
from typing import Any

import pytest

from adsmod_ml.services.jobs import JobManager


###############################################################################
def _successful_thread_job() -> dict[str, str]:
    return {"value": "thread"}


###############################################################################
def _failing_thread_job() -> dict[str, str]:
    raise RuntimeError("thread failed")


###############################################################################
def _successful_process_job(stop_event: Any) -> dict[str, str]:
    del stop_event
    return {"value": "process"}


###############################################################################
def _cancellable_process_job(stop_event: Any) -> dict[str, str]:
    while not stop_event.is_set():
        time.sleep(0.01)
    return {"value": "cancelled"}


###############################################################################
def _wait_for_terminal(manager: JobManager, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        status = manager.get_job_status(job_id)
        if status and status["status"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not reach a terminal state.")


###############################################################################
def _wait_for_cleanup(manager: JobManager, job_id: str) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        with manager.lock:
            clean = (
                job_id not in manager.threads
                and job_id not in manager.processes
                and job_id not in manager.job_configs
            )
        if clean:
            return
        time.sleep(0.02)
    raise AssertionError(f"Execution bookkeeping for {job_id} was not released.")


###############################################################################
def test_thread_success_releases_execution_bookkeeping() -> None:
    manager = JobManager()
    job_id = manager.start_job("test", _successful_thread_job)

    status = _wait_for_terminal(manager, job_id)
    _wait_for_cleanup(manager, job_id)

    assert status["status"] == "completed"
    assert status["result"] == {"value": "thread"}
    assert job_id in manager.jobs


###############################################################################
def test_thread_exception_releases_execution_bookkeeping() -> None:
    manager = JobManager()
    job_id = manager.start_job("test", _failing_thread_job)

    status = _wait_for_terminal(manager, job_id)
    _wait_for_cleanup(manager, job_id)

    assert status["status"] == "failed"
    assert status["error"] == "thread failed"
    assert job_id in manager.jobs


###############################################################################
def test_process_start_failure_is_failed_and_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    context = multiprocessing.get_context("spawn")
    process_type = type(context.Process(target=_successful_process_job, args=(None,)))

    def fail_start(process: Any) -> None:
        del process
        raise RuntimeError("process start failed")

    monkeypatch.setattr(process_type, "start", fail_start)
    manager = JobManager()
    job_id = manager.start_job("test", _successful_process_job, run_mode="process")

    status = _wait_for_terminal(manager, job_id)
    _wait_for_cleanup(manager, job_id)

    assert status["status"] == "failed"
    assert status["error"] == "process start failed"
    assert manager.processes == {}
    assert job_id in manager.jobs


###############################################################################
def test_process_monitor_failure_is_failed_and_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_monitor(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("process monitor failed")

    manager = JobManager()
    monkeypatch.setattr(manager, "consume_process_messages", fail_monitor)
    job_id = manager.start_job("test", _cancellable_process_job, run_mode="process")

    status = _wait_for_terminal(manager, job_id)
    _wait_for_cleanup(manager, job_id)

    assert status["status"] == "failed"
    assert status["error"] == "process monitor failed"
    assert manager.processes == {}


###############################################################################
def test_process_cancellation_preserves_cancelled_status() -> None:
    manager = JobManager()
    job_id = manager.start_job("test", _cancellable_process_job, run_mode="process")

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        with manager.lock:
            if job_id in manager.processes:
                break
        time.sleep(0.02)
    assert manager.cancel_job(job_id) is True

    status = _wait_for_terminal(manager, job_id)
    _wait_for_cleanup(manager, job_id)

    assert status["status"] == "cancelled"
    assert manager.processes == {}

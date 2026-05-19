"""
In-memory job queue for tracking file processing jobs.

Each job progresses through stages, updating a shared dict that the
/job/{id}/status endpoint polls. Thread-safe via asyncio.Lock.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class JobStatus(str, Enum):
    QUEUED      = "queued"
    RUNNING     = "running"
    DONE        = "done"
    ERROR       = "error"


@dataclass
class Job:
    id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0           # 0.0 – 1.0
    stage_label: str = "Queued…"
    eta_seconds: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> dict:
        elapsed = (time.time() - self.started_at) if self.started_at else 0
        if self.progress > 0.01 and self.progress < 1.0 and elapsed > 0:
            rate = self.progress / elapsed
            remaining = (1.0 - self.progress) / rate
        else:
            remaining = None
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "stage_label": self.stage_label,
            "eta_seconds": round(remaining, 0) if remaining else None,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobQueue:
    """Singleton in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    def create_job(self, filename: str) -> Job:
        job = Job(id=str(uuid.uuid4()), filename=filename)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        progress: float,
        stage_label: str,
        status: JobStatus = JobStatus.RUNNING,
    ) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.progress = progress
            job.stage_label = stage_label
            job.status = status
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = time.time()

    def complete(self, job_id: str, result: dict) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.DONE
            job.progress = 1.0
            job.stage_label = "Complete"
            job.result = result
            job.finished_at = time.time()

    def fail(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.ERROR
            job.stage_label = "Error"
            job.error = error
            job.finished_at = time.time()

    def make_progress_callback(self, job_id: str) -> Callable:
        """Returns a callback(progress_float, label) for use in long tasks."""
        def callback(progress: float, label: str) -> None:
            self.update(job_id, progress, label, JobStatus.RUNNING)
        return callback


# Global singleton
job_queue = JobQueue()

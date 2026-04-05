"""Simple sequential generation job queue."""
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum
import threading
import queue
import uuid

from core.logger import get_logger
logger = get_logger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str = ""
    params: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    progress: int = 0


class GenerationQueue:
    """Thread-safe sequential job queue."""
    def __init__(self):
        self._jobs: list = []
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._paused = False
        self._running = False
        self.on_job_started: Optional[Callable] = None
        self.on_job_done: Optional[Callable] = None
        self.on_queue_empty: Optional[Callable] = None

    def add_job(self, job: GenerationJob) -> str:
        with self._lock:
            self._jobs.append(job)
        self._queue.put(job.job_id)
        return job.job_id

    def cancel_job(self, job_id: str):
        with self._lock:
            for job in self._jobs:
                if job.job_id == job_id and job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def clear(self):
        with self._lock:
            for j in self._jobs:
                if j.status == JobStatus.PENDING:
                    j.status = JobStatus.CANCELLED

    def get_jobs(self) -> list:
        with self._lock:
            return list(self._jobs)

    def get_pending_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs if j.status == JobStatus.PENDING)

    @property
    def is_paused(self) -> bool:
        return self._paused

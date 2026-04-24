"""The JobManager singleton class manages all job states."""

from __future__ import annotations

import time
import asyncio
import logging
import traceback

from fastapi import HTTPException

from zeus.optimizer.pipeline_frequency.common import (
    JobInfo,
    PFOServerSettings,
    FrequencySchedule,
    ProfilingResult,
    RankInfo,
    save_prof,
    save_sched,
    save_ranks,
)
from zeus.utils.async_utils import create_task

GLOBAL_JOB_MANAGER: JobManager | None = None

logger = logging.getLogger(__name__)


class JobManager:
    """A singleton class that manages all states."""

    def __init__(self, pfo_settings: PFOServerSettings) -> None:
        """Initialize the job manager."""
        raise NotImplementedError

    def register_job(self, job_info: JobInfo) -> None:
        """Prepare internal state for a new job.

        This method will be invoked exactly once by the global rank 0 (master) process.
        """
        pass

    def register_rank(self, job_id: str, rank_info: RankInfo) -> None:
        """Register rank-specific information for an already registered job.

        This method will be invoked `world_size` number of times (once per rank).
        """
        pass

    async def get_frequency_schedule(self, job_id: str, rank: int) -> FrequencySchedule:
        """Get the next frequency schedule for a rank.

        This method will be called `world_size` number of times (once per rank).
        All ranks will block on this method untill everyone reports their
        profiling results and calls this method.

        When an internal scheduler error happened at any point of servicing the
        job, clients will be notified through this API with a 500 Internal Error.
        """
        pass

    def report_profiling_result(self, job_id: str, result: ProfilingResult) -> None:
        """Send the profiling result to the job task and immediately return.

        This method will be called `world_size` number of times - one for each rank.
        """
        pass

    async def _cleanup_task(
        self,
        cleanup_period: int,
        max_idle_time: int,
    ) -> None:
        """Periodically evict job states.

        Args:
            cleanup_period: How often to run the cleanup task, in seconds.
            max_idle_time: Maximum amount of time a job can be idle for, in seconds.
        """
        raise NotImplementedError

    async def _job_task(self, job_id: str, dump_data: bool) -> None:
        """Coalese requests and responses of each rank and interface with the scheduler."""
        pass


def init_global_job_manager(pfo_settings: PFOServerSettings) -> None:
    """Instantiate the global singleton `JobManager`."""
    raise NotImplementedError


def get_global_job_manager() -> JobManager:
    """Fetch the global singleton `JobManager`."""
    raise NotImplementedError

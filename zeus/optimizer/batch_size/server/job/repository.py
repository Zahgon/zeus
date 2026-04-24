"""Repository for manipulating Job table."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio.session import AsyncSession

from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import JobTable
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOServiceBadOperationError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.job.commands import (
    CreateJob,
    UpdateExpDefaultBs,
    UpdateGeneratorState,
    UpdateJobMinCost,
    UpdateJobStage,
)
from zeus.optimizer.batch_size.server.job.models import JobState

logger = logging.getLogger(__name__)


class JobStateRepository(DatabaseRepository):
    """Repository that provides basic interfaces to interact with Job table."""

    def __init__(self, session: AsyncSession):
        """Set db session and intialize job. We are working with only one job per session."""
        raise NotImplementedError

    async def get_job(self, job_id: str) -> JobState | None:
        """Get job State, which includes jobSpec + batch_sizes(list[int]), without specific states of each batch_size.

        Args:
            job_id: Job id.

        Returns:
            set fetched_job and return `JobState` if we found a job, unless return None.
        """
        pass

    def get_job_from_session(self, job_id: str) -> JobState | None:
        """Get a job that was fetched from this session.

        Args:
            job_id: Job id.

        Returns:
            Corresponding `JobState`. If none was found, return None.
        """
        pass

    def update_exp_default_bs(self, updated_bs: UpdateExpDefaultBs) -> None:
        """Update exploration default batch size on fetched job.

        Args:
            updated_bs: Job Id and new batch size.
        """
        pass

    def update_stage(self, updated_stage: UpdateJobStage) -> None:
        """Update stage on fetched job.

        Args:
            updated_stage: Job Id and new stage.
        """
        pass

    def update_min(self, updated_min: UpdateJobMinCost) -> None:
        """Update exploration min training cost and corresponding batch size on fetched job.

        Args:
            updated_min: Job Id, new min cost and batch size.
        """
        pass

    def update_generator_state(self, updated_state: UpdateGeneratorState) -> None:
        """Update generator state on fetched job.

        Args:
            updated_state: Job Id and new generator state.
        """
        pass

    def create_job(self, new_job: CreateJob) -> None:
        """Create a new job by adding a new job to the session.

        Args:
            new_job: Job configuration for a new job.
        """
        pass

    def check_job_fetched(self, job_id: str) -> bool:
        """Check if this job is already fetched before.

        Args:
            job_id: Job id.

        Returns:
            True if this job was fetched and in session. Otherwise, return false.
        """
        pass

    async def delete_job(self, job_id: str) -> bool:
        """Delete the job of a given job_Id.

        Args:
            job_id: Job id.

        Returns:
            True if the job got deleted.
        """
        pass

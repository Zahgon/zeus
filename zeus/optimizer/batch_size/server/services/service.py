"""Zeus batch size optimizer service layer."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
from numpy.random import Generator as np_Generator
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateConcurrentTrial,
    CreateExplorationTrial,
    CreateMabTrial,
    CreateTrial,
    ReadTrial,
    UpdateTrial,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerJob,
    GaussianTsArmState,
    Trial,
    TrialResultsPerBs,
)
from zeus.optimizer.batch_size.server.batch_size_state.repository import (
    BatchSizeStateRepository,
)
from zeus.optimizer.batch_size.server.database.schema import TrialStatus, TrialType
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
from zeus.optimizer.batch_size.server.job.repository import JobStateRepository
from zeus.optimizer.batch_size.server.services.commands import (
    GetNormal,
    GetRandomChoices,
    UpdateArm,
)
from zeus.utils.metric import zeus_cost


class ZeusService:
    """Zeus Service that interacts with database using repository.

    Provides application layer methods to communicate with database.
    Each method is one or more number of db operations that have to be done at the same time.
    """

    def __init__(self, db_session: AsyncSession):
        """Set up repositories to use to talk to database."""
        raise NotImplementedError

    async def get_arms(self, job_id: str) -> list[GaussianTsArmState]:
        """Get GaussianTs arm states for all arms(job_id, batch size).

        Args:
            job_id: Job id

        Returns:
            list of arms
        """
        pass

    async def get_arm(self, bs: BatchSizeBase) -> GaussianTsArmState | None:
        """Get arm state for one arm.

        Args:
            bs: (job_id, batch size) pair that represents one arm

        Returns:
            Result arm state or None if we cannot find that arm
        """
        pass

    async def get_explorations_of_job(self, job_id: str) -> ExplorationsPerJob:
        """Get all explorations we have done for that job.

        Args:
            job_id: Job id

        Returns:
            list of explorations per each batch size
        """
        pass

    def update_trial(self, updated_trial: UpdateTrial) -> None:
        """Update trial.

        (1) update the corresponding trial.
        (2) we update the min training cost observed so far if we have to.

        Args:
            updated_trial: Result of training that batch size

        Raises:
            [`ZeusBSOServiceBadOperationError`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOServiceBadOperationError]: When we didn't fetch the job or trial during this session. This operation should have
                    fetched the job and trial first. Also, check if trial type is matching with fetched trial's type.
        """
        pass

    def update_arm_state(
        self,
        arm: UpdateArm,
    ) -> None:
        """Update arm state.

        Args:
            arm: Updated arm state.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job or trial during this session. This operation should have
                    fetched the job and trial first. Also, check if trial type is matching with fetched trial's type.
        """
        pass

    def update_exp_default_bs(self, updated_default_bs: UpdateExpDefaultBs) -> None:
        """Update the default batch size for exploration.

        Args:
            updated_default_bs: Job Id and new default batch size

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        pass

    async def create_trial(self, trial: CreateExplorationTrial | CreateMabTrial | CreateConcurrentTrial) -> ReadTrial:
        """Create a new trial.

        Args:
            trial: New trial to create.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        pass

    def get_random_choices(self, choice: GetRandomChoices) -> np.ndarray[Any, Any]:
        """Get randome choices based on job's seed.

        If seed is not None (set by the user) we get the random choices from the generator that is stored in the database.
        Otherwise, we get random choices based on random seed.

        Args:
            choice: Job id and list of choices

        Returns:
            reuslt random choices

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        pass

    def get_normal(self, arg: GetNormal) -> float:
        """Sample from normal distribution and update the generator state if seed was set.

        Args:
            arg: args for `numpy.random.normal`, which is loc(mean of distribution) and scale(stdev of distribution)

        Returns:
            Drawn sample.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        pass

    async def get_job(self, job_id: str) -> JobState | None:
        """Get job from database.

        Args:
            job_id: Job Id

        Returns:
            JobState if we found one, None if we couldn't find a job matching the job id.
        """
        pass

    async def get_trial(self, trial: ReadTrial) -> Trial | None:
        """Get a trial from database.

        Args:
            trial: (Job Id, batch size, trial_number) triplet.

        Returns:
            Trial if we found one, None if we couldn't find a job matching trial.
        """
        pass

    def create_job(self, new_job: CreateJob) -> None:
        """Create a new job.

        Args:
            new_job: Configuration of a new job
        """
        pass

    async def get_trial_results_of_bs(self, bs: BatchSizeBase) -> TrialResultsPerBs:
        """Load window size amount of results for a given batch size. If window size <= 0, load all of them.

        Args:
            bs: (job_id, batch size) pair.

        Returns:
            list of windowed measurements in descending order for that (job_id, batch size)

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        pass

    def create_arms(self, new_arms: list[GaussianTsArmState]) -> None:
        """Create GuassianTs arms for the job.

        Args:
            new_arms: List of new arm states

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        pass

    def update_job_stage(self, updated_stage: UpdateJobStage) -> None:
        """Update the job stage (Pruning -> MAB).

        Args:
            updated_stage: Updated stage.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        pass

    async def delete_job(self, job_id: str) -> bool:
        """Delete the job.

        Args:
            job_id: ID of the job.

        Returns:
            True if the job is deleted. False if none was deleted
        """
        pass

    def _get_generator(self, job_id: str) -> tuple[np_Generator, bool]:
        """Get generator based on job_id. If mab_seed is not none, we should update the state after using generator.

        Returns:
            Tuple of [Generator, if we should update state]
        """
        pass

    def _get_job(self, job_id: str) -> JobState:
        """Get the job from the session. If we couldn't find the job, raise a `ZeusBSOServiceBadOperationError`."""
        pass

    def _get_trial(self, trial: ReadTrial) -> Trial:
        """Get the job from the session. If we couldn't find the trial, raise a `ZeusBSOServiceBadOperationError`."""
        pass

    def _check_job_fetched(self, job_id: str) -> None:
        """Check if we fetched the job in the current session. If we didn't raise a `ZeusBSOServiceBadOperationError`."""
        pass

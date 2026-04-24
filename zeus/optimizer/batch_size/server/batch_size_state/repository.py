"""Repository for batch size states(Trial, Gaussian Ts arm state)."""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio.session import AsyncSession

from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateTrial,
    ReadTrial,
    UpdateTrial,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerJob,
    GaussianTsArmState,
    Trial,
    TrialResult,
    TrialResultsPerBs,
)
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import (
    GaussianTsArmStateTable,
    TrialStatus,
    TrialTable,
    TrialType,
)
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOValueError

logger = logging.getLogger(__name__)


class BatchSizeStateRepository(DatabaseRepository):
    """Repository for handling batch size related operations."""

    def __init__(self, session: AsyncSession):
        """Set db session and intialize fetched trial. We are only updating one trial per session."""
        raise NotImplementedError

    async def get_next_trial_number(self, job_id: str) -> int:
        """Get next trial number of a given job. Trial number starts from 1 and increase by 1 at a time."""
        pass

    async def get_trial_results_of_bs(self, batch_size: BatchSizeBase, window_size: int) -> TrialResultsPerBs:
        """Load window size amount of results for a given batch size. If window size <= 0, load all of them.

        From all trials, we filter succeeded one since failed/dispatched ones doesn't have a valid result.

        Args:
            batch_size (BatchSizeBase): The batch size object.
            window_size (int): The size of the measurement window.

        Returns:
            TrialResultsPerBs: trial results for the given batch size.
        """
        pass

    async def get_arms(self, job_id: str) -> list[GaussianTsArmState]:
        """Retrieve Gaussian Thompson Sampling arms for a given job.

        Args:
            job_id (str): The ID of the job.

        Returns:
            List[GaussianTsArmStateModel]: List of Gaussian Thompson Sampling arms. These arms are all "good" arms (converged during pruning stage).
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        pass

    async def get_arm(self, bs: BatchSizeBase) -> GaussianTsArmState | None:
        """Retrieve Gaussian Thompson Sampling arm for a given job id and batch size.

        Args:
            bs (BatchSizeBase): The batch size object.

        Returns:
            Optional[GaussianTsArmStateModel]: Gaussian Thompson Sampling arm if found, else None.
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        pass

    async def get_trial(self, trial: ReadTrial) -> Trial | None:
        """Get a corresponding trial.

        Args:
            trial: job_id, batch_size, trial_number.

        Returns:
            Found Trial. If none found, return None.
        """
        pass

    def get_trial_from_session(self, trial: ReadTrial) -> Trial | None:
        """Fetch a trial from the session."""
        pass

    def create_trial(self, trial: CreateTrial) -> None:
        """Create a trial in db.

        Refer to `CreateTrial`[zeus.optimizer.batch_size.server.batch_size_state.models.CreateTrial] for attributes.

        Args:
            trial (CreateTrial): The trial to add.
        """
        pass

    def updated_current_trial(self, updated_trial: UpdateTrial) -> None:
        """Update trial in the database (report the result of trial).

        Args:
            updated_trial (UpdateTrial): The updated trial. Refer to `UpdateTrial`[zeus.optimizer.batch_size.server.batch_size_state.models.UpdateTrial] for attributes.
        """
        pass

    def create_arms(self, new_arms: list[GaussianTsArmState]) -> None:
        """Create Gaussian Thompson Sampling arms in the database.

        Args:
            new_arms (List[GaussianTsArmStateModel]): List of new arms to create.
                Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        pass

    def update_arm_state(self, updated_mab_state: GaussianTsArmState) -> None:
        """Update Gaussian Thompson Sampling arm state in db.

        Args:
            updated_mab_state (GaussianTsArmStateModel): The updated arm state.
                Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        pass

    async def get_explorations_of_job(self, job_id: str) -> ExplorationsPerJob:
        """Retrieve succeeded or ongoing explorations for a given job.

        Args:
            job_id: ID of the job

        Returns:
            ExplorationsPerJob: Explorations for the given batch size.
            Refer to `ExplorationsPerJob`[zeus.optimizer.batch_size.server.batch_size_state.models.ExplorationsPerJob] for attributes.
        """
        pass

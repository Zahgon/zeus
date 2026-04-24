"""Provides report/next_batch_size during pruning stage."""

from __future__ import annotations

import logging

from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateConcurrentTrial,
    CreateExplorationTrial,
    ReadTrial,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import ExplorationsPerJob
from zeus.optimizer.batch_size.server.database.schema import TrialStatus
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOServerRuntimeError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.job.models import JobState
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.utils.metric import zeus_cost

logger = logging.getLogger(__name__)


class PruningExploreManager:
    """Pruning manager that manges the batch size states in pruning stage."""

    def __init__(self, service: ZeusService):
        """Set up zeus service."""
        raise NotImplementedError

    async def next_batch_size(
        self,
        job: JobState,
        exploration_history: ExplorationsPerJob,
    ) -> ReadTrial | list[int]:
        """Find the next batch size to explore.

        Three cases possible.
        1. Pruninig Stage : There is a batch size that has not explored during the round.
        2. Concurrent job : There is an exploration with "Dispatched" state.
        3. Mab stage : All batch sizes have been explored and round is over.

        Args:
            job: state of the job
            exploration_history: all "succeeded" explorations that we have done for that job

        Returns:
            Return the batch size to use during Pruning stage.
            If Pruning stage was over, return None.

        Raises:
            `ZeusBSOValueError`: If the value is invalid. EX) default batch size is not in the converged batch size list.
        """
        pass

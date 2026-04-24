"""Thompson Sampling policy for Gaussian bandits. MAB related logic is implented here."""

from __future__ import annotations

import logging

import numpy as np

from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    ReadTrial,
    UpdateTrial,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerJob,
    GaussianTsArmState,
)
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOServiceBadOperationError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.job.commands import UpdateJobStage
from zeus.optimizer.batch_size.server.job.models import JobState, Stage
from zeus.optimizer.batch_size.server.services.commands import (
    GetNormal,
    GetRandomChoices,
    UpdateArm,
)
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.utils.metric import zeus_cost

logger = logging.getLogger(__name__)


class GaussianTS:
    """Thompson Sampling policy for Gaussian bandits.

    For each arm, the reward is modeled as a Gaussian distribution with
    known precision. The conjugate priors are also Gaussian distributions.
    """

    def __init__(self, service: ZeusService):
        """Set up zeus service to interact with database."""
        raise NotImplementedError

    def _fit_arm(
        self,
        bs_base: BatchSizeBase,
        prior_mean: float,
        prior_precision: float,
        rewards: np.ndarray,
    ) -> GaussianTsArmState:
        """Update the parameter distribution for one arm.

        Reference: <https://en.wikipedia.org/wiki/Conjugate_prior>

        Args:
            bs_base: job id and batch size tha represents this arm
            prior_mean: Mean of the belief prior distribution.
            prior_precision: Precision of the belief prior distribution.
            rewards: Array of rewards observed by pulling that arm.

        Returns:
            Updated arm state
        """
        pass

    def predict(
        self,
        job_id: str,
        prior_precision: float,
        num_exploration: int,
        arms: list[GaussianTsArmState],
    ) -> int:
        """Return the arm with the largest sampled expected reward.

        Args:
            job_id: job id
            prior_precision: Precision of the belief prior distribution.
            num_exploration: How many static explorations to run when no observations are available.
            arms: list of arms

        Returns:
            batch size to use
        """
        pass

    async def construct_mab(
        self, job: JobState, evidence: ExplorationsPerJob, good_bs: list[int]
    ) -> list[GaussianTsArmState]:
        """Construct arms and initialize them.

        Args:
            job: state of job.
            evidence: Completed explorations. We create arms based on the explorations we have done during pruning stage.
            good_bs: Converged batch size list.

        Returns:
            list of arms that we created

        Raises:
            `ValueError`: If exploration states is invalid (ex. number of pruning rounds doesn't corresponds)
            `ZeusBSOValueError`: No converged batch sizes from pruning stage.
        """
        pass

    async def report(self, job: JobState, trial_result: UpdateTrial) -> None:
        """Based on the measurement, update the arm state.

        Args:
            job: state of the job
            trial_result: result of training (job id, batch_size, trial_number)

        Raises:
            `ZeusBSOValueError`: When the arm (job id, batch_size) doesn't exist
        """
        pass

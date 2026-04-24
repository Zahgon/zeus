"""Batch size optimizer top-most layer that provides register/report/predict."""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np

from zeus.optimizer.batch_size.common import (
    JobSpecFromClient,
    TrialId,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateMabTrial,
    ReadTrial,
    UpdateTrial,
)
from zeus.optimizer.batch_size.server.database.schema import TrialStatus
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOJobConfigMismatchError,
    ZeusBSOServerNotFoundError,
    ZeusBSOServiceBadOperationError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.explorer import PruningExploreManager
from zeus.optimizer.batch_size.server.job.commands import CreateJob
from zeus.optimizer.batch_size.server.job.models import Stage
from zeus.optimizer.batch_size.server.mab import GaussianTS
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.utils.metric import zeus_cost

logger = logging.getLogger(__name__)


class ZeusBatchSizeOptimizer:
    """Batch size optimizer server. Manages which stage the job is in and call corresponding manager (pruning or mab)."""

    def __init__(self, service: ZeusService) -> None:
        """Initialize the server. Set the service, pruning manager, and mab.

        Args:
            service: ZeusService for interacting with database
        """
        raise NotImplementedError

    async def register_job(self, job: JobSpecFromClient) -> bool:
        """Register a job that user submitted. If the job id already exists, check if it is identical with previously registered configuration.

        Args:
            job: job configuration

        Returns:
            True if a job is regiested, False if a job already exists and identical with previous configuration

        Raises:
            [`ZeusBSOJobConfigMismatchError`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOJobConfigMismatchError]: In the case of existing job, if job configuration doesn't match with previously registered config
        """
        pass

    async def predict(self, job_id: str) -> TrialId:
        """Return a batch size to use.

        Args:
            job_id: Id of job

        Returns:
            batch size to use

        Raises:
            [`ZeusBSOValueError`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOValueError]: If the job id is unknown, or creating a mab failed due to no converged batch size
        """
        pass

    async def report(self, result: TrainingResult) -> ReportResponse:
        """Report the training result. Stop train if the train is converged or reached max epochs or reached early stop threshold. Otherwise, keep training.

        Args:
            result: result of training [`TrainingResult`][zeus.optimizer.batch_size.common.TrainingResult].

        Returns:
            Decision on training [`ReportResponse`][zeus.optimizer.batch_size.common.ReportResponse].
        """
        pass

    async def end_trial(self, trial_id: TrialId) -> None:
        """Mark the trial as finished. If status is still `Dispatched` make the trial as `Failed`.

        Args:
            trial_id: Unique identifier of trial

        Raises:
            [`ZeusBSOServerNotFound`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOServerNotFound]: If there is no corresponding trial.
        """
        pass

    async def delete_job(self, job_id: str) -> None:
        """Delete a job.

        Args:
            job_id: ID of a job.

        Returns:
            True if the job is deleted. False if none was deleted
        """
        pass

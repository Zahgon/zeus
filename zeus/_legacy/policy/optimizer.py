"""Implementations of various optimization policies.

[`JITPowerLimitOptimizer`][zeus._legacy.policy.optimizer.JITPowerLimitOptimizer] and
[`PruningGTSBatchSizeOptimizer`][zeus._legacy.policy.optimizer.PruningGTSBatchSizeOptimizer]
are the implementations used in Zeus's publication.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Generator

import numpy as np

from zeus._legacy.job import Job
from zeus._legacy.policy.interface import BatchSizeOptimizer, PowerLimitOptimizer
from zeus._legacy.policy.mab import GaussianTS


class GTSBatchSizeOptimizer(BatchSizeOptimizer):
    """One Gaussian Thompson Sampling MAB for each job."""

    # ruff: noqa: D417
    def __init__(
        self,
        learn_reward_precision: bool,
        reward_precision: float = 0.0,
        prior_mean: float = 0.0,
        prior_precision: float = 0.0,
        num_exploration: int = 1,
        seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        """Initialze the optimizer.

        Refer to the constructor of [`GaussianTS`][zeus._legacy.policy.mab.GaussianTS]
        for descriptions of other arguments.

        Args:
            learn_reward_precision: Whether to learn the reward precision of
                each arm as we accumulate observations.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the batch size optimizer."""
        pass

    def register_job(self, job: Job, batch_sizes: list[int]) -> None:
        """Instantiate a new GaussianTS MAB for the new job."""
        pass

    def predict(self, job: Job) -> int:
        """Return the batch size to use for the job."""
        pass

    def observe(self, job: Job, batch_size: int, cost: float, converged: bool | None = None) -> None:
        """Learn from the cost of using the given batch size for the job."""
        raise NotImplementedError


class PruningExploreManager:
    """Helper class that generates batch sizes to explore and prune."""

    def __init__(
        self,
        batch_sizes: list[int],
        default: int,
        num_pruning_rounds: int = 2,
    ) -> None:
        """Initialze the object.

        Args:
            batch_sizes: The initial set of batch sizes to prune from.
            default: The default batch size (b0) to begin exploration from.
            num_pruning_rounds: How many rounds to run pruning.
        """
        raise NotImplementedError

    def _exploration_engine(
        self,
    ) -> Generator[int | None, tuple[int, float, bool], list[int]]:
        """Drive pruning exploration.

        Yields the batch size to be explored.
        The caller should `send` a tuple of (explored batch size, cost, whether reached).
        As a safety measure, the explored batch size must match the most recently yielded
        batch size, and otherwise a `RuntimeError` is raised.
        Finally, when exploration is over, returns a sorted list of batch sizes that
        survived pruning.
        """
        raise NotImplementedError

    def next_batch_size(self) -> int:
        """Return the next batch size to explore.

        Raises `StopIteration` when pruning exploration phase is over.
        The exception instance contains the final set of batch sizes to consider.
        Access it through `exception.value`.
        """
        pass

    def report_batch_size_result(self, batch_size: int, cost: float, reached: bool) -> None:
        """Report whether the previous batch size reached the target metric.

        Args:
            batch_size: The batch size which this cost observation is from.
            cost: The energy-time cost of running the job with this batch size.
            reached: Whether the job reached the target metric.
        """
        raise NotImplementedError


class PruningGTSBatchSizeOptimizer(BatchSizeOptimizer):
    """One Gaussian Thompson Sampling MAB for each job with double pruning exploration."""

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_precision: float = 0.0,
        window_size: int = 0,
        concurrency: bool = False,
        seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        """Initialze the optimizer.

        Refer to the constructor of [`GaussianTS`][zeus._legacy.policy.mab.GaussianTS]
        for descriptions of other arguments.

        Args:
            window_size: Size of the window for the MAB (for drift handling).
            concurrency: Whether to support concurrent job submissions.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the batch size optimizer."""
        pass

    def register_job(self, job: Job, batch_sizes: list[int]) -> None:
        """Register the job."""
        pass

    def predict(self, job: Job) -> int:
        """Return the batch size to use for the job."""
        pass

    def observe(self, job: Job, batch_size: int, cost: float, converged: bool | None = None) -> None:
        """Learn from the cost of using the given batch size for the job."""
        raise NotImplementedError

    def _get_history_for_bs(self, job: Job, batch_size: int) -> list[float]:
        """Return the windowed history for the given job's batch size."""
        raise NotImplementedError

    def _construct_mab(self, job: Job, batch_sizes: list[int]) -> None:
        """When exploration is over, this method is called to construct and learn GTS."""
        pass


class JITPowerLimitOptimizer(PowerLimitOptimizer):
    """Returns the best power limit to use for the job & batch size."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the object."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the power limit optimizer."""
        pass

    def predict(self, job: Job, batch_size: int) -> int | None:
        """Return the best power limit for the job, or None if unknown."""
        pass

    def observe(self, job: Job, batch_size: int, power_limit: int, cost: float) -> None:
        """Learn from the cost of using the given knobs for the job."""
        raise NotImplementedError

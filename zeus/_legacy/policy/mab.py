"""Multi-Armed Bandit implementations."""

from __future__ import annotations

import warnings

import numpy as np


class GaussianTS:
    """Thompson Sampling policy for Gaussian bandits.

    For each arm, the reward is modeled as a Gaussian distribution with
    known precision. The conjugate priors are also Gaussian distributions.
    """

    def __init__(
        self,
        arms: list[int],
        reward_precision: list[float] | float,
        prior_mean: float = 0.0,
        prior_precision: float = 0.0,
        num_exploration: int = 1,
        seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        """Initialze the object.

        Args:
            arms: Bandit arm values to use.
            reward_precision: Precision (inverse variance) of the reward distribution.
                Pass in a list of `float`s to set the reward precision differently for
                each arm.
            prior_mean: Mean of the belief prior distribution.
            prior_precision: Precision of the belief prior distribution.
            num_exploration: How many static explorations to run when no observations
                are available.
            seed: The random seed to use.
            verbose: Whether to print out what's going on.
        """
        raise NotImplementedError

    def fit(
        self,
        decisions: list[int] | np.ndarray,
        rewards: list[float] | np.ndarray,
        reset: bool,
    ) -> None:
        """Fit the bandit on the given list of observations.

        Args:
            decisions: A list of arms chosen.
            rewards: A list of rewards that resulted from choosing the arms in `decisions`.
            reset: Whether to reset all arms.
        """
        raise NotImplementedError

    def fit_arm(self, arm: int, rewards: np.ndarray, reset: bool) -> None:
        """Update the parameter distribution for one arm.

        Reference: <https://en.wikipedia.org/wiki/Conjugate_prior>

        Args:
            arm: Arm to fit.
            rewards: Array of rewards observed by pulling that arm.
            reset: Whether to reset the parameters of the arm before fitting.
        """
        raise NotImplementedError

    def predict(self) -> int:
        """Return the arm with the largest sampled expected reward."""
        pass

    def predict_expectations(self) -> dict[int, float]:
        """Sample the expected reward for each arm.

        Assumes that each arm has been explored at least once. Otherwise,
        a value will be sampled from the prior.

        Returns:
            A mapping from every arm to their sampled expected reward.
        """
        pass

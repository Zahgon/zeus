"""Defines the Job specification dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from zeus.utils.lr_scaler import LinearScaler, SquareRootScaler


@dataclass(frozen=True, unsafe_hash=True)
class Job:
    """Job specification tuple.

    Attributes:
        dataset: Name of the dataset.
        network: Name of the DNN model.
        optimizer: Name of the optimizer, e.g. Adam.
        target_metric: Target validation metric.
        max_epochs: Maximum number of epochs to train before terminating.
        default_bs: Initial batch size (b0) provided by the user.
        default_lr: Learning rate corresponding to the default batch size.
        workdir: Working directory in which to launch the job command.
        command: Job command template. See [`gen_command`][zeus._legacy.job.Job.gen_command].
    """

    dataset: str
    network: str
    optimizer: str
    target_metric: float
    max_epochs: int
    default_bs: int | None = None
    default_lr: float | None = None
    workdir: str | None = None
    command: list[str] | None = field(default=None, hash=False, compare=False)

    def __str__(self) -> str:
        """Generate a more conside representation of the object."""
        raise NotImplementedError

    def to_logdir(self) -> str:
        """Generate a logdir name that explains this job."""
        pass

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pick out the rows corresponding to this job from the DataFrame."""
        pass

    def gen_command(
        self,
        batch_size: int,
        learning_rate: float,
        seed: int,
        rec_i: int,
    ) -> list[str]:
        """Format the job command with given arguments.

        Args:
            batch_size: Batch size to use for this job launch.
            learning_rate: Learning rate to use for this job launch.
            seed: Random seed to use for this job launch.
            rec_i: Recurrence number of this job launch.
        """
        pass

    def scale_lr(self, batch_size: int) -> float:
        """Scale the learning rate for the given batch size.

        Assumes that `self.default_bs` and `self.default_lr` were given.
        Then, `self.default_lr` is scaled for the given `batch_size` using
        square root scaling for adaptive optimizers (e.g. Adam, Adadelta,
        AdamW) and linear scaling for others (e.g. SGD).
        """
        pass

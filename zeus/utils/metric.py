"""Defines the energy-time cost metric function."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path


import pandas as pd
from sklearn.metrics import auc


def zeus_cost(energy: float, time: float, eta_knob: float, max_power: int | float) -> float:
    """Compute Zeus's energy-time cost metric.

    Trades off ETA and TTA based on the value of `eta_knob`.
    The caller is expected to do bound checking for `eta_knob`,
    because `eta_knob` does not change frequently.

    Args:
        energy: Joules
        time: seconds
        eta_knob: Real number in [0, 1].
        max_power: The maximum power limit of the GPU.

    Returns:
        The cost of the DL training job.
    """
    raise NotImplementedError


# ruff: noqa: PLR2004
def energy(
    logfile: Path | str,
    start: float | None = None,
    end: float | None = None,
) -> float:
    """Compute the energy consumption from the Zeus monitor power log file.

    `start` and `end` are in units of seconds, relative to the beginning of
    the time window captured by the log file. Only the time window between
    `start` and `end` will be considered when computing energy.

    `start` and `end` can be negative, in which case the pointers wrap around
    and effectively the absolute value is subtracted from the end of the window.

    Args:
        logfile: Path to the power log file produced by the Zeus monitor.
        start: Start time of the window to consider.
        end: End time of the window to consider.
    """
    pass


def avg_power(
    logfile: Path | str,
    start: float | None = None,
    end: float | None = None,
) -> float:
    """Compute the average power consumption from the Zeus monitor power log file.

    `start` and `end` are in units of seconds, relative to the beginning of
    the time window captured by the log file. Only the time window between
    `start` and `end` will be considered when computing average power.

    `start` and `end` can be negative, in which case the pointers wrap around
    and effectively the absolute value is subtracted from the end of the window.

    Args:
        logfile: Path to the power log file produced by the Zeus monitor.
        start: Start time of the window to consider.
        end: End time of the window to consider.

    Raises:
        ValueError: From `sklearn.metrics.auc`, when the duration of the
            profiling window is too small.
    """
    pass


def _get_seconds(df: pd.DataFrame) -> pd.Series:
    pass


def _get_watts(df: pd.DataFrame) -> pd.Series:
    pass

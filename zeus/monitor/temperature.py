"""Monitor the temperature of GPUs."""

from __future__ import annotations

import bisect
import collections
import logging
import multiprocessing as mp
import weakref
from time import time, sleep
from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING

from zeus.device.gpu.common import ZeusGPUNotSupportedError
from zeus.device import get_gpus
from zeus.utils.multiprocessing import warn_if_global_in_subprocess

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass
    from multiprocessing.context import SpawnProcess

logger = logging.getLogger(__name__)


def _cleanup_temperature_process(
    stop_event: EventClass,
    process: SpawnProcess,
) -> None:
    """Idempotent cleanup function for temperature monitoring process."""
    pass


@dataclass
class TemperatureSample:
    """A single temperature measurement sample."""

    timestamp: float
    gpu_index: int
    temperature_c: int


class TemperatureMonitor:
    """Monitor GPU temperature over time.

    This class provides:

    1. Continuous temperature monitoring in a background process
    2. Timeline export with deduplication
    3. Point-in-time temperature queries

    !!! Note
        The current implementation only supports cases where all GPUs are homogeneous
        (i.e., the same model).

    !!! Warning
        This monitor uses multiprocessing with the spawn start method to poll temperature
        in a background process. Spawned processes re-import your main module, so keep
        heavy setup under `if __name__ == "__main__":` or inside functions.
        See also the "Safe importing of main module" section in the [Python documentation](
        https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        update_period: float = 1.0,
        max_samples_per_gpu: int | None = None,
    ) -> None:
        """Initialize the temperature monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the temperature monitor in seconds.
                Defaults to 1.0 second. Temperature typically doesn't change as
                rapidly as power, so a longer update period is reasonable.
            max_samples_per_gpu: Maximum number of temperature samples to keep per GPU
                in memory. If None (default), unlimited samples are kept.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the monitoring process."""
        pass

    def _process_temperature_queue_data(self) -> None:
        """Process all pending temperature samples from the queue."""
        pass

    def get_temperature_timeline(
        self,
        gpu_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[int, list[tuple[float, int]]]:
        """Get temperature timeline for specific GPU(s).

        Args:
            gpu_index: Specific GPU index, or None for all GPUs
            start_time: Start time filter (unix timestamp)
            end_time: End time filter (unix timestamp)

        Returns:
            Dictionary mapping GPU indices to timeline data.
            Timeline data is list of (timestamp, temperature_celsius) tuples.
        """
        pass

    def get_temperature(self, time: float | None = None) -> dict[int, int] | None:
        """Get the GPU temperature at a specific time point.

        Args:
            time: Time point to get the temperature at. If None, get the temperature
                at the last recorded time point.

        Returns:
            A dictionary mapping GPU indices to the temperature of the GPU at the
            specified time point. If there are no temperature readings, return None.
        """
        pass


def _temperature_polling_process(
    gpu_indices: list[int],
    data_queue: mp.Queue,
    ready_event: EventClass,
    stop_event: EventClass,
    update_period: float,
) -> None:
    """Polling process for GPU temperature with deduplication."""
    pass

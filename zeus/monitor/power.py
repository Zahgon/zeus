"""Monitor the power usage of GPUs."""

from __future__ import annotations

import bisect
import collections
import logging
import multiprocessing as mp
import weakref
from enum import Enum
from time import time, sleep
from dataclasses import dataclass
from queue import Empty
from typing import Literal, Callable, TYPE_CHECKING

from sklearn.metrics import auc

from zeus.device.gpu.common import ZeusGPUNotSupportedError
from zeus.device import get_gpus
from zeus.utils.multiprocessing import warn_if_global_in_subprocess

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass
    from multiprocessing.context import SpawnProcess

logger = logging.getLogger(__name__)


def infer_counter_update_period(gpu_indicies: list[int]) -> float:
    """Infer the update period of the GPU power counter.

    GPU power counters can update as slow as 10 Hz depending on the GPU model, so
    there's no need to poll them too faster than that. This function infers the
    update period for each unique GPU model and selects the fastest-updating
    period detected. Then, it returns half the period to ensure that the
    counter is polled at least twice per update period.
    """
    raise NotImplementedError


def _infer_counter_update_period_single(gpu_index: int) -> float:
    """Infer the update period of the GPU power counter for a single GPU."""
    raise NotImplementedError


class PowerDomain(Enum):
    """Power measurement domains with different update characteristics."""

    DEVICE_INSTANT = "device_instant"
    DEVICE_AVERAGE = "device_average"
    MEMORY_AVERAGE = "memory_average"


@dataclass
class PowerSample:
    """A single power measurement sample."""

    timestamp: float
    gpu_index: int
    power_mw: float


def _cleanup_processes(
    stop_events: dict[PowerDomain, EventClass],
    processes: dict[PowerDomain, SpawnProcess],
) -> None:
    """Idempotent cleanup function for power monitoring processes."""
    pass


class PowerMonitor:
    """Enhanced PowerMonitor with multiple power domains and timeline export.

    This class provides:

    1. Multiple power domains: device instant, device average, and memory average
    2. Timeline export with independent deduplication per domain
    3. Separate processes for each power domain (2-3 processes depending on GPU support)
    4. Backward compatibility with existing PowerMonitor interface

    !!! Note
        The current implementation only supports cases where all GPUs are homegeneous
        (i.e., the same model).

    !!! Warning
        This monitor uses multiprocessing with the spawn start method to poll power in
        background processes. Spawned processes re-import your main module, so keep heavy
        initialization (for example, model loading) under `if __name__ == "__main__":` or
        inside functions.
        See also the "Safe importing of main module" section in the [Python documentation](
        https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        update_period: float | None = None,
        max_samples_per_gpu: int | None = None,
        power_domains: list[PowerDomain | Literal["device_instant", "device_average", "memory_average"]] | None = None,
    ) -> None:
        """Initialize the enhanced power monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the power monitor in seconds. If None,
                infer the update period by max speed polling the power counter for
                each GPU model.
            max_samples_per_gpu: Maximum number of power samples to keep per GPU per domain
                in memory. If None (default), unlimited samples are kept.
            power_domains: Power domains to monitor. If None, monitor all supported domains.
        """
        raise NotImplementedError

    def _determine_supported_domains(self) -> list[PowerDomain]:
        """Determine which power domains are supported by the current GPUs."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop all monitoring processes."""
        pass

    def _process_queue_data(self, domain: PowerDomain) -> None:
        """Process all pending samples from a specific domain's queue."""
        raise NotImplementedError

    def _process_all_queue_data(self) -> None:
        """Process all pending samples from all domain queues."""
        pass

    def get_power_timeline(
        self,
        power_domain: PowerDomain | Literal["device_instant", "device_average", "memory_average"],
        gpu_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[int, list[tuple[float, float]]]:
        """Get power timeline for specific power domain and GPU(s).

        Args:
            power_domain: Power domain to query
            gpu_index: Specific GPU index, or None for all GPUs
            start_time: Start time filter (unix timestamp from time.time() or similar)
            end_time: End time filter (unix timestamp from time.time() or similar)

        Returns:
            Dictionary mapping GPU indices to timeline data with deduplication.
            Timeline data is list of (timestamp, power_watts) tuples.
        """
        raise NotImplementedError

    def get_all_power_timelines(
        self,
        gpu_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, dict[int, list[tuple[float, float]]]]:
        """Get all power timelines organized by power domain.

        Args:
            gpu_index: Specific GPU index, or None for all GPUs
            start_time: Start time filter (unix timestamp from time.time() or similar)
            end_time: End time filter (unix timestamp from time.time() or similar)

        Returns:
            Dictionary with power domain names as keys and each value is a dict
            mapping GPU indices to timeline data.
        """
        pass

    def get_energy(self, start_time: float, end_time: float) -> dict[int, float] | None:
        """Get the energy used by the GPUs between two times (backward compatibility).

        Uses device instant power for energy calculation.

        Args:
            start_time: Start time of the interval, from time.time().
            end_time: End time of the interval, from time.time().

        Returns:
            A dictionary mapping GPU indices to the energy used by the GPU between the
            two times. If there are no power readings, return None.
        """
        raise NotImplementedError

    def get_power(self, time: float | None = None) -> dict[int, float] | None:
        """Get the instant power usage of the GPUs at a specific time point.

        Uses device instant power for compatibility.

        Args:
            time: Time point to get the power usage at. If None, get the power usage
                at the last recorded time point.

        Returns:
            A dictionary mapping GPU indices to the power usage of the GPU at the
            specified time point. If there are no power readings, return None.
        """
        pass


def _domain_polling_process(
    power_domain: PowerDomain,
    gpu_indices: list[int],
    data_queue: mp.Queue,
    ready_event: EventClass,
    stop_event: EventClass,
    update_period: float,
) -> None:
    """Polling process for a specific power domain with deduplication."""
    pass

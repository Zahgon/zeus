"""Measure the GPU time and energy consumption of a block of code."""

from __future__ import annotations

import os
import logging
import warnings
from typing import Literal
from time import time
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property

from zeus.monitor.power import PowerMonitor
from zeus.utils.framework import sync_execution as sync_execution_fn
from zeus.utils.multiprocessing import warn_if_global_in_subprocess
from zeus.device import get_gpus, get_cpus, get_soc
from zeus.device.gpu.common import ZeusGPUInitError, EmptyGPUs
from zeus.device.cpu.common import ZeusCPUInitError, ZeusCPUNoPermissionError, EmptyCPUs
from zeus.device.soc.common import ZeusSoCInitError, SoCMeasurement, EmptySoC

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    """Measurement result of one window.

    Attributes:
        time: Time elapsed (in seconds) during the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d)  and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
        soc_energy: If your machine contains an SoC (e.g., Apple silicon), metrics for various
            subsystems of the SoC (e.g., the on-chip CPU, the on-chip GPU) will all be included
            within this field.
    """

    time: float
    gpu_energy: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None

    soc_energy: SoCMeasurement | None = None

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        pass


@dataclass
class MeasurementState:
    """Measurement state to keep track of measurements in start_window.

    Used in ZeusMonitor to map string keys of measurements to this dataclass.

    Attributes:
        time: The beginning timestamp of the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d)  and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
    """

    time: float
    gpu_energy: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        pass


class ZeusMonitor:
    """Measure the GPU energy and time consumption of a block of code.

    Works for multi-GPU and heterogeneous GPU types. Aware of `CUDA_VISIBLE_DEVICES`.
    For instance, if `CUDA_VISIBLE_DEVICES=2,3`, GPU index `1` passed into `gpu_indices`
    will be interpreted as CUDA device `3`.

    You can mark the beginning and end of a measurement window, during which the GPU
    energy and time consumed will be recorded. Multiple concurrent measurement windows
    are supported.

    For Volta or newer GPUs, energy consumption is measured very cheaply with the
    `nvmlDeviceGetTotalEnergyConsumption` API. On older architectures, this API is
    not supported, so a separate Python process is used to poll `nvmlDeviceGetPowerUsage`
    to get power samples over time, which are integrated to compute energy consumption.

    !!! Warning
        This monitor may start helper processes for power polling on some hardware using
        the spawn start method. Spawned processes re-import your main module, so keep heavy
        initialization (for example, model loading) under `if __name__ == "__main__":` or
        inside functions.
        See also the "Safe importing of main module" section in the [Python documentation](
        https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).

    ## Integration Example

    ```python
    from zeus.monitor import ZeusMonitor

    def training():
        \"\"\"A dummy training function.\"\"\"
        import time
        time.sleep(5)

    # Make sure to protect the entry point of the program to avoid monitoring
    # subprocesses from re-executing the main module.
    if __name__ == "__main__":
        # Time/Energy measurements for four GPUs will begin and end at the same time.
        gpu_indices = [0, 1, 2, 3]
        monitor = ZeusMonitor(gpu_indices)

        # Mark the beginning of a measurement window. You can use any string
        # as the window name, but make sure it's unique.
        monitor.begin_window("entire_training")

        # Actual work
        training()

        # Mark the end of a measurement window and retrieve the measurment result.
        result = monitor.end_window("entire_training")

        # Print the measurement result.
        print(f"Training consumed {result.total_energy} Joules.")
        for gpu_idx, gpu_energy in result.gpu_energy.items():
            print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")
    ```

    Attributes:
        gpu_indices (`list[int]`): Indices of all the CUDA devices to monitor, from the
            DL framework's perspective after applying `CUDA_VISIBLE_DEVICES`.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
        approx_instant_energy: bool = False,
        log_file: str | Path | None = None,
        sync_execution_with: Literal["torch", "jax", "cupy"] = "torch",
    ) -> None:
        """Instantiate the monitor.

        Args:
            gpu_indices: Indices of all the CUDA devices to monitor. Time/Energy measurements
                will begin and end at the same time for these GPUs (i.e., synchronized).
                If None, all the GPUs available will be used. `CUDA_VISIBLE_DEVICES`
                is respected if set, e.g., GPU index `1` passed into `gpu_indices` when
                `CUDA_VISIBLE_DEVICES=2,3` will be interpreted as CUDA device `3`.
                `CUDA_VISIBLE_DEVICES`s formatted with comma-separated indices are supported.
            cpu_indices: Indices of the CPU packages to monitor. If None, all CPU packages will
                be used.
            approx_instant_energy: When the execution time of a measurement window is
                shorter than the NVML energy counter's update period, energy consumption may
                be observed as zero. In this case, if `approx_instant_energy` is True, the
                window's energy consumption will be approximated by multiplying the current
                instantaneous power consumption with the window's execution time. This should
                be a better estimate than zero, but it's still an approximation.
            log_file: Path to the log CSV file. If `None`, logging will be disabled.
            sync_execution_with: Deep learning framework to use to synchronize CPU/GPU computations.
                Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
                See [`sync_execution`][zeus.utils.framework.sync_execution] for more details.
        """
        raise NotImplementedError

    def _get_instant_power(self) -> tuple[dict[int, float], float]:
        """Measure the power consumption of all GPUs at the current time."""
        raise NotImplementedError

    def begin_window(self, key: str, sync_execution: bool = True, restart: bool = False) -> None:
        """Begin a new measurement window.

        Args:
            key: Unique name of the measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
            restart: If True and the window already exists, cancel the existing window
                and start a new one. This is useful in interactive environments like
                Jupyter notebooks where a cell may crash between `begin_window` and
                `end_window`, leaving the window in a stale state.
        """
        raise NotImplementedError

    def end_window(self, key: str, sync_execution: bool = True, cancel: bool = False) -> Measurement:
        """End a measurement window and return the time and energy consumption.

        Args:
            key: Name of an active measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
            cancel: Whether to cancel the measurement window. If `True`, the measurement
                window is assumed to be cancelled and discarded. Thus, an empty Measurement
                object will be returned and the measurement window will not be recorded in
                the log file either. `sync_execution` is still respected.
        """
        raise NotImplementedError

    def reset_windows(self) -> None:
        """Reset the monitor by removing all active measurement windows.

        Any ongoing measurements will be cancelled and their data will be lost.
        """
        pass

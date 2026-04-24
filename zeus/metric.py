"""Track and export energy and power metrics via Prometheus."""

from __future__ import annotations

import abc
import time
import warnings
from typing import Sequence
import multiprocessing as mp
from multiprocessing.context import SpawnProcess
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Histogram,
    Counter,
    Gauge,
    push_to_gateway,
)

from zeus.monitor.power import PowerMonitor
from zeus.monitor.energy import ZeusMonitor
from zeus.utils.framework import sync_execution as sync_execution_fn
from zeus.device.cpu import get_cpus


@dataclass
class MonitoringProcessState:
    """Represents the state of a monitoring window."""

    queue: mp.Queue
    proc: SpawnProcess


class Metric(abc.ABC):
    """Abstract base class for all metric types in Zeus.

    Defines a common interface for metrics, ensuring consistent behavior
    for `begin_window` and `end_window` operations.
    """

    @abc.abstractmethod
    def begin_window(self, name: str, sync_execution: bool = True) -> None:
        """Start a new measurement window.

        Args:
            name (str): Name of the measurement window.
            sync_execution (bool): Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window.
        """
        pass

    @abc.abstractmethod
    def end_window(self, name: str, sync_execution: bool = True) -> None:
        """End the current measurement window and report metrics.

        Args:
            name (str): Name of the measurement window.
            sync_execution (bool): Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window.
        """
        pass


class EnergyHistogram(Metric):
    """Measures the energy consumption a code range and exports a histogram metrics.

    Tracks energy consumption for GPUs, CPUs, and DRAM as Prometheus Histogram metrics.
    """

    def __init__(
        self,
        cpu_indices: list,
        gpu_indices: list,
        pushgateway_url: str,
        job: str,
        gpu_bucket_range: Sequence[float] = [50.0, 100.0, 200.0, 500.0, 1000.0],
        cpu_bucket_range: Sequence[float] = [10.0, 50.0, 100.0, 500.0, 1000.0],
        dram_bucket_range: Sequence[float] = [5.0, 10.0, 20.0, 50.0, 150.0],
    ) -> None:
        """Initialize the EnergyHistogram class.

        Sets up the Prometheus Histogram metrics to track energy consumption for GPUs, CPUs, and DRAMs.
        The data will be collected and pushed to the Prometheus Push Gateway at regular intervals.

        Args:
            cpu_indices (list): List of CPU indices to monitor.
            gpu_indices (list): List of GPU indices to monitor.
            pushgateway_url (str): URL of the Prometheus Push Gateway where metrics will be pushed.
            job (str): Name of the Prometheus job to associate with the energy metrics.
            gpu_bucket_range (list[float], optional): Bucket ranges for GPU energy histograms.
                Defaults to [50.0, 100.0, 200.0, 500.0, 1000.0].
            cpu_bucket_range (list[float], optional): Bucket ranges for CPU energy histograms.
                Defaults to [10.0, 20.0, 50.0, 100.0, 200.0].
            dram_bucket_range (list[float], optional): Bucket ranges for DRAM energy histograms.
                Defaults to [5.0, 10.0, 20.0, 50.0, 150.0].

        Raises:
            ValueError: If any of the bucket ranges (GPU, CPU, DRAM) is an empty list.
        """
        raise NotImplementedError

    def begin_window(self, name: str, sync_execution: bool = True) -> None:
        """Begin the energy monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool): Whether to execute synchronously. Defaults to True. If assigned True, calls sync_execution_fn with the defined gpu
        """
        raise NotImplementedError

    def end_window(self, name: str, sync_execution: bool = True) -> None:
        """End the current energy monitoring window and record the energy data.

        Retrieves the energy consumption data (for GPUs, CPUs, and DRAMs) for the monitoring window
        and updates the corresponding Histogram metrics. The data is then pushed to the Prometheus Push Gateway.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool): Whether to execute synchronously. Defaults to True.
        """
        raise NotImplementedError


class EnergyCumulativeCounter(Metric):
    """EnergyCumulativeCounter class to monitor and record cumulative energy consumption.

    This class tracks GPU, CPU, and DRAM energy usage over time, and records the data as Prometheus Counter metrics.
    The energy consumption metrics are periodically updated and pushed to a Prometheus Push Gateway for monitoring and analysis.

    The cumulative nature of the Counter ensures that energy values are always incremented over time, never reset,
    which is ideal for tracking continuously increasing values like energy usage.
    """

    def __init__(
        self,
        cpu_indices: list,
        gpu_indices: list,
        update_period: int,
        pushgateway_url: str,
        job: str,
    ) -> None:
        """Initialize the EnergyCumulativeCounter.

        Args:
            cpu_indices (list): List of CPU indices to monitor.
            gpu_indices (list): List of GPU indices to monitor.
            update_period: The time interval (in seconds) at which energy measurements are updated.
            pushgateway_url: The URL for the Prometheus Push Gateway where the metrics will be pushed.
            job: The name of the job to be associated with the Prometheus metrics.
        """
        raise NotImplementedError

    def begin_window(self, name: str, sync_execution: bool = False) -> None:
        """Begin the energy monitoring window.

        Starts a new multiprocessing process that monitors energy usage periodically
        and pushes the results to the Prometheus Push Gateway.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        raise NotImplementedError

    def end_window(self, name: str, sync_execution: bool = False) -> None:
        """End the energy monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        raise NotImplementedError


def energy_monitoring_loop(
    name: str,
    pipe: mp.Queue,
    cpu_indices: list,
    gpu_indices: list,
    update_period: int,
    pushgateway_url: str,
    job: str,
) -> None:
    """Runs in a separate process to collect and update energy consumption metrics (for GPUs, CPUs, and DRAM).

    Args:
        name (str): The user-defined name of the monitoring window (used as a label for Prometheus metrics).
        pipe (mp.Queue): A multiprocessing queue for inter-process communication, used to signal when to stop the process.
        cpu_indices (list): List of CPU indices to monitor.
        gpu_indices (list): List of GPU indices to monitor.
        update_period (int): The interval (in seconds) between consecutive energy data updates.
        pushgateway_url (str): The URL of the Prometheus Push Gateway where the metrics will be pushed.
        job (str): The name of the Prometheus job associated with these metrics.
    """
    pass


class PowerGauge(Metric):
    """PowerGauge class to monitor and record power consumption.

    This class tracks GPU power usage in real time and records it as **Prometheus Gauge** metrics.
    The Gauge metric type is suitable for tracking values that can go up and down over time, like power consumption.

    Power usage data is collected at regular intervals and pushed to a Prometheus Push Gateway for monitoring.
    """

    def __init__(
        self,
        gpu_indices: list,
        update_period: int,
        pushgateway_url: str,
        job: str,
    ) -> None:
        """Initialize the PowerGauge metric.

        Args:
            gpu_indices (list[int]): List of GPU indices to monitor for power consumption.
            update_period (int): Interval (in seconds) between consecutive power measurements.
            pushgateway_url (str): URL of the Prometheus Push Gateway where Gauge metrics are pushed.
            job (str): Name of the Prometheus job to associate with the power metrics.
        """
        raise NotImplementedError

    def begin_window(self, name: str, sync_execution: bool = False) -> None:
        """Begin the power monitoring window.

        Starts a new multiprocessing process that runs the power monitoring loop.
        The process collects real-time power consumption data and updates the corresponding
        Gauge metrics in Prometheus.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        raise NotImplementedError

    def end_window(self, name: str, sync_execution: bool = False) -> None:
        """End the power monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        raise NotImplementedError


def power_monitoring_loop(
    name: str,
    pipe: mp.Queue,
    gpu_indices: list[int],
    update_period: int,
    pushgateway_url: str,
    job: str,
) -> None:
    """Runs in a separate process and periodically collects power consumption data for each GPU and pushes the results to the Prometheus Push Gateway.

    Args:
        name (str): Unique name for the monitoring window (used as a label in Prometheus metrics).
        pipe (multiprocessing.Queue): Queue to receive control signals (e.g., "stop").
        gpu_indices (list[int]): List of GPU indices to monitor for power consumption.
        update_period (int): Interval (in seconds) between consecutive power data polls.
        pushgateway_url (str): URL of the Prometheus Push Gateway where metrics are pushed.
        job (str): Name of the Prometheus job to associate with the metrics.
    """
    pass

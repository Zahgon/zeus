"""Abstraction layer for CPU devices.

The main function of this module is [`get_cpus`][zeus.device.cpu.get_cpus],
which returns a CPU Manager object specific to the platform.
"""
from __future__ import annotations
import os
from typing import Literal
from zeus.device.cpu.common import CPUs, ZeusCPUInitError
from zeus.device.cpu.rapl import rapl_is_available, RAPLCPUs
_cpus: CPUs | None = None

def get_current_cpu_index(pid: int | Literal['current']='current') -> int:
    """Retrieves the specific CPU index (socket) where the given PID is running.

    If no PID is given or pid is "current", the CPU index returned is of the CPU running the current process.

    !!! Note
        Linux schedulers can preempt and reschedule processes to different CPUs. To prevent this from happening
        during monitoring, use `taskset` to pin processes to specific CPUs.
    """
    raise NotImplementedError()

def get_cpus() -> CPUs:
    """Initialize and return a singleton CPU monitoring object for INTEL CPUs.

    The function returns a CPU management object that aims to abstract the underlying CPU monitoring libraries
    (RAPL for Intel CPUs).

    This function attempts to initialize CPU mointoring using RAPL. If this attempt fails, it raises
    a ZeusErrorInit exception.
    """
    raise NotImplementedError()
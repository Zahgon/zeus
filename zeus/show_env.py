"""Collect information about the environment and display it.

- Python version
- Package availability and versions: Zeus, PyTorch, JAX, CuPy.
- NVIDIA GPU availability: Number of GPUs and models.
- AMD GPU availability: Number of GPUs and models.
- Intel RAPL availability: Number of CPUs and whether DRAM measurements are available.
- Zeusd daemon connectivity and capabilities (when configured).
"""

from __future__ import annotations

import datetime
import logging
import os
import platform
import shutil
import time

import zeus
from zeus.device.cpu.rapl import RAPLCPU, ZeusdRAPLCPU
from zeus.device.exception import ZeusBaseCPUError, ZeusBaseGPUError, ZeusBaseSoCError
from zeus.utils.zeusd import ZeusdClient, ZeusdConfig
from zeus.utils import framework
from zeus.device import get_gpus, get_cpus, get_soc
from zeus.device.cpu import RAPLCPUs
from zeus.device.gpu.common import ZeusGPUInitError, EmptyGPUs
from zeus.device.cpu.common import ZeusCPUInitError, EmptyCPUs
from zeus.device.soc.common import ZeusSoCInitError, EmptySoC
from zeus.device.soc.apple import AppleSilicon
from zeus.device.soc.jetson import Jetson

SECTION_SEPARATOR = "-" * shutil.get_terminal_size().columns


def show_env():
    """Collect information about the environment and display it."""
    raise NotImplementedError


if __name__ == "__main__":
    # For the `python -m zeus.show_env` usage
    logging.basicConfig(level=logging.INFO, format="  [%(asctime)s] [%(name)s:%(lineno)d] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    show_env()

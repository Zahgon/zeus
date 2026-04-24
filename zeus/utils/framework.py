"""Utilities for framework-specific code."""

from __future__ import annotations

import logging
import types
from typing import Literal
from functools import lru_cache

logger = logging.getLogger(__name__)
MODULE_CACHE: dict[str, types.ModuleType] = {}


@lru_cache(maxsize=1)
def torch_is_available(ensure_available: bool = False, ensure_cuda: bool = True) -> bool:
    """Check if PyTorch is available."""
    raise NotImplementedError


@lru_cache(maxsize=1)
def jax_is_available(ensure_available: bool = False, ensure_cuda: bool = True) -> bool:
    """Check if JAX is available."""
    raise NotImplementedError


@lru_cache(maxsize=1)
def cupy_is_available(ensure_available: bool = False) -> bool:
    """Check if CuPy is available."""
    raise NotImplementedError


def sync_execution(gpu_devices: list[int], sync_with: Literal["torch", "jax", "cupy"] = "torch") -> None:
    """Block until all computations on the specified devices are finished.

    PyTorch only runs GPU computations asynchronously, so synchronizing computations
    for the given GPU devices is done by calling `torch.cuda.synchronize` on each
    device. On the other hand, JAX runs both CPU and GPU computations asynchronously,
    but by default it only has a single CPU device (id=0). Therefore for JAX, all GPU
    devices passed in and the CPU device (id=0) are synchronized.

    CuPy is independent of the specific ML framework and calls the CUDA API directly,
    but you'll need to install it separately and ensure that it is compatible with
    the ML framework you are using.

    !!! Note
        `jax.device_put` with `block_until_ready` is used to synchronize computations
        on JAX devices. This is a workaround to the lack of a direct API for
        synchronizing computations on JAX devices. Tracking issue:
        https://github.com/google/jax/issues/4335

    !!! Note
        Across the Zeus library, an integer device index corresponds to a single whole
        physical device. This is usually what you want, except when using more advanced
        device partitioning (e.g., using `--xla_force_host_platform_device_count` in JAX
        to partition CPUs into more pieces). In such cases, you probably want to opt out
        from using this function and handle synchronization manually at the appropriate
        granularity.

    Args:
        gpu_devices: GPU device indices to synchronize.
        sync_with: Deep learning framework to use to synchronize computations.
            Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
    """
    raise NotImplementedError


def all_reduce(object: list[int] | list[float], operation: Literal["sum", "max"]) -> list[int] | list[float]:
    """Reduce objects from all replicas through the specified operation.

    If the current execution is not distributed, the object is returned as is.
    """
    raise NotImplementedError


def is_distributed() -> bool:
    """Check if the current execution is distributed across multiple devices."""
    raise NotImplementedError


def get_rank() -> int:
    """Return the rank of the current process, or 0 if not distributed."""
    raise NotImplementedError


def get_world_size() -> int:
    """Return the number of processes, or 1 if not distributed."""
    raise NotImplementedError

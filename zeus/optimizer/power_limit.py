"""Optimizers that select the optimum power limit.

This module contains the following pieces:

- [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer]
  is the main class that implements the state machine
  and the logic for profiling power limits and selecting
  the optimum power limit.
- [`PowerLimitMeasurement`][zeus.optimizer.power_limit.PowerLimitMeasurement] and various
  state classes are helpers that support the state machine.
- [`OptimumSelector`][zeus.optimizer.power_limit.OptimumSelector]
  is an abstract base class for selecting the optimum power limit
  from a list of power limit profiling results. There are concrete classes
  that implement different selection strategies, like
  [minimizing energy][zeus.optimizer.power_limit.Energy],
  [minimizing time][zeus.optimizer.power_limit.Time],
  [minimizing the Zeus time-energy cost][zeus.optimizer.power_limit.ZeusCost],
  or [selecting the lowest power limit that meets the given maximum training time slowdown factor][zeus.optimizer.power_limit.MaxSlowdownConstraint].
- [`HFGlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer]
  is a wrapper for the Hugging Face `TrainerCallback` class that uses `GlobalPowerLimitOptimizer`.
"""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from abc import ABC, abstractmethod

from zeus.callback import Callback
from zeus.monitor import ZeusMonitor
from zeus.utils.framework import all_reduce, is_distributed
from zeus.utils.metric import zeus_cost
from zeus.utils.pydantic_v1 import BaseModel, PositiveInt, PositiveFloat
from zeus.device import get_gpus
from zeus.device.gpu import ZeusGPUNoPermissionError

from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


class OptimumSelector(ABC):
    """Base class for optimum power limit selectors."""

    @abstractmethod
    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""


class Energy(OptimumSelector):
    """Selects the power limit that minimizes energy consumption."""

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        raise NotImplementedError


class Time(OptimumSelector):
    """Selects the power limit that minimizes training time.

    This may not necessarily choose the maximum power limit, as time profiling
    results can be slightly noisy. However, we believe that's actually better
    because it means that training time is very similar among higher power limits,
    but lower power limit will consume less power.
    """

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        raise NotImplementedError


class ZeusCost(OptimumSelector):
    r"""Selects the power limit that minimizes a linear Zeus time-energy cost function.

    Cost function is $\eta \cdot \text{Energy} + (1 - \eta) \cdot \text{MaxPower} \cdot \text{Time}$.
    """

    def __init__(self, eta_knob: float, world_size: int = 1) -> None:
        r"""Initialize the selector.

        Args:
            eta_knob: The $0 \le \eta \le 1$ knob for the Zeus time-energy cost function.
            world_size: The number of GPUs in the training job. Defaults to 1.
        """
        raise NotImplementedError

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        raise NotImplementedError


class MaxSlowdownConstraint(OptimumSelector):
    """Selects the minumum power limit that does not slow down training by more than the given factor."""

    def __init__(self, factor: float) -> None:
        """Initialize the selector.

        Args:
            factor: The maximum allowed slowdown factor. Greater than or equal to 1.0.
        """
        raise NotImplementedError

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        raise NotImplementedError


class Ready(BaseModel):
    """State for when we are ready to start measuring the next power limit.

    Initial state of the state machine if no previous profiling results were given.
    `Ready` -> `Warmup` after `step`'th `on_step_begin`.
    """

    next_power_limit: PositiveInt
    steps: PositiveInt


class Warmup(BaseModel):
    """State for when we are warming up for a power limit.

    `Warmup` -> `Profiling` on the `steps`'th `on_step_begin`.
    `Warmup` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_begin`.
    """

    current_power_limit: PositiveInt
    steps: PositiveInt


class Profiling(BaseModel):
    """State for when we are profiling a power limit.

    `Profiling` -> `Warmup` after `steps`'th `on_step_begin` and
        there are still power limits left to profile.
    `Profiling` -> `Done` after `steps`'th `on_step_begin` and
        there are no more power limits left to profile.
    `Profiling` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_begin`.
    """

    current_power_limit: PositiveInt
    steps: PositiveInt


class Done(BaseModel):
    """State for when we are done profiling all power limits.

    Initial state of the state machine if previous profiling results were given.
    Final state of the state machine in any case.
    """

    optimal_power_limit: PositiveInt


class PowerLimitMeasurement(BaseModel):
    """POD for GPU energy and time measurements for one power limit (W)."""

    power_limit: PositiveInt  # In Watts.
    energy: PositiveFloat
    time: PositiveFloat


class _PowerLimitMeasurementList(BaseModel):
    """Proxy class to save and load a list of `PowerLimitMeasurement`s."""

    measurements: list[PowerLimitMeasurement]


class GlobalPowerLimitOptimizer(Callback):
    """Optimizer for the power limit knob.

    This optimizer uses the JIT profiling log to determine the optimal power limit.

    ## Usage with distributed data parallelism

    The global power limit optimizer expects one process to control each GPU used for training.
    For instance, `torchrun` will automatically spawn one process for each GPU on the node.
    Correspondingly, the [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] instance passed in
    should be monitoring **one GPU**: the one being managed by the current process. The index of
    this GPU would typically match the local rank of the process. In the case of PyTorch, users would have
    called `torch.cuda.set_device` early on, so `torch.cuda.current_device` will give you the GPU index.
    `GlobalPowerLimitOptimizer` will internally do an AllReduce across all GPUs to aggregate
    time and energy measurements, and then select the globally optimal power limit.


    ```python
    monitor = ZeusMonitor(gpu_indices=[local_rank])  # pass in local rank to gpu_indices.
    plo = GlobalPowerLimitOptimizer(monitor)
    ```
    """

    def __init__(
        self,
        monitor: ZeusMonitor,
        optimum_selector: OptimumSelector | None = None,
        wait_steps: int = 1,
        warmup_steps: int = 10,
        profile_steps: int = 40,
        pl_step: int = 25,
        profile_path: str | Path | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        GPU indices to profile and optimize for are taken from `monitor.gpu_indices`.

        Args:
            monitor: `ZeusMonitor` instance used to profile GPU time and energy consumption.
            optimum_selector: The optimum selector to use. If not given, use `ZeusCost` with \eta=0.5.
            wait_steps: Number of steps to pass by before doing anything at the beginning.
                Useful if you have something like `torch.backends.cudnn.benchmark=True`,
                because the first iteration won't be representative of the rest of the iterations.
            warmup_steps: Number of warmup iterations for each power limit.
            profile_steps: Number of profie iterations for each power limit.
            pl_step: The stride between power limits to explore, in unites of Watts.
            profile_path: If the path points to an existing file, load the profile from the file
                and do not run any profiling. If the path points to a non-existing file, profile
                and save the profile to the file. If `None`, do not save or load any profile.
        """
        raise NotImplementedError

    def on_epoch_end(self) -> None:
        """Mark the end of a training epoch."""
        pass

    def on_step_begin(self) -> None:
        """Mark the beginning of a training step."""
        pass

    def _set_power_limit(self, power_limit: int) -> None:
        """Set the power limit for all GPUs.

        Args:
            power_limit: The power limit to set, in milliWatts.
        """
        raise NotImplementedError

    def _compute_optimal_power_limit(self) -> int:
        """Compute the optimal power limit in milliWatts."""
        raise NotImplementedError

    def _save_profile(self) -> None:
        """Save JIT profiling results and the optimal power limit to a JSON file."""
        pass


# Only import HuggingFace Classes when type checking, to avoid hard dependency on HuggingFace Transformers
if TYPE_CHECKING:
    from transformers.training_args import TrainingArguments
    from transformers.trainer_callback import TrainerState, TrainerControl

try:
    from transformers.trainer_callback import TrainerCallback

    transformers_available = True
except ModuleNotFoundError:
    transformers_available = False
    TrainerCallback = object  # ty: ignore[invalid-assignment]


class HFGlobalPowerLimitOptimizer(TrainerCallback):
    """[Wrapped for Hugging Face Trainer Callback] Optimizer for the power limit knob.

    This optimizer uses the JIT profiling log to determine the optimal power limit.
    See [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer]
    for the underlying optimizer implementation.
    """

    def __init__(
        self,
        monitor: ZeusMonitor,
        optimum_selector: OptimumSelector | None = None,
        wait_steps: int = 1,
        warmup_steps: int = 10,
        profile_steps: int = 40,
        pl_step: int = 25,
        profile_path: str | Path | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        GPU indices to profile and optimize for are taken from `monitor.gpu_indices`.

        Args:
            monitor: `ZeusMonitor` instance used to profile GPU time and energy consumption.
            optimum_selector: The optimum selector to use. If not given, use `ZeusCost` with \eta=0.5.
            wait_steps: Number of steps to pass by before doing anything at the beginning.
                Useful if you have something like `torch.backends.cudnn.benchmark=True`,
                because the first iteration won't be representative of the rest of the iterations.
            warmup_steps: Number of warmup iterations for each power limit.
            profile_steps: Number of profie iterations for each power limit.
            pl_step: The stride between power limits to explore, in unites of Watts.
            profile_path: If the path points to an existing file, load the profile from the file
                and do not run any profiling. If the path points to a non-existing file, profile
                and save the profile to the file. If `None`, do not save or load any profile.
        """
        raise NotImplementedError

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Mark the end of a training epoch."""
        pass

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Mark the beginning of a training step."""
        pass

"""Controller that sets the GPU's frequency in a non-blocking fashion."""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp

from zeus.device import get_gpus
from zeus.device.gpu import ZeusGPUNotSupportedError


class FrequencyController:
    """Spawns a separate process that sets the GPU frequency."""

    def __init__(self, device_id: int = 0) -> None:
        """Instantiate the frequency controller.

        Args:
            device_id: Device ID of the GPU to control.
        """
        raise NotImplementedError

    def set_frequency(self, frequency: int) -> None:
        """Set the GPU's frequency asynchronously.

        If `frequency` is zero, returns without doing anything.
        """
        pass

    def end(self) -> None:
        """Stop the controller process."""
        pass

    def _controller_process(self, device_id: int) -> None:
        """Receive frequency values through a queue and apply it."""
        pass

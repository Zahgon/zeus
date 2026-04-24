"""Classes that enclose learning rate scaling rules."""

import math
from dataclasses import dataclass


@dataclass
class SquareRootScaler:
    """Square root scaling.

    Args:
        bs: The initial batch size
        lr: The initial learning rate
    """

    bs: int
    lr: float

    def compute_lr(self, new_bs: int) -> float:
        """Compute the scaled learning rate given the new batch size."""
        pass


@dataclass
class LinearScaler:
    """Linear scaling.

    Args:
        bs: The initial batch size
        lr: The initial learning rate
    """

    bs: int
    lr: float

    def compute_lr(self, new_bs: int) -> float:
        """Compute the scaled learning rate given the new batch size."""
        pass

"""Infrastructure for calling callbacks."""

from __future__ import annotations


class Callback:
    """Base class for callbacks."""

    def on_train_begin(self) -> None:
        """Called at the beginning of training."""

    def on_train_end(self) -> None:
        """Called at the end of training."""

    def on_epoch_begin(self) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch."""

    def on_step_begin(self) -> None:
        """Called at the beginning of each step."""

    def on_step_end(self) -> None:
        """Called at the end of each step."""

    def on_evaluate(self, metric: float) -> None:
        """Called after evaluating the model."""

    def on_instruction_begin(self, name: str) -> None:
        """Called at the beginning of pipeline instructions like forward or backward."""

    def on_instruction_end(self, name: str) -> None:
        """Called at the end of pipeline instructions like forward or backward."""


class CallbackSet(Callback):
    """A set of callbacks."""

    def __init__(self, callbacks: list[Callback]) -> None:
        """Initialize the callback set."""
        raise NotImplementedError

    def on_train_begin(self) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self) -> None:
        """Called at the beginning of each step."""
        pass

    def on_step_end(self) -> None:
        """Called at the end of each step."""
        pass

    def on_evaluate(self, metric: float) -> None:
        """Called after evaluating the model."""
        pass

    def on_instruction_begin(self, name: str) -> None:
        """Called at the beginning of pipeline instructions like forward or backward."""
        pass

    def on_instruction_end(self, name: str) -> None:
        """Called at the end of pipeline instructions like forward or backward."""
        pass

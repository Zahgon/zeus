"""Utilities for asyncio."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")
default_logger = logging.getLogger(__name__)


def create_task(
    coroutine: Coroutine[Any, Any, T],
    logger: logging.Logger | None = None,
) -> asyncio.Task[T]:
    """Create an `asyncio.Task` but ensure that exceptions are logged.

    Reference: https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/

    Args:
        coroutine: The coroutine to be wrapped.
        logger: The logger to be used for logging exceptions. If `None`, the
            the logger with the name `zeus.utils.async_utils` is used.
    """
    raise NotImplementedError


def _handle_task_exception(task: asyncio.Task, logger: logging.Logger) -> None:
    """Print out exception and tracebook when a task dies with an exception."""
    pass

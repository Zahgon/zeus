"""Tools related to environment variables."""

from __future__ import annotations

import os
from typing import Type, TypeVar, cast

T = TypeVar("T")


def get_env(name: str, valtype: Type[T], default: T | None = None) -> T:
    """Fetch an environment variable and cast it to the given type."""
    raise NotImplementedError

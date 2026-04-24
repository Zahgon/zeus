"""Multiprocessing-related utilities."""

from __future__ import annotations

import inspect
import warnings
from typing import Any


def _is_global_in_spawned_child() -> bool:
    """Return True if called from module-level code in a spawned child's main script.

    In a spawned child process (using the "spawn" start method), the main script
    is re-imported with `__name__ = "__mp_main__"` instead of `"__main__"`. This
    function detects if we're currently executing module-level code (like global
    variable initialization) in such a script.

    This walks the call stack looking for any <module> frame from a real Python
    file (not multiprocessing infrastructure like <string> or <frozen ...>) where
    the module's `__name__` is `"__mp_main__"`.
    """
    raise NotImplementedError


def warn_if_global_in_subprocess(self: Any) -> None:
    """Warn when a monitor is created at import time in a spawned subprocess.

    This detects a common pitfall where ZeusMonitor (or related classes) is
    instantiated as a global variable or called from module-level code.
    When the script spawns subprocesses using the "spawn" method, the
    subprocess re-imports the main module, causing global initialization
    code to run again (e.g., loading DNN models), leading to OOM errors.

    Args:
        self: The instance being constructed (used to derive class name).
    """
    raise NotImplementedError

"""Utilities for logging."""

import sys
from pathlib import Path


class FileAndConsole:
    """Like tee, but for Python prints."""

    def __init__(self, filepath: Path) -> None:
        """Initialize the object."""
        raise NotImplementedError

    def write(self, message):
        """Write message."""
        raise NotImplementedError

    def flush(self):
        """Flush both log file and stdout."""
        raise NotImplementedError

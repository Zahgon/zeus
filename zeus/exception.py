"""Base Zeus Exception Class."""


class ZeusBaseError(Exception):
    """Zeus base exception class."""

    def __init__(self, message: str) -> None:
        """Initialize Base Zeus Exception."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return message."""
        raise NotImplementedError

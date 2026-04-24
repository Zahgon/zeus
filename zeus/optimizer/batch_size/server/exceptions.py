"""Zeus server exceptions."""

from zeus.exception import ZeusBaseError


class ZeusBSOServerBaseError(ZeusBaseError):
    """Base error class for BSO server."""

    def __init__(self, msg: str):
        """Set status code."""
        raise NotImplementedError


class ZeusBSOJobConfigMismatchError(ZeusBSOServerBaseError):
    """When the job configuration doesn't align for the same job_id."""

    def __init__(self, msg: str):
        """Set status code."""
        raise NotImplementedError


class ZeusBSOValueError(ZeusBSOServerBaseError):
    """When the certain value is invalid."""

    def __init__(self, msg: str):
        """Set status code."""
        raise NotImplementedError


class ZeusBSOServerNotFoundError(ZeusBSOServerBaseError):
    """Resource we are looking for is not found."""

    def __init__(self, msg: str):
        """Set status code."""
        raise NotImplementedError


class ZeusBSOServiceBadOperationError(ZeusBSOServerBaseError):
    """When the operation doesn't meet requirements. ex) fetching measurements before fetching a job."""

    def __init__(self, msg: str):
        """Set status code."""
        raise NotImplementedError


class ZeusBSOServerRuntimeError(ZeusBSOServerBaseError):
    """Initialization or other errors during runtime."""

    def __init__(self, msg: str):
        """Set status code."""
        raise NotImplementedError

"""Managing database connection.

Heavily inspired by https://praciano.com.br/fastapi-and-async-sqlalchemy-20-with-pytest-done-right.html
and https://medium.com/@tclaitken/setting-up-a-fastapi-app-with-async-sqlalchemy-2-0-pydantic-v2-e6c540be4308
"""

from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from zeus.optimizer.batch_size.server.config import settings
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOServerRuntimeError


class DatabaseSessionManager:
    """Session manager class."""

    def __init__(self, host: str, engine_kwargs: dict[str, Any] | None = None):
        """Create async engine and session maker."""
        raise NotImplementedError

    async def close(self):
        """Close connection."""
        raise NotImplementedError

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """Connect to db."""
        raise NotImplementedError

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get session from session maker."""
        raise NotImplementedError


# Initialize session manager.
sessionmanager = DatabaseSessionManager(settings.database_url, {"echo": settings.echo_sql})


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Get db session from session manager. Used with fastapi dependency injection."""
    raise NotImplementedError

"""
Async PostgreSQL connection pool.

Call get_pool() anywhere in the app to borrow a connection.
Call setup_db() once at startup to register the pgvector extension
and create tables.
"""

from __future__ import annotations

import psycopg
from psycopg_pool import AsyncConnectionPool

from api.core.config import get_settings
from api.db.schema import create_tables

_pool: AsyncConnectionPool | None = None


async def get_pool() -> AsyncConnectionPool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=2,
            max_size=10,
            open=False,
        )
        await _pool.open()
    return _pool


async def setup_db() -> None:
    """
    Run once at application startup.
    Enables pgvector and creates tables if they don't exist.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await create_tables(conn)


async def close_pool() -> None:
    """Gracefully close the pool on application shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

"""
Database schema — session_chunks and session_meta.

session_chunks  : one row per embedded chunk produced during the cold seed call.
session_meta    : one row per session, holds the topic centroid used by the
                  relevance guard and the embedding dimension for that session.

Tables are created with IF NOT EXISTS so setup_db() is safely idempotent.

Vector dimensions are read from settings at table-creation time so the schema
stays consistent with whichever embedding model is configured in .env.
"""

from __future__ import annotations

import psycopg

from api.core.config import get_settings


async def create_tables(conn: psycopg.AsyncConnection) -> None:
    settings = get_settings()
    dims = settings.embedding_dimensions

    # pgvector requires the dimension to be specified at column definition time.
    # We use a parameterised string here — dims comes from trusted config, not
    # user input, so direct interpolation is safe.
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS session_chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id  UUID NOT NULL,
            content     TEXT NOT NULL,
            metadata    JSONB DEFAULT '{{}}'::jsonb,
            embedding   vector({dims}),
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at  TIMESTAMPTZ
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS session_meta (
            session_id      UUID PRIMARY KEY,
            topic           TEXT,
            centroid        vector({dims}),
            embedding_dims  INTEGER NOT NULL DEFAULT {dims},
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at      TIMESTAMPTZ
        )
    """)

    # HNSW index for fast ANN queries on chunks, scoped per session.
    # Created with IF NOT EXISTS (Postgres 15+). On older versions this will
    # error harmlessly if the index already exists — acceptable for a prototype.
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON session_chunks
        USING hnsw (embedding vector_cosine_ops)
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_session_id
        ON session_chunks (session_id)
    """)


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

async def insert_chunk(
    conn: psycopg.AsyncConnection,
    session_id: str,
    content: str,
    embedding: list[float],
    metadata: dict | None = None,
    expires_at=None,
) -> None:
    await conn.execute(
        """
        INSERT INTO session_chunks (session_id, content, embedding, metadata, expires_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (session_id, content, embedding, metadata or {}, expires_at),
    )


async def get_top_k_chunks(
    conn: psycopg.AsyncConnection,
    session_id: str,
    query_embedding: list[float],
    k: int = 5,
) -> list[dict]:
    """Return the top-k chunks closest to query_embedding within the session."""
    rows = await conn.execute(
        """
        SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score
        FROM session_chunks
        WHERE session_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, session_id, query_embedding, k),
    )
    return [
        {"content": r[0], "metadata": r[1], "score": r[2]}
        async for r in rows
    ]


# ---------------------------------------------------------------------------
# Session meta helpers
# ---------------------------------------------------------------------------

async def upsert_session_meta(
    conn: psycopg.AsyncConnection,
    session_id: str,
    topic: str,
    centroid: list[float],
    expires_at=None,
) -> None:
    settings = get_settings()
    await conn.execute(
        """
        INSERT INTO session_meta (session_id, topic, centroid, embedding_dims, expires_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (session_id) DO UPDATE
            SET centroid = EXCLUDED.centroid,
                topic    = EXCLUDED.topic
        """,
        (session_id, topic, centroid, settings.embedding_dimensions, expires_at),
    )


async def get_session_meta(
    conn: psycopg.AsyncConnection,
    session_id: str,
) -> dict | None:
    rows = await conn.execute(
        "SELECT session_id, topic, centroid, embedding_dims FROM session_meta WHERE session_id = %s",
        (session_id,),
    )
    row = await rows.fetchone()
    if row is None:
        return None
    return {
        "session_id": str(row[0]),
        "topic": row[1],
        "centroid": row[2],
        "embedding_dims": row[3],
    }

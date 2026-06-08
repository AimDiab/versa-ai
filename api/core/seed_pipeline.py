"""
Seed pipeline — cold path orchestrator.

Runs once per session on the user's first query. Responsibilities:

  1. Build a structured seed prompt from the query
  2. Call the LLM for a comprehensive, section-headed response (heavy call)
  3. Extract the direct answer to return to the user immediately
  4. Chunk the full response
  5. Embed each chunk using the configured embedding provider
  6. Compute the session centroid (mean of all chunk embeddings)
  7. Store chunks and session metadata (including centroid) in pgvector
  8. Return a SeedResult containing the direct answer and storage stats
"""

from __future__ import annotations

from dataclasses import dataclass

from api.core.chunker import chunk_response, Chunk
from api.core.config import get_settings, get_completion_provider, get_embedding_provider
from api.core.seed_prompt import build_seed_prompt, extract_direct_answer
from api.db.client import get_pool
from api.db.schema import insert_chunk, upsert_session_meta
from api.providers.base import CompletionOptions


@dataclass
class SeedResult:
    session_id: str
    direct_answer: str          # Returned to the user as the first message
    chunk_count: int            # Number of chunks stored
    topic: str                  # Derived topic label stored in session_meta


class SeedPipeline:
    """
    Orchestrates the cold-path seed flow for a new session.

    Usage:
        pipeline = SeedPipeline()
        result = await pipeline.run(session_id="uuid", query="What are black holes?")
        # result.direct_answer → stream back to user
    """

    async def run(self, session_id: str, query: str) -> SeedResult:
        settings = get_settings()
        completion_provider = get_completion_provider()
        embedding_provider = get_embedding_provider()

        # 1. Build prompt
        seed_prompt = build_seed_prompt(query)
        options = CompletionOptions(
            model=self._completion_model(settings),
            system_prompt=seed_prompt.system_prompt,
            temperature=0.3,    # Lower temperature for factual, structured output
            max_tokens=4096,
        )

        # 2. Call LLM — single heavy completion, not streaming
        completion = await completion_provider.complete(
            prompt=seed_prompt.user_prompt,
            options=options,
        )
        llm_response = completion.content

        # 3. Extract direct answer for the user
        direct_answer = extract_direct_answer(llm_response)

        # 4. Chunk full response
        chunks = chunk_response(llm_response)

        # 5 & 6. Embed each chunk and collect vectors
        embeddings: list[list[float]] = []
        for chunk in chunks:
            result = await embedding_provider.embed(chunk.content)
            embeddings.append(result.vector)

        # 7. Compute centroid — mean of all embedding vectors
        centroid = _compute_centroid(embeddings)

        # 8. Derive a topic label from the query (first 120 chars, cleaned)
        topic = _derive_topic(query)

        # 9. Store everything in the DB
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.transaction():
                for chunk, embedding in zip(chunks, embeddings):
                    await insert_chunk(
                        conn=conn,
                        session_id=session_id,
                        content=chunk.content,
                        embedding=embedding,
                        metadata=chunk.metadata,
                    )

                await upsert_session_meta(
                    conn=conn,
                    session_id=session_id,
                    topic=topic,
                    centroid=centroid,
                )

        return SeedResult(
            session_id=session_id,
            direct_answer=direct_answer,
            chunk_count=len(chunks),
            topic=topic,
        )

    def _completion_model(self, settings) -> str:
        if settings.active_provider == "anthropic":
            return settings.anthropic_completion_model
        return settings.openai_completion_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """
    Compute the mean of a list of embedding vectors.
    The centroid represents the session's topic as a single point in
    embedding space, used by the relevance guard on subsequent queries.
    """
    if not embeddings:
        raise ValueError("Cannot compute centroid of an empty embedding list.")

    dims = len(embeddings[0])
    n = len(embeddings)
    centroid = [
        sum(e[i] for e in embeddings) / n
        for i in range(dims)
    ]
    return centroid


def _derive_topic(query: str) -> str:
    """
    Derive a short topic label from the user's query.
    Used as a human-readable label in session_meta — not used for logic.
    """
    cleaned = query.strip().rstrip("?").strip()
    return cleaned[:120] if len(cleaned) > 120 else cleaned

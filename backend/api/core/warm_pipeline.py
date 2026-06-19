"""
Warm pipeline — retrieval and response for established sessions.

Runs on every turn after the cold seed has been stored. Responsibilities:

  1. Retrieve top-k chunks from pgvector using the query embedding
  2. Assemble retrieved chunks into a context block
  3. Build a minimal prompt: context + user query
  4. Stream a lite LLM response (format and grammar only — no new knowledge)
  5. Yield text chunks to the caller for streaming to the UI

The warm path makes no heavy LLM calls. The LLM's only job here is to
present the retrieved context conversationally. All factual content comes
from the database.
"""

from __future__ import annotations

from typing import AsyncIterable

from api.core.context_assembler import assemble_context
from api.core.config import get_completion_provider, get_settings
from api.core.session_router import RouteDecision
from api.db.client import get_pool
from api.db.schema import get_top_k_chunks
from api.providers.base import CompletionOptions


_SYSTEM_PROMPT = """\
You are a conversational assistant. Answer the user's question using only \
the context provided below. Be concise and natural — this is a conversation, \
not a report.

Rules:
- Use only information present in the context. Do not introduce outside knowledge.
- If the context does not contain enough information to answer, say so honestly.
- Keep responses brief. Two to four sentences is usually enough.
- Match the user's conversational tone.\
"""

_TOP_K = 5


class WarmPipeline:
    """
    Handles the warm path for an established session.

    Usage:
        pipeline = WarmPipeline()
        async for chunk in pipeline.run(decision):
            # stream chunk to the UI
    """

    async def run(self, decision: RouteDecision) -> AsyncIterable[str]:
        """
        Retrieve relevant context and stream a lite LLM response.

        Args:
            decision: A RouteDecision with path=WARM, carrying the session,
                      query text, and query_embedding.

        Yields:
            Text chunks from the LLM as they stream in.
        """
        settings = get_settings()
        completion_provider = get_completion_provider()

        # 1. Retrieve top-k chunks from pgvector
        pool = await get_pool()
        async with pool.connection() as conn:
            chunks = await get_top_k_chunks(
                conn=conn,
                session_id=decision.session.session_id,
                query_embedding=decision.query_embedding,
                k=_TOP_K,
            )

        # 2. Assemble context
        context = assemble_context(chunks, max_chunks=_TOP_K)

        # 3. Build minimal prompt
        prompt = _build_prompt(decision.query, context.context_text)

        # 4. Stream lite LLM response
        options = CompletionOptions(
            model=self._model(settings),
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=512,
        )

        async for chunk in completion_provider.stream(prompt, options):
            yield chunk

    def _model(self, settings) -> str:
        if settings.active_provider == "anthropic":
            return settings.anthropic_completion_model
        return settings.openai_completion_model


def _build_prompt(query: str, context_text: str) -> str:
    return (
        f"CONTEXT:\n\n{context_text}\n\n"
        f"---\n\n"
        f"QUESTION: {query}"
    )

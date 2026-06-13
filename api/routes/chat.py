"""
Chat endpoint.

Single POST /chat route — the sole HTTP entry point for the frontend.

Flow:
  1. Embed the incoming query (cheap, local or API)
  2. Run the SessionRouter to decide the path
  3. Dispatch to the appropriate pipeline
  4. Stream the response back as Server-Sent Events

SSE event shape (all payloads are JSON):
  {"type": "meta",    "path": "cold"|"warm",  "session_id": "..."}
  {"type": "chunk",   "text": "..."}
  {"type": "deflect", "message": "...",        "session_id": "..."}
  {"type": "done"}
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.core.config import get_embedding_provider
from api.core.seed_pipeline import SeedPipeline
from api.core.session_router import RouteDecision, SessionRouter
from api.core.warm_pipeline import WarmPipeline


router = APIRouter()

_DEFLECT_MESSAGE = (
    "That's a bit outside what we've been exploring. "
    "Want to start a fresh conversation about that?"
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    query: str


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    """Format a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_cold(session_id: str, query: str) -> AsyncGenerator[str, None]:
    """Cold path — seed, then yield the direct answer as a single chunk."""
    pipeline = SeedPipeline()
    result = await pipeline.run(session_id, query)
    yield _sse({"type": "meta", "path": "cold", "session_id": session_id})
    yield _sse({"type": "chunk", "text": result.direct_answer})
    yield _sse({"type": "done"})


async def _stream_warm(decision: RouteDecision) -> AsyncGenerator[str, None]:
    """Warm path — retrieve and stream lite LLM chunks."""
    pipeline = WarmPipeline()
    yield _sse({"type": "meta", "path": "warm", "session_id": decision.session.session_id})
    async for chunk in pipeline.run(decision):
        yield _sse({"type": "chunk", "text": chunk})
    yield _sse({"type": "done"})


async def _stream_deflect(decision: RouteDecision) -> AsyncGenerator[str, None]:
    """Deflect path — return a canned message with no LLM spend."""
    session_id = decision.session.session_id if decision.session else ""
    yield _sse({"type": "deflect", "message": _DEFLECT_MESSAGE, "session_id": session_id})
    yield _sse({"type": "done"})


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Process a user message and stream a response.

    Args:
        request: session_id and query text.

    Returns:
        A StreamingResponse with Content-Type text/event-stream.
        Each event is a JSON object — see module docstring for the schema.
    """
    # 1. Embed the query (used for routing and warm retrieval)
    embedding_provider = get_embedding_provider()
    embed_result = await embedding_provider.embed(request.query)
    query_embedding = embed_result.embedding

    # 2. Route — zero LLM calls, one DB lookup, one cosine check
    session_router = SessionRouter()
    decision = await session_router.route(
        session_id=request.session_id,
        query=request.query,
        query_embedding=query_embedding,
    )

    # 3. Dispatch
    if decision.is_deflect:
        generator = _stream_deflect(decision)
    elif decision.is_cold:
        generator = _stream_cold(request.session_id, request.query)
    else:
        generator = _stream_warm(decision)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )

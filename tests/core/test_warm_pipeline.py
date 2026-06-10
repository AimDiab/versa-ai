"""
Tests for WarmPipeline — retrieval, context assembly, and streaming.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.core.warm_pipeline import WarmPipeline, _build_prompt
from api.core.session_router import RouteDecision, RoutePath, SessionMeta
from api.core.relevance_guard import GuardResult


# ---------------------------------------------------------------------------
# Constants + factories
# ---------------------------------------------------------------------------

SESSION_ID = "00000000-0000-0000-0000-000000000001"
QUERY = "How do black holes form?"
QUERY_EMBEDDING = [1.0, 0.0, 0.0]

MOCK_CHUNKS = [
    {
        "content": "Formation — Stellar Collapse\n\nMassive stars collapse when fuel is exhausted.",
        "metadata": {"section": "Formation", "subsection": "Stellar Collapse", "chunk_index": 0},
        "score": 0.95,
    },
    {
        "content": "Types\n\nStellar, supermassive, and intermediate black holes exist.",
        "metadata": {"section": "Types", "chunk_index": 1},
        "score": 0.82,
    },
]


def make_route_decision(query_embedding=None):
    return RouteDecision(
        path=RoutePath.WARM,
        session=SessionMeta(
            session_id=SESSION_ID,
            topic="black holes",
            centroid=[1.0, 0.0, 0.0],
            embedding_dims=3,
        ),
        guard=GuardResult(is_relevant=True, score=0.95, threshold=0.70),
        query=QUERY,
        query_embedding=query_embedding or QUERY_EMBEDDING,
    )


def make_mock_db_pool(chunks=None):
    mock_conn_ctx = AsyncMock()
    mock_conn = AsyncMock()
    mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pool = AsyncMock()
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    return mock_pool, mock_conn


@pytest.fixture
def mock_settings(monkeypatch):
    from api.core import config
    config.get_settings.cache_clear()
    monkeypatch.setenv("ACTIVE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    yield
    config.get_settings.cache_clear()


async def run_warm_pipeline(chunks=None, stream_chunks=None):
    """Run the warm pipeline with all external deps mocked. Returns collected output."""
    mock_pool, _ = make_mock_db_pool()
    stream_output = stream_chunks or ["Black ", "holes ", "form ", "when..."]

    async def mock_stream(*args, **kwargs):
        for chunk in stream_output:
            yield chunk

    mock_provider = AsyncMock()
    mock_provider.stream = mock_stream

    decision = make_route_decision()

    with patch("api.core.warm_pipeline.get_pool", return_value=AsyncMock(return_value=mock_pool)), \
         patch("api.core.warm_pipeline.get_top_k_chunks", new=AsyncMock(return_value=chunks or MOCK_CHUNKS)), \
         patch("api.core.warm_pipeline.get_completion_provider", return_value=mock_provider):
        pipeline = WarmPipeline()
        collected = []
        async for chunk in pipeline.run(decision):
            collected.append(chunk)

    return collected


# ---------------------------------------------------------------------------
# WarmPipeline.run()
# ---------------------------------------------------------------------------

class TestWarmPipelineRun:
    async def test_yields_text_chunks(self, mock_settings):
        result = await run_warm_pipeline()
        assert len(result) > 0
        assert all(isinstance(c, str) for c in result)

    async def test_streams_all_llm_chunks(self, mock_settings):
        stream_chunks = ["The ", "answer ", "is here."]
        result = await run_warm_pipeline(stream_chunks=stream_chunks)
        assert result == stream_chunks

    async def test_retrieves_chunks_for_correct_session(self, mock_settings):
        mock_pool, _ = make_mock_db_pool()
        get_chunks_mock = AsyncMock(return_value=MOCK_CHUNKS)

        async def mock_stream(*args, **kwargs):
            yield "response"

        mock_provider = AsyncMock()
        mock_provider.stream = mock_stream
        decision = make_route_decision()

        with patch("api.core.warm_pipeline.get_pool", return_value=AsyncMock(return_value=mock_pool)), \
             patch("api.core.warm_pipeline.get_top_k_chunks", get_chunks_mock), \
             patch("api.core.warm_pipeline.get_completion_provider", return_value=mock_provider):
            pipeline = WarmPipeline()
            async for _ in pipeline.run(decision):
                pass

        call_kwargs = get_chunks_mock.call_args.kwargs
        assert call_kwargs["session_id"] == SESSION_ID

    async def test_passes_query_embedding_to_retrieval(self, mock_settings):
        mock_pool, _ = make_mock_db_pool()
        get_chunks_mock = AsyncMock(return_value=MOCK_CHUNKS)

        async def mock_stream(*args, **kwargs):
            yield "response"

        mock_provider = AsyncMock()
        mock_provider.stream = mock_stream
        decision = make_route_decision(query_embedding=QUERY_EMBEDDING)

        with patch("api.core.warm_pipeline.get_pool", return_value=AsyncMock(return_value=mock_pool)), \
             patch("api.core.warm_pipeline.get_top_k_chunks", get_chunks_mock), \
             patch("api.core.warm_pipeline.get_completion_provider", return_value=mock_provider):
            pipeline = WarmPipeline()
            async for _ in pipeline.run(decision):
                pass

        call_kwargs = get_chunks_mock.call_args.kwargs
        assert call_kwargs["query_embedding"] == QUERY_EMBEDDING

    async def test_empty_chunks_still_streams_response(self, mock_settings):
        result = await run_warm_pipeline(chunks=[], stream_chunks=["I don't have enough context."])
        assert result == ["I don't have enough context."]


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_context(self):
        prompt = _build_prompt("How do black holes form?", "Some context here.")
        assert "Some context here." in prompt

    def test_contains_query(self):
        prompt = _build_prompt("How do black holes form?", "Some context.")
        assert "How do black holes form?" in prompt

    def test_context_before_question(self):
        prompt = _build_prompt("The question", "The context")
        assert prompt.index("The context") < prompt.index("The question")

    def test_prompt_labels_sections(self):
        prompt = _build_prompt("query", "context")
        assert "CONTEXT" in prompt
        assert "QUESTION" in prompt

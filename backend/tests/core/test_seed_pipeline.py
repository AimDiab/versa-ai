"""
Tests for SeedPipeline — orchestration with mocked LLM, embedding provider, and DB.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.core.seed_pipeline import SeedPipeline, SeedResult, _derive_topic
from api.providers.base import CompletionResult, EmbeddingResult, UsageStats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE = """\
## Direct Answer
Black holes are regions of spacetime where gravity is so strong that \
nothing, not even light, can escape.

## Formation
Stars that exceed a certain mass can collapse into black holes when \
they exhaust their nuclear fuel.

## Types
Stellar black holes, supermassive black holes, and intermediate black holes \
are the three main categories.
"""

SESSION_ID = "00000000-0000-0000-0000-000000000001"
QUERY = "What are black holes?"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_mock_completion(content: str = MOCK_LLM_RESPONSE):
    return CompletionResult(
        content=content,
        model="claude-haiku-4-5",
        usage=UsageStats(input_tokens=100, output_tokens=500),
    )


def make_mock_embedding(dims: int = 384):
    return EmbeddingResult(
        vector=[0.1] * dims,
        model="BAAI/bge-small-en-v1.5",
        usage=UsageStats(),
    )


def make_mock_db_pool():
    """
    Build a mock pool that satisfies the two nested async context managers
    the pipeline uses:

        async with pool.connection() as conn:
            async with conn.transaction():

    Returns (pool, conn) so individual tests can attach extra behaviour
    to the connection (e.g. capturing insert_chunk calls).
    """
    mock_tx = AsyncMock()
    mock_tx.__aenter__ = AsyncMock(return_value=None)
    mock_tx.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_tx)

    mock_conn_ctx = AsyncMock()
    mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pool = AsyncMock()
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    return mock_pool, mock_conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(monkeypatch):
    from api.core import config
    config.get_settings.cache_clear()
    monkeypatch.setenv("ACTIVE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def mock_completion_provider():
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=make_mock_completion())
    return provider


@pytest.fixture
def mock_embedding_provider():
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=make_mock_embedding())
    return provider


# ---------------------------------------------------------------------------
# Helper — run the pipeline with all external deps mocked
# ---------------------------------------------------------------------------

async def run_pipeline(
    completion_provider,
    embedding_provider,
    insert_chunk_mock=None,
    upsert_meta_mock=None,
):
    pool, _ = make_mock_db_pool()

    with patch("api.core.seed_pipeline.get_completion_provider", return_value=completion_provider), \
         patch("api.core.seed_pipeline.get_embedding_provider", return_value=embedding_provider), \
         patch("api.core.seed_pipeline.get_pool", return_value=AsyncMock(return_value=pool)), \
         patch("api.core.seed_pipeline.insert_chunk", new=insert_chunk_mock or AsyncMock()), \
         patch("api.core.seed_pipeline.upsert_session_meta", new=upsert_meta_mock or AsyncMock()):
        return await SeedPipeline().run(SESSION_ID, QUERY)


# ---------------------------------------------------------------------------
# SeedPipeline.run()
# ---------------------------------------------------------------------------

class TestSeedPipelineRun:
    async def test_returns_seed_result(self, mock_settings, mock_completion_provider, mock_embedding_provider):
        result = await run_pipeline(mock_completion_provider, mock_embedding_provider)
        assert isinstance(result, SeedResult)

    async def test_direct_answer_extracted(self, mock_settings, mock_completion_provider, mock_embedding_provider):
        result = await run_pipeline(mock_completion_provider, mock_embedding_provider)
        assert "nothing" in result.direct_answer.lower() or "gravity" in result.direct_answer.lower()
        assert "Formation" not in result.direct_answer

    async def test_chunk_count_reflects_stored_chunks(self, mock_settings, mock_completion_provider, mock_embedding_provider):
        insert_mock = AsyncMock()
        result = await run_pipeline(mock_completion_provider, mock_embedding_provider, insert_chunk_mock=insert_mock)
        assert result.chunk_count == insert_mock.call_count
        assert result.chunk_count > 0

    async def test_embed_called_once_per_chunk(self, mock_settings, mock_completion_provider, mock_embedding_provider):
        result = await run_pipeline(mock_completion_provider, mock_embedding_provider)
        assert mock_embedding_provider.embed.call_count == result.chunk_count

    async def test_session_id_stored_in_result(self, mock_settings, mock_completion_provider, mock_embedding_provider):
        result = await run_pipeline(mock_completion_provider, mock_embedding_provider)
        assert result.session_id == SESSION_ID


# ---------------------------------------------------------------------------
# _derive_topic
# ---------------------------------------------------------------------------

class TestDeriveTopic:
    def test_strips_trailing_question_mark(self):
        assert _derive_topic("What are black holes?") == "What are black holes"

    def test_strips_whitespace(self):
        assert _derive_topic("  black holes  ") == "black holes"

    def test_truncates_long_queries(self):
        assert len(_derive_topic("a" * 200)) <= 120

    def test_short_query_unchanged(self):
        assert _derive_topic("black holes") == "black holes"

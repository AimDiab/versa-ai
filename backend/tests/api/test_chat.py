"""
Tests for POST /api/chat.

All external dependencies are mocked:
  - get_embedding_provider  → returns a mock that yields a fixed embedding
  - SessionRouter.route     → controlled per-test to return cold/warm/deflect
  - SeedPipeline.run        → returns a fixed SeedResult
  - WarmPipeline.run        → async-generates fixed text chunks

SSE parsing helper: each "data: {...}\n\n" line is parsed back into a dict
so tests can assert on typed event payloads rather than raw strings.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from api.core.session_router import RouteDecision, RoutePath, SessionMeta
from api.core.relevance_guard import GuardResult
from api.core.seed_pipeline import SeedResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_ID = "00000000-0000-0000-0000-000000000001"
QUERY = "How do black holes form?"
QUERY_EMBEDDING = [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sse(response_text: str) -> list[dict]:
    """Parse SSE response body into a list of event dicts."""
    events = []
    for line in response_text.strip().splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def make_session_meta():
    return SessionMeta(
        session_id=SESSION_ID,
        topic="black holes",
        centroid=QUERY_EMBEDDING,
        embedding_dims=3,
    )


def make_warm_decision():
    return RouteDecision(
        path=RoutePath.WARM,
        session=make_session_meta(),
        guard=GuardResult(is_relevant=True, score=0.95, threshold=0.70),
        query=QUERY,
        query_embedding=QUERY_EMBEDDING,
    )


def make_cold_decision():
    return RouteDecision(
        path=RoutePath.COLD,
        query=QUERY,
        query_embedding=QUERY_EMBEDDING,
    )


def make_deflect_decision():
    return RouteDecision(
        path=RoutePath.DEFLECT,
        session=make_session_meta(),
        guard=GuardResult(is_relevant=False, score=0.20, threshold=0.70),
        query="What is the weather today?",
        query_embedding=QUERY_EMBEDDING,
    )


def make_mock_embedding_provider():
    mock = AsyncMock()
    result = MagicMock()
    result.embedding = QUERY_EMBEDDING
    mock.embed = AsyncMock(return_value=result)
    return mock


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
def client():
    # Use TestClient in a context that skips the lifespan (no real DB needed).
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Warm path
# ---------------------------------------------------------------------------

class TestWarmPath:
    def test_returns_200(self, client, mock_settings):
        async def mock_stream(*args, **kwargs):
            yield "response"

        mock_warm = MagicMock()
        mock_warm.run = mock_stream
        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=make_warm_decision())

        with patch("api.routes.chat.get_embedding_provider", return_value=make_mock_embedding_provider()), \
             patch("api.routes.chat.SessionRouter", return_value=mock_router), \
             patch("api.routes.chat.WarmPipeline", return_value=mock_warm):
            response = client.post("/api/chat", json={"session_id": SESSION_ID, "query": QUERY})

        assert response.status_code == 200

    def test_content_type_is_sse(self, client, mock_settings):
        async def mock_stream(*args, **kwargs):
            yield "response"

        mock_warm = MagicMock()
        mock_warm.run = mock_stream
        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=make_warm_decision())

        with patch("api.routes.chat.get_embedding_provider", return_value=make_mock_embedding_provider()), \
             patch("api.routes.chat.SessionRouter", return_value=mock_router), \
             patch("api.routes.chat.WarmPipeline", return_value=mock_warm):
            response = client.post("/api/chat", json={"session_id": SESSION_ID, "query": QUERY})

        assert "text/event-stream" in response.headers["content-type"]

    def test_meta_event_has_warm_path(self, client, mock_settings):
        async def mock_stream(*args, **kwargs):
            yield "response"

        mock_warm = MagicMock()
        mock_warm.run = mock_stream
        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=make_warm_decision())

        with patch("api.routes.chat.get_embedding_provider", return_value=make_mock_embedding_provider()), \
             patch("api.routes.chat.SessionRouter", return_value=mock_router), \
             patch("api.routes.chat.WarmPipeline", return_value=mock_warm):
            response = client.post("/api/chat", json={"session_id": SESSION_ID, "query": QUERY})

        events = parse_sse(response.text)
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["path"] == "warm"
        assert meta["session_id"] == SESSION_ID

    def test_chunk_events_contain_text(self, client, mock_settings):
        stream_chunks = ["Black ", "holes ", "form."]

        async def mock_stream(*args, **kwargs):
            for chunk in stream_chunks:
                yield chunk

        mock_warm = MagicMock()
        mock_warm.run = mock_stream
        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=make_warm_decision())

        with patch("api.routes.chat.get_embedding_provider", return_value=make_mock_embedding_provider()), \
             patch("api.routes.chat.SessionRouter", return_value=mock_router), \
             patch("api.routes.chat.WarmPipeline", return_value=mock_warm):
            response = client.post("/api/chat", json={"session_id": SESSION_ID, "query": QUERY})

        events = parse_sse(response.text)
        chunk_events = [e for e in events if e["type"] == "chunk"]
        assert [e["text"] for e in chunk_events] == stream_chunks

    def test_done_event_is_last(self, client, mock_settings):
        async def mock_stream(*args, **kwargs):
            yield "response"

        mock_warm = MagicMock()
        mock_warm.run = mock_stream
        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=make_warm_decision())

        with patch("api.routes.chat.get_embedding_provider", return_value=make_mock_embedding_provider()), \
             patch("api.routes.chat.SessionRouter", return_value=mock_router), \
             patch("api.routes.chat.WarmPipeline", return_value=mock_warm):
            response = client.post("/api/chat", json={"session_id": SESSION_ID, "query": QUERY})

        events = parse_sse(response.text)
        assert events[-1]["type"] == "done"


# ---------------------------------------------------------------------------
# Cold path
# ---------------------------------------------------------------------------

class TestColdPath:
    def _run(self, client, mock_settings, direct_answer="Black holes form when stars collapse."):
        seed_result = SeedResult(
            session_id=SESSION_ID,
            direct_answer=direct_answer,
            chunk_count=4,
            topic="black holes",
        )
        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=make_cold_decision())
        mock_seed = AsyncMock()
        mock_seed.run = AsyncMock(return_value=seed_result)

        with patch("api.routes.chat.get_embedding_provider", return_value=make_mock_embedding_provider()), \
             patch("api.routes.chat.SessionRouter", return_value=mock_router), \
             patch("api.routes.chat.SeedPipeline", return_value=mock_seed):
            return client.post("/api/chat", json={"session_id": SESSION_ID, "query": QUERY})

    def test_meta_event_has_cold_path(self, client, mock_settings):
        response = self._run(client, mock_settings)
        events = parse_sse(response.text)
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["path"] == "cold"

    def test_chunk_event_contains_direct_answer(self, client, mock_settings):
        answer = "Black holes form when stars collapse."
        response = self._run(client, mock_settings, direct_answer=answer)
        events = parse_sse(response.text)
        chunk = next(e for e in events if e["type"] == "chunk")
        assert chunk["text"] == answer

    def test_done_event_present(self, client, mock_settings):
        response = self._run(client, mock_settings)
        events = parse_sse(response.text)
        assert any(e["type"] == "done" for e in events)


# ---------------------------------------------------------------------------
# Deflect path
# ---------------------------------------------------------------------------

class TestDeflectPath:
    def _run(self, client, mock_settings):
        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=make_deflect_decision())

        with patch("api.routes.chat.get_embedding_provider", return_value=make_mock_embedding_provider()), \
             patch("api.routes.chat.SessionRouter", return_value=mock_router):
            return client.post("/api/chat", json={"session_id": SESSION_ID, "query": "What's the weather?"})

    def test_deflect_event_type(self, client, mock_settings):
        response = self._run(client, mock_settings)
        events = parse_sse(response.text)
        assert any(e["type"] == "deflect" for e in events)

    def test_deflect_message_not_empty(self, client, mock_settings):
        response = self._run(client, mock_settings)
        events = parse_sse(response.text)
        deflect = next(e for e in events if e["type"] == "deflect")
        assert len(deflect["message"]) > 0

    def test_deflect_includes_session_id(self, client, mock_settings):
        response = self._run(client, mock_settings)
        events = parse_sse(response.text)
        deflect = next(e for e in events if e["type"] == "deflect")
        assert deflect["session_id"] == SESSION_ID

    def test_done_event_follows_deflect(self, client, mock_settings):
        response = self._run(client, mock_settings)
        events = parse_sse(response.text)
        assert events[-1]["type"] == "done"

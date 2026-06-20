"""
Tests for SessionRouter.

The database and RelevanceGuard are mocked — no real DB connections made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.core.session_router import SessionRouter, RoutePath, RouteDecision, SessionMeta
from api.core.relevance_guard import GuardResult


SESSION_ID = "00000000-0000-0000-0000-000000000001"

MOCK_SESSION_META = {
    "session_id": SESSION_ID,
    "topic": "black holes",
    "centroid": [1.0, 0.0, 0.0],
    "embedding_dims": 384,
}

ON_TOPIC_EMBEDDING = [1.0, 0.0, 0.0]      # identical to centroid → score = 1.0
OFF_TOPIC_EMBEDDING = [0.0, 1.0, 0.0]     # orthogonal to centroid → score = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_router(threshold: float = 0.70) -> SessionRouter:
    from api.core.relevance_guard import RelevanceGuard
    return SessionRouter(guard=RelevanceGuard(threshold=threshold))


def mock_pool(session_meta: dict | None):
    """
    Return a mock pool whose connection() context manager yields a
    connection that returns session_meta from get_session_meta().
    """
    mock_conn = AsyncMock()
    mock_conn_ctx = AsyncMock()
    mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pool_obj = AsyncMock()
    mock_pool_obj.connection = MagicMock(return_value=mock_conn_ctx)

    return mock_pool_obj, mock_conn, session_meta


# ---------------------------------------------------------------------------
# Cold path
# ---------------------------------------------------------------------------

class TestColdPath:
    async def test_cold_when_no_session_exists(self):
        router = make_router()
        mock_pool_obj, mock_conn, _ = mock_pool(None)

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool_obj)), \
             patch("api.core.session_router.get_session_meta", return_value=AsyncMock(return_value=None)):
            decision = await router.route(SESSION_ID, "What are black holes?", ON_TOPIC_EMBEDDING)

        assert decision.path == RoutePath.COLD
        assert decision.is_cold is True
        assert decision.session is None

    async def test_cold_decision_has_no_guard_result(self):
        router = make_router()

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta", return_value=AsyncMock(return_value=None)):
            decision = await router.route(SESSION_ID, "Hello", ON_TOPIC_EMBEDDING)

        assert decision.guard is None

    async def test_cold_decision_stores_query(self):
        router = make_router()

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta", return_value=AsyncMock(return_value=None)):
            decision = await router.route(SESSION_ID, "Tell me about neutron stars", ON_TOPIC_EMBEDDING)

        assert decision.query == "Tell me about neutron stars"


# ---------------------------------------------------------------------------
# Warm path
# ---------------------------------------------------------------------------

class TestWarmPath:
    async def test_warm_when_session_exists_and_on_topic(self):
        router = make_router(threshold=0.70)

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta",
                   return_value=AsyncMock(return_value=MOCK_SESSION_META)):
            decision = await router.route(SESSION_ID, "How massive are black holes?", ON_TOPIC_EMBEDDING)

        assert decision.path == RoutePath.WARM
        assert decision.is_warm is True

    async def test_warm_decision_has_session_populated(self):
        router = make_router(threshold=0.70)

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta",
                   return_value=AsyncMock(return_value=MOCK_SESSION_META)):
            decision = await router.route(SESSION_ID, "query", ON_TOPIC_EMBEDDING)

        assert isinstance(decision.session, SessionMeta)
        assert decision.session.session_id == SESSION_ID
        assert decision.session.topic == "black holes"

    async def test_warm_decision_has_guard_result(self):
        router = make_router(threshold=0.70)

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta",
                   return_value=AsyncMock(return_value=MOCK_SESSION_META)):
            decision = await router.route(SESSION_ID, "query", ON_TOPIC_EMBEDDING)

        assert isinstance(decision.guard, GuardResult)
        assert decision.guard.is_relevant is True
        assert decision.guard.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Deflect path
# ---------------------------------------------------------------------------

class TestDeflectPath:
    async def test_deflect_when_session_exists_but_off_topic(self):
        router = make_router(threshold=0.70)

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta",
                   return_value=AsyncMock(return_value=MOCK_SESSION_META)):
            decision = await router.route(SESSION_ID, "What is the best pasta recipe?", OFF_TOPIC_EMBEDDING)

        assert decision.path == RoutePath.DEFLECT
        assert decision.is_deflect is True

    async def test_deflect_decision_has_low_guard_score(self):
        router = make_router(threshold=0.70)

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta",
                   return_value=AsyncMock(return_value=MOCK_SESSION_META)):
            decision = await router.route(SESSION_ID, "Best pasta?", OFF_TOPIC_EMBEDDING)

        assert decision.guard.score < decision.guard.threshold
        assert decision.guard.is_relevant is False

    async def test_deflect_still_populates_session(self):
        router = make_router(threshold=0.70)

        with patch("api.core.session_router.get_pool", return_value=AsyncMock(return_value=mock_pool(None)[0])), \
             patch("api.core.session_router.get_session_meta",
                   return_value=AsyncMock(return_value=MOCK_SESSION_META)):
            decision = await router.route(SESSION_ID, "Best pasta?", OFF_TOPIC_EMBEDDING)

        # Session is preserved even on deflect — user can return to it
        assert decision.session is not None
        assert decision.session.topic == "black holes"


# ---------------------------------------------------------------------------
# RouteDecision convenience properties
# ---------------------------------------------------------------------------

class TestRouteDecisionProperties:
    def test_is_cold_true_only_for_cold(self):
        d = RouteDecision(path=RoutePath.COLD)
        assert d.is_cold is True
        assert d.is_warm is False
        assert d.is_deflect is False

    def test_is_warm_true_only_for_warm(self):
        d = RouteDecision(path=RoutePath.WARM)
        assert d.is_warm is True
        assert d.is_cold is False
        assert d.is_deflect is False

    def test_is_deflect_true_only_for_deflect(self):
        d = RouteDecision(path=RoutePath.DEFLECT)
        assert d.is_deflect is True
        assert d.is_cold is False
        assert d.is_warm is False

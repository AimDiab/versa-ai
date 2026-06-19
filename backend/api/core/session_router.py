"""
Session Router.

The single entry point for every incoming user query.
Decides which path to take and returns a RouteDecision:

  cold    — no session exists yet; caller should run the seed pipeline
  warm    — session exists and query is on-topic; caller should retrieve + respond
  deflect — session exists but query is off-topic; caller should return deflect message
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from api.core.config import get_settings
from api.core.relevance_guard import RelevanceGuard, GuardResult
from api.db.client import get_pool
from api.db.schema import get_session_meta


class RoutePath(str, Enum):
    COLD = "cold"
    WARM = "warm"
    DEFLECT = "deflect"


@dataclass
class SessionMeta:
    session_id: str
    topic: str
    centroid: list[float]
    embedding_dims: int


@dataclass
class RouteDecision:
    path: RoutePath
    session: SessionMeta | None = None        # None on cold path
    guard: GuardResult | None = None          # set on warm + deflect
    query: str = ""
    query_embedding: list[float] = field(default_factory=list)  # carried for warm retrieval

    @property
    def is_cold(self) -> bool:
        return self.path == RoutePath.COLD

    @property
    def is_warm(self) -> bool:
        return self.path == RoutePath.WARM

    @property
    def is_deflect(self) -> bool:
        return self.path == RoutePath.DEFLECT


class SessionRouter:
    """
    Routes an incoming query to the correct pipeline path.

    Typical usage (in a FastAPI endpoint):

        router = SessionRouter()
        decision = await router.route(session_id, query, query_embedding)

        if decision.is_cold:
            # run seed pipeline, store chunks + centroid, then respond
        elif decision.is_warm:
            # retrieve top-k chunks, format response with lite LLM call
        elif decision.is_deflect:
            # return deflect message, prompt user to start new conversation
    """

    def __init__(self, guard: RelevanceGuard | None = None):
        settings = get_settings()
        self._guard = guard or RelevanceGuard(threshold=settings.relevance_threshold)

    async def route(
        self,
        session_id: str,
        query: str,
        query_embedding: list[float],
    ) -> RouteDecision:
        """
        Determine the path for this query.

        query_embedding should already be computed by the caller using the
        configured embedding provider — the router itself makes no LLM calls.

        Steps:
          1. Look up session_meta for session_id in the DB.
          2. If not found → cold.
          3. If found → run relevance guard against session centroid.
          4. Guard passes → warm. Guard fails → deflect.
        """
        pool = await get_pool()
        async with pool.connection() as conn:
            meta_row = await get_session_meta(conn, session_id)

        if meta_row is None:
            return RouteDecision(
                path=RoutePath.COLD,
                query=query,
                query_embedding=query_embedding,
            )

        session = SessionMeta(
            session_id=meta_row["session_id"],
            topic=meta_row["topic"],
            centroid=meta_row["centroid"],
            embedding_dims=meta_row["embedding_dims"],
        )

        guard_result = self._guard.check(
            centroid=session.centroid,
            query_embedding=query_embedding,
        )

        path = RoutePath.WARM if guard_result.is_relevant else RoutePath.DEFLECT

        return RouteDecision(
            path=path,
            session=session,
            guard=guard_result,
            query=query,
            query_embedding=query_embedding,
        )

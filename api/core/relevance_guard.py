"""
Relevance Guard.

Pure cosine similarity check — no DB calls, no LLM calls.
Given a session centroid and an incoming query embedding, returns whether
the query is on-topic for this session.

The guard fires before any LLM spend on the warm path, so off-topic
queries cost only one embed() call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GuardResult:
    is_relevant: bool
    score: float          # cosine similarity, 0.0–1.0
    threshold: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a value in [-1, 1]; in practice embedding vectors are non-negative
    so results fall in [0, 1].
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RelevanceGuard:
    """
    Checks whether an incoming query is relevant to the current session.

    Usage:
        guard = RelevanceGuard(threshold=0.70)
        result = guard.check(session_centroid, query_embedding)
        if not result.is_relevant:
            # deflect
    """

    def __init__(self, threshold: float = 0.70):
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold

    def check(
        self,
        centroid: list[float],
        query_embedding: list[float],
    ) -> GuardResult:
        """
        Compare query_embedding against the session centroid.
        Returns GuardResult with is_relevant=True if score >= threshold.
        """
        score = cosine_similarity(centroid, query_embedding)
        return GuardResult(
            is_relevant=score >= self.threshold,
            score=round(score, 4),
            threshold=self.threshold,
        )

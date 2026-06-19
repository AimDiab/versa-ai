"""
Tests for RelevanceGuard and cosine_similarity.

Pure math — no mocks, no I/O.
"""

import math
import pytest

from api.core.relevance_guard import RelevanceGuard, cosine_similarity, GuardResult
from tests.conftest import make_unit_vector


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_0(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_1(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_known_angle_45_degrees(self):
        # Two unit vectors 45° apart → cos(45°) ≈ 0.707
        a = make_unit_vector(2, angle_deg=0)
        b = make_unit_vector(2, angle_deg=45)
        assert cosine_similarity(a, b) == pytest.approx(math.cos(math.radians(45)), abs=1e-6)

    def test_zero_vector_returns_0(self):
        zero = [0.0, 0.0, 0.0]
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(zero, v) == 0.0

    def test_uniform_vectors_return_1(self):
        # All-equal components → same direction → similarity = 1
        a = [0.5] * 384
        b = [0.5] * 384
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_high_dimensional_vectors(self):
        a = [1.0] + [0.0] * 1535
        b = [0.9] + [0.0] * 1535
        # Both point in same direction → similarity = 1
        assert cosine_similarity(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RelevanceGuard construction
# ---------------------------------------------------------------------------

class TestRelevanceGuardInit:
    def test_valid_threshold_accepted(self):
        guard = RelevanceGuard(threshold=0.75)
        assert guard.threshold == 0.75

    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            RelevanceGuard(threshold=0.0)

    def test_threshold_one_raises(self):
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            RelevanceGuard(threshold=1.0)

    def test_default_threshold_is_0_70(self):
        guard = RelevanceGuard()
        assert guard.threshold == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# RelevanceGuard.check()
# ---------------------------------------------------------------------------

class TestRelevanceGuardCheck:
    def test_identical_vectors_pass(self):
        guard = RelevanceGuard(threshold=0.70)
        v = [1.0, 0.0, 0.0]
        result = guard.check(centroid=v, query_embedding=v)
        assert result.is_relevant is True
        assert result.score == pytest.approx(1.0)

    def test_orthogonal_vectors_deflect(self):
        guard = RelevanceGuard(threshold=0.70)
        centroid = [1.0, 0.0, 0.0]
        query = [0.0, 1.0, 0.0]
        result = guard.check(centroid=centroid, query_embedding=query)
        assert result.is_relevant is False
        assert result.score == pytest.approx(0.0)

    def test_score_at_threshold_passes(self):
        # Construct two vectors with exactly cosine similarity = threshold
        threshold = 0.70
        angle = math.degrees(math.acos(threshold))
        centroid = make_unit_vector(2, angle_deg=0)
        query = make_unit_vector(2, angle_deg=angle)
        guard = RelevanceGuard(threshold=threshold)
        result = guard.check(centroid=centroid, query_embedding=query)
        assert result.is_relevant is True

    def test_score_just_below_threshold_deflects(self):
        threshold = 0.70
        angle = math.degrees(math.acos(threshold)) + 1.0  # slightly more than threshold angle
        centroid = make_unit_vector(2, angle_deg=0)
        query = make_unit_vector(2, angle_deg=angle)
        guard = RelevanceGuard(threshold=threshold)
        result = guard.check(centroid=centroid, query_embedding=query)
        assert result.is_relevant is False

    def test_guard_result_contains_threshold(self):
        guard = RelevanceGuard(threshold=0.65)
        v = [1.0, 0.0]
        result = guard.check(centroid=v, query_embedding=v)
        assert result.threshold == 0.65

    def test_score_is_rounded_to_4_decimal_places(self):
        guard = RelevanceGuard(threshold=0.70)
        # Use vectors where similarity is not a round number
        a = make_unit_vector(2, angle_deg=30)
        b = make_unit_vector(2, angle_deg=0)
        result = guard.check(centroid=a, query_embedding=b)
        assert result.score == round(result.score, 4)

    def test_returns_guard_result_dataclass(self):
        guard = RelevanceGuard(threshold=0.70)
        v = [1.0, 0.0]
        result = guard.check(centroid=v, query_embedding=v)
        assert isinstance(result, GuardResult)

    def test_high_dimensional_similar_vectors_pass(self, embedding_384, similar_embedding_384):
        guard = RelevanceGuard(threshold=0.70)
        result = guard.check(centroid=embedding_384, query_embedding=similar_embedding_384)
        assert result.is_relevant is True

    def test_high_dimensional_orthogonal_vectors_deflect(self, embedding_384, orthogonal_embedding_384):
        guard = RelevanceGuard(threshold=0.70)
        result = guard.check(centroid=embedding_384, query_embedding=orthogonal_embedding_384)
        assert result.is_relevant is False

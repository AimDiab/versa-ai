"""
Shared pytest fixtures.
"""

import math
import pytest


# ---------------------------------------------------------------------------
# Sample embedding vectors
# ---------------------------------------------------------------------------

def make_unit_vector(dims: int, angle_deg: float = 0.0) -> list[float]:
    """
    Return a unit vector of the given dimensionality.
    For dims >= 2, angle_deg rotates in the first two dimensions.
    Useful for constructing vectors with a known cosine similarity.
    """
    v = [0.0] * dims
    angle_rad = math.radians(angle_deg)
    v[0] = math.cos(angle_rad)
    if dims > 1:
        v[1] = math.sin(angle_rad)
    return v


@pytest.fixture
def embedding_384() -> list[float]:
    """A normalised 384-dim embedding (fastembed dimensions)."""
    base = [0.1] * 384
    norm = math.sqrt(sum(x ** 2 for x in base))
    return [x / norm for x in base]


@pytest.fixture
def embedding_1536() -> list[float]:
    """A normalised 1536-dim embedding (OpenAI text-embedding-3-small dimensions)."""
    base = [0.1] * 1536
    norm = math.sqrt(sum(x ** 2 for x in base))
    return [x / norm for x in base]


@pytest.fixture
def similar_embedding_384(embedding_384) -> list[float]:
    """A 384-dim embedding close to embedding_384 (high cosine similarity)."""
    perturbed = [x + 0.001 for x in embedding_384]
    norm = math.sqrt(sum(x ** 2 for x in perturbed))
    return [x / norm for x in perturbed]


@pytest.fixture
def orthogonal_embedding_384() -> list[float]:
    """A 384-dim embedding orthogonal to embedding_384 (cosine similarity ≈ 0)."""
    v = [0.0] * 384
    v[0] = 1.0   # embedding_384 has all equal components; this is near-orthogonal
    return v


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def openai_env(monkeypatch):
    """Set env vars for an OpenAI-configured run."""
    monkeypatch.setenv("ACTIVE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)


@pytest.fixture
def anthropic_env(monkeypatch):
    """Set env vars for an Anthropic-configured run."""
    monkeypatch.setenv("ACTIVE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)

"""
Tests for api/core/config.py.

Each test clears the lru_cache on get_settings() so env var changes
take effect cleanly between tests.
"""

import pytest

from api.core.config import get_settings, _load_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache before and after every test in this module."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# ACTIVE_PROVIDER validation
# ---------------------------------------------------------------------------

class TestActiveProvider:
    def test_openai_provider_accepted(self, openai_env):
        s = _load_settings()
        assert s.active_provider == "openai"

    def test_anthropic_provider_accepted(self, anthropic_env):
        s = _load_settings()
        assert s.active_provider == "anthropic"

    def test_invalid_provider_raises(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_PROVIDER", "gemini")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        with pytest.raises(ValueError, match="ACTIVE_PROVIDER must be 'openai' or 'anthropic'"):
            _load_settings()

    def test_empty_provider_raises(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_PROVIDER", "")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        with pytest.raises(ValueError, match="ACTIVE_PROVIDER must be 'openai' or 'anthropic'"):
            _load_settings()

    def test_provider_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_PROVIDER", "OpenAI")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        s = _load_settings()
        assert s.active_provider == "openai"


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

class TestApiKeyValidation:
    def test_openai_key_required_for_openai_provider(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_PROVIDER", "openai")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            _load_settings()

    def test_anthropic_key_required_for_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_PROVIDER", "anthropic")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            _load_settings()

    def test_unused_key_can_be_absent(self, openai_env):
        # Anthropic key not set, but that's fine when ACTIVE_PROVIDER=openai
        s = _load_settings()
        assert s.anthropic_api_key is None

    def test_api_key_stored_in_settings(self, openai_env):
        s = _load_settings()
        assert s.openai_api_key == "sk-test-openai"


# ---------------------------------------------------------------------------
# DATABASE_URL
# ---------------------------------------------------------------------------

class TestDatabaseUrl:
    def test_missing_database_url_raises(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError, match="DATABASE_URL is not set"):
            _load_settings()

    def test_database_url_stored(self, openai_env):
        s = _load_settings()
        assert s.database_url == "postgresql://user:pass@localhost:5432/testdb"


# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------

class TestEmbeddingDimensions:
    def test_openai_defaults_to_1536(self, openai_env):
        s = _load_settings()
        assert s.embedding_dimensions == 1536

    def test_anthropic_defaults_to_384(self, anthropic_env):
        s = _load_settings()
        assert s.embedding_dimensions == 384

    def test_explicit_override_respected(self, openai_env, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
        s = _load_settings()
        assert s.embedding_dimensions == 768


# ---------------------------------------------------------------------------
# Relevance threshold
# ---------------------------------------------------------------------------

class TestRelevanceThreshold:
    def test_defaults_to_0_70(self, openai_env):
        s = _load_settings()
        assert s.relevance_threshold == pytest.approx(0.70)

    def test_explicit_override_respected(self, openai_env, monkeypatch):
        monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.85")
        s = _load_settings()
        assert s.relevance_threshold == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

class TestModelDefaults:
    def test_openai_completion_model_default(self, openai_env):
        s = _load_settings()
        assert s.openai_completion_model == "gpt-4o-mini"

    def test_anthropic_completion_model_default(self, anthropic_env):
        s = _load_settings()
        assert s.anthropic_completion_model == "claude-haiku-4-5"

    def test_openai_embedding_model_default(self, openai_env):
        s = _load_settings()
        assert s.openai_embedding_model == "text-embedding-3-small"

    def test_fastembed_model_default(self, anthropic_env):
        s = _load_settings()
        assert s.fastembed_model == "BAAI/bge-small-en-v1.5"

    def test_completion_model_override(self, openai_env, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPLETION_MODEL", "gpt-4o")
        s = _load_settings()
        assert s.openai_completion_model == "gpt-4o"

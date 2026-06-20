"""
Configuration layer.

Loads .env on import, validates required values, and exposes:
  - Settings  — typed config object
  - settings  — singleton instance
  - get_completion_provider()  — returns the configured ILLMProvider for completions
  - get_embedding_provider()   — returns the configured ILLMProvider for embeddings
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


ProviderName = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class Settings:
    active_provider: ProviderName

    # API keys — one will be None depending on active_provider
    openai_api_key: str | None
    anthropic_api_key: str | None

    # Model overrides (fall back to provider defaults if not set)
    openai_completion_model: str
    anthropic_completion_model: str
    openai_embedding_model: str
    fastembed_model: str

    # Database
    database_url: str

    # Embedding vector dimensions — must match the embedding model in use.
    # openai/text-embedding-3-small = 1536, fastembed/bge-small-en-v1.5 = 384
    embedding_dimensions: int

    # Relevance guard cosine similarity threshold (0–1).
    # Queries scoring below this are deflected as off-topic.
    relevance_threshold: float


def _load_settings() -> Settings:
    raw = os.getenv("ACTIVE_PROVIDER", "").strip().lower()
    if raw not in ("openai", "anthropic"):
        raise ValueError(
            f"ACTIVE_PROVIDER must be 'openai' or 'anthropic', got: '{raw}'. "
            "Check your .env file."
        )
    active_provider: ProviderName = raw  # type: ignore[assignment]

    openai_key = os.getenv("OPENAI_API_KEY") or None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or None

    if active_provider == "openai" and not openai_key:
        raise ValueError(
            "ACTIVE_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Add it to your .env file."
        )
    if active_provider == "anthropic" and not anthropic_key:
        raise ValueError(
            "ACTIVE_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file."
        )

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError(
            "DATABASE_URL is not set. "
            "Add it to your .env file, e.g. postgresql://user:pass@localhost:5432/versaai"
        )

    # Derive default embedding dimensions from provider unless explicitly set
    default_dims = 1536 if active_provider == "openai" else 384
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", str(default_dims)))

    relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.70"))

    return Settings(
        active_provider=active_provider,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        openai_completion_model=os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini"),
        anthropic_completion_model=os.getenv("ANTHROPIC_COMPLETION_MODEL", "claude-haiku-4-5"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        fastembed_model=os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        database_url=database_url,
        embedding_dimensions=embedding_dimensions,
        relevance_threshold=relevance_threshold,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _load_settings()


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------

def get_completion_provider():
    """
    Returns the ILLMProvider to use for all completion calls (complete/stream).
    Instantiated fresh each call but cheap — clients are stateless.
    """
    from api.providers import AnthropicProvider, OpenAIProvider

    s = get_settings()
    if s.active_provider == "anthropic":
        return AnthropicProvider(
            api_key=s.anthropic_api_key,
            default_model=s.anthropic_completion_model,
        )
    return OpenAIProvider(
        api_key=s.openai_api_key,
        default_model=s.openai_completion_model,
        embedding_model=s.openai_embedding_model,
    )


def get_embedding_provider():
    """
    Returns the ILLMProvider to use for all embed() calls.

    - ACTIVE_PROVIDER=openai   → OpenAIProvider (text-embedding-3-small)
    - ACTIVE_PROVIDER=anthropic → FastEmbedProvider (local, no key needed)
    """
    from api.providers import OpenAIProvider
    from api.providers.fastembed_provider import FastEmbedProvider

    s = get_settings()
    if s.active_provider == "openai":
        return OpenAIProvider(
            api_key=s.openai_api_key,
            embedding_model=s.openai_embedding_model,
        )
    return FastEmbedProvider(model_name=s.fastembed_model)

"""
FastEmbed provider — local embeddings, no API key required.

Used automatically when ACTIVE_PROVIDER=anthropic, since Anthropic
does not expose an embedding API. Runs fully offline after the model
is downloaded on first use (~40MB for bge-small-en-v1.5).

complete() and stream() are not supported — this provider exists
solely to satisfy the embed() contract.
"""

from typing import AsyncIterable, Optional

from .base import (
    ILLMProvider,
    CompletionOptions,
    CompletionResult,
    EmbeddingResult,
    Model,
    UsageStats,
)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Embedding dimension for bge-small-en-v1.5 — update if you switch models.
# text-embedding-3-small (OpenAI) = 1536 dims
# bge-small-en-v1.5 (fastembed)   = 384 dims
# Both are compatible with pgvector; just keep the dimension consistent
# across ingestion and query time.
_MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class FastEmbedProvider(ILLMProvider):
    """
    ILLMProvider adapter wrapping fastembed for local CPU-based embeddings.

    The fastembed TextEmbedding model is loaded lazily on first embed() call
    and cached for the lifetime of the process.

    Usage:
        provider = FastEmbedProvider()
        result = await provider.embed("black holes and spacetime curvature")
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None  # lazy init

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    async def embed(self, text: str) -> EmbeddingResult:
        import asyncio

        model = self._get_model()

        # fastembed is synchronous — run in executor to avoid blocking the
        # FastAPI event loop.
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: list(model.embed([text])),
        )

        vector = embeddings[0].tolist()

        return EmbeddingResult(
            vector=vector,
            model=self._model_name,
            usage=UsageStats(input_tokens=0, output_tokens=0),
        )

    async def complete(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
    ) -> CompletionResult:
        raise NotImplementedError(
            "FastEmbedProvider only supports embed(). "
            "Use AnthropicProvider or OpenAIProvider for completions."
        )

    async def stream(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
    ) -> AsyncIterable[str]:
        raise NotImplementedError(
            "FastEmbedProvider only supports embed(). "
            "Use AnthropicProvider or OpenAIProvider for completions."
        )

    async def list_models(self) -> list[Model]:
        return [
            Model(
                id=self._model_name,
                name=self._model_name,
                provider="fastembed",
                context_window=512,
                supports_streaming=False,
                supports_embeddings=True,
            )
        ]

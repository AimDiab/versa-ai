"""
ILLMProvider — base interface and shared types.

All LLM provider adapters must implement this interface.
The orchestrator only ever talks to ILLMProvider; it never imports a
concrete provider directly.

Embedding note:
  Anthropic does not expose an embedding API. embed() is defined on the
  interface so the orchestrator can call it uniformly, but providers that
  don't support embeddings should raise NotImplementedError. In practice,
  the session router is configured with a dedicated embedding provider
  (defaulting to OpenAIProvider) separately from whichever provider
  handles completions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterable, Optional


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class CompletionOptions:
    """
    Parameters forwarded to the underlying model call.
    Providers map these to their own SDK equivalents and silently ignore
    options they don't support.
    """
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: Optional[str] = None
    # Arbitrary extra kwargs passed through verbatim to the provider SDK.
    extra: dict = field(default_factory=dict)


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CompletionResult:
    content: str
    model: str
    usage: UsageStats = field(default_factory=UsageStats)


@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str
    usage: UsageStats = field(default_factory=UsageStats)


@dataclass
class Model:
    id: str
    name: str
    provider: str
    context_window: Optional[int] = None
    supports_streaming: bool = True
    supports_embeddings: bool = False


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class ILLMProvider(ABC):
    """
    Abstract base class for all LLM provider adapters.

    Implement all four methods for a fully capable provider.
    Raise NotImplementedError on methods the underlying service doesn't
    support (e.g. embed() on Anthropic).
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
    ) -> CompletionResult:
        """
        Single-shot completion. Returns the full response once the model
        finishes. Use this for cold-path seed generation where the entire
        dataset is needed before the parser begins chunking.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
    ) -> AsyncIterable[str]:
        """
        Streaming completion. Yields text chunks as they arrive.
        Use this for warm-path responses so the UI renders incrementally.
        """
        ...

    @abstractmethod
    async def embed(
        self,
        text: str,
    ) -> EmbeddingResult:
        """
        Produce a vector embedding for the given text.

        Providers that don't support embeddings should raise NotImplementedError.
        The session router routes embed() calls to the configured embedding
        provider (OpenAIProvider by default), which may differ from the
        provider used for completions.
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[Model]:
        """
        Return available models from this provider. Used by the session
        router to validate the configured model exists before a seed call.
        """
        ...

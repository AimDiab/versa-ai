"""
OpenAI provider adapter.

Wraps the `openai` Python SDK behind ILLMProvider.
Supports complete(), stream(), and embed().

This is the default embedding provider for the session router — even if
completions are handled by AnthropicProvider, embed() calls should be
routed here using text-embedding-3-small.
"""

from openai import AsyncOpenAI
from typing import AsyncIterable, Optional

from .base import (
    ILLMProvider,
    CompletionOptions,
    CompletionResult,
    EmbeddingResult,
    Model,
    UsageStats,
)


_KNOWN_MODELS = [
    Model(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        context_window=128_000,
        supports_streaming=True,
        supports_embeddings=False,
    ),
    Model(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        provider="openai",
        context_window=128_000,
        supports_streaming=True,
        supports_embeddings=False,
    ),
    Model(
        id="o3",
        name="o3",
        provider="openai",
        context_window=200_000,
        supports_streaming=True,
        supports_embeddings=False,
    ),
    Model(
        id="text-embedding-3-small",
        name="Text Embedding 3 Small",
        provider="openai",
        context_window=8_191,
        supports_streaming=False,
        supports_embeddings=True,
    ),
    Model(
        id="text-embedding-3-large",
        name="Text Embedding 3 Large",
        provider="openai",
        context_window=8_191,
        supports_streaming=False,
        supports_embeddings=True,
    ),
]

_DEFAULT_COMPLETION_MODEL = "gpt-4o-mini"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIProvider(ILLMProvider):
    """
    ILLMProvider adapter for OpenAI models.

    Handles both completions and embeddings. Use as the embedding provider
    alongside any completion provider (including AnthropicProvider).

    Usage:
        provider = OpenAIProvider(api_key="sk-...")
        result = await provider.complete("Summarise quantum entanglement")
        embedding = await provider.embed("quantum entanglement")
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = _DEFAULT_COMPLETION_MODEL,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ):
        self._client = AsyncOpenAI(api_key=api_key)
        self._default_model = default_model
        self._embedding_model = embedding_model

    def _resolve_options(self, options: Optional[CompletionOptions]) -> CompletionOptions:
        if options is None:
            return CompletionOptions(model=self._default_model)
        return options

    async def complete(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
    ) -> CompletionResult:
        opts = self._resolve_options(options)

        messages = []
        if opts.system_prompt:
            messages.append({"role": "system", "content": opts.system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=opts.model,
            messages=messages,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            **opts.extra,
        )

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return CompletionResult(
            content=choice.message.content or "",
            model=response.model,
            usage=UsageStats(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    async def stream(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
    ) -> AsyncIterable[str]:
        opts = self._resolve_options(options)

        messages = []
        if opts.system_prompt:
            messages.append({"role": "system", "content": opts.system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=opts.model,
            messages=messages,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            stream=True,
            **opts.extra,
        )

        async for chunk in await self._client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def embed(self, text: str) -> EmbeddingResult:
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )

        return EmbeddingResult(
            vector=response.data[0].embedding,
            model=response.model,
            usage=UsageStats(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0,
            ),
        )

    async def list_models(self) -> list[Model]:
        return list(_KNOWN_MODELS)

"""
Anthropic provider adapter.

Wraps the `anthropic` Python SDK behind ILLMProvider.
Supports complete() and stream(). embed() is not supported by Anthropic
and will raise NotImplementedError — route embedding calls to OpenAIProvider.
"""

import anthropic

from .base import (
    ILLMProvider,
    CompletionOptions,
    CompletionResult,
    EmbeddingResult,
    Model,
    UsageStats,
)
from typing import AsyncIterable, Optional


# Models available at time of writing — list_models() returns these.
# Update as Anthropic releases new versions.
_KNOWN_MODELS = [
    Model(
        id="claude-opus-4-5",
        name="Claude Opus 4.5",
        provider="anthropic",
        context_window=200_000,
        supports_streaming=True,
        supports_embeddings=False,
    ),
    Model(
        id="claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        context_window=200_000,
        supports_streaming=True,
        supports_embeddings=False,
    ),
    Model(
        id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        context_window=200_000,
        supports_streaming=True,
        supports_embeddings=False,
    ),
]

_DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicProvider(ILLMProvider):
    """
    ILLMProvider adapter for Anthropic's Claude models.

    Usage:
        provider = AnthropicProvider(api_key="sk-ant-...")
        result = await provider.complete("Tell me about black holes")
    """

    def __init__(self, api_key: str, default_model: str = _DEFAULT_MODEL):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._default_model = default_model

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

        kwargs = dict(
            model=opts.model,
            max_tokens=opts.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **opts.extra,
        )
        if opts.system_prompt:
            kwargs["system"] = opts.system_prompt

        # temperature is not supported on thinking models; guard it
        if opts.temperature is not None:
            kwargs["temperature"] = opts.temperature

        response = await self._client.messages.create(**kwargs)

        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        return CompletionResult(
            content=content,
            model=response.model,
            usage=UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    async def stream(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
    ) -> AsyncIterable[str]:
        opts = self._resolve_options(options)

        kwargs = dict(
            model=opts.model,
            max_tokens=opts.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **opts.extra,
        )
        if opts.system_prompt:
            kwargs["system"] = opts.system_prompt
        if opts.temperature is not None:
            kwargs["temperature"] = opts.temperature

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, text: str) -> EmbeddingResult:
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Configure an OpenAIProvider as the embedding provider instead."
        )

    async def list_models(self) -> list[Model]:
        return list(_KNOWN_MODELS)

"""
Tests for AnthropicProvider.

The anthropic SDK is mocked throughout — no real API calls are made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from api.providers.anthropic_provider import AnthropicProvider, _KNOWN_MODELS, _DEFAULT_MODEL
from api.providers.base import CompletionOptions, CompletionResult, UsageStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return AnthropicProvider(api_key="sk-ant-test", default_model=_DEFAULT_MODEL)


def _make_mock_response(content: str, model: str = "claude-haiku-4-5",
                        input_tokens: int = 10, output_tokens: int = 20):
    """Build a mock that looks like an Anthropic messages response."""
    block = MagicMock()
    block.text = content
    response = MagicMock()
    response.content = [block]
    response.model = model
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:
    async def test_returns_completion_result(self, provider):
        mock_response = _make_mock_response("Black holes warp spacetime.")

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=mock_response)):
            result = await provider.complete("Tell me about black holes")

        assert isinstance(result, CompletionResult)
        assert result.content == "Black holes warp spacetime."
        assert result.model == "claude-haiku-4-5"

    async def test_usage_stats_populated(self, provider):
        mock_response = _make_mock_response("Hello", input_tokens=5, output_tokens=15)

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=mock_response)):
            result = await provider.complete("Hi")

        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 15
        assert result.usage.total_tokens == 20

    async def test_system_prompt_forwarded(self, provider):
        mock_response = _make_mock_response("ok")
        create_mock = AsyncMock(return_value=mock_response)

        opts = CompletionOptions(model=_DEFAULT_MODEL, system_prompt="You are a helpful tutor.")
        with patch.object(provider._client.messages, "create", new=create_mock):
            await provider.complete("Explain gravity", options=opts)

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["system"] == "You are a helpful tutor."

    async def test_uses_default_model_when_no_options(self, provider):
        mock_response = _make_mock_response("ok")
        create_mock = AsyncMock(return_value=mock_response)

        with patch.object(provider._client.messages, "create", new=create_mock):
            await provider.complete("Hello")

        assert create_mock.call_args.kwargs["model"] == _DEFAULT_MODEL

    async def test_uses_model_from_options(self, provider):
        mock_response = _make_mock_response("ok", model="claude-opus-4-5")
        create_mock = AsyncMock(return_value=mock_response)

        opts = CompletionOptions(model="claude-opus-4-5")
        with patch.object(provider._client.messages, "create", new=create_mock):
            await provider.complete("Hello", options=opts)

        assert create_mock.call_args.kwargs["model"] == "claude-opus-4-5"

    async def test_multiple_content_blocks_concatenated(self, provider):
        block1, block2 = MagicMock(), MagicMock()
        block1.text = "Hello "
        block2.text = "world"
        response = MagicMock()
        response.content = [block1, block2]
        response.model = _DEFAULT_MODEL
        response.usage.input_tokens = 5
        response.usage.output_tokens = 5

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=response)):
            result = await provider.complete("Hi")

        assert result.content == "Hello world"


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------

class TestStream:
    async def test_yields_text_chunks(self, provider):
        chunks = ["Hello", " ", "world"]

        async def mock_text_stream():
            for chunk in chunks:
                yield chunk

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_stream_ctx.text_stream = mock_text_stream()

        with patch.object(provider._client.messages, "stream", return_value=mock_stream_ctx):
            collected = []
            async for chunk in provider.stream("Tell me something"):
                collected.append(chunk)

        assert collected == ["Hello", " ", "world"]

    async def test_system_prompt_forwarded_in_stream(self, provider):
        async def mock_text_stream():
            yield "ok"

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_stream_ctx.text_stream = mock_text_stream()

        stream_mock = MagicMock(return_value=mock_stream_ctx)
        opts = CompletionOptions(model=_DEFAULT_MODEL, system_prompt="Be concise.")

        with patch.object(provider._client.messages, "stream", stream_mock):
            async for _ in provider.stream("Hi", options=opts):
                pass

        assert stream_mock.call_args.kwargs["system"] == "Be concise."


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

class TestEmbed:
    async def test_raises_not_implemented(self, provider):
        with pytest.raises(NotImplementedError, match="Anthropic does not provide an embedding API"):
            await provider.embed("some text")


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------

class TestListModels:
    async def test_returns_known_models(self, provider):
        models = await provider.list_models()
        assert models == list(_KNOWN_MODELS)

    async def test_no_model_supports_embeddings(self, provider):
        models = await provider.list_models()
        assert all(not m.supports_embeddings for m in models)

    async def test_all_models_support_streaming(self, provider):
        models = await provider.list_models()
        assert all(m.supports_streaming for m in models)

    async def test_all_models_have_anthropic_provider(self, provider):
        models = await provider.list_models()
        assert all(m.provider == "anthropic" for m in models)

"""
Tests for OpenAIProvider.

The openai SDK is mocked throughout — no real API calls are made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.providers.openai_provider import (
    OpenAIProvider,
    _KNOWN_MODELS,
    _DEFAULT_COMPLETION_MODEL,
    _DEFAULT_EMBEDDING_MODEL,
)
from api.providers.base import CompletionOptions, CompletionResult, EmbeddingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return OpenAIProvider(api_key="sk-test-openai")


def _make_completion_response(content: str, model: str = "gpt-4o-mini",
                               prompt_tokens: int = 10, completion_tokens: int = 20):
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def _make_embedding_response(vector: list[float], model: str = "text-embedding-3-small",
                              prompt_tokens: int = 8):
    data = MagicMock()
    data.embedding = vector
    response = MagicMock()
    response.data = [data]
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    return response


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:
    async def test_returns_completion_result(self, provider):
        mock_response = _make_completion_response("The speed of light is 3×10⁸ m/s.")

        with patch.object(provider._client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_response)):
            result = await provider.complete("What is the speed of light?")

        assert isinstance(result, CompletionResult)
        assert result.content == "The speed of light is 3×10⁸ m/s."
        assert result.model == "gpt-4o-mini"

    async def test_usage_stats_populated(self, provider):
        mock_response = _make_completion_response("ok", prompt_tokens=12, completion_tokens=8)

        with patch.object(provider._client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_response)):
            result = await provider.complete("Hi")

        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 8
        assert result.usage.total_tokens == 20

    async def test_system_prompt_added_as_system_message(self, provider):
        mock_response = _make_completion_response("ok")
        create_mock = AsyncMock(return_value=mock_response)

        opts = CompletionOptions(model=_DEFAULT_COMPLETION_MODEL, system_prompt="You are a physicist.")
        with patch.object(provider._client.chat.completions, "create", new=create_mock):
            await provider.complete("Explain quarks", options=opts)

        messages = create_mock.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are a physicist."}
        assert messages[1] == {"role": "user", "content": "Explain quarks"}

    async def test_no_system_message_when_not_provided(self, provider):
        mock_response = _make_completion_response("ok")
        create_mock = AsyncMock(return_value=mock_response)

        with patch.object(provider._client.chat.completions, "create", new=create_mock):
            await provider.complete("Hello")

        messages = create_mock.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    async def test_uses_default_model_when_no_options(self, provider):
        mock_response = _make_completion_response("ok")
        create_mock = AsyncMock(return_value=mock_response)

        with patch.object(provider._client.chat.completions, "create", new=create_mock):
            await provider.complete("Hello")

        assert create_mock.call_args.kwargs["model"] == _DEFAULT_COMPLETION_MODEL


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------

class TestStream:
    async def test_yields_text_chunks(self, provider):
        chunks = ["The ", "answer ", "is 42."]

        async def mock_stream():
            for text in chunks:
                chunk = MagicMock()
                chunk.choices[0].delta.content = text
                yield chunk

        with patch.object(provider._client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_stream())):
            collected = []
            async for chunk in provider.stream("What is the answer?"):
                collected.append(chunk)

        assert collected == chunks

    async def test_skips_none_delta_content(self, provider):
        async def mock_stream():
            for text in [None, "Hello", None, " world"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = text
                yield chunk

        with patch.object(provider._client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_stream())):
            collected = []
            async for chunk in provider.stream("Hi"):
                collected.append(chunk)

        assert collected == ["Hello", " world"]


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

class TestEmbed:
    async def test_returns_embedding_result(self, provider):
        vector = [0.1] * 1536
        mock_response = _make_embedding_response(vector)

        with patch.object(provider._client.embeddings, "create",
                          new=AsyncMock(return_value=mock_response)):
            result = await provider.embed("black holes")

        assert isinstance(result, EmbeddingResult)
        assert result.vector == vector
        assert result.model == "text-embedding-3-small"

    async def test_uses_configured_embedding_model(self, provider):
        vector = [0.1] * 1536
        mock_response = _make_embedding_response(vector, model="text-embedding-3-large")
        create_mock = AsyncMock(return_value=mock_response)

        custom_provider = OpenAIProvider(
            api_key="sk-test",
            embedding_model="text-embedding-3-large",
        )
        with patch.object(custom_provider._client.embeddings, "create", new=create_mock):
            await custom_provider.embed("some text")

        assert create_mock.call_args.kwargs["model"] == "text-embedding-3-large"

    async def test_usage_stats_populated(self, provider):
        mock_response = _make_embedding_response([0.1] * 1536, prompt_tokens=6)

        with patch.object(provider._client.embeddings, "create",
                          new=AsyncMock(return_value=mock_response)):
            result = await provider.embed("hello")

        assert result.usage.input_tokens == 6
        assert result.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------

class TestListModels:
    async def test_returns_known_models(self, provider):
        models = await provider.list_models()
        assert models == list(_KNOWN_MODELS)

    async def test_embedding_models_flagged(self, provider):
        models = await provider.list_models()
        embedding_models = [m for m in models if m.supports_embeddings]
        assert any("embedding" in m.id for m in embedding_models)

    async def test_all_models_have_openai_provider(self, provider):
        models = await provider.list_models()
        assert all(m.provider == "openai" for m in models)

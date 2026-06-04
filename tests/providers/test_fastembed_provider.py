"""
Tests for FastEmbedProvider.

fastembed is mocked — no model is downloaded during tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from api.providers.fastembed_provider import FastEmbedProvider, _DEFAULT_MODEL
from api.providers.base import EmbeddingResult


@pytest.fixture
def provider():
    return FastEmbedProvider()


class TestEmbed:
    async def test_returns_embedding_result(self, provider):
        fake_vector = np.array([0.1] * 384)
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([fake_vector])

        with patch("api.providers.fastembed_provider.FastEmbedProvider._get_model",
                   return_value=mock_model):
            result = await provider.embed("black holes")

        assert isinstance(result, EmbeddingResult)
        assert len(result.vector) == 384
        assert result.model == _DEFAULT_MODEL

    async def test_vector_converted_to_list(self, provider):
        fake_vector = np.array([0.5] * 384)
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([fake_vector])

        with patch("api.providers.fastembed_provider.FastEmbedProvider._get_model",
                   return_value=mock_model):
            result = await provider.embed("test")

        assert isinstance(result.vector, list)

    async def test_uses_configured_model_name(self):
        custom_model = "BAAI/bge-base-en-v1.5"
        provider = FastEmbedProvider(model_name=custom_model)

        fake_vector = np.array([0.1] * 768)
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([fake_vector])

        with patch("api.providers.fastembed_provider.FastEmbedProvider._get_model",
                   return_value=mock_model):
            result = await provider.embed("test")

        assert result.model == custom_model

    async def test_usage_stats_are_zero(self, provider):
        fake_vector = np.array([0.1] * 384)
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([fake_vector])

        with patch("api.providers.fastembed_provider.FastEmbedProvider._get_model",
                   return_value=mock_model):
            result = await provider.embed("test")

        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    async def test_model_loaded_lazily(self, provider):
        assert provider._model is None  # not loaded yet

        fake_vector = np.array([0.1] * 384)
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([fake_vector])

        with patch("fastembed.TextEmbedding", return_value=mock_model):
            await provider.embed("trigger load")

        assert provider._model is not None


class TestCompleteAndStream:
    async def test_complete_raises_not_implemented(self, provider):
        with pytest.raises(NotImplementedError, match="FastEmbedProvider only supports embed"):
            await provider.complete("hello")

    async def test_stream_raises_not_implemented(self, provider):
        with pytest.raises(NotImplementedError, match="FastEmbedProvider only supports embed"):
            async for _ in provider.stream("hello"):
                pass


class TestListModels:
    async def test_returns_one_model(self, provider):
        models = await provider.list_models()
        assert len(models) == 1

    async def test_model_supports_embeddings(self, provider):
        models = await provider.list_models()
        assert models[0].supports_embeddings is True

    async def test_model_does_not_support_streaming(self, provider):
        models = await provider.list_models()
        assert models[0].supports_streaming is False

    async def test_provider_is_fastembed(self, provider):
        models = await provider.list_models()
        assert models[0].provider == "fastembed"

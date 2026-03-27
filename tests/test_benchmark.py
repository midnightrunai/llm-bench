"""Tests for llm_bench.benchmark module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_bench.benchmark import BenchmarkConfig, run_benchmark
from llm_bench.providers.base import ProviderResponse


def make_mock_response(model: str, provider: str, latency_ms: float = 100.0) -> ProviderResponse:
    return ProviderResponse(
        model=model,
        provider=provider,
        content="This is a test response.",
        input_tokens=50,
        output_tokens=20,
        latency_ms=latency_ms,
    )


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig(
            models=["gpt-4o"],
            prompts=["Hello world"],
        )
        assert config.n_runs == 3
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.judge_model is None

    def test_custom_config(self):
        config = BenchmarkConfig(
            models=["gpt-4o", "claude-3-5-sonnet"],
            prompts=["Test prompt 1", "Test prompt 2"],
            n_runs=5,
            temperature=0.7,
            judge_model="gpt-4o-mini",
        )
        assert len(config.models) == 2
        assert len(config.prompts) == 2
        assert config.n_runs == 5


class TestRunBenchmark:
    @pytest.mark.asyncio
    async def test_basic_benchmark(self):
        """Test that benchmark runs and returns correct structure."""
        config = BenchmarkConfig(
            models=["gpt-4o"],
            prompts=["Test prompt"],
            n_runs=2,
        )

        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.timed_complete = AsyncMock(
            return_value=make_mock_response("gpt-4o", "openai")
        )
        mock_provider.cost_per_1k_requests = MagicMock(return_value=0.05)

        with patch("llm_bench.benchmark.resolve_provider", return_value=mock_provider):
            result = await run_benchmark(config)

        assert "gpt-4o" in result.metrics
        m = result.metrics["gpt-4o"]
        assert m.n_total == 2
        assert m.n_success == 2
        assert m.latency_p50_ms > 0

    @pytest.mark.asyncio
    async def test_multiple_models(self):
        """Test benchmarking multiple models."""
        config = BenchmarkConfig(
            models=["gpt-4o", "claude-3-5-sonnet"],
            prompts=["Hello"],
            n_runs=1,
        )

        def make_provider(name, provider_name):
            mock = MagicMock()
            mock.name = provider_name
            mock.timed_complete = AsyncMock(
                return_value=make_mock_response(name, provider_name)
            )
            mock.cost_per_1k_requests = MagicMock(return_value=0.01)
            return mock

        providers = {
            "gpt-4o": make_provider("gpt-4o", "openai"),
            "claude-3-5-sonnet": make_provider("claude-3-5-sonnet", "anthropic"),
        }

        with patch("llm_bench.benchmark.resolve_provider", side_effect=lambda m: providers[m]):
            result = await run_benchmark(config)

        assert "gpt-4o" in result.metrics
        assert "claude-3-5-sonnet" in result.metrics

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that errors are captured and don't crash the benchmark."""
        config = BenchmarkConfig(
            models=["bad-model"],
            prompts=["Test"],
            n_runs=1,
        )

        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.timed_complete = AsyncMock(
            return_value=ProviderResponse(
                model="bad-model",
                provider="test",
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=100.0,
                error="API Error: Invalid key",
            )
        )
        mock_provider.cost_per_1k_requests = MagicMock(return_value=0.0)

        with patch("llm_bench.benchmark.resolve_provider", return_value=mock_provider):
            result = await run_benchmark(config)

        m = result.metrics["bad-model"]
        assert m.n_errors == 1
        assert m.n_success == 0

    @pytest.mark.asyncio
    async def test_to_dict(self):
        """Test that results serialize to dict correctly."""
        config = BenchmarkConfig(
            models=["gpt-4o"],
            prompts=["Test"],
            n_runs=1,
        )

        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.timed_complete = AsyncMock(
            return_value=make_mock_response("gpt-4o", "openai", latency_ms=123.4)
        )
        mock_provider.cost_per_1k_requests = MagicMock(return_value=0.05)

        with patch("llm_bench.benchmark.resolve_provider", return_value=mock_provider):
            result = await run_benchmark(config)

        d = result.to_dict()
        assert "timestamp" in d
        assert "results" in d
        assert "gpt-4o" in d["results"]
        assert d["results"]["gpt-4o"]["n_success"] == 1

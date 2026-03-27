"""Tests for llm_bench.metrics module."""

import pytest
from llm_bench.metrics import ModelMetrics, _percentile
from llm_bench.providers.base import ProviderResponse


def make_response(latency_ms: float, input_tokens: int = 100, output_tokens: int = 50, error: str | None = None) -> ProviderResponse:
    return ProviderResponse(
        model="test-model",
        provider="test",
        content="test response" if not error else "",
        input_tokens=input_tokens if not error else 0,
        output_tokens=output_tokens if not error else 0,
        latency_ms=latency_ms,
        error=error,
    )


class TestPercentile:
    def test_single_value(self):
        assert _percentile([100.0], 50) == 100.0

    def test_p50_of_sorted_list(self):
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        assert _percentile(values, 50) == 300.0

    def test_p95(self):
        values = sorted([float(i) for i in range(1, 101)])
        result = _percentile(values, 95)
        assert 94.0 <= result <= 96.0

    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_two_values(self):
        assert _percentile([100.0, 200.0], 50) == 150.0


class TestModelMetrics:
    def test_compute_basic(self):
        metrics = ModelMetrics(model="gpt-4o", provider="openai")
        metrics.responses = [
            make_response(100.0),
            make_response(200.0),
            make_response(150.0),
        ]
        metrics.compute()

        assert metrics.n_total == 3
        assert metrics.n_success == 3
        assert metrics.n_errors == 0
        assert metrics.latency_p50_ms == 150.0
        assert metrics.latency_mean_ms == pytest.approx(150.0)
        assert metrics.latency_min_ms == 100.0
        assert metrics.latency_max_ms == 200.0

    def test_compute_with_errors(self):
        metrics = ModelMetrics(model="gpt-4o", provider="openai")
        metrics.responses = [
            make_response(100.0),
            make_response(0.0, error="API error"),
            make_response(150.0),
        ]
        metrics.compute()

        assert metrics.n_total == 3
        assert metrics.n_success == 2
        assert metrics.n_errors == 1
        assert len(metrics.errors) == 1
        assert "API error" in metrics.errors[0]

    def test_compute_all_errors(self):
        metrics = ModelMetrics(model="gpt-4o", provider="openai")
        metrics.responses = [
            make_response(0.0, error="Error 1"),
            make_response(0.0, error="Error 2"),
        ]
        metrics.compute()

        assert metrics.n_success == 0
        assert metrics.latency_p50_ms == 0.0

    def test_to_dict(self):
        metrics = ModelMetrics(model="gpt-4o", provider="openai")
        metrics.responses = [make_response(100.0, input_tokens=50, output_tokens=25)]
        metrics.compute()

        d = metrics.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["provider"] == "openai"
        assert d["n_success"] == 1
        assert "latency" in d
        assert "tokens" in d
        assert d["tokens"]["avg_input"] == 50.0
        assert d["tokens"]["avg_output"] == 25.0

    def test_compute_empty(self):
        metrics = ModelMetrics(model="gpt-4o", provider="openai")
        metrics.compute()
        assert metrics.n_total == 0
        assert metrics.latency_p50_ms == 0.0

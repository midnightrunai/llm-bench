"""Statistical metrics computation for llm-bench."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from llm_bench.providers.base import ProviderResponse


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model across N runs."""

    model: str
    provider: str
    responses: list[ProviderResponse] = field(default_factory=list)

    # Computed fields (populated by compute())
    n_total: int = 0
    n_success: int = 0
    n_errors: int = 0

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0

    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_total_tokens: float = 0.0

    cost_per_1k_requests: float = 0.0
    quality_score: float | None = None

    errors: list[str] = field(default_factory=list)

    def compute(self, provider_instance=None) -> "ModelMetrics":
        """Compute all metrics from raw responses."""
        self.n_total = len(self.responses)
        successful = [r for r in self.responses if r.success]
        failed = [r for r in self.responses if not r.success]

        self.n_success = len(successful)
        self.n_errors = len(failed)
        self.errors = [r.error for r in failed if r.error]

        if not successful:
            return self

        latencies = sorted(r.latency_ms for r in successful)
        self.latency_mean_ms = statistics.mean(latencies)
        self.latency_min_ms = latencies[0]
        self.latency_max_ms = latencies[-1]
        self.latency_p50_ms = _percentile(latencies, 50)
        self.latency_p95_ms = _percentile(latencies, 95)

        self.avg_input_tokens = statistics.mean(r.input_tokens for r in successful)
        self.avg_output_tokens = statistics.mean(r.output_tokens for r in successful)
        self.avg_total_tokens = statistics.mean(r.total_tokens for r in successful)

        if provider_instance:
            self.cost_per_1k_requests = provider_instance.cost_per_1k_requests(
                model=self.model,
                avg_input_tokens=self.avg_input_tokens,
                avg_output_tokens=self.avg_output_tokens,
            )

        return self

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_errors": self.n_errors,
            "latency": {
                "p50_ms": round(self.latency_p50_ms, 2),
                "p95_ms": round(self.latency_p95_ms, 2),
                "mean_ms": round(self.latency_mean_ms, 2),
                "min_ms": round(self.latency_min_ms, 2),
                "max_ms": round(self.latency_max_ms, 2),
            },
            "tokens": {
                "avg_input": round(self.avg_input_tokens, 1),
                "avg_output": round(self.avg_output_tokens, 1),
                "avg_total": round(self.avg_total_tokens, 1),
            },
            "cost_per_1k_requests_usd": round(self.cost_per_1k_requests, 4),
            "quality_score": self.quality_score,
            "errors": self.errors,
        }


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_values[-1]
    fraction = index - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])

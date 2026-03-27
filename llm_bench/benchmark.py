"""Core benchmark runner for llm-bench."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable

from llm_bench.judge import JudgeScore, score_responses_batch
from llm_bench.metrics import ModelMetrics
from llm_bench.providers import resolve_provider
from llm_bench.providers.base import BaseProvider, ProviderResponse


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    models: list[str]
    prompts: list[str]
    system: str | None = None
    n_runs: int = 3
    temperature: float = 0.0
    max_tokens: int = 1024
    max_concurrent: int = 5
    judge_model: str | None = None
    timeout_seconds: float = 60.0
    provider_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Full results for a benchmark run."""

    config: BenchmarkConfig
    metrics: dict[str, ModelMetrics]  # model -> metrics
    duration_seconds: float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "config": {
                "models": self.config.models,
                "n_prompts": len(self.config.prompts),
                "n_runs": self.config.n_runs,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "judge_model": self.config.judge_model,
            },
            "results": {model: m.to_dict() for model, m in self.metrics.items()},
        }


async def run_benchmark(
    config: BenchmarkConfig,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> BenchmarkResult:
    """Run a full benchmark and return results."""
    import datetime

    start = time.perf_counter()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    # Resolve providers
    providers: dict[str, BaseProvider] = {}
    for model in config.models:
        if model in config.provider_overrides:
            from llm_bench.providers import PROVIDER_REGISTRY
            provider_name = config.provider_overrides[model]
            providers[model] = PROVIDER_REGISTRY[provider_name]()
        else:
            providers[model] = resolve_provider(model)

    # Initialize metrics collectors
    metrics: dict[str, ModelMetrics] = {
        model: ModelMetrics(model=model, provider=providers[model].name)
        for model in config.models
    }

    # Build all tasks
    semaphore = asyncio.Semaphore(config.max_concurrent)
    total_calls = len(config.models) * len(config.prompts) * config.n_runs
    completed_calls = 0

    async def run_single(model: str, prompt: str) -> ProviderResponse:
        nonlocal completed_calls
        provider = providers[model]
        async with semaphore:
            try:
                resp = await asyncio.wait_for(
                    provider.timed_complete(
                        model=model,
                        prompt=prompt,
                        system=config.system,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    ),
                    timeout=config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                resp = ProviderResponse(
                    model=model,
                    provider=provider.name,
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=config.timeout_seconds * 1000,
                    error=f"Timeout after {config.timeout_seconds}s",
                )
        completed_calls += 1
        if progress_callback:
            progress_callback(model, completed_calls, total_calls)
        return resp

    # Run all prompts × runs
    for prompt in config.prompts:
        tasks = []
        for model in config.models:
            for _ in range(config.n_runs):
                tasks.append((model, run_single(model, prompt)))

        responses = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

        for (model, _), resp in zip(tasks, responses):
            if isinstance(resp, Exception):
                resp = ProviderResponse(
                    model=model,
                    provider=providers[model].name,
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    error=str(resp),
                )
            metrics[model].responses.append(resp)

        # Quality scoring (per prompt, score the first response from each model)
        if config.judge_model:
            judge_provider = resolve_provider(config.judge_model)
            first_responses = {
                model: next(
                    (r for r in metrics[model].responses[-config.n_runs:] if r.success),
                    None,
                )
                for model in config.models
            }
            valid_responses = [r for r in first_responses.values() if r is not None]
            if valid_responses:
                scores: list[JudgeScore | None] = await score_responses_batch(
                    judge_provider=judge_provider,
                    judge_model=config.judge_model,
                    prompt=prompt,
                    responses=valid_responses,
                    max_concurrent=3,
                )
                for model, score in zip(
                    [m for m, r in first_responses.items() if r is not None],
                    scores,
                ):
                    if score is not None:
                        # Accumulate quality scores (average across prompts)
                        current = metrics[model].quality_score
                        if current is None:
                            metrics[model].quality_score = score.composite
                        else:
                            metrics[model].quality_score = (current + score.composite) / 2

    # Compute final stats
    for model in config.models:
        metrics[model].compute(provider_instance=providers[model])

    duration = time.perf_counter() - start
    return BenchmarkResult(
        config=config,
        metrics=metrics,
        duration_seconds=duration,
        timestamp=timestamp,
    )

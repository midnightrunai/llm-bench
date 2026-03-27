"""Base provider interface for llm-bench."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """Standardized response from any LLM provider."""

    model: str
    provider: str
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def success(self) -> bool:
        return self.error is None


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"

    # Pricing in USD per 1M tokens (input, output)
    PRICING: dict[str, tuple[float, float]] = {}

    def cost_per_1k_requests(
        self,
        model: str,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> float:
        """Calculate estimated cost per 1000 requests in USD."""
        if model not in self.PRICING:
            # Try prefix match
            for key in self.PRICING:
                if model.startswith(key) or key.startswith(model):
                    model = key
                    break
            else:
                return 0.0

        input_price, output_price = self.PRICING[model]
        cost_per_request = (avg_input_tokens * input_price + avg_output_tokens * output_price) / 1_000_000
        return cost_per_request * 1_000

    @abstractmethod
    async def complete(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a completion request and return a standardized response."""
        ...

    async def timed_complete(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Wrapper that measures wall-clock latency."""
        start = time.perf_counter()
        try:
            response = await self.complete(
                model=model,
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.latency_ms = elapsed_ms
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ProviderResponse(
                model=model,
                provider=self.name,
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=elapsed_ms,
                error=str(exc),
            )

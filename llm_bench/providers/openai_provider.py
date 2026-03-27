"""OpenAI provider for llm-bench."""

from __future__ import annotations

import os
from typing import Any

from llm_bench.providers.base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    """OpenAI API provider (GPT-4o, GPT-4, GPT-3.5, O1, etc.)."""

    name = "openai"

    PRICING = {
        # USD per 1M tokens (input, output)
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "o1": (15.00, 60.00),
        "o1-mini": (3.00, 12.00),
        "o3-mini": (1.10, 4.40),
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install llm-bench[openai]"
                )
        return self._client

    async def complete(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ProviderResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # O1 models don't support system messages or temperature
        if model.startswith("o1") or model.startswith("o3"):
            messages = [m for m in messages if m["role"] != "system"]
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        usage = response.usage
        return ProviderResponse(
            model=model,
            provider=self.name,
            content=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=0.0,  # Set by timed_complete
            raw=response.model_dump(),
        )

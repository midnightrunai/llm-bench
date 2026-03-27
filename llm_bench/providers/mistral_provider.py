"""Mistral AI provider for llm-bench."""

from __future__ import annotations

import os
from typing import Any

from llm_bench.providers.base import BaseProvider, ProviderResponse


class MistralProvider(BaseProvider):
    """Mistral AI API provider."""

    name = "mistral"

    PRICING = {
        # USD per 1M tokens (input, output)
        "mistral-large": (3.00, 9.00),
        "mistral-medium": (2.70, 8.10),
        "mistral-small": (1.00, 3.00),
        "codestral": (1.00, 3.00),
        "mistral-7b": (0.25, 0.25),
        "mixtral-8x7b": (0.70, 0.70),
        "mixtral-8x22b": (2.00, 6.00),
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable."
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from mistralai import Mistral
                self._client = Mistral(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "mistralai not installed. Run: pip install llm-bench[mistral]"
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

        response = await self.client.chat.complete_async(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = response.usage
        return ProviderResponse(
            model=model,
            provider=self.name,
            content=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=0.0,
            raw={},
        )

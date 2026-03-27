"""Groq provider for llm-bench."""

from __future__ import annotations

import os
from typing import Any

from llm_bench.providers.base import BaseProvider, ProviderResponse


class GroqProvider(BaseProvider):
    """Groq API provider (ultra-fast inference)."""

    name = "groq"

    PRICING = {
        # USD per 1M tokens (input, output)
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "llama3-70b-8192": (0.59, 0.79),
        "llama3-8b-8192": (0.05, 0.08),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "gemma2-9b-it": (0.20, 0.20),
        "gemma-7b-it": (0.07, 0.07),
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable."
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: pip install llm-bench[groq]"
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
            latency_ms=0.0,
            raw={},
        )

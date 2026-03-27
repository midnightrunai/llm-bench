"""Google Gemini provider for llm-bench."""

from __future__ import annotations

import os
from typing import Any

from llm_bench.providers.base import BaseProvider, ProviderResponse


class GeminiProvider(BaseProvider):
    """Google Gemini API provider."""

    name = "gemini"

    PRICING = {
        # USD per 1M tokens (input, output) - Gemini pricing
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.0-flash-lite": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.0-pro": (0.50, 1.50),
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable."
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai not installed. Run: pip install llm-bench[gemini]"
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
        import asyncio

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model_instance = self.client.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system,
        )

        # Gemini SDK is sync — run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model_instance.generate_content(prompt),
        )

        # Count tokens
        try:
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
        except Exception:
            input_tokens = 0
            output_tokens = 0

        return ProviderResponse(
            model=model,
            provider=self.name,
            content=response.text if hasattr(response, "text") else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            raw={"candidates": len(response.candidates) if hasattr(response, "candidates") else 0},
        )

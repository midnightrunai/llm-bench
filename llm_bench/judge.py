"""LLM-as-judge for quality scoring in llm-bench."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from llm_bench.providers.base import BaseProvider, ProviderResponse


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of LLM responses.
You will be given a prompt and a response. Rate the response on two dimensions:
1. Coherence (1-10): Is the response well-structured, clear, and logically consistent?
2. Relevance (1-10): Does the response directly address what was asked?

Respond ONLY with this exact format:
COHERENCE: <score>
RELEVANCE: <score>
REASONING: <one sentence explanation>"""

JUDGE_USER_TEMPLATE = """PROMPT: {prompt}

RESPONSE: {response}

Rate this response."""


@dataclass
class JudgeScore:
    coherence: float
    relevance: float
    reasoning: str
    composite: float = 0.0

    def __post_init__(self):
        self.composite = (self.coherence + self.relevance) / 2


async def score_response(
    judge_provider: BaseProvider,
    judge_model: str,
    prompt: str,
    response: str,
) -> JudgeScore | None:
    """Score a single response using a judge model."""
    try:
        result = await judge_provider.complete(
            model=judge_model,
            prompt=JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response),
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=256,
        )

        if not result.success:
            return None

        return _parse_judge_response(result.content)
    except Exception:
        return None


async def score_responses_batch(
    judge_provider: BaseProvider,
    judge_model: str,
    prompt: str,
    responses: list[ProviderResponse],
    max_concurrent: int = 5,
) -> list[JudgeScore | None]:
    """Score multiple responses concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(resp: ProviderResponse) -> JudgeScore | None:
        if not resp.success or not resp.content:
            return None
        async with semaphore:
            return await score_response(judge_provider, judge_model, prompt, resp.content)

    return await asyncio.gather(*[score_one(r) for r in responses])


def _parse_judge_response(text: str) -> JudgeScore | None:
    """Parse judge output into a JudgeScore."""
    try:
        coherence_match = re.search(r"COHERENCE:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        relevance_match = re.search(r"RELEVANCE:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE)

        if not coherence_match or not relevance_match:
            # Try to extract any numbers
            numbers = re.findall(r"\b([1-9]|10)\b", text)
            if len(numbers) >= 2:
                return JudgeScore(
                    coherence=float(numbers[0]),
                    relevance=float(numbers[1]),
                    reasoning="(parsed from unstructured response)",
                )
            return None

        return JudgeScore(
            coherence=min(10.0, max(1.0, float(coherence_match.group(1)))),
            relevance=min(10.0, max(1.0, float(relevance_match.group(1)))),
            reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
        )
    except Exception:
        return None

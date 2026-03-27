"""YAML config file parsing for llm-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptConfig:
    """A single prompt definition in a YAML config."""
    text: str
    name: str | None = None
    system: str | None = None


@dataclass
class YAMLBenchmarkConfig:
    """Full YAML config structure."""
    models: list[str]
    prompts: list[PromptConfig]
    n_runs: int = 3
    temperature: float = 0.0
    max_tokens: int = 1024
    max_concurrent: int = 5
    judge_model: str | None = None
    timeout_seconds: float = 60.0
    system: str | None = None
    output: str | None = None
    output_format: str = "table"  # table | json | both


def load_yaml_config(path: str | Path) -> YAMLBenchmarkConfig:
    """Load and validate a YAML benchmark config file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml not installed. Run: pip install llm-bench")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(data)}")

    models = data.get("models", [])
    if not models:
        raise ValueError("Config must specify at least one model under 'models:'")

    raw_prompts = data.get("prompts", [])
    if not raw_prompts:
        raise ValueError("Config must specify at least one prompt under 'prompts:'")

    prompts: list[PromptConfig] = []
    for item in raw_prompts:
        if isinstance(item, str):
            prompts.append(PromptConfig(text=item))
        elif isinstance(item, dict):
            prompts.append(PromptConfig(
                text=item.get("text", item.get("prompt", "")),
                name=item.get("name"),
                system=item.get("system"),
            ))

    if not prompts or not all(p.text for p in prompts):
        raise ValueError("All prompts must have non-empty text")

    return YAMLBenchmarkConfig(
        models=models,
        prompts=prompts,
        n_runs=data.get("n_runs", data.get("runs", 3)),
        temperature=data.get("temperature", 0.0),
        max_tokens=data.get("max_tokens", 1024),
        max_concurrent=data.get("max_concurrent", data.get("concurrency", 5)),
        judge_model=data.get("judge_model", data.get("judge")),
        timeout_seconds=data.get("timeout_seconds", data.get("timeout", 60.0)),
        system=data.get("system"),
        output=data.get("output"),
        output_format=data.get("output_format", data.get("format", "table")),
    )

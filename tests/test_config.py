"""Tests for llm_bench.config module."""

import pytest
import tempfile
import os
from pathlib import Path

from llm_bench.config import load_yaml_config, PromptConfig, YAMLBenchmarkConfig


MINIMAL_CONFIG = """
models:
  - gpt-4o
  - claude-3-5-sonnet
prompts:
  - "What is 2+2?"
"""

FULL_CONFIG = """
models:
  - gpt-4o
  - gemini-2.0-flash

prompts:
  - text: "Classify the sentiment of: I love this!"
    name: sentiment
  - text: "Write a haiku about Python."
    name: haiku

n_runs: 5
temperature: 0.5
max_tokens: 512
judge_model: gpt-4o-mini
output: results.json
output_format: both
"""

INVALID_NO_MODELS = """
prompts:
  - "Hello"
"""

INVALID_NO_PROMPTS = """
models:
  - gpt-4o
"""


def write_yaml(content: str) -> Path:
    """Write YAML content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestLoadYAMLConfig:
    def test_minimal_config(self):
        path = write_yaml(MINIMAL_CONFIG)
        try:
            cfg = load_yaml_config(path)
            assert cfg.models == ["gpt-4o", "claude-3-5-sonnet"]
            assert len(cfg.prompts) == 1
            assert cfg.prompts[0].text == "What is 2+2?"
            assert cfg.n_runs == 3  # default
        finally:
            os.unlink(path)

    def test_full_config(self):
        path = write_yaml(FULL_CONFIG)
        try:
            cfg = load_yaml_config(path)
            assert len(cfg.models) == 2
            assert len(cfg.prompts) == 2
            assert cfg.prompts[0].name == "sentiment"
            assert cfg.prompts[1].name == "haiku"
            assert cfg.n_runs == 5
            assert cfg.temperature == 0.5
            assert cfg.max_tokens == 512
            assert cfg.judge_model == "gpt-4o-mini"
            assert cfg.output == "results.json"
        finally:
            os.unlink(path)

    def test_invalid_no_models(self):
        path = write_yaml(INVALID_NO_MODELS)
        try:
            with pytest.raises(ValueError, match="model"):
                load_yaml_config(path)
        finally:
            os.unlink(path)

    def test_invalid_no_prompts(self):
        path = write_yaml(INVALID_NO_PROMPTS)
        try:
            with pytest.raises(ValueError, match="prompt"):
                load_yaml_config(path)
        finally:
            os.unlink(path)

# llm-bench

[![PyPI version](https://badge.fury.io/py/llm-benchmarker.svg)](https://pypi.org/project/llm-benchmarker/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/midnightrunai/llm-bench/actions/workflows/test.yml/badge.svg)](https://github.com/midnightrunai/llm-bench/actions/workflows/test.yml)

**Benchmark any LLM against your actual prompts.** Compare OpenAI, Anthropic, Gemini, Mistral, Groq — latency, cost, quality, side by side.

```
pip install llm-benchmarker
```

---

## Quick Start

```bash
# Install with the providers you need
pip install "llm-benchmarker[openai,anthropic,gemini]"

# Set your API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...

# Run a benchmark
llm-bench run \
  --prompt "Classify the sentiment: 'I love this product!'" \
  --models gpt-4o,claude-3-5-sonnet,gemini-2.0-flash
```

Output:
```
llm-bench results — 3 runs × 1 prompt(s)

  Model                     Provider    p50 (ms)  p95 (ms)  Avg tokens ↑↓   $/1k req  Errors
  ────────────────────────────────────────────────────────────────────────────────────────────
  gemini-2.0-flash          gemini          312       445       15↑  18↓     $0.003      —
  gpt-4o                    openai          487       623       15↑  22↓     $0.059      —
  claude-3-5-sonnet         anthropic       891      1204       15↑  31↓     $0.121      —
```

---

## Installation

```bash
# Core only
pip install llm-benchmarker

# With specific providers
pip install "llm-benchmarker[openai]"
pip install "llm-benchmarker[anthropic]"
pip install "llm-benchmarker[gemini]"
pip install "llm-benchmarker[mistral]"
pip install "llm-benchmarker[groq]"

# All providers
pip install "llm-benchmarker[all]"
```

---

## Usage

### CLI

```bash
# Single prompt, multiple models
llm-bench run \
  --prompt "Write a Python function to reverse a string" \
  --models gpt-4o-mini,claude-3-5-haiku,gemini-2.0-flash

# Multiple prompts
llm-bench run \
  --prompt "What is 2+2?" \
  --prompt "Explain quantum entanglement simply." \
  --models gpt-4o,claude-3-5-sonnet

# With quality scoring (LLM-as-judge)
llm-bench run \
  --prompt "Summarize the French Revolution in 3 sentences." \
  --models gpt-4o,claude-3-5-sonnet \
  --judge gpt-4o-mini

# Save to JSON (for CI/CD)
llm-bench run \
  --prompt "Hello" \
  --models gpt-4o \
  --json > results.json

# Use a YAML config for complex benchmarks
llm-bench run --config benchmark.yaml
```

### YAML Config

Generate a starter config:
```bash
llm-bench init
```

Or create `benchmark.yaml`:
```yaml
models:
  - gpt-4o
  - claude-3-5-sonnet
  - gemini-2.0-flash
  - llama-3.3-70b-versatile  # Groq

prompts:
  - text: "Classify sentiment: 'Great product, fast shipping!'"
    name: positive_sentiment

  - text: "Debug this: def fib(n): return fib(n-1) + fib(n-2)"
    name: code_debug

# Optional: score response quality with a judge model
judge_model: gpt-4o-mini

n_runs: 5
temperature: 0.0
max_tokens: 512
output: results.json
```

Run it:
```bash
llm-bench run --config benchmark.yaml
```

### JSON Output (CI/CD)

```bash
llm-bench run --config benchmark.yaml --json
```

Output schema:
```json
{
  "timestamp": "2026-03-27T03:00:00Z",
  "duration_seconds": 12.4,
  "config": {
    "models": ["gpt-4o", "claude-3-5-sonnet"],
    "n_prompts": 2,
    "n_runs": 3
  },
  "results": {
    "gpt-4o": {
      "model": "gpt-4o",
      "provider": "openai",
      "n_success": 6,
      "n_errors": 0,
      "latency": {
        "p50_ms": 487.2,
        "p95_ms": 623.1,
        "mean_ms": 501.4
      },
      "tokens": {
        "avg_input": 15.0,
        "avg_output": 22.3
      },
      "cost_per_1k_requests_usd": 0.059,
      "quality_score": 8.4
    }
  }
}
```

---

## Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o3-mini |
| **Anthropic** | claude-opus-4, claude-sonnet-4, claude-3-5-sonnet, claude-3-5-haiku, claude-3-haiku |
| **Gemini** | gemini-2.5-pro, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro, gemini-1.5-flash |
| **Mistral** | mistral-large, mistral-small, codestral, mixtral-8x7b |
| **Groq** | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it |

See all: `llm-bench list-models`

---

## Environment Variables

| Variable | Provider |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Google Gemini |
| `MISTRAL_API_KEY` | Mistral |
| `GROQ_API_KEY` | Groq |

---

## Options Reference

```
llm-bench run [OPTIONS]

  --prompt, -p TEXT        Prompt to benchmark (repeatable)
  --models, -m TEXT        Comma-separated model list
  --config, -c PATH        YAML config file
  --runs, -n INTEGER       Runs per prompt per model [default: 3]
  --temperature, -t FLOAT  Sampling temperature [default: 0.0]
  --max-tokens INTEGER     Max output tokens [default: 1024]
  --judge TEXT             Judge model for quality scoring
  --system, -s TEXT        System prompt
  --output, -o PATH        Save results to JSON file
  --json                   Output JSON to stdout
  --concurrency INTEGER    Max concurrent requests [default: 5]
  --timeout FLOAT          Request timeout in seconds [default: 60.0]
  --verbose, -v            Show progress
```

---

## Python API

```python
import asyncio
from llm_bench.benchmark import BenchmarkConfig, run_benchmark
from llm_bench.reporter import print_results_table

config = BenchmarkConfig(
    models=["gpt-4o", "claude-3-5-sonnet"],
    prompts=["What is the meaning of life?"],
    n_runs=3,
    judge_model="gpt-4o-mini",
)

result = asyncio.run(run_benchmark(config))
print_results_table(result)

# Access raw data
for model, metrics in result.metrics.items():
    print(f"{model}: p50={metrics.latency_p50_ms:.0f}ms, quality={metrics.quality_score}")
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/midnightrunai/llm-bench
cd llm-bench
pip install -e ".[dev]"
pytest
```

---

## License

MIT © [Midnight Run](https://midnightrun.ai)

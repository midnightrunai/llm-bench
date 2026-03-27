"""Command-line interface for llm-bench."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from llm_bench import __version__


def _make_progress_callback(verbose: bool):
    """Create a progress callback for the benchmark runner."""
    if not verbose:
        return None

    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        from rich.console import Console

        console = Console(stderr=True)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )

        task_id = None
        ctx = {"progress": progress, "task_id": task_id, "started": False}

        def callback(model: str, completed: int, total: int):
            if not ctx["started"]:
                ctx["progress"].start()
                ctx["started"] = True
                ctx["task_id"] = ctx["progress"].add_task(
                    "Running benchmark...", total=total
                )
            ctx["progress"].update(
                ctx["task_id"],
                completed=completed,
                description=f"[dim]Testing {model}...[/dim]",
            )
            if completed >= total:
                ctx["progress"].stop()

        return callback
    except ImportError:
        def simple_callback(model: str, completed: int, total: int):
            print(f"  [{completed}/{total}] {model}", file=sys.stderr)
        return simple_callback


@click.group()
@click.version_option(version=__version__, prog_name="llm-bench")
def cli():
    """llm-bench — benchmark any LLM against your actual prompts.

    \b
    Examples:
      llm-bench run --prompt "Classify: I love this!" --models gpt-4o,claude-3-5-sonnet
      llm-bench run --config benchmark.yaml
      llm-bench run --prompt "Hello" --models gpt-4o --judge gpt-4o-mini
      llm-bench list-models
    """
    pass


@cli.command()
@click.option(
    "--prompt", "-p",
    multiple=True,
    help="Prompt(s) to benchmark. Can be specified multiple times.",
)
@click.option(
    "--models", "-m",
    help="Comma-separated list of models (e.g. gpt-4o,claude-3-5-sonnet,gemini-2.0-flash).",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML config file.",
)
@click.option(
    "--runs", "-n",
    default=3,
    show_default=True,
    help="Number of runs per prompt per model.",
)
@click.option(
    "--temperature", "-t",
    default=0.0,
    show_default=True,
    help="Sampling temperature (0.0 = deterministic).",
)
@click.option(
    "--max-tokens",
    default=1024,
    show_default=True,
    help="Maximum output tokens per request.",
)
@click.option(
    "--judge",
    help="Judge model for quality scoring (e.g. gpt-4o-mini).",
)
@click.option(
    "--system", "-s",
    help="System prompt to use for all requests.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Path to save JSON results.",
)
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=False,
    help="Output results as JSON (stdout).",
)
@click.option(
    "--concurrency",
    default=5,
    show_default=True,
    help="Max concurrent API requests.",
)
@click.option(
    "--timeout",
    default=60.0,
    show_default=True,
    help="Timeout in seconds per request.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Show progress during benchmark.",
)
def run(
    prompt: tuple[str, ...],
    models: Optional[str],
    config: Optional[Path],
    runs: int,
    temperature: float,
    max_tokens: int,
    judge: Optional[str],
    system: Optional[str],
    output: Optional[Path],
    output_format: str,
    concurrency: int,
    timeout: float,
    verbose: bool,
):
    """Run a benchmark across one or more LLM models.

    \b
    Quick start:
      llm-bench run --prompt "What is 2+2?" --models gpt-4o,claude-3-5-sonnet

    \b
    With YAML config:
      llm-bench run --config my_bench.yaml

    \b
    Output to JSON (for CI/CD):
      llm-bench run --prompt "..." --models gpt-4o --json
    """
    from llm_bench.benchmark import BenchmarkConfig, run_benchmark
    from llm_bench.reporter import print_results_table, save_json, print_json

    # Build config from CLI args or YAML file
    if config:
        from llm_bench.config import load_yaml_config
        yaml_cfg = load_yaml_config(config)
        bench_config = BenchmarkConfig(
            models=yaml_cfg.models,
            prompts=[p.text for p in yaml_cfg.prompts],
            system=system or yaml_cfg.system,
            n_runs=runs if runs != 3 else yaml_cfg.n_runs,
            temperature=temperature if temperature != 0.0 else yaml_cfg.temperature,
            max_tokens=max_tokens if max_tokens != 1024 else yaml_cfg.max_tokens,
            max_concurrent=concurrency if concurrency != 5 else yaml_cfg.max_concurrent,
            judge_model=judge or yaml_cfg.judge_model,
            timeout_seconds=timeout if timeout != 60.0 else yaml_cfg.timeout_seconds,
        )
        if not output and yaml_cfg.output:
            output = Path(yaml_cfg.output)
        if output_format not in ("json",) and yaml_cfg.output_format != "table":
            output_format = yaml_cfg.output_format
    else:
        if not prompt:
            raise click.UsageError(
                "Provide at least one --prompt or use --config to specify a YAML file."
            )
        if not models:
            raise click.UsageError(
                "Provide --models (comma-separated) or use --config."
            )

        model_list = [m.strip() for m in models.split(",") if m.strip()]
        bench_config = BenchmarkConfig(
            models=model_list,
            prompts=list(prompt),
            system=system,
            n_runs=runs,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=concurrency,
            judge_model=judge,
            timeout_seconds=timeout,
        )

    if verbose and output_format != "json":
        click.echo(
            f"Running benchmark: {len(bench_config.models)} model(s) × "
            f"{len(bench_config.prompts)} prompt(s) × {bench_config.n_runs} runs",
            err=True,
        )

    progress_callback = _make_progress_callback(verbose and output_format != "json")

    try:
        result = asyncio.run(run_benchmark(bench_config, progress_callback=progress_callback))
    except KeyboardInterrupt:
        click.echo("\nBenchmark cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if output_format == "json":
        print_json(result)
    else:
        print_results_table(result)
        if output:
            save_json(result, output)
        elif output_format == "both":
            pass  # Already printed table


@cli.command("list-models")
def list_models():
    """List all supported models and their providers."""
    from llm_bench.providers import MODEL_TO_PROVIDER, PROVIDER_REGISTRY

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            title="Supported Models",
            box=box.SIMPLE_HEAD,
            header_style="bold dim",
        )
        table.add_column("Model", style="bold")
        table.add_column("Provider", style="dim")

        grouped: dict[str, list[str]] = {}
        for model, provider in MODEL_TO_PROVIDER.items():
            grouped.setdefault(provider, []).append(model)

        for provider in sorted(grouped):
            for model in grouped[provider]:
                table.add_row(model, provider)

        console.print(table)
        console.print(
            f"[dim]Set the appropriate API key env var: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, GROQ_API_KEY[/dim]"
        )

    except ImportError:
        for model, provider in sorted(MODEL_TO_PROVIDER.items()):
            print(f"{model:<40} {provider}")


@cli.command()
@click.argument("output", type=click.Path(path_type=Path), default=Path("benchmark.yaml"))
def init(output: Path):
    """Generate a starter YAML config file.

    \b
    Example:
      llm-bench init
      llm-bench init my_benchmark.yaml
    """
    template = """\
# llm-bench configuration
# Run with: llm-bench run --config benchmark.yaml

models:
  - gpt-4o
  - claude-3-5-sonnet
  - gemini-2.0-flash

prompts:
  - text: "Classify the sentiment of this text as positive, negative, or neutral: 'I absolutely loved the product but the shipping was terrible.'"
    name: sentiment_classification

  - text: "Write a Python function that checks if a string is a palindrome."
    name: code_generation

  - text: "Summarize the following in 2 sentences: The theory of relativity encompasses two theories by Albert Einstein: special relativity and general relativity."
    name: summarization

# Number of runs per prompt per model (for statistical reliability)
n_runs: 3

# Optional: use a judge model to score response quality (coherence + relevance)
# judge_model: gpt-4o-mini

# Temperature (0.0 = deterministic, better for benchmarking)
temperature: 0.0

# Max output tokens per request
max_tokens: 512

# Save results to JSON
# output: results.json
"""
    if output.exists():
        click.confirm(f"{output} already exists. Overwrite?", abort=True)
    output.write_text(template)
    click.echo(f"Created {output}")
    click.echo(f"Edit it, then run: llm-bench run --config {output}")


if __name__ == "__main__":
    cli()

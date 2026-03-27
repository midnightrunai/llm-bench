"""Rich terminal output and JSON reporting for llm-bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_bench.benchmark import BenchmarkResult
    from llm_bench.metrics import ModelMetrics


def print_results_table(result: "BenchmarkResult") -> None:
    """Print a rich terminal table with benchmark results."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.text import Text
    except ImportError:
        _print_plain_table(result)
        return

    console = Console()
    metrics = result.metrics

    if not metrics:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(
        title=f"[bold]llm-bench results[/bold] — {result.config.n_runs} runs × {len(result.config.prompts)} prompt(s)",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        expand=True,
    )

    has_quality = any(m.quality_score is not None for m in metrics.values())
    has_pricing = any(m.cost_per_1k_requests > 0 for m in metrics.values())

    table.add_column("Model", style="bold", min_width=24)
    table.add_column("Provider", style="dim", min_width=10)
    table.add_column("p50 (ms)", justify="right", min_width=10)
    table.add_column("p95 (ms)", justify="right", min_width=10)
    table.add_column("Avg tokens ↑↓", justify="right", min_width=14)
    if has_pricing:
        table.add_column("$/1k req", justify="right", min_width=10)
    if has_quality:
        table.add_column("Quality", justify="right", min_width=9)
    table.add_column("Errors", justify="right", min_width=7)

    # Sort by p50 latency
    sorted_metrics = sorted(metrics.values(), key=lambda m: m.latency_p50_ms)

    for i, m in enumerate(sorted_metrics):
        # Color the fastest in green
        latency_style = "green" if i == 0 and m.latency_p50_ms > 0 else ""
        cost_style = ""
        if has_pricing and m.cost_per_1k_requests > 0:
            if m.cost_per_1k_requests == min(
                x.cost_per_1k_requests for x in sorted_metrics if x.cost_per_1k_requests > 0
            ):
                cost_style = "green"

        tokens_str = f"{int(m.avg_input_tokens)}↑ {int(m.avg_output_tokens)}↓"
        latency_p50 = f"{m.latency_p50_ms:.0f}" if m.latency_p50_ms > 0 else "—"
        latency_p95 = f"{m.latency_p95_ms:.0f}" if m.latency_p95_ms > 0 else "—"

        row = [
            m.model,
            m.provider,
            Text(latency_p50, style=latency_style),
            f"{latency_p95}",
            tokens_str,
        ]

        if has_pricing:
            cost_str = f"${m.cost_per_1k_requests:.3f}" if m.cost_per_1k_requests > 0 else "—"
            row.append(Text(cost_str, style=cost_style))

        if has_quality:
            quality_str = f"{m.quality_score:.1f}/10" if m.quality_score is not None else "—"
            row.append(quality_str)

        error_str = str(m.n_errors) if m.n_errors > 0 else "—"
        row.append(Text(error_str, style="red" if m.n_errors > 0 else ""))

        table.add_row(*row)

    console.print()
    console.print(table)
    console.print(
        f"[dim]Total time: {result.duration_seconds:.1f}s | "
        f"Requests: {sum(m.n_total for m in metrics.values())}[/dim]"
    )
    console.print()


def _print_plain_table(result: "BenchmarkResult") -> None:
    """Fallback plain-text table for environments without rich."""
    metrics = result.metrics
    print(f"\nllm-bench results ({result.config.n_runs} runs)\n")
    print(f"{'Model':<30} {'p50ms':>8} {'p95ms':>8} {'$/1k':>8} {'Errors':>7}")
    print("-" * 65)
    for m in sorted(metrics.values(), key=lambda x: x.latency_p50_ms):
        cost = f"${m.cost_per_1k_requests:.3f}" if m.cost_per_1k_requests > 0 else "—"
        print(
            f"{m.model:<30} "
            f"{m.latency_p50_ms:>7.0f} "
            f"{m.latency_p95_ms:>7.0f} "
            f"{cost:>8} "
            f"{m.n_errors:>7}"
        )
    print()


def save_json(result: "BenchmarkResult", output_path: str | Path) -> None:
    """Save benchmark results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Results saved to {output_path}")


def print_json(result: "BenchmarkResult") -> None:
    """Print benchmark results as JSON to stdout."""
    print(json.dumps(result.to_dict(), indent=2))

"""Rich-based logging and metric tracking utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class MetricLogger:
    """Logs training metrics to console and writes to JSON history file."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = self.log_dir / "metrics_history.jsonl"
        else:
            self.history_file = None
        self.history: List[Dict[str, Any]] = []

    def log(self, step: int, metrics: Dict[str, Any], display: bool = True):
        record = {"step": step, **metrics}
        self.history.append(record)

        if self.history_file:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if display:
            formatted = " | ".join(
                f"[cyan]{k}[/cyan]: [bold yellow]{v:.4f}[/bold yellow]"
                if isinstance(v, float)
                else f"[cyan]{k}[/cyan]: {v}"
                for k, v in metrics.items()
            )
            console.log(f"[bold green]Step {step}[/bold green] -> {formatted}")

    def summary_table(self, title: str = "Training Summary") -> Table:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        if not self.history:
            return table

        keys = list(self.history[-1].keys())
        for k in keys:
            table.add_column(k, justify="right")

        # Show first, middle, last few rows
        indices = [0]
        if len(self.history) > 2:
            indices.append(len(self.history) // 2)
        if len(self.history) > 1:
            indices.append(len(self.history) - 1)

        for idx in set(indices):
            row = self.history[idx]
            table.add_row(
                *(
                    f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k])
                    for k in keys
                )
            )
        return table

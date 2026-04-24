"""Plotting helpers for benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_time(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot median training time by implementation."""

    _ensure_parent(output_path)
    labels = _labels(results)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, results["median_time"])
    plt.ylabel("Median training time (s)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_speedup(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot speedup for rows where it is defined."""

    _plot_line_metric(results, "speedup", "Speedup", output_path)


def plot_efficiency(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot efficiency for rows where it is defined."""

    _plot_line_metric(results, "efficiency", "Efficiency", output_path)


def plot_loss_curve(
    loss_histories: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    """Plot loss histories captured from custom gradient descent runs."""

    _ensure_parent(output_path)
    plt.figure(figsize=(10, 5))
    for label, history in loss_histories.items():
        if history:
            plt.plot(range(1, len(history) + 1), history, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_line_metric(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    filtered = results.dropna(subset=[metric, "processes"]).copy()
    if filtered.empty:
        return
    _ensure_parent(output_path)
    plt.figure(figsize=(8, 5))
    for implementation, group in filtered.groupby("implementation"):
        ordered = group.sort_values("processes")
        plt.plot(
            ordered["processes"],
            ordered[metric],
            marker="o",
            label=str(implementation),
        )
    plt.xlabel("Processes")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _labels(results: pd.DataFrame) -> list[str]:
    labels = []
    for row in results.itertuples(index=False):
        suffix = "" if pd.isna(row.processes) else f" ({int(row.processes)}p)"
        labels.append(f"{row.implementation}{suffix}")
    return labels


def _ensure_parent(output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

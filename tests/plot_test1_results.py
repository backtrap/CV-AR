"""전체 파이프라인 test1 결과 그래프 2개를 생성한다."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test1"
CSV_RESULT = RESULT_DIR / "test1_res.csv"
OUTPUT_DIR = RESULT_DIR / "test1_graphs"


def load_rows() -> list[dict]:
    rows = []
    with CSV_RESULT.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            payload = json.loads(row["json"])
            results = payload["vlm_result"]["results"]
            bottle = next(
                (
                    item
                    for item in results
                    if item["identity"]["class_name"].strip().lower() == "bottle"
                ),
                None,
            )
            rows.append(
                {
                    "experiment": int(row["experiment"]),
                    "word_limit": int(row["word_limit"]),
                    "seconds": float(row["vlm_inference_seconds"]),
                    "classes": [item["identity"]["class_name"] for item in results],
                    "bottle_environment": (
                        bottle["corrected_spatial_relation"]["environment_relative"]
                        if bottle
                        else "not_detected"
                    ),
                    "bottle_social": (
                        bottle["semantic_state"]["social_state"]
                        if bottle
                        else "not_detected"
                    ),
                }
            )
    if not rows:
        raise RuntimeError("test1_res.csv에 실험 결과가 없습니다.")
    return rows


def apply_style() -> None:
    korean_font = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = (
        font_manager.FontProperties(fname=korean_font).get_name()
        if korean_font.exists()
        else "DejaVu Sans"
    )
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "font.family": font_family,
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
        }
    )


def plot_latency(rows: list[dict]) -> None:
    limits = np.array([row["word_limit"] for row in rows], dtype=float)
    seconds = np.array([row["seconds"] for row in rows], dtype=float)
    order = np.argsort(limits)
    trend = np.poly1d(np.polyfit(limits, seconds, 1))
    correlation = float(np.corrcoef(limits, seconds)[0, 1])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(
        limits,
        seconds,
        marker="o",
        markersize=5.5,
        linewidth=1.7,
        color="#2563EB",
        alpha=0.85,
        label="Experiment",
    )
    ax.plot(
        limits[order],
        trend(limits[order]),
        linewidth=2.3,
        color="#DC2626",
        label="Linear trend",
    )
    ax.invert_xaxis()
    ax.set_title("VLM Inference Time by Micro CoT Word Limit (Full Pipeline)")
    ax.set_xlabel("Reasoning word limit (30 to 0)")
    ax.set_ylabel("VLM inference time (seconds)")
    ax.set_xticks(range(30, -1, -2))
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.96,
        f"Pearson r = {correlation:.2f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_word_limit_vs_vlm_latency.png", dpi=180)
    plt.close(fig)


def plot_classification_heatmap(rows: list[dict]) -> None:
    frequencies = Counter(label for row in rows for label in row["classes"])
    labels = [label for label, _ in frequencies.most_common()]
    matrix = np.array(
        [[int(label in row["classes"]) for row in rows] for label in labels],
        dtype=int,
    )

    fig_height = max(6.5, 0.34 * len(labels) + 2.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_title("VLM Object Classification Across Micro CoT Word Limits")
    ax.set_xlabel("Experiment (word limit decreases left to right)")
    ax.set_ylabel("VLM class label")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(row["experiment"]) for row in rows], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    top = ax.secondary_xaxis("top")
    top.set_xticks(range(len(rows)))
    top.set_xticklabels([str(row["word_limit"]) for row in rows], rotation=90, fontsize=8)
    top.set_xlabel("Reasoning word limit")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_classification_changes.png", dpi=180)
    plt.close(fig)


def plot_bottle_state(rows: list[dict]) -> None:
    environment_colors = {
        "on_floor": "#16A34A",
        "on_surface": "#2563EB",
        "elevated": "#7C3AED",
        "floating": "#F97316",
        "held": "#DC2626",
        "not_detected": "#CBD5E1",
    }
    social_colors = {
        "available": "#16A34A",
        "held_by_user": "#DC2626",
        "in_use_by_other": "#F97316",
        "not_detected": "#CBD5E1",
    }

    environment_values = [row["bottle_environment"] for row in rows]
    social_values = [row["bottle_social"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 5.7), sharex=True)
    panels = [
        (axes[0], environment_values, environment_colors, "Environment relation"),
        (axes[1], social_values, social_colors, "Social state"),
    ]
    for ax, values, color_map, label in panels:
        categories = list(color_map)
        encoded = np.array([[categories.index(value) for value in values]])
        ax.imshow(
            encoded,
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap([color_map[value] for value in categories]),
            vmin=-0.5,
            vmax=len(categories) - 0.5,
        )
        ax.grid(False)
        ax.set_yticks([0])
        ax.set_yticklabels([label])
        present_categories = [category for category in categories if category in values]
        ax.legend(
            handles=[
                Patch(facecolor=color_map[category], label=category)
                for category in present_categories
            ],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
            frameon=False,
        )

    axes[0].set_title("Bottle State Across Micro CoT Word Limits")
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels([str(row["experiment"]) for row in rows], fontsize=8)
    axes[1].set_xlabel("Experiment (word limit decreases left to right)")

    top = axes[0].secondary_xaxis("top")
    top.set_xticks(range(len(rows)))
    top.set_xticklabels([str(row["word_limit"]) for row in rows], rotation=90, fontsize=8)
    top.set_xlabel("Reasoning word limit")

    fig.subplots_adjust(top=0.78, bottom=0.14, right=0.80, hspace=0.42)
    fig.savefig(OUTPUT_DIR / "03_bottle_state_changes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    apply_style()
    plot_latency(rows)
    plot_classification_heatmap(rows)
    plot_bottle_state(rows)
    print(f"Saved 3 graphs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

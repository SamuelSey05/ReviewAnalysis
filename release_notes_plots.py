import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from comparison_models import RankedResult

logger = logging.getLogger(__name__)


def plot_mttr_comparison(app_summaries: dict, output_file: str = "results/mttr_comparison.png") -> None:
    """Plot MTTR comparison across apps as a bar chart with 95% CI error bars.

    Args:
        app_summaries (dict): Mapping of app name to summary dict containing
            a `times_to_resolutions` list.
        output_file (str, optional): Output image path. Defaults to
            "results/mttr_comparison.png".
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    app_names = [name.upper() for name in app_summaries.keys()]
    mttrs = []
    errors = []

    for summary in app_summaries.values():
        values = summary.get("times_to_resolutions", [])
        if values:
            values_array = np.array(values, dtype=float)
            mttrs.append(float(np.mean(values_array)))
        else:
            values_array = None
            mttrs.append(0.0)

        if values_array is not None and values_array.size > 1:
            std = float(np.std(values_array, ddof=1))
            sem = std / float(np.sqrt(values_array.size))
            errors.append(1.96 * sem)
        else:
            errors.append(0.0)

    ax.bar(
        app_names,
        mttrs,
        yerr=errors,
        capsize=6,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        error_kw={"elinewidth": 1.5},
    )
    ax.set_ylabel("Days", fontsize=12)
    ax.set_title("Mean Time To Resolution (MTTR) by App", fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    for i, (v, err) in enumerate(zip(mttrs, errors)):
        ax.text(i, v + err + 2, f"{v:.1f} ± {err:.1f}", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    logger.info(f"MTTR comparison plot saved to {output_file}")
    plt.close()


def plot_resolution_time_distribution(app_summaries: dict, output_file: str = "results/resolution_time_distribution.png") -> None:
    """Plot a boxplot distribution of resolution times for each app.

    Args:
        app_summaries (dict): Mapping of app name to summary dict containing
            a `times_to_resolutions` list.
        output_file (str, optional): Output image path. Defaults to
            "results/resolution_time_distribution.png".
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        [summary.get("times_to_resolutions", []) for summary in app_summaries.values()],
        tick_labels=[name.upper() for name in app_summaries.keys()],
        patch_artist=True,
        widths=0.6,
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Days to Resolution", fontsize=12)
    ax.set_title("Distribution of Resolution Times by App", fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    logger.info(f"Resolution time distribution plot saved to {output_file}")
    plt.close()


def plot_specificity_comparison(all_results: list[RankedResult], output_file: str = "results/similarity_vs_resolution_time.png") -> None:
    """Plot similarity score versus time-to-resolution for ranked results.

    Args:
        all_results (list[RankedResult]): Ranked comparison results.
        output_file (str, optional): Output image path. Defaults to
            "results/similarity_vs_resolution_time.png".
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    similarities = [result.similarity for _, result in all_results]
    time_to_resolution = [result.time_diff_days for _, result in all_results]

    ax.scatter(similarities, time_to_resolution, alpha=0.6)
    ax.set_xlabel("Similarity Score", fontsize=12)
    ax.set_ylabel("Time to Resolution (Days)", fontsize=12)
    ax.set_title("Similarity vs Time to Resolution", fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    logger.info(f"Similarity vs Time to Resolution plot saved to {output_file}")
    plt.close()


def plot_aspect_density_comparison(aspect_metrics: dict, aspect_labels: list[str], output_file: str = "results/aspect_density_comparison.png") -> None:
    """Plot match density by aspect as a bar chart.

    Args:
        aspect_metrics (dict): Per-aspect metrics including `match_density`.
        aspect_labels (list[str]): Aspect label names indexed by aspect id.
        output_file (str, optional): Output image path. Defaults to
            "results/aspect_density_comparison.png".
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    aspects = [aspect_labels[aspect] for aspect in aspect_metrics.keys()]
    densities = [metrics["match_density"] for metrics in aspect_metrics.values()]

    ax.bar(
        aspects,
        densities,
        color='#1f77b4',
        alpha=0.7,
    )
    ax.set_ylabel("Match Density", fontsize=12)
    ax.set_title("Match Density by Aspect", fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    logger.info(f"Aspect density comparison plot saved to {output_file}")
    plt.close()

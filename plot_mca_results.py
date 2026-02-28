#!/usr/bin/env python3
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


MCA_RE = re.compile(r"size_(\d+)_run_(\d+)\.txt$")
EXAMPLE_RE = re.compile(r"size_(\d+)_run_(\d+)\.txt$")

METHOD_LABELS = {
    "fp64": "FP64",
    "fp32": "FP32",
}

SETTING_ORDER = ["baseline", "mca"]

SETTING_DESCRIPTIONS = {
    "baseline": "baseline: native compiler (no MCA)",
    "mca": "mca: instrumented (MCA)",
}


def compute_true_norm(size):
    """Compute ||x_true|| where x_true = [1, 2, 3, ..., size]."""
    sum_of_squares = size * (size + 1) * (2 * size + 1) / 6
    return math.sqrt(sum_of_squares)


def compute_relative_error(error_vs_true_mean, true_norm):
    """Calculate relative error = ||x - x_true|| / ||x_true||."""
    if error_vs_true_mean is None or true_norm is None or true_norm == 0:
        return float("nan")
    return error_vs_true_mean / true_norm


def parse_results_file(path):
    current = None
    result = {
        "fp64": None,
        "fp32": None,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line.startswith("--- FP64 Solution") or line.startswith("--- Solutions"):
                current = "fp64"
                continue
            if line.startswith("--- FP32 Solution"):
                current = "fp32"
                continue

            if current is None:
                continue

            if line.startswith("Error vs true:"):
                result[current] = float(line.split()[-1])

    return result


def parse_example_results_file(path):
    """Parse example_results files with 'Naive FP64/FP32 Solution' format."""
    current = None
    result = {
        "fp64": None,
        "fp32": None,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line.startswith("--- Naive FP64 Solution"):
                current = "fp64"
                continue
            if line.startswith("--- Naive FP32 Solution"):
                current = "fp32"
                continue
            if line.startswith("--- Mixed Precision"):
                current = None
                continue

            if current is None:
                continue

            if line.startswith("Error vs true:"):
                result[current] = float(line.split()[-1])

    return result


def collect_mca_results(results_dir):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for path in results_dir.glob("size_*_run_*.txt"):
        match = MCA_RE.match(path.name)
        if not match:
            continue
        size = int(match.group(1))
        parsed = parse_results_file(path)

        for method, error_vs_true in parsed.items():
            if error_vs_true is not None:
                data[size]["mca"][method].append(error_vs_true)

    result = defaultdict(lambda: defaultdict(dict))
    for size, settings in data.items():
        for setting, methods in settings.items():
            for method, values in methods.items():
                result[size][setting][method] = mean_or_nan(values)

    return result


def collect_example_results(results_dir):
    """Collect uninstrumented results from example_results."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for path in results_dir.glob("size_*_run_*.txt"):
        match = EXAMPLE_RE.match(path.name)
        if not match:
            continue
        size = int(match.group(1))
        parsed = parse_example_results_file(path)

        for method, error_vs_true in parsed.items():
            if error_vs_true is not None:
                data[size]["baseline"][method].append(error_vs_true)

    result = defaultdict(lambda: defaultdict(dict))
    for size, settings in data.items():
        for setting, methods in settings.items():
            for method, values in methods.items():
                result[size][setting][method] = mean_or_nan(values)

    return result


def mean_or_nan(values):
    if not values:
        return float("nan")
    return float(np.nanmean(values))


def build_relative_errors(settings_data, true_norm):
    relative_errors = defaultdict(dict)
    for setting, methods in settings_data.items():
        for method, error_vs_true_mean in methods.items():
            relative_errors[setting][method] = compute_relative_error(
                error_vs_true_mean,
                true_norm,
            )
    return relative_errors


def plot_relative_error_summary(size, relative_errors, output_dir):
    labels = ["fp64", "fp32"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax1 = axes[0]
    baseline_values = [
        relative_errors.get("baseline", {}).get(label, float("nan"))
        for label in labels
    ]
    bars1 = ax1.bar(x, baseline_values, width=0.6, alpha=0.8, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_LABELS[label] for label in labels])
    ax1.set_yscale("log")
    ax1.set_ylabel("Relative Error ||x-x_true||/||x_true||", fontsize=11)
    ax1.set_title(
        f"Relative Error of Uninstrumented Native Implementations for Gaussian Solver\n"
        f"(Matrix Size {size}x{size})",
        fontsize=11
    )
    ax1.grid(True, axis="y", which="major", linestyle="--", alpha=0.4)
    ax1.bar_label(bars1, fmt="%.2e")

    ax2 = axes[1]
    mca_values = [
        relative_errors.get("mca", {}).get(label, float("nan"))
        for label in labels
    ]
    bars2 = ax2.bar(x, mca_values, width=0.6, alpha=0.8, color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_LABELS[label] for label in labels])
    ax2.set_yscale("log")
    ax2.set_ylabel("Relative Error", fontsize=11)
    ax2.set_title(
        f"Relative Error of FP64 & FP32 Version for Gaussian Solver using VFC MCA Backend\n"
        f"Matrix Size {size}x{size}",
        fontsize=11
    )
    ax2.grid(True, axis="y", which="major", linestyle="--", alpha=0.4)
    ax2.bar_label(bars2, fmt="%.2e")

    output_path = output_dir / f"mca_size_{size}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    results_dir = Path(__file__).resolve().parent
    example_results_dir = results_dir.parent.parent / "example_results"

    mca_data = collect_mca_results(results_dir)
    uninstrumented_data = collect_example_results(example_results_dir)

    sizes = sorted(set(mca_data.keys()) | set(uninstrumented_data.keys()))
    if not sizes:
        print("No results found.")
        return

    print("=" * 100)
    print("RELATIVE ERROR ANALYSIS")
    print("=" * 100)
    print()

    for size in sizes:
        true_norm = compute_true_norm(size)
        settings_data = defaultdict(dict)
        if size in uninstrumented_data:
            settings_data.update(uninstrumented_data[size])
        if size in mca_data:
            settings_data.update(mca_data[size])

        relative_errors = build_relative_errors(settings_data, true_norm)

        print(f"Size {size}x{size}:")
        print(f"  x_true = [1, 2, 3, ..., {size}] (true solution)")
        print(f"  True norm ||x_true|| = {true_norm:.6e}")
        print()

        for setting in [s for s in SETTING_ORDER if s in settings_data]:
            print(f"  {SETTING_DESCRIPTIONS.get(setting, setting)}")
            for method in ["fp64", "fp32"]:
                error_vs_true_mean = settings_data[setting].get(method)

                if error_vs_true_mean is None:
                    continue

                relative_error = compute_relative_error(error_vs_true_mean, true_norm)

                print(f"    {METHOD_LABELS[method]}:")
                print(f"      ||x - x_true|| = {error_vs_true_mean:.6e}")
                print(f"      relative_error = ||x - x_true|| / ||x_true||")
                print(
                    f"      relative_error = {error_vs_true_mean:.6e} / {true_norm:.6e} = {relative_error:.6e}"
                )
                print()

        output_path = plot_relative_error_summary(size, relative_errors, results_dir)
        print(f"Plot saved to: {output_path}")
        print("-" * 100)
        print()


if __name__ == "__main__":
    main()

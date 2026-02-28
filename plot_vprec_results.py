#!/usr/bin/env python3
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation


FILENAME_RE = re.compile(r"size_(\d+)_([a-z0-9]+)_run_(\d+)\.txt$")
BASELINE_RE = re.compile(r"size_(\d+)_run_(\d+)\.txt$")

METHOD_LABELS = {
    "fp64": "FP64",
    "fp32": "FP32",
    "ir": "IR",
}

SETTING_ORDER = ["baseline", "fp64", "fp32", "bfloat16", "fp16"]

SETTING_DESCRIPTIONS = {
    "baseline": "baseline: native compiler (no VPREC)",
    "fp64": "fp64: binary64 mantissa 53, exponent 11",
    "fp32": "fp32: binary32 mantissa 24, exponent 8",
    "bfloat16": "bfloat16: preset bfloat16 (mantissa 8, exponent 8)",
    "fp16": "fp16: preset binary16 (mantissa 11, exponent 5)",
}


def compute_true_norm(size):
    """Compute ||x_true|| where x_true = [1, 2, 3, ..., size]
    Using the formula: ||x_true||^2 = sum(i^2 for i=1 to size) = size*(size+1)*(2*size+1)/6
    """
    sum_of_squares = size * (size + 1) * (2 * size + 1) / 6
    return math.sqrt(sum_of_squares)


def compute_relative_error(error_vs_true_mean, true_norm):
    """Calculate relative error = ||x - x_true|| / ||x_true||"""
    if error_vs_true_mean is None or true_norm is None or true_norm == 0:
        return float('nan')
    return error_vs_true_mean / true_norm


def parse_results_file(path):
    """Extract ||x-x_true|| from result file for all methods."""
    current_method = None
    result = {
        "fp64": None,
        "fp32": None,
        "ir": None,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("--- Naive FP64"):
                current_method = "fp64"
                continue
            if line.startswith("--- Naive FP32"):
                current_method = "fp32"
                continue
            if line.startswith("--- Mixed Precision"):
                current_method = "ir"
                continue

            if current_method is None:
                continue

            if line.startswith("Error vs true:"):
                value = line.split()[-1]
                result[current_method] = float(value)

    return result


def collect_vprec_results(results_dir):
    """Collect error_vs_true values, compute means."""
    data = defaultdict(lambda: defaultdict(dict))

    for path in results_dir.glob("size_*_run_*.txt"):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        size = int(match.group(1))
        setting = match.group(2)

        parsed = parse_results_file(path)
        for method, error_vs_true in parsed.items():
            if error_vs_true is not None:
                if (size, setting, method) not in data:
                    data[(size, setting, method)] = []
                data[(size, setting, method)].append(error_vs_true)

    # Convert to means
    result = defaultdict(lambda: defaultdict(dict))
    for (size, setting, method), values in data.items():
        result[size][setting][method] = mean_or_nan(values)

    return result


def collect_baseline_results(results_dir):
    """Collect error_vs_true values from baseline, compute means."""
    data = defaultdict(lambda: defaultdict(dict))

    for path in results_dir.glob("size_*_run_*.txt"):
        match = BASELINE_RE.match(path.name)
        if not match:
            continue
        size = int(match.group(1))
        setting = "baseline"

        parsed = parse_results_file(path)
        for method, error_vs_true in parsed.items():
            if error_vs_true is not None:
                if (size, setting, method) not in data:
                    data[(size, setting, method)] = []
                data[(size, setting, method)].append(error_vs_true)

    # Convert to means
    result = defaultdict(lambda: defaultdict(dict))
    for (size, setting, method), values in data.items():
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
    left_labels = ["FP64", "FP32", "BF16 (VPREC)"]
    left_settings = ["fp64", "fp32", "bfloat16"]
    left_methods = ["fp64", "fp32", "fp64"]
    left_values = [
        relative_errors.get(setting, {}).get(method, float("nan"))
        for setting, method in zip(left_settings, left_methods)
    ]

    right_labels = [
        "Mixed Precision (FP64/FP32)",
        "FP32 (VPREC)",
        "BF16 (VPREC)",
    ]
    right_settings = ["fp64", "fp32", "bfloat16"]
    right_values = [
        relative_errors.get(setting, {}).get("ir", float("nan"))
        for setting in right_settings
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    bars0 = axes[0].bar(left_labels, left_values, color="#1f77b4")
    axes[0].bar_label(bars0, fmt='%.2e')
    axes[0].set_title("Relative Error for Gaussian Solver under Different Precision Formats on 500x500 Matrix")
    axes[0].set_ylabel("Relative Error")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(LogFormatterSciNotation())
    axes[0].grid(True, axis="y", which="both", linestyle="--", alpha=0.4)

    bars1 = axes[1].bar(right_labels, right_values, color="#ff7f0e")
    axes[1].bar_label(bars1, fmt='%.2e')
    axes[1].set_title(
        "Relative Error for IR Mixed-Precision, FP32, BF16 measured on 500x500 Matrix"
    )
    axes[1].set_ylabel("Relative Error")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(LogFormatterSciNotation())
    axes[1].grid(True, axis="y", which="both", linestyle="--", alpha=0.4)

    for ax in axes:
        ax.set_xlabel("Precision setting")
        ax.tick_params(axis="x", rotation=15)

    output_path = output_dir / f"relative_error_summary_size_{size}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    results_dir = Path(__file__).resolve().parent
    baseline_dir = results_dir.parent / "example_results"

    data = collect_vprec_results(results_dir)
    if baseline_dir.exists():
        baseline_data = collect_baseline_results(baseline_dir)
        # Merge baseline data into main data
        for size, settings in baseline_data.items():
            for setting, methods in settings.items():
                data[size][setting].update(methods)

    if not data:
        print("No result files found.")
        return

    print("=" * 100)
    print("RELATIVE ERROR ANALYSIS")
    print("=" * 100)
    print()

    for size in sorted(data.keys()):
        true_norm = compute_true_norm(size)
        relative_errors = build_relative_errors(data[size], true_norm)
        
        print(f"Size {size}x{size}:")
        print(f"  x_true = [1, 2, 3, ..., {size}] (true solution)")
        print(f"  True norm ||x_true|| = {true_norm:.6e}")
        print()
        
        settings_data = data[size]
        settings = [s for s in SETTING_ORDER if s in settings_data]

        for setting in settings:
            print(f"  {SETTING_DESCRIPTIONS.get(setting, setting)}")
            for method in ["fp64", "fp32", "ir"]:
                error_vs_true_mean = settings_data[setting].get(method)
                
                if error_vs_true_mean is not None:
                    relative_error = compute_relative_error(error_vs_true_mean, true_norm)
                    
                    print(f"    {METHOD_LABELS[method]}:")
                    print(f"      ||x - x_true|| = {error_vs_true_mean:.6e}")
                    print(f"      relative_error = ||x - x_true|| / ||x_true||")
                    print(f"      relative_error = {error_vs_true_mean:.6e} / {true_norm:.6e} = {relative_error:.6e}")
                    print()
        
                output_path = plot_relative_error_summary(size, relative_errors, results_dir)
                print(f"Plot saved to: {output_path}")
        print("-" * 100)
        print()


if __name__ == "__main__":
    main()

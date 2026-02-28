#!/usr/bin/env python3
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LOG_RE = re.compile(r"size_(\d+)_(fp64|fp32|ir)_run_(\d+)_valgrind\.log$")
ALLOC_RE = re.compile(r"total heap usage: .*?,\s*([0-9,]+) bytes allocated")

MODE_LABELS = {
    "fp64": "FP64",
    "fp32": "FP32",
    "ir": "IR",
}


def parse_allocated_bytes(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = ALLOC_RE.search(line)
            if match:
                return int(match.group(1).replace(",", ""))
    return None


def collect_results(results_dir):
    data = defaultdict(lambda: defaultdict(list))

    for path in results_dir.glob("size_*_valgrind.log"):
        match = LOG_RE.match(path.name)
        if not match:
            continue
        size = int(match.group(1))
        mode = match.group(2)
        allocated = parse_allocated_bytes(path)
        if allocated is not None:
            data[size][mode].append(allocated)

    return data


def mean_or_nan(values):
    if not values:
        return float("nan")
    return float(np.nanmean(values))


def plot_heap_alloc_for_size(size, stats, output_dir):
    modes = ["fp64", "fp32", "ir"]
    values = [stats[mode] for mode in modes]
    values_mb = [value / 1e6 for value in values]
    min_val = min(values_mb)
    max_val = max(values_mb)
    padding = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.bar(
        [MODE_LABELS[m] for m in modes],
        values_mb,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    ax.set_title(f"Valgrind Memcheck: Average Heap Allocated (Matrix Size {size}x{size})")
    ax.set_ylabel("Allocated bytes (MB)")
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.grid(True, axis="y", linestyle=":", color="#b0b0b0", alpha=0.7)

    output_path = output_dir / "heap_alloc_plot.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    results_dir = Path(__file__).resolve().parent
    data = collect_results(results_dir)

    if not data:
        print("No Valgrind logs found.")
        return

    target_size = 1000
    if target_size not in data:
        print(f"No results found for size {target_size}x{target_size}.")
        return

    for size, modes in sorted(data.items()):
        if size != target_size:
            continue
        stats = {mode: mean_or_nan(modes.get(mode, [])) for mode in ["fp64", "fp32", "ir"]}
        output_path = plot_heap_alloc_for_size(size, stats, results_dir)

        print(f"Size {size}x{size}:")
        for mode in ["fp64", "fp32", "ir"]:
            value = stats[mode]
            print(f"  {MODE_LABELS[mode]}: {value:,.0f} bytes")
        print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()

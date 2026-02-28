#!/usr/bin/env python3
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

SIZE_RE = re.compile(r"size_(\d+)_run_(\d+)\.txt$")

SECTION_MAP = {
    "--- Naive FP64 Solution ---": "fp64",
    "--- Naive FP32 Solution ---": "fp32",
    "--- Mixed Precision with Iterative Refinement ---": "ir",
}


def parse_time_file(path):
    current = None
    result = {"fp64": None, "fp32": None, "ir": None}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if line in SECTION_MAP:
                current = SECTION_MAP[line]
                continue

            if current is None:
                continue

            if line.startswith("Time:"):
                value = line.split()[1]
                result[current] = float(value)
                current = None

    return result


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def relative_to_fp64(value, fp64_mean):
    if fp64_mean is None or fp64_mean == 0:
        return float("nan")
    return value / fp64_mean


def main():
    results_dir = Path(__file__).resolve().parent
    data = defaultdict(lambda: defaultdict(list))

    for path in results_dir.glob("size_*_run_*.txt"):
        match = SIZE_RE.match(path.name)
        if not match:
            continue
        size = int(match.group(1))
        parsed = parse_time_file(path)
        for method, value in parsed.items():
            if value is not None:
                data[size][method].append(value)

    if not data:
        print("No results found.")
        return

    print("=" * 100)
    print("TIME SUMMARY (mean over runs, microseconds)")
    print("=" * 100)
    print()

    sizes_sorted = sorted(data.keys())
    fp32_rel = []
    ir_rel = []
    fp64_rel = []

    for size in sizes_sorted:
        fp64_mean = mean(data[size].get("fp64", []))
        fp32_mean = mean(data[size].get("fp32", []))
        ir_mean = mean(data[size].get("ir", []))

        print(f"Size {size}x{size}:")
        print(f"  FP64 mean time: {fp64_mean:.2f} us")
        print(f"  FP32 mean time: {fp32_mean:.2f} us")
        print(f"  IR mean time:   {ir_mean:.2f} us")
        print()

        fp32_rel.append(relative_to_fp64(fp32_mean, fp64_mean))
        ir_rel.append(relative_to_fp64(ir_mean, fp64_mean))
        fp64_rel.append(1.0 if fp64_mean == fp64_mean else float("nan"))

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(sizes_sorted, fp32_rel, marker="o", label="FP32 / FP64")
    ax.plot(sizes_sorted, ir_rel, marker="s", label="IR / FP64")
    ax.plot(sizes_sorted, fp64_rel, label="FP64 / FP64", color="green")
    ax.set_xlabel("Matrix size")
    ax.set_ylabel("Relative time (vs FP64)")
    ax.set_title("Relative Runtime vs FP64 Baseline")
    ax.set_xticks(sizes_sorted)
    ax.set_xticklabels([str(size) for size in sizes_sorted])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    output_path = results_dir / "relative_time_vs_fp64.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()

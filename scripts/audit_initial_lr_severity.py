#!/usr/bin/env python3
"""First-step Adam severity analysis across counterfactual initial LRs.

For Adam with zero initial moment states, fixed pre-step parameters, fixed
clipped gradient, and no weight decay, the first parameter displacement is
linear in the learning rate:

    Delta(theta; alpha) = (alpha / alpha_ref) * Delta(theta; alpha_ref).

Therefore the first-step update-to-parameter ratio scales by the same factor.
This script applies that exact first-step scaling to the canonical LR=1.0
unsafe candidate transitions and compares the resulting ratios with the
first-100-step safe guard thresholds.
"""

import argparse
import csv
import json
import re
from pathlib import Path


INITIAL_LRS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001)
REFERENCE_LR = 1.0
REFERENCE_CONDITION = "buggy_no_guard"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit first-step severity across counterfactual initial LRs."
    )
    parser.add_argument("--semantics-csv", type=Path, required=True)
    parser.add_argument("--threshold-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_run_identity(source_member):
    model_match = re.search(r"(medium-4|large-6)", source_member)
    seed_match = re.search(r"seed(\d+)", source_member)

    if model_match is None or seed_match is None:
        raise ValueError(
            f"Could not parse model/seed from source member: {source_member}"
        )

    return model_match.group(1), int(seed_match.group(1))


def main():
    args = parse_args()

    with args.threshold_summary.open(encoding="utf-8") as handle:
        threshold_data = json.load(handle)

    with args.semantics_csv.open(newline="", encoding="utf-8") as handle:
        semantics_rows = list(csv.DictReader(handle))

    reference_rows = [
        row for row in semantics_rows
        if row["condition"] == REFERENCE_CONDITION
    ]

    output_rows = []

    for row in reference_rows:
        dataset = row["dataset"]
        model, seed = parse_run_identity(row["source_member"])

        reference_ratio = float(row["candidate_step1_ratio"])
        threshold = float(
            threshold_data["datasets"][dataset]["first_100_threshold_kappa10"]
        )

        for initial_lr in INITIAL_LRS:
            projected_ratio = (
                reference_ratio * initial_lr / REFERENCE_LR
            )

            output_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "initial_lr": initial_lr,
                    "reference_lr": REFERENCE_LR,
                    "reference_step1_ratio": reference_ratio,
                    "projected_step1_ratio": projected_ratio,
                    "threshold_kappa10": threshold,
                    "margin_over_threshold": projected_ratio / threshold,
                    "exceeds_threshold": projected_ratio > threshold,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = args.output_dir / "initial_lr_severity_per_run.csv"

    fieldnames = [
        "dataset",
        "model",
        "seed",
        "initial_lr",
        "reference_lr",
        "reference_step1_ratio",
        "projected_step1_ratio",
        "threshold_kappa10",
        "margin_over_threshold",
        "exceeds_threshold",
    ]

    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary_rows = []

    for dataset in sorted({row["dataset"] for row in output_rows}):
        for initial_lr in INITIAL_LRS:
            subset = [
                row for row in output_rows
                if row["dataset"] == dataset
                and row["initial_lr"] == initial_lr
            ]

            projected = [row["projected_step1_ratio"] for row in subset]
            margins = [row["margin_over_threshold"] for row in subset]

            summary_rows.append(
                {
                    "dataset": dataset,
                    "initial_lr": initial_lr,
                    "runs": len(subset),
                    "projected_ratio_min": min(projected),
                    "projected_ratio_max": max(projected),
                    "threshold_kappa10": subset[0]["threshold_kappa10"],
                    "minimum_margin_over_threshold": min(margins),
                    "all_runs_exceed_threshold": all(
                        row["exceeds_threshold"] for row in subset
                    ),
                }
            )

    summary_path = args.output_dir / "initial_lr_severity_summary.csv"

    summary_fields = [
        "dataset",
        "initial_lr",
        "runs",
        "projected_ratio_min",
        "projected_ratio_max",
        "threshold_kappa10",
        "minimum_margin_over_threshold",
        "all_runs_exceed_threshold",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Initial-LR first-step severity audit complete")
    print()

    for row in summary_rows:
        print(
            f"{row['dataset']:<12} "
            f"lr={row['initial_lr']:<5g} | "
            f"r1=[{row['projected_ratio_min']:.8g}, "
            f"{row['projected_ratio_max']:.8g}] | "
            f"min margin={row['minimum_margin_over_threshold']:.2f}x | "
            f"all exceed={row['all_runs_exceed_threshold']}"
        )

    print(f"\nSaved: {summary_path.resolve()}")
    print(f"Saved: {per_run_path.resolve()}")


if __name__ == "__main__":
    main()

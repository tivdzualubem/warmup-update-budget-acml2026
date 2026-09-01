#!/usr/bin/env python3

import csv
import json
import math
import re
import zipfile
from pathlib import Path

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory containing acml_raw_agnews.zip and acml_raw_dbpedia_core.zip",
    )
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        required=True,
        help="Path to threshold_window_audit.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rebuttal_2026/lr_assertion_vs_update_guard"),
    )
    return parser.parse_args()

WARMUP_STEPS = 8000
BASE_LR = 1e-4

D_MODEL = {
    "medium-4": 128,
    "large-6": 256,
}

SAFE_CONDITIONS = {
    "stable_with_guard",
    "fixed_order_with_guard",
}

BUGGY_AFTER_STEP = {
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
}


def stable_lr(step):
    if step < WARMUP_STEPS:
        scale = step / WARMUP_STEPS
    else:
        scale = 1.0
    return BASE_LR * scale


def legacy_lr(model_name, step):
    d_model = D_MODEL[model_name]
    step = max(1, step)
    return (d_model ** -0.5) * min(
        step ** -0.5,
        step * WARMUP_STEPS ** -1.5,
    )


def intended_lr(condition, model_name, step):
    if condition in {
        "fixed_order_with_guard",
        "buggy_no_guard",
        "buggy_failfast_guard",
        "buggy_rescue_guard",
    }:
        return legacy_lr(model_name, step)

    if condition in {
        "stable_with_guard",
        "single_bad_first_step_then_stable",
    }:
        return stable_lr(step)

    raise ValueError(condition)


def load_thresholds(threshold_csv):
    safe_max = {}

    with threshold_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["is_safe"] != "True":
                continue

            dataset = row["dataset"]
            value = float(row["max_ratio_window"])
            safe_max[dataset] = max(
                safe_max.get(dataset, 0.0),
                value,
            )

    return {
        dataset: 10.0 * value
        for dataset, value in safe_max.items()
    }


def parse_identity(member):
    name = Path(member).name

    match = re.match(
        r"(ag_news|dbpedia_14)_(medium-4|large-6)_(.+)_seed(\d+)_early_logs\.json",
        name,
    )

    if not match:
        return None

    dataset, model, condition, seed = match.groups()

    return dataset, model, condition, int(seed)


def main():
    args = parse_args()

    archives = {
        "ag_news": args.artifact_root / "acml_raw_agnews.zip",
        "dbpedia_14": args.artifact_root / "acml_raw_dbpedia_core.zip",
    }

    thresholds = load_thresholds(args.threshold_csv)
    rows = []

    for expected_dataset, archive_path in archives.items():
        with zipfile.ZipFile(archive_path) as zf:
            members = [
                name
                for name in zf.namelist()
                if name.endswith("_early_logs.json")
            ]

            for member in members:
                parsed = parse_identity(member)
                if parsed is None:
                    continue

                dataset, model, condition, seed = parsed

                if dataset != expected_dataset:
                    continue

                records = json.loads(zf.read(member))
                if not records:
                    continue

                first = records[0]

                if int(first["step"]) != 1:
                    continue

                intended = intended_lr(condition, model, 1)

                if condition in BUGGY_AFTER_STEP:
                    candidate_lr = float(first["lr_before_step"])
                else:
                    candidate_lr = float(first["lr_used"])

                lr_abs_error = abs(candidate_lr - intended)
                lr_assertion_violation = not math.isclose(
                    candidate_lr,
                    intended,
                    rel_tol=1e-9,
                    abs_tol=1e-15,
                )

                candidate_ratio = float(
                    first.get(
                        "unsafe_update_to_param_ratio",
                        first["update_to_param_ratio"],
                    )
                )

                threshold = thresholds[dataset]
                ratio_guard_violation = candidate_ratio > threshold

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "condition": condition,
                    "seed": seed,
                    "candidate_lr": candidate_lr,
                    "intended_lr": intended,
                    "candidate_to_intended_lr_ratio": candidate_lr / intended,
                    "lr_abs_error": lr_abs_error,
                    "lr_assertion_violation": lr_assertion_violation,
                    "candidate_update_ratio": candidate_ratio,
                    "threshold": threshold,
                    "update_guard_violation": ratio_guard_violation,
                    "source_member": member,
                })

    rows.sort(
        key=lambda x: (
            x["dataset"],
            x["model"],
            x["condition"],
            x["seed"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = args.output_dir / "lr_assertion_vs_update_guard.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 88)
    print("LR ASSERTION VS UPDATE-RATIO GUARD")
    print("=" * 88)

    for dataset in sorted(set(r["dataset"] for r in rows)):
        subset = [r for r in rows if r["dataset"] == dataset]

        print(f"\n{dataset}")
        print(f"  total first-step runs: {len(subset)}")

        for condition in sorted(set(r["condition"] for r in subset)):
            cond = [r for r in subset if r["condition"] == condition]

            lr_flags = sum(r["lr_assertion_violation"] for r in cond)
            update_flags = sum(r["update_guard_violation"] for r in cond)

            print(
                f"  {condition:<34} "
                f"LR assertion {lr_flags}/{len(cond)} | "
                f"update guard {update_flags}/{len(cond)}"
            )

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

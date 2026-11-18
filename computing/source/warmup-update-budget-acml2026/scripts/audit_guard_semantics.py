#!/usr/bin/env python3
"""Audit candidate and committed first-step update ratios.

This script verifies the semantics of the unsafe, fail-fast, rescue, and
single-bad conditions directly from the archived early-step logs.

The candidate ratio is measured immediately after the candidate optimizer
transition. For rescue runs, the committed ratio is measured after rollback
and replay at the safe learning rate.
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path


CONDITIONS = (
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "single_bad_first_step_then_stable",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit candidate versus committed first-step update ratios."
    )
    parser.add_argument("--agnews-zip", type=Path, required=True)
    parser.add_argument("--dbpedia-core-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_archive(dataset, zip_path):
    rows = []

    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith("_early_logs.json"):
                continue

            condition = next(
                (condition for condition in CONDITIONS if condition in member),
                None,
            )
            if condition is None:
                continue

            with archive.open(member) as handle:
                records = json.load(handle)

            step1 = next(
                record for record in records if int(record["step"]) == 1
            )

            candidate = float(
                step1.get(
                    "unsafe_update_to_param_ratio",
                    step1["update_to_param_ratio"],
                )
            )
            committed = float(step1["update_to_param_ratio"])

            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "candidate_step1_ratio": candidate,
                    "committed_step1_ratio": committed,
                    "candidate_to_committed_ratio":
                        candidate / committed if committed != 0 else None,
                    "source_member": member,
                }
            )

    expected = len(CONDITIONS) * 2 * 3
    if len(rows) != expected:
        raise RuntimeError(
            f"{dataset}: expected {expected} unsafe-condition runs, "
            f"found {len(rows)}"
        )

    return rows


def main():
    args = parse_args()

    for path in (args.agnews_zip, args.dbpedia_core_zip):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = []
    rows.extend(audit_archive("ag_news", args.agnews_zip))
    rows.extend(audit_archive("dbpedia_14", args.dbpedia_core_zip))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "condition",
        "candidate_step1_ratio",
        "committed_step1_ratio",
        "candidate_to_committed_ratio",
        "source_member",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Guard semantics audit complete")

    for dataset in ("ag_news", "dbpedia_14"):
        print(f"\n{dataset}")

        dataset_rows = [row for row in rows if row["dataset"] == dataset]

        for condition in CONDITIONS:
            condition_rows = [
                row for row in dataset_rows
                if row["condition"] == condition
            ]

            candidates = [
                row["candidate_step1_ratio"] for row in condition_rows
            ]
            committed = [
                row["committed_step1_ratio"] for row in condition_rows
            ]

            print(
                f"{condition}: "
                f"candidate=[{min(candidates):.8g}, {max(candidates):.8g}] | "
                f"committed=[{min(committed):.8g}, {max(committed):.8g}]"
            )

    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

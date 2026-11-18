#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_expansion.src.kaggle_scratch_pilot import (
    atomic_json,
    build_thresholds,
    calibration_jobs,
    diagnostic_jobs,
    main_jobs,
    run_jobs,
    summarize,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=[
            "calibration",
            "main",
            "diagnostic",
            "summary",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--output-root",
        default=(
            "publication_expansion/"
            "results/kaggle_pilot/"
            "scratch"
        ),
    )

    parser.add_argument(
        "--threshold-file",
        default=(
            "publication_expansion/"
            "results/kaggle_pilot/"
            "scratch_thresholds.json"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(
        args.output_root
    )

    threshold_path = Path(
        args.threshold_file
    )

    def thresholds():
        if not threshold_path.exists():
            raise SystemExit(
                "scratch pilot threshold file "
                "missing; run calibration first"
            )

        return json.loads(
            threshold_path.read_text()
        )

    if args.phase in {
        "calibration",
        "all",
    }:
        jobs = calibration_jobs()

        print(
            "SCRATCH PILOT CALIBRATION "
            f"JOBS={len(jobs)}"
        )

        run_jobs(
            jobs,
            output_root=root,
            device=args.device,
            num_workers=args.num_workers,
            execute=args.execute,
        )

        if args.execute:
            manifest = build_thresholds(
                root
            )

            atomic_json(
                threshold_path,
                manifest,
            )

            print(
                "WROTE:",
                threshold_path,
            )

    if args.phase in {
        "main",
        "all",
    }:
        manifest = thresholds()

        jobs = main_jobs(
            manifest
        )

        print(
            "SCRATCH PILOT MAIN "
            f"JOBS={len(jobs)}"
        )

        run_jobs(
            jobs,
            output_root=root,
            device=args.device,
            num_workers=args.num_workers,
            execute=args.execute,
        )

    if args.phase in {
        "diagnostic",
        "all",
    }:
        manifest = thresholds()

        jobs = diagnostic_jobs(
            manifest
        )

        print(
            "SCRATCH PILOT DIAGNOSTIC "
            f"JOBS={len(jobs)}"
        )

        run_jobs(
            jobs,
            output_root=root,
            device=args.device,
            num_workers=args.num_workers,
            execute=args.execute,
        )

    if args.phase in {
        "summary",
        "all",
    }:
        if args.execute:
            result = summarize(
                root
            )

            print(
                "SCRATCH PILOT SUMMARY "
                f"ROWS={len(result['rows'])}"
            )


if __name__ == "__main__":
    main()

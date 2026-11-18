#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_expansion.src.pretrained_matrix import (
    build_jobs,
    run_matrix,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "safe",
            "nonguarded",
            "rescue",
            "diagnostic",
        ],
    )

    parser.add_argument(
        "--output-root",
        default=(
            "publication_expansion/results/"
            "pretrained"
        ),
    )

    parser.add_argument(
        "--threshold-file",
        default=(
            "publication_expansion/results/"
            "pretrained/"
            "loso_thresholds.json"
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
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

    thresholds = None

    if args.phase in {
        "rescue",
        "diagnostic",
    }:
        path = Path(
            args.threshold_file
        )

        if not path.exists():
            raise SystemExit(
                "phase requires completed "
                "LOSO threshold manifest"
            )

        thresholds = json.loads(
            path.read_text()
        )

    jobs = build_jobs(
        args.phase,
        threshold_manifest=thresholds,
    )

    print(
        f"PHASE={args.phase} "
        f"JOBS={len(jobs)} "
        f"EXECUTE={args.execute}"
    )

    run_matrix(
        jobs,
        output_root=args.output_root,
        device=args.device,
        num_workers=args.num_workers,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_expansion.src.scratch_matrix import (
    compute_loso_thresholds,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        default=(
            "publication_expansion/results/"
            "scratch"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "publication_expansion/results/"
            "scratch/loso_thresholds.json"
        ),
    )

    parser.add_argument(
        "--kappa",
        type=float,
        default=10.0,
    )

    args = parser.parse_args()

    manifest = compute_loso_thresholds(
        Path(args.output_root),
        kappa=args.kappa,
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output.with_suffix(
        output.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            manifest,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    os.replace(tmp, output)

    print(
        f"WROTE {len(manifest['records'])} "
        f"LOSO thresholds: {output}"
    )


if __name__ == "__main__":
    main()

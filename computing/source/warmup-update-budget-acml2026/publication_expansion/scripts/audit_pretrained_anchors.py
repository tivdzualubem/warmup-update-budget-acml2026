#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_expansion.src.pretrained_matrix import (
    audit_matched_anchors,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        default=(
            "publication_expansion/results/"
            "pretrained"
        ),
    )

    args = parser.parse_args()

    audit = audit_matched_anchors(
        Path(args.output_root)
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Allow this script to run directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

from publication_expansion.src.scratch_dataset_adapter import (
    prepare_scratch_dataset_from_examples,
)


REVISION_CONFIG = (
    REPO_ROOT
    / "publication_expansion"
    / "configs"
    / "scratch_dataset_revisions_v1.json"
)

revision_config = json.loads(
    REVISION_CONFIG.read_text()
)

DATASETS = {
    "ag_news": {
        "repo": revision_config["datasets"]["ag_news"]["repository"],
        "revision": revision_config["datasets"]["ag_news"]["revision"],
        "text_field": "text",
    },
    "dbpedia_14": {
        "repo": revision_config["datasets"]["dbpedia_14"]["repository"],
        "revision": revision_config["datasets"]["dbpedia_14"]["revision"],
        "text_field": "content",
    },
}


def vocab_sha256(vocab):
    payload = json.dumps(
        vocab,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()



output = {
    "schema_version": 1,
    "split_seed": 42,
    "validation_ratio": 0.15,
    "train_fraction": 1.0,
    "datasets": {},
}

for dataset_name, cfg in DATASETS.items():
    print("\n" + "=" * 80)
    print("VERIFYING:", dataset_name)

    revision = cfg["revision"]

    print("repo:", cfg["repo"])
    print("frozen revision:", revision)

    raw = load_dataset(
        cfg["repo"],
        revision=revision,
    )

    train = raw["train"]
    test = raw["test"]

    train_texts = [
        row[cfg["text_field"]]
        for row in train
    ]

    train_labels = [
        row["label"]
        for row in train
    ]

    test_texts = [
        row[cfg["text_field"]]
        for row in test
    ]

    test_labels = [
        row["label"]
        for row in test
    ]

    prepared = prepare_scratch_dataset_from_examples(
        dataset_name=dataset_name,
        train_texts_full=train_texts,
        train_labels_full=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        max_length=128,
        split_seed=42,
        val_ratio=0.15,
        train_fraction=1.0,
    )

    record = dict(prepared.manifest)

    record["hf_revision"] = revision
    record["hf_fingerprints"] = {
        "train": getattr(
            train,
            "_fingerprint",
            None,
        ),
        "test": getattr(
            test,
            "_fingerprint",
            None,
        ),
    }

    record["vocab_sha256"] = vocab_sha256(
        prepared.vocab
    )

    output["datasets"][dataset_name] = record

    print(
        "train / validation / test:",
        len(prepared.train_dataset),
        len(prepared.validation_dataset),
        len(prepared.test_dataset),
    )

    print(
        "vocab size:",
        len(prepared.vocab),
    )

    print(
        "train split hash:",
        record["split"]["train_indices_sha256"],
    )

    print(
        "validation split hash:",
        record["split"]["validation_indices_sha256"],
    )

    print(
        "vocab hash:",
        record["vocab_sha256"],
    )


out = Path(
    "publication_expansion/manifests/"
    "scratch_fixed_split_manifest_v1.json"
)

out.write_text(
    json.dumps(
        output,
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)

print("\nWROTE:", out)

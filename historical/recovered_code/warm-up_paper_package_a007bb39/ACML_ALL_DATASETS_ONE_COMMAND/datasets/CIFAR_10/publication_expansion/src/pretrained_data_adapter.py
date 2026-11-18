"""Data preparation for pretrained publication experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "pretrained_sources_v1.json"
)

FIXED_SPLIT_SEED = 42
TEXT_MAX_LENGTH = 128
WNUT_MAX_LENGTH = 128
CIFAR_VALIDATION_SIZE = 5000


def load_sources() -> Dict:
    data = json.loads(CONFIG_PATH.read_text())

    if data.get("status") != "frozen":
        raise RuntimeError(
            "pretrained source manifest is not frozen"
        )

    return data


def dataset_source(name: str) -> Dict[str, str]:
    sources = load_sources()

    if name not in sources["datasets"]:
        raise ValueError(
            f"unsupported pretrained dataset: {name}"
        )

    return sources["datasets"][name]


def load_frozen_dataset(name: str):
    src = dataset_source(name)

    return load_dataset(
        src["repository"],
        revision=src["revision"],
    )


# ============================================================
# AG NEWS / DBPEDIA
# ============================================================

SEQUENCE_SPECS = {
    "ag_news": {
        "text_field": "text",
        "num_labels": 4,
    },
    "dbpedia_14": {
        "text_field": "content",
        "num_labels": 14,
    },
}


def tokenize_sequence_batch(
    examples,
    *,
    tokenizer,
    dataset_name: str,
    max_length: int = TEXT_MAX_LENGTH,
):
    if dataset_name not in SEQUENCE_SPECS:
        raise ValueError(
            f"unsupported sequence dataset: {dataset_name}"
        )

    text_field = SEQUENCE_SPECS[
        dataset_name
    ]["text_field"]

    encoded = tokenizer(
        examples[text_field],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    encoded["labels"] = examples["label"]

    return encoded


def deterministic_dataset_fraction(
    dataset,
    *,
    fraction: float,
    seed: int = FIXED_SPLIT_SEED,
):
    """Deterministic training-only subset for non-scientific pilots."""

    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "fraction must satisfy 0 < fraction <= 1"
        )

    if fraction == 1.0:
        return dataset

    n = len(dataset)
    n_keep = max(
        1,
        int(n * fraction),
    )

    rng = np.random.RandomState(seed)

    indices = rng.choice(
        n,
        size=n_keep,
        replace=False,
    )

    return dataset.select(
        indices.tolist()
    )


def prepare_sequence_dataset(
    *,
    dataset_name: str,
    tokenizer,
    train_fraction: float = 1.0,
    fraction_seed: int = FIXED_SPLIT_SEED,
):
    raw = load_frozen_dataset(dataset_name)

    spec = SEQUENCE_SPECS[dataset_name]

    train = raw["train"]

    # Fixed publication split:
    # 15% validation from official train, seed 42.
    split = train.train_test_split(
        test_size=0.15,
        seed=FIXED_SPLIT_SEED,
        shuffle=True,
    )

    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = raw["test"]

    train_ds = deterministic_dataset_fraction(
        train_ds,
        fraction=train_fraction,
        seed=fraction_seed,
    )

    remove_columns = train.column_names

    def encode(batch):
        return tokenize_sequence_batch(
            batch,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
        )

    train_ds = train_ds.map(
        encode,
        batched=True,
        remove_columns=remove_columns,
    )

    val_ds = val_ds.map(
        encode,
        batched=True,
        remove_columns=remove_columns,
    )

    test_ds = test_ds.map(
        encode,
        batched=True,
        remove_columns=test_ds.column_names,
    )

    columns = [
        "input_ids",
        "attention_mask",
        "labels",
    ]

    train_ds.set_format(
        type="torch",
        columns=columns,
    )
    val_ds.set_format(
        type="torch",
        columns=columns,
    )
    test_ds.set_format(
        type="torch",
        columns=columns,
    )

    return {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
        "num_labels": spec["num_labels"],
    }


# ============================================================
# WNUT-17
# ============================================================

def align_first_subtoken_labels(
    *,
    word_ids: Sequence[int | None],
    word_labels: Sequence[int],
) -> List[int]:
    """
    Label only the first subtoken of each original word.
    Special tokens and continuation subtokens receive -100.
    """

    aligned = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)

        elif word_id != previous_word_id:
            aligned.append(
                int(word_labels[word_id])
            )

        else:
            aligned.append(-100)

        previous_word_id = word_id

    return aligned


def tokenize_wnut_batch(
    examples,
    *,
    tokenizer,
    max_length: int = WNUT_MAX_LENGTH,
):
    encoded = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
    )

    labels = []

    for batch_index, word_labels in enumerate(
        examples["ner_tags"]
    ):
        word_ids = encoded.word_ids(
            batch_index=batch_index
        )

        labels.append(
            align_first_subtoken_labels(
                word_ids=word_ids,
                word_labels=word_labels,
            )
        )

    encoded["labels"] = labels

    return encoded


def prepare_wnut_dataset(*, tokenizer):
    raw = load_frozen_dataset("wnut_17")

    label_feature = (
        raw["train"]
        .features["ner_tags"]
        .feature
    )

    label_names = list(
        label_feature.names
    )

    id2label = {
        i: label
        for i, label in enumerate(
            label_names
        )
    }

    label2id = {
        label: i
        for i, label in id2label.items()
    }

    def encode(batch):
        return tokenize_wnut_batch(
            batch,
            tokenizer=tokenizer,
        )

    prepared = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split = raw[split_name]

        split = split.map(
            encode,
            batched=True,
            remove_columns=split.column_names,
        )

        split.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "labels",
            ],
        )

        prepared[split_name] = split

    prepared["num_labels"] = len(
        label_names
    )
    prepared["id2label"] = id2label
    prepared["label2id"] = label2id

    return prepared


# ============================================================
# CIFAR-10
# ============================================================

@dataclass(frozen=True)
class StratifiedSplit:
    train_indices: List[int]
    validation_indices: List[int]


def make_cifar_stratified_split(
    labels: Sequence[int],
    *,
    validation_size: int = CIFAR_VALIDATION_SIZE,
    split_seed: int = FIXED_SPLIT_SEED,
) -> StratifiedSplit:
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    classes, counts = np.unique(
        labels,
        return_counts=True,
    )

    if validation_size <= 0:
        raise ValueError(
            "validation_size must be positive"
        )

    if validation_size >= len(labels):
        raise ValueError(
            "validation_size must be smaller "
            "than the dataset"
        )

    # CIFAR-10 is balanced. For the frozen 5000-image
    # validation set this gives exactly 500 per class.
    proportions = (
        counts / counts.sum()
    )

    raw_allocations = (
        proportions * validation_size
    )

    allocations = np.floor(
        raw_allocations
    ).astype(int)

    remainder = (
        validation_size
        - int(allocations.sum())
    )

    fractional = (
        raw_allocations - allocations
    )

    order = np.argsort(
        -fractional,
        kind="stable",
    )

    for idx in order[:remainder]:
        allocations[idx] += 1

    rng = np.random.RandomState(
        split_seed
    )

    validation = []

    for cls, amount in zip(
        classes,
        allocations,
    ):
        cls_indices = np.flatnonzero(
            labels == cls
        )

        permuted = rng.permutation(
            cls_indices
        )

        validation.extend(
            permuted[:amount].tolist()
        )

    validation = np.asarray(
        validation,
        dtype=np.int64,
    )

    # Shuffle selected validation indices deterministically
    # so they are not class-block ordered.
    validation = rng.permutation(
        validation
    )

    validation_set = set(
        validation.tolist()
    )

    train = [
        i
        for i in range(len(labels))
        if i not in validation_set
    ]

    return StratifiedSplit(
        train_indices=train,
        validation_indices=(
            validation.tolist()
        ),
    )


class CifarImageDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        indices,
        *,
        processor,
    ):
        self.dataset = hf_dataset
        self.indices = list(indices)
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        example = self.dataset[
            self.indices[index]
        ]

        processed = self.processor(
            images=example["img"],
            return_tensors="pt",
        )

        return {
            "pixel_values":
                processed["pixel_values"][0],
            "labels":
                torch.tensor(
                    int(example["label"]),
                    dtype=torch.long,
                ),
        }


def prepare_cifar10_dataset(
    *,
    processor,
):
    raw = load_frozen_dataset("cifar10")

    labels = raw["train"]["label"]

    split = make_cifar_stratified_split(
        labels,
        validation_size=(
            CIFAR_VALIDATION_SIZE
        ),
        split_seed=FIXED_SPLIT_SEED,
    )

    label_names = list(
        raw["train"]
        .features["label"]
        .names
    )

    id2label = {
        i: label
        for i, label in enumerate(
            label_names
        )
    }

    label2id = {
        label: i
        for i, label in id2label.items()
    }

    test_indices = list(
        range(len(raw["test"]))
    )

    return {
        "train": CifarImageDataset(
            raw["train"],
            split.train_indices,
            processor=processor,
        ),
        "validation": CifarImageDataset(
            raw["train"],
            split.validation_indices,
            processor=processor,
        ),
        "test": CifarImageDataset(
            raw["test"],
            test_indices,
            processor=processor,
        ),
        "num_labels": len(
            label_names
        ),
        "id2label": id2label,
        "label2id": label2id,
        "split": split,
    }

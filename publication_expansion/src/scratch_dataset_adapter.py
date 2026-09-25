"""Publication-grade AG News / DBpedia scratch dataset adapter.

Preserves the original scratch preprocessing while enforcing the new
fixed-split contract:
    - official train/test splits
    - validation derived only from official train
    - split_seed fixed independently of experiment seed
    - vocabulary built from final training subset only
    - lowercase whitespace tokenization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from publication_expansion.src.scratch_data_contract import (
    DEFAULT_VAL_RATIO,
    FIXED_SPLIT_SEED,
    make_split_indices,
    split_manifest,
)


DATASET_SPECS = {
    "ag_news": {
        "hf_name": "fancyzhx/ag_news",
        "text_field": "text",
        "label_field": "label",
        "num_classes": 4,
    },
    "dbpedia_14": {
        "hf_name": "fancyzhx/dbpedia_14",
        "text_field": "content",
        "label_field": "label",
        "num_classes": 14,
    },
}


def get_dataset_spec(dataset_name: str) -> Dict[str, object]:
    try:
        return dict(DATASET_SPECS[dataset_name])
    except KeyError as exc:
        raise ValueError(
            f"unsupported scratch dataset: {dataset_name}"
        ) from exc


def build_vocab(
    texts: Sequence[str],
    *,
    min_freq: int = 2,
    max_vocab_size: int = 30000,
) -> Dict[str, int]:
    from collections import Counter

    if min_freq < 1:
        raise ValueError("min_freq must be >= 1")

    if max_vocab_size < 2:
        raise ValueError(
            "max_vocab_size must be >= 2"
        )

    counter = Counter()

    for text in texts:
        counter.update(text.lower().split())

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, freq in counter.most_common(
        max_vocab_size - 2
    ):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def text_to_indices(
    text: str,
    vocab: Dict[str, int],
    *,
    max_length: int = 128,
) -> List[int]:
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    tokens = text.lower().split()[:max_length]

    indices = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    padding_length = max_length - len(indices)

    if padding_length > 0:
        indices.extend(
            [vocab["<PAD>"]] * padding_length
        )

    return indices


class ScratchTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Dict[str, int],
        *,
        max_length: int = 128,
    ):
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must have equal length"
            )

        self.texts = list(texts)
        self.labels = [int(x) for x in labels]
        self.vocab = dict(vocab)
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.tensor(
                text_to_indices(
                    self.texts[idx],
                    self.vocab,
                    max_length=self.max_length,
                ),
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        }


@dataclass
class PreparedScratchDataset:
    train_dataset: ScratchTextDataset
    validation_dataset: ScratchTextDataset
    test_dataset: ScratchTextDataset
    vocab: Dict[str, int]
    num_classes: int
    manifest: Dict[str, object]


def prepare_scratch_dataset_from_examples(
    *,
    dataset_name: str,
    train_texts_full: Sequence[str],
    train_labels_full: Sequence[int],
    test_texts: Sequence[str],
    test_labels: Sequence[int],
    max_length: int = 128,
    split_seed: int = FIXED_SPLIT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    train_fraction: float = 1.0,
) -> PreparedScratchDataset:
    """Prepare scratch datasets from already-loaded examples."""

    spec = get_dataset_spec(dataset_name)

    if len(train_texts_full) != len(
        train_labels_full
    ):
        raise ValueError(
            "official train texts/labels length mismatch"
        )

    if len(test_texts) != len(test_labels):
        raise ValueError(
            "official test texts/labels length mismatch"
        )

    split = make_split_indices(
        len(train_texts_full),
        split_seed=split_seed,
        val_ratio=val_ratio,
        train_fraction=train_fraction,
    )

    train_texts = [
        train_texts_full[i]
        for i in split.train
    ]

    train_labels = [
        train_labels_full[i]
        for i in split.train
    ]

    validation_texts = [
        train_texts_full[i]
        for i in split.validation
    ]

    validation_labels = [
        train_labels_full[i]
        for i in split.validation
    ]

    vocab = build_vocab(
        train_texts,
        min_freq=2,
        max_vocab_size=30000,
    )

    manifest = {
        "dataset_name": dataset_name,
        "hf_name": spec["hf_name"],
        "text_field": spec["text_field"],
        "label_field": spec["label_field"],
        "num_classes": spec["num_classes"],
        "official_train_count":
            len(train_texts_full),
        "official_test_count":
            len(test_texts),
        "max_length": int(max_length),
        "vocab_policy": {
            "tokenization":
                "lowercase_whitespace_split",
            "pad_token": "<PAD>",
            "pad_index": 0,
            "unk_token": "<UNK>",
            "unk_index": 1,
            "min_freq": 2,
            "max_vocab_size": 30000,
            "source":
                "final_training_subset_only",
        },
        "split": split_manifest(
            split,
            split_seed=split_seed,
            val_ratio=val_ratio,
            train_fraction=train_fraction,
        ),
        "vocab_size": len(vocab),
    }

    return PreparedScratchDataset(
        train_dataset=ScratchTextDataset(
            train_texts,
            train_labels,
            vocab,
            max_length=max_length,
        ),
        validation_dataset=ScratchTextDataset(
            validation_texts,
            validation_labels,
            vocab,
            max_length=max_length,
        ),
        test_dataset=ScratchTextDataset(
            test_texts,
            test_labels,
            vocab,
            max_length=max_length,
        ),
        vocab=vocab,
        num_classes=int(spec["num_classes"]),
        manifest=manifest,
    )


def load_scratch_hf_dataset(
    dataset_name: str,
    *,
    max_length: int = 128,
    split_seed: int = FIXED_SPLIT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    train_fraction: float = 1.0,
) -> PreparedScratchDataset:
    """Load official HF splits and apply the frozen scratch protocol.

    datasets is imported lazily so pure unit tests do not need network
    access or the Hugging Face package installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face 'datasets' is required "
            "for real dataset loading"
        ) from exc

    spec = get_dataset_spec(dataset_name)

    raw = load_dataset(spec["hf_name"])

    official_train = raw["train"]
    official_test = raw["test"]

    train_texts_full = [
        item[spec["text_field"]]
        for item in official_train
    ]

    train_labels_full = [
        item[spec["label_field"]]
        for item in official_train
    ]

    test_texts = [
        item[spec["text_field"]]
        for item in official_test
    ]

    test_labels = [
        item[spec["label_field"]]
        for item in official_test
    ]

    prepared = prepare_scratch_dataset_from_examples(
        dataset_name=dataset_name,
        train_texts_full=train_texts_full,
        train_labels_full=train_labels_full,
        test_texts=test_texts,
        test_labels=test_labels,
        max_length=max_length,
        split_seed=split_seed,
        val_ratio=val_ratio,
        train_fraction=train_fraction,
    )

    # Record dataset fingerprints when exposed by datasets.
    prepared.manifest["hf_fingerprints"] = {
        split_name: getattr(
            raw[split_name],
            "_fingerprint",
            None,
        )
        for split_name in ["train", "test"]
    }

    return prepared

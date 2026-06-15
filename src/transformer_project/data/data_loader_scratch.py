"""
Generic scratch-model data loader for AG News and DBPedia.
Supports:
- official train/test split
- validation split from training only
- train-size fractions
- simple evaluation-time shifts
"""

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
import numpy as np


def get_dataset_spec(dataset_name: str):
    if dataset_name == "ag_news":
        return {
            "hf_name": "fancyzhx/ag_news",
            "text_field": "text",
            "num_classes": 4
        }
    elif dataset_name == "dbpedia_14":
        return {
            "hf_name": "fancyzhx/dbpedia_14",
            "text_field": "content",
            "num_classes": 14
        }
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_vocab(texts, min_freq=2, max_vocab_size=30000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab_size - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def apply_shift_to_text(text, shift_config=None, idx=0):
    if shift_config is None:
        return text

    shift_name = shift_config.get("name", "clean")
    if shift_name == "clean":
        return text

    tokens = text.lower().split()

    if shift_name == "truncation":
        max_tokens = shift_config.get("max_tokens", 32)
        tokens = tokens[:max_tokens]
        return " ".join(tokens)

    if shift_name == "unk_corruption":
        rate = shift_config.get("rate", 0.2)
        seed = shift_config.get("seed", 42)
        rng = np.random.RandomState(seed + idx)

        if len(tokens) == 0:
            return text

        n_corrupt = int(round(rate * len(tokens)))
        n_corrupt = min(n_corrupt, len(tokens))

        if n_corrupt > 0:
            corrupt_idx = rng.choice(len(tokens), size=n_corrupt, replace=False)
            for j in corrupt_idx:
                tokens[j] = "__UNKSHIFT__"

        return " ".join(tokens)

    raise ValueError(f"Unknown shift name: {shift_name}")


def text_to_indices(text, vocab, max_length=128):
    tokens = text.lower().split()[:max_length]
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    padding_length = max_length - len(indices)
    if padding_length > 0:
        indices.extend([vocab["<PAD>"]] * padding_length)

    return indices


class ScratchTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=128, shift_config=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.shift_config = shift_config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        shifted_text = apply_shift_to_text(text, self.shift_config, idx=idx)
        input_ids = text_to_indices(shifted_text, self.vocab, self.max_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_text_dataset(
    dataset_name: str,
    max_length=128,
    split_seed=42,
    val_ratio=0.15,
    train_fraction=1.0
):
    spec = get_dataset_spec(dataset_name)

    print(f"Loading {dataset_name}...")
    dataset = load_dataset(spec["hf_name"])

    official_train = dataset["train"]
    official_test = dataset["test"]

    train_texts_full = [item[spec["text_field"]] for item in official_train]
    train_labels_full = [item["label"] for item in official_train]

    test_texts = [item[spec["text_field"]] for item in official_test]
    test_labels = [item["label"] for item in official_test]

    print(f"Official split sizes: Train={len(train_texts_full):,} | Test={len(test_texts):,}")

    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(len(train_texts_full))

    n_val = int(val_ratio * len(train_texts_full))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    if train_fraction < 1.0:
        frac_rng = np.random.RandomState(split_seed + 1)
        n_train_keep = max(1, int(round(train_fraction * len(train_indices))))
        sampled = frac_rng.choice(train_indices, size=n_train_keep, replace=False)
        train_indices = np.array(sampled)

    train_texts = [train_texts_full[i] for i in train_indices]
    train_labels = [train_labels_full[i] for i in train_indices]

    val_texts = [train_texts_full[i] for i in val_indices]
    val_labels = [train_labels_full[i] for i in val_indices]

    print(
        f"Split (seed={split_seed}, train_fraction={train_fraction:.2f}): "
        f"Train={len(train_texts):,} | Val={len(val_texts):,} | Test={len(test_texts):,}"
    )

    vocab = build_vocab(train_texts, min_freq=2, max_vocab_size=30000)
    print(f"Vocabulary size: {len(vocab):,} (built from final training subset only)")

    train_dataset = ScratchTextDataset(train_texts, train_labels, vocab, max_length=max_length, shift_config=None)
    val_dataset = ScratchTextDataset(val_texts, val_labels, vocab, max_length=max_length, shift_config=None)
    test_dataset = ScratchTextDataset(test_texts, test_labels, vocab, max_length=max_length, shift_config=None)

    return train_dataset, val_dataset, test_dataset, vocab, spec["num_classes"]


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=pin
    )
    return train_loader, val_loader, test_loader


def build_shifted_test_loader(base_test_dataset, shift_config, batch_size=32):
    shifted_dataset = ScratchTextDataset(
        texts=base_test_dataset.texts,
        labels=base_test_dataset.labels,
        vocab=base_test_dataset.vocab,
        max_length=base_test_dataset.max_length,
        shift_config=shift_config
    )

    pin = torch.cuda.is_available()
    return DataLoader(
        shifted_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=pin
    )

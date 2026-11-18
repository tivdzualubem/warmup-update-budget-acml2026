"""
Pretrained-model data loading for AG News and DBPedia.
Official train/test split is preserved.
Validation is created only from the training split.
"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import torch


def _get_dataset_config(dataset_name: str):
    if dataset_name == "ag_news":
        return {
            "hf_name": "ag_news",
            "text_field": "text",
            "num_classes": 4
        }
    elif dataset_name == "dbpedia_14":
        return {
            "hf_name": "dbpedia_14",
            "text_field": "content",
            "num_classes": 14
        }
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def load_pretrained_dataset(
    dataset_name: str,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    split_seed: int = 42,
    val_ratio: float = 0.15,
    batch_size: int = 8
):
    cfg = _get_dataset_config(dataset_name)
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(cfg["hf_name"])

    official_train = dataset["train"]
    official_test = dataset["test"]

    train_size = len(official_train)
    test_size = len(official_test)

    print(f"Official split sizes: Train={train_size:,} | Test={test_size:,}")

    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(train_size)

    n_val = int(val_ratio * train_size)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()

    train_split = official_train.select(train_indices)
    val_split = official_train.select(val_indices)
    test_split = official_test

    print(
        f"Split (seed={split_seed}): "
        f"Train={len(train_split):,} | Val={len(val_split):,} | Test={len(test_split):,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_batch(batch):
        return tokenizer(
            batch[cfg["text_field"]],
            truncation=True,
            max_length=max_length
        )

    train_split = train_split.map(tokenize_batch, batched=True)
    val_split = val_split.map(tokenize_batch, batched=True)
    test_split = test_split.map(tokenize_batch, batched=True)

    train_split = train_split.rename_column("label", "labels")
    val_split = val_split.rename_column("label", "labels")
    test_split = test_split.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_split.set_format(type="torch", columns=keep_cols)
    val_split.set_format(type="torch", columns=keep_cols)
    test_split.set_format(type="torch", columns=keep_cols)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=pin
    )
    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=pin
    )
    test_loader = DataLoader(
        test_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=pin
    )

    return train_loader, val_loader, test_loader, tokenizer, cfg["num_classes"]

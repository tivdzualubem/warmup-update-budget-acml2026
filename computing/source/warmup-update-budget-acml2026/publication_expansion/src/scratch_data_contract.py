"""Reproducible scratch-data split and loader contracts.

Key rule:
    data_split_seed and experiment_seed are distinct.

The validation split is determined only by data_split_seed.
Training shuffle order is determined only by experiment_seed.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


FIXED_SPLIT_SEED = 42
DEFAULT_VAL_RATIO = 0.15


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray


def make_split_indices(
    n_examples: int,
    *,
    split_seed: int = FIXED_SPLIT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    train_fraction: float = 1.0,
) -> SplitIndices:
    """Reproduce the scratch split policy deterministically."""
    if n_examples < 2:
        raise ValueError("n_examples must be >= 2")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must lie strictly between 0 and 1")

    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must lie in (0, 1]")

    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(n_examples)

    n_val = int(val_ratio * n_examples)

    if n_val < 1:
        raise ValueError("validation split would be empty")

    val_indices = indices[:n_val].copy()
    train_indices = indices[n_val:].copy()

    if train_fraction < 1.0:
        frac_rng = np.random.RandomState(split_seed + 1)

        n_keep = max(
            1,
            int(round(train_fraction * len(train_indices))),
        )

        train_indices = np.asarray(
            frac_rng.choice(
                train_indices,
                size=n_keep,
                replace=False,
            ),
            dtype=np.int64,
        )

    return SplitIndices(
        train=np.asarray(train_indices, dtype=np.int64),
        validation=np.asarray(val_indices, dtype=np.int64),
    )


def indices_sha256(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    arr = np.ascontiguousarray(arr)

    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())

    return h.hexdigest()


def split_manifest(
    split: SplitIndices,
    *,
    split_seed: int,
    val_ratio: float,
    train_fraction: float,
) -> Dict[str, object]:
    return {
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "train_fraction": float(train_fraction),
        "train_count": int(len(split.train)),
        "validation_count": int(len(split.validation)),
        "train_indices_sha256": indices_sha256(split.train),
        "validation_indices_sha256": indices_sha256(
            split.validation
        ),
    }


def make_loader_generator(
    experiment_seed: int,
) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(experiment_seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the DataLoader worker seed."""
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    experiment_seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    generator = make_loader_generator(
        experiment_seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=seed_worker,
    )

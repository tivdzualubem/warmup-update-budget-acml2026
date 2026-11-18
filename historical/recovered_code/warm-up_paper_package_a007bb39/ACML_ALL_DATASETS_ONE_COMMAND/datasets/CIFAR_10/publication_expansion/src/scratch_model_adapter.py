"""Thin adapter around the original ACML scratch Transformer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SRC = REPO_ROOT / "src"

if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from transformer_project.models.transformer import (  # noqa: E402
    TransformerClassifier,
    create_model_configs,
)


SUPPORTED_MODELS = {"medium-4", "large-6"}


def get_scratch_model_config(name: str):
    configs = {
        config["name"]: dict(config)
        for config in create_model_configs()
    }

    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"unsupported publication scratch model: {name}"
        )

    return configs[name]


def build_scratch_model(
    *,
    model_name: str,
    vocab_size: int,
    num_classes: int,
    max_seq_len: int = 128,
    dropout: float = 0.1,
):
    cfg = get_scratch_model_config(model_name)

    return TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=max_seq_len,
        dropout=dropout,
        pad_idx=0,
    )


def build_scratch_adam(
    model,
    *,
    initial_lr: float,
):
    return torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.0,
    )

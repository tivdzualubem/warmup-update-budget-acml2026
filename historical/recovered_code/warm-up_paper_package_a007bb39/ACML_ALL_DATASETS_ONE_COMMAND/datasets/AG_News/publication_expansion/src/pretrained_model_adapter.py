"""Shared pretrained model, processor, and AdamW construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "pretrained_sources_v1.json"
)

MODEL_KEYS = {
    "distilbert": "distilbert_base_uncased",
    "deit_tiny": "deit_tiny_patch16_224",
}


def load_sources() -> Dict:
    data = json.loads(CONFIG_PATH.read_text())

    if data.get("status") != "frozen":
        raise RuntimeError(
            "pretrained source manifest is not frozen"
        )

    return data


def model_source(model_family: str) -> Dict[str, str]:
    if model_family not in MODEL_KEYS:
        raise ValueError(
            f"unsupported model family: {model_family}"
        )

    sources = load_sources()

    return sources["models"][
        MODEL_KEYS[model_family]
    ]


def load_distilbert_tokenizer():
    src = model_source("distilbert")

    return AutoTokenizer.from_pretrained(
        src["repository"],
        revision=src["revision"],
        use_fast=True,
    )


def load_deit_processor():
    src = model_source("deit_tiny")

    return AutoImageProcessor.from_pretrained(
        src["repository"],
        revision=src["revision"],
    )


def build_sequence_classifier(
    *,
    num_labels: int,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
):
    src = model_source("distilbert")

    kwargs = {
        "revision": src["revision"],
        "num_labels": num_labels,
    }

    if id2label is not None:
        kwargs["id2label"] = id2label

    if label2id is not None:
        kwargs["label2id"] = label2id

    return AutoModelForSequenceClassification.from_pretrained(
        src["repository"],
        **kwargs,
    )


def build_token_classifier(
    *,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
):
    src = model_source("distilbert")

    return AutoModelForTokenClassification.from_pretrained(
        src["repository"],
        revision=src["revision"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def build_image_classifier(
    *,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
):
    src = model_source("deit_tiny")

    return AutoModelForImageClassification.from_pretrained(
        src["repository"],
        revision=src["revision"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


_NORM_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
)


def _no_decay_parameter_ids(model: nn.Module):
    ids = set()

    for module in model.modules():
        if isinstance(module, _NORM_TYPES):
            for parameter in module.parameters(
                recurse=False
            ):
                ids.add(id(parameter))

    for name, parameter in model.named_parameters():
        if name.endswith(".bias"):
            ids.add(id(parameter))

    return ids


def adamw_parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float = 0.01,
):
    no_decay_ids = _no_decay_parameter_ids(
        model
    )

    decay = []
    no_decay = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if id(parameter) in no_decay_ids:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    if not decay:
        raise RuntimeError(
            "AdamW decay group is empty"
        )

    if not no_decay:
        raise RuntimeError(
            "AdamW no-decay group is empty"
        )

    return [
        {
            "params": decay,
            "weight_decay": float(
                weight_decay
            ),
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]


def build_pretrained_adamw(
    model: nn.Module,
    *,
    initial_lr: float,
    weight_decay: float = 0.01,
):
    return torch.optim.AdamW(
        adamw_parameter_groups(
            model,
            weight_decay=weight_decay,
        ),
        lr=float(initial_lr),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

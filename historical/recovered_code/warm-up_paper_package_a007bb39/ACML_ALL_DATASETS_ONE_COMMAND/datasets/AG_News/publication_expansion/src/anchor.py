"""Deterministic matched-branch anchor utilities."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import struct
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn


def _hash_update(h, obj: Any) -> None:
    if obj is None:
        h.update(b"N")
        return

    if isinstance(obj, bool):
        h.update(b"B1" if obj else b"B0")
        return

    if isinstance(obj, int):
        h.update(b"I")
        h.update(str(obj).encode())
        return

    if isinstance(obj, float):
        h.update(b"F")
        h.update(struct.pack("!d", obj))
        return

    if isinstance(obj, str):
        h.update(b"S")
        data = obj.encode("utf-8")
        h.update(str(len(data)).encode())
        h.update(b":")
        h.update(data)
        return

    if isinstance(obj, bytes):
        h.update(b"Y")
        h.update(str(len(obj)).encode())
        h.update(b":")
        h.update(obj)
        return

    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        h.update(b"T")
        h.update(str(t.dtype).encode())
        h.update(json.dumps(list(t.shape)).encode())

        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(raw)
        return

    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(b"A")
        h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
        return

    if isinstance(obj, dict):
        h.update(b"D")
        for key in sorted(obj.keys(), key=repr):
            _hash_update(h, key)
            _hash_update(h, obj[key])
        return

    if isinstance(obj, (list, tuple)):
        h.update(b"L" if isinstance(obj, list) else b"U")
        for item in obj:
            _hash_update(h, item)
        return

    raise TypeError(
        f"unsupported object type for deterministic hashing: {type(obj)}"
    )


def object_sha256(obj: Any) -> str:
    h = hashlib.sha256()
    _hash_update(h, obj)
    return h.hexdigest()


def model_fingerprint(model: nn.Module) -> str:
    return object_sha256(model.state_dict())


def optimizer_fingerprint(
    optimizer: torch.optim.Optimizer,
) -> str:
    return object_sha256(optimizer.state_dict())


def optimizer_non_lr_fingerprint(
    optimizer: torch.optim.Optimizer,
) -> str:
    state = copy.deepcopy(optimizer.state_dict())

    for group in state["param_groups"]:
        group.pop("lr", None)
        group.pop("initial_lr", None)

    return object_sha256(state)


def gradient_fingerprint(model: nn.Module) -> str:
    grads = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    return object_sha256(grads)


def batch_fingerprint(batch: Any) -> str:
    return object_sha256(batch)


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = [
            x.clone()
            for x in torch.cuda.get_rng_state_all()
        ]

    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if (
        torch.cuda.is_available()
        and state["torch_cuda"] is not None
    ):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_anchor(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "rng_state": capture_rng_state(),
        "batch": copy.deepcopy(batch),
        "metadata": copy.deepcopy(metadata or {}),
    }


def restore_anchor(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    anchor: Dict[str, Any],
) -> None:
    model.load_state_dict(
        copy.deepcopy(anchor["model_state"])
    )

    optimizer.load_state_dict(
        copy.deepcopy(anchor["optimizer_state"])
    )

    restore_rng_state(
        copy.deepcopy(anchor["rng_state"])
    )


def anchor_fingerprints(
    anchor: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    return {
        "model_state":
            object_sha256(anchor["model_state"]),
        "optimizer_state":
            object_sha256(anchor["optimizer_state"]),
        "rng_state":
            object_sha256(anchor["rng_state"]),
        "batch":
            None if anchor["batch"] is None
            else object_sha256(anchor["batch"]),
        "metadata":
            object_sha256(anchor["metadata"]),
    }

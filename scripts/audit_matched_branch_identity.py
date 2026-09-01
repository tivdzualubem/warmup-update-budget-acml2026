#!/usr/bin/env python3
"""Audit first-step identity across paired scheduler conditions.

For each dataset/model/seed combination, this diagnostic independently
recreates each scheduler condition from the same seed and compares:

- initialized trainable parameters
- first minibatch
- RNG state immediately before the first forward pass
- first logits
- first loss
- clipped first gradient

The optimizer/scheduler action is intentionally not executed. The purpose is
to verify that paired conditions are identical up to the point where their
learning-rate execution semantics diverge.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


CONDITIONS = (
    "stable_with_guard",
    "fixed_order_with_guard",
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "single_bad_first_step_then_stable",
)

MODELS = ("medium-4", "large-6")
SEEDS = (42, 123, 456)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("ag_news", "dbpedia_14"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_tensor(tensor):
    x = tensor.detach().cpu().contiguous()
    return sha256_bytes(x.numpy().tobytes())


def hash_named_tensors(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        x = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(x.shape)).encode("utf-8"))
        digest.update(str(x.dtype).encode("utf-8"))
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def hash_model(model):
    return hash_named_tensors(
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def hash_gradients(model):
    tensors = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            raise RuntimeError(f"Missing gradient for {name}")
        tensors.append((name, param.grad))
    return hash_named_tensors(tensors)


def hash_batch(batch):
    digest = hashlib.sha256()
    for key in sorted(batch):
        tensor = batch[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def hash_rng_state(device):
    digest = hashlib.sha256()

    digest.update(repr(random.getstate()).encode("utf-8"))
    digest.update(
        np.random.get_state()[1].tobytes()
    )
    digest.update(torch.get_rng_state().cpu().numpy().tobytes())

    if device.type == "cuda":
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())

    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    repo_src = args.repo.resolve() / "src" / "transformer_project"
    if not repo_src.is_dir():
        raise FileNotFoundError(repo_src)

    sys.path.insert(0, str(repo_src))

    from data.data_loader_scratch import (
        load_text_dataset,
        get_dataloaders,
    )
    from models.transformer import (
        TransformerClassifier,
        create_model_configs,
    )

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    results = []

    for seed in SEEDS:
        print(f"\nLoading {args.dataset} split for seed {seed} ...")

        train_dataset, val_dataset, test_dataset, vocab, num_classes = (
            load_text_dataset(
                args.dataset,
                max_length=128,
                split_seed=seed,
                train_fraction=1.0,
            )
        )

        for model_name in MODELS:
            cfg = next(
                config
                for config in create_model_configs()
                if config["name"] == model_name
            )

            print(f"  Auditing {model_name}")

            for condition in CONDITIONS:
                set_seed(seed)

                train_loader, _, _ = get_dataloaders(
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    batch_size=32,
                )

                model = TransformerClassifier(
                    vocab_size=len(vocab),
                    num_classes=num_classes,
                    d_model=cfg["d_model"],
                    n_layers=cfg["n_layers"],
                    n_heads=cfg["n_heads"],
                    d_ff=cfg["d_ff"],
                    max_seq_len=128,
                ).to(device)

                model.train()

                model_hash = hash_model(model)

                batch = next(iter(train_loader))
                batch_hash = hash_batch(batch)

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                rng_before_forward = hash_rng_state(device)

                model.zero_grad(set_to_none=True)

                logits = model(input_ids)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                results.append(
                    {
                        "dataset": args.dataset,
                        "model": model_name,
                        "seed": seed,
                        "condition": condition,
                        "model_hash": model_hash,
                        "batch_hash": batch_hash,
                        "rng_before_forward_hash": rng_before_forward,
                        "logits_hash": hash_tensor(logits),
                        "loss": float(loss.item()),
                        "gradient_hash": hash_gradients(model),
                    }
                )

                print(
                    f"    {condition:<34} "
                    f"loss={loss.item():.8f}"
                )

    checks = []

    fields = (
        "model_hash",
        "batch_hash",
        "rng_before_forward_hash",
        "logits_hash",
        "loss",
        "gradient_hash",
    )

    for seed in SEEDS:
        for model_name in MODELS:
            subset = [
                row
                for row in results
                if row["seed"] == seed
                and row["model"] == model_name
            ]

            check = {
                "dataset": args.dataset,
                "model": model_name,
                "seed": seed,
            }

            for field in fields:
                values = [row[field] for row in subset]

                if field == "loss":
                    identical = max(values) == min(values)
                else:
                    identical = len(set(values)) == 1

                check[f"{field}_identical"] = identical

            check["all_identity_checks_pass"] = all(
                check[f"{field}_identical"]
                for field in fields
            )

            checks.append(check)

    payload = {
        "dataset": args.dataset,
        "device": str(device),
        "conditions": list(CONDITIONS),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "runs": results,
        "identity_checks": checks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n" + "=" * 88)
    print("MATCHED-BRANCH IDENTITY SUMMARY")
    print("=" * 88)

    for row in checks:
        print(
            f"{row['dataset']:<12} "
            f"{row['model']:<9} "
            f"seed={row['seed']:<3} | "
            f"all checks pass={row['all_identity_checks_pass']}"
        )

    passed = sum(
        row["all_identity_checks_pass"] for row in checks
    )

    print(f"\nPassed: {passed}/{len(checks)} model/seed comparisons")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Independent publication scratch-experiment job runner."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from publication_expansion.src.anchor import (
    batch_fingerprint,
    capture_rng_state,
    gradient_fingerprint,
    model_fingerprint,
    object_sha256,
    optimizer_non_lr_fingerprint,
    restore_rng_state,
)
from publication_expansion.src.scheduler_semantics import (
    legacy_transformer_lr,
    stable_lr,
)
from publication_expansion.src.scratch_data_contract import (
    make_dataloader,
)
from publication_expansion.src.scratch_dataset_adapter import (
    load_scratch_hf_dataset,
)
from publication_expansion.src.scratch_model_adapter import (
    build_scratch_adam,
    build_scratch_model,
)
from publication_expansion.src.transition_executor import (
    transactional_step,
)


EXPERIMENT_SEEDS = {42, 123, 456, 789, 2026}
DATASETS = {"ag_news", "dbpedia_14"}
MODELS = {"medium-4", "large-6"}

CONDITIONS = {
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "stable_with_guard",
    "fixed_order_with_guard",
    "single_bad_first_step_then_stable",
}

GUARDED_CONDITIONS = {
    "buggy_failfast_guard",
    "buggy_rescue_guard",
}

FULL_TRAINING_CONDITIONS = {
    "buggy_no_guard",
    "buggy_rescue_guard",
    "stable_with_guard",
    "fixed_order_with_guard",
    "single_bad_first_step_then_stable",
}

SPLIT_SEED = 42
VAL_RATIO = 0.15
TRAIN_FRACTION = 1.0
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 30
PATIENCE = 5

GRAD_CLIP = 1.0
GUARD_WINDOW = 100
TRACE_WINDOW = 300

BASE_LR = 1e-4
LEGACY_INITIAL_LR = 1.0
WARMUP_STEPS = 8000
TOTAL_STEPS = 50000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )

    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_revisions() -> Dict[str, str]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "scratch_dataset_revisions_v1.json"
    )

    data = json.loads(path.read_text())

    return {
        name: cfg["revision"]
        for name, cfg in data["datasets"].items()
    }


def initial_lr_for_condition(
    condition: str,
) -> float:
    if condition == "stable_with_guard":
        return BASE_LR

    return LEGACY_INITIAL_LR


def step_semantics(
    *,
    condition: str,
    step: int,
    d_model: int,
    threshold: Optional[float],
):
    """Return executor semantics for one 1-based transition."""

    if step < 1:
        raise ValueError("step must be one-based")

    legacy = legacy_transformer_lr(
        step,
        d_model=d_model,
        warmup_steps=WARMUP_STEPS,
    )

    stable = stable_lr(
        step,
        base_lr=BASE_LR,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS,
    )

    if condition == "buggy_no_guard":
        return {
            "trace_intended_lr": legacy,
            "executor_intended_lr": legacy,
            "mode": "late_no_guard",
            "threshold": None,
            "action": "schedule_after_step",
        }

    if condition == "buggy_failfast_guard":
        if step <= GUARD_WINDOW:
            return {
                "trace_intended_lr": legacy,
                "executor_intended_lr": legacy,
                "mode": "failfast",
                "threshold": threshold,
                "action":
                    "schedule_after_step_guard_monitor",
            }

        return {
            "trace_intended_lr": legacy,
            "executor_intended_lr": legacy,
            "mode": "late_no_guard",
            "threshold": None,
            "action": "schedule_after_step",
        }

    if condition == "buggy_rescue_guard":
        if step <= GUARD_WINDOW:
            return {
                "trace_intended_lr": legacy,
                "executor_intended_lr": legacy,
                "mode": "rescue",
                "threshold": threshold,
                "action":
                    "schedule_after_step_guard_monitor",
            }

        return {
            "trace_intended_lr": legacy,
            "executor_intended_lr": legacy,
            "mode": "late_no_guard",
            "threshold": None,
            "action": "schedule_after_step",
        }

    if condition == "fixed_order_with_guard":
        return {
            "trace_intended_lr": legacy,
            "executor_intended_lr": legacy,
            "mode": "safe",
            "threshold": None,
            "action": "schedule_before_step",
        }

    if condition == "stable_with_guard":
        return {
            "trace_intended_lr": stable,
            "executor_intended_lr": stable,
            "mode": "safe",
            "threshold": None,
            "action": "schedule_before_step",
        }

    if condition == "single_bad_first_step_then_stable":
        if step == 1:
            # The intended stable target is S(1), but the actual first
            # transition consumes and retains the optimizer entry LR=1.
            return {
                "trace_intended_lr": stable,
                "executor_intended_lr":
                    LEGACY_INITIAL_LR,
                "mode": "late_no_guard",
                "threshold": None,
                "action":
                    "single_bad_first_step",
            }

        return {
            "trace_intended_lr": stable,
            "executor_intended_lr": stable,
            "mode": "safe",
            "threshold": None,
            "action":
                "stable_schedule_before_step",
        }

    raise ValueError(
        f"unknown condition: {condition}"
    )


def macro_f1(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
) -> float:
    scores = []

    for c in range(num_classes):
        tp = np.sum(
            (labels == c) & (preds == c)
        )
        fp = np.sum(
            (labels != c) & (preds == c)
        )
        fn = np.sum(
            (labels == c) & (preds != c)
        )

        denom = 2 * tp + fp + fn

        scores.append(
            0.0 if denom == 0
            else (2.0 * tp) / denom
        )

    return float(np.mean(scores))


def evaluate(
    model,
    loader,
    device,
    num_classes: int,
):
    model.eval()

    losses = []
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            logits = model(input_ids)

            loss = F.cross_entropy(
                logits,
                labels,
            )

            losses.append(float(loss.item()))
            probs_all.append(
                torch.softmax(
                    logits,
                    dim=1,
                ).cpu().numpy()
            )
            labels_all.append(
                labels.cpu().numpy()
            )

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = probs.argmax(axis=1)

    acc = float(
        np.mean(preds == labels)
    )

    f1 = macro_f1(
        labels,
        preds,
        num_classes,
    )

    confidences = probs.max(axis=1)
    correct = preds == labels

    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)

    for i in range(15):
        if i == 14:
            mask = (
                (confidences >= edges[i])
                & (confidences <= edges[i + 1])
            )
        else:
            mask = (
                (confidences >= edges[i])
                & (confidences < edges[i + 1])
            )

        if np.any(mask):
            ece += (
                np.mean(mask)
                * abs(
                    np.mean(correct[mask])
                    - np.mean(confidences[mask])
                )
            )

    clipped = np.clip(
        probs,
        1e-12,
        1.0,
    )

    nll = float(
        -np.mean(
            np.log(
                clipped[
                    np.arange(len(labels)),
                    labels,
                ]
            )
        )
    )

    one_hot = np.zeros(
        (len(labels), num_classes),
        dtype=np.float64,
    )

    one_hot[
        np.arange(len(labels)),
        labels,
    ] = 1.0

    brier = float(
        np.mean(
            np.sum(
                (probs - one_hot) ** 2,
                axis=1,
            )
        )
    )

    return {
        "loss": float(np.mean(losses)),
        "accuracy": acc,
        "macro_f1": f1,
        "ece": float(ece),
        "nll": nll,
        "brier": brier,
    }


def resolve_scratch_run_controls(
    *,
    pilot_mode: bool,
    train_fraction: float,
    epochs: int,
    max_optimizer_steps: int | None,
):
    """Keep frozen publication defaults immutable outside pilot mode."""

    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(
            "train_fraction must satisfy 0 < fraction <= 1"
        )

    if epochs < 1:
        raise ValueError(
            "epochs must be >= 1"
        )

    if (
        max_optimizer_steps is not None
        and max_optimizer_steps < 1
    ):
        raise ValueError(
            "max_optimizer_steps must be >= 1"
        )

    if not pilot_mode:
        if train_fraction != TRAIN_FRACTION:
            raise ValueError(
                "reduced train_fraction requires --pilot-mode"
            )

        if epochs != NUM_EPOCHS:
            raise ValueError(
                "non-default epochs require --pilot-mode"
            )

        if max_optimizer_steps is not None:
            raise ValueError(
                "max optimizer steps requires --pilot-mode"
            )

    return {
        "pilot_mode": bool(pilot_mode),
        "train_fraction":
            float(train_fraction),
        "epochs": int(epochs),
        "max_optimizer_steps":
            max_optimizer_steps,
    }


def run_job(args) -> int:
    if args.dataset not in DATASETS:
        raise ValueError("unsupported dataset")

    if args.model not in MODELS:
        raise ValueError("unsupported model")

    if args.seed not in EXPERIMENT_SEEDS:
        raise ValueError(
            "seed is not in the frozen seed set"
        )

    if args.condition not in CONDITIONS:
        raise ValueError("unsupported condition")

    if (
        args.condition in GUARDED_CONDITIONS
        and args.threshold is None
    ):
        raise ValueError(
            "guarded condition requires --threshold"
        )

    controls = resolve_scratch_run_controls(
        pilot_mode=args.pilot_mode,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        max_optimizer_steps=args.max_optimizer_steps,
    )

    run_epochs = controls["epochs"]

    device = torch.device(
        args.device
        if args.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    job_id = (
        f"{args.dataset}__{args.model}"
        f"__seed{args.seed}"
        f"__{args.condition}"
    )

    result_dir = (
        Path(args.output_root)
        / args.dataset
        / args.model
        / f"seed{args.seed}"
        / args.condition
    )

    result_file = result_dir / "result.json"

    if result_file.exists() and not args.force:
        result = json.loads(
            result_file.read_text()
        )

        if result.get("job_complete"):
            print(
                f"SKIP completed job: {job_id}"
            )
            return 0

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    revisions = load_revisions()

    job_manifest = {
        "job_id": job_id,
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "condition": args.condition,
        "threshold": args.threshold,
        "split_seed": SPLIT_SEED,
        "validation_ratio": VAL_RATIO,
        "train_fraction":
            controls["train_fraction"],
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs": run_epochs,
        "pilot_mode":
            controls["pilot_mode"],
        "scientific_results":
            not controls["pilot_mode"],
        "max_optimizer_steps":
            controls["max_optimizer_steps"],
        "patience": PATIENCE,
        "gradient_clip": GRAD_CLIP,
        "guard_window_steps": GUARD_WINDOW,
        "trace_window_steps": TRACE_WINDOW,
        "hf_revision":
            revisions[args.dataset],
        "device": str(device),
    }

    atomic_json(
        result_dir / "job_manifest.json",
        job_manifest,
    )

    print("JOB:", job_id)
    print("DEVICE:", device)

    set_seed(args.seed)

    prepared = load_scratch_hf_dataset(
        args.dataset,
        max_length=MAX_LENGTH,
        split_seed=SPLIT_SEED,
        val_ratio=VAL_RATIO,
        train_fraction=controls[
            "train_fraction"
        ],
        revision=revisions[args.dataset],
    )

    pin = device.type == "cuda"

    train_loader = make_dataloader(
        prepared.train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    val_loader = make_dataloader(
        prepared.validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    test_loader = make_dataloader(
        prepared.test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    train_eval_loader = make_dataloader(
        prepared.train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_scratch_model(
        model_name=args.model,
        vocab_size=len(prepared.vocab),
        num_classes=prepared.num_classes,
        max_seq_len=MAX_LENGTH,
    ).to(device)

    optimizer = build_scratch_adam(
        model,
        initial_lr=initial_lr_for_condition(
            args.condition
        ),
    )

    d_model = model.d_model

    checkpoint_file = (
        result_dir / "resume_checkpoint.pt"
    )
    best_state_file = (
        result_dir / "best_state_resume.pt"
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    trace_rows = []
    anchor_evidence = None

    terminal_reason = None

    if (
        checkpoint_file.exists()
        and not args.force
    ):
        print("RESUME:", checkpoint_file)

        ckpt = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model_state"]
        )
        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

        restore_rng_state(
            ckpt["rng_state"]
        )

        if train_loader.generator is not None:
            train_loader.generator.set_state(
                ckpt[
                    "train_loader_generator_state"
                ]
            )

        start_epoch = ckpt["next_epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt[
            "best_val_loss"
        ]
        patience_counter = ckpt[
            "patience_counter"
        ]
        history = ckpt["history"]
        trace_rows = ckpt["trace_rows"]
        anchor_evidence = ckpt[
            "anchor_evidence"
        ]

    for epoch in range(
        start_epoch,
        run_epochs,
    ):
        model.train()

        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            step = global_step + 1

            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if step == 1:
                pre_model_hash = (
                    model_fingerprint(model)
                )

                pre_optimizer_non_lr = (
                    optimizer_non_lr_fingerprint(
                        optimizer
                    )
                )

                batch_hash = (
                    batch_fingerprint(batch)
                )

                rng_hash = object_sha256(
                    capture_rng_state()
                )

            logits = model(input_ids)

            loss = F.cross_entropy(
                logits,
                labels,
            )

            if not torch.isfinite(loss):
                terminal_reason = (
                    "nonfinite_loss"
                )
                break

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )

            if step == 1:
                anchor_evidence = {
                    "model_state":
                        pre_model_hash,
                    "optimizer_non_lr":
                        pre_optimizer_non_lr,
                    "first_batch":
                        batch_hash,
                    "rng_before_forward":
                        rng_hash,
                    "logits":
                        object_sha256(
                            logits.detach()
                        ),
                    "loss":
                        object_sha256(
                            loss.detach()
                        ),
                    "clipped_gradient":
                        gradient_fingerprint(
                            model
                        ),
                }

            semantics = step_semantics(
                condition=args.condition,
                step=step,
                d_model=d_model,
                threshold=args.threshold,
            )

            # Capture the optimizer LR exactly as it exists on
            # transition entry, before scheduler/guard action.
            entry_lr = float(
                optimizer.param_groups[0]["lr"]
            )

            transition = transactional_step(
                model=model,
                optimizer=optimizer,
                intended_lr=semantics[
                    "executor_intended_lr"
                ],
                threshold=semantics[
                    "threshold"
                ],
                mode=semantics["mode"],
            )

            global_step = step

            action = semantics["action"]

            if transition.guard_triggered:
                if transition.replayed:
                    action = "rescue_replay"
                elif transition.stopped:
                    action = "failfast_stop"

            if step <= TRACE_WINDOW:
                trace_rows.append(
                    {
                        "step": int(step),
                        "entry_lr":
                            float(entry_lr),
                        "scheduler_or_guard_action":
                            action,
                        "intended_lr":
                            float(
                                semantics[
                                    "trace_intended_lr"
                                ]
                            ),
                        "candidate_lr_consumed":
                            float(
                                transition
                                .candidate_lr_by_group[0]
                            ),
                        "candidate_update_ratio":
                            float(
                                transition
                                .candidate_update_ratio
                            ),
                        "guard_triggered":
                            bool(
                                transition
                                .guard_triggered
                            ),
                        "committed_lr_consumed":
                            float(
                                transition
                                .committed_lr_by_group[0]
                            ),
                        "committed_update_ratio":
                            float(
                                transition
                                .committed_update_ratio
                            ),
                        "actual_exit_lr":
                            float(
                                transition
                                .actual_exit_lr_by_group[0]
                            ),
                    }
                )

            epoch_losses.append(
                float(loss.item())
            )

            preds = logits.argmax(dim=1)

            epoch_correct += int(
                (preds == labels)
                .sum()
                .item()
            )

            epoch_total += int(
                labels.numel()
            )

            if transition.stopped:
                terminal_reason = (
                    "guard_failfast"
                )
                break

            if (
                controls["max_optimizer_steps"]
                is not None
                and global_step
                >= controls["max_optimizer_steps"]
            ):
                terminal_reason = (
                    "pilot_max_steps"
                )
                break

        if terminal_reason is not None:
            break

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            prepared.num_classes,
        )

        train_epoch_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss":
                float(
                    np.mean(epoch_losses)
                ),
            "train_accuracy":
                float(
                    epoch_correct
                    / max(1, epoch_total)
                ),
            "validation":
                val_metrics,
        }

        history.append(
            train_epoch_metrics
        )

        print(
            f"epoch={epoch + 1} "
            f"step={global_step} "
            f"train_loss="
            f"{train_epoch_metrics['train_loss']:.4f} "
            f"val_loss="
            f"{val_metrics['loss']:.4f}"
        )

        if (
            val_metrics["loss"]
            < best_val_loss
        ):
            best_val_loss = (
                val_metrics["loss"]
            )
            patience_counter = 0

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            atomic_torch_save(
                best_state_file,
                best_state,
            )
        else:
            patience_counter += 1

        checkpoint = {
            "model_state":
                model.state_dict(),
            "optimizer_state":
                optimizer.state_dict(),
            "rng_state":
                capture_rng_state(),
            "train_loader_generator_state":
                train_loader.generator.get_state(),
            "next_epoch":
                epoch + 1,
            "global_step":
                global_step,
            "best_val_loss":
                best_val_loss,
            "patience_counter":
                patience_counter,
            "history":
                history,
            "trace_rows":
                trace_rows,
            "anchor_evidence":
                anchor_evidence,
        }

        atomic_torch_save(
            checkpoint_file,
            checkpoint,
        )

        if patience_counter >= PATIENCE:
            terminal_reason = (
                "early_stopping"
            )
            break

    if terminal_reason is None:
        terminal_reason = (
            "max_epochs"
        )

    is_full_condition = (
        args.condition
        in FULL_TRAINING_CONDITIONS
    )

    train_metrics = None
    test_metrics = None

    if (
        is_full_condition
        and best_state_file.exists()
    ):
        best_state = torch.load(
            best_state_file,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            best_state
        )

        train_metrics = evaluate(
            model,
            train_eval_loader,
            device,
            prepared.num_classes,
        )

        test_metrics = evaluate(
            model,
            test_loader,
            device,
            prepared.num_classes,
        )

    result = {
        "job_complete": True,
        "job_id": job_id,
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "condition": args.condition,
        "threshold": args.threshold,
        "pilot_mode":
            controls["pilot_mode"],
        "scientific_results":
            not controls["pilot_mode"],
        "train_fraction":
            controls["train_fraction"],
        "configured_epochs":
            run_epochs,
        "max_optimizer_steps":
            controls["max_optimizer_steps"],
        "terminal_reason":
            terminal_reason,
        "global_steps":
            global_step,
        "best_validation_loss":
            (
                None
                if math.isinf(best_val_loss)
                else float(best_val_loss)
            ),
        "train_metrics":
            train_metrics,
        "test_metrics":
            test_metrics,
        "generalization_gap_accuracy":
            (
                None
                if (
                    train_metrics is None
                    or test_metrics is None
                )
                else float(
                    train_metrics["accuracy"]
                    - test_metrics["accuracy"]
                )
            ),
        "anchor_evidence":
            anchor_evidence,
        "dataset_manifest":
            prepared.manifest,
        "trace_file":
            "trace.json",
        "history_file":
            "history.json",
    }

    atomic_json(
        result_dir / "trace.json",
        trace_rows,
    )

    atomic_json(
        result_dir / "history.json",
        history,
    )

    atomic_json(
        result_file,
        result,
    )

    atomic_json(
        result_dir / "status.json",
        {
            "job_id": job_id,
            "status": "complete",
            "terminal_reason":
                terminal_reason,
        },
    )

    checkpoint_file.unlink(
        missing_ok=True
    )
    best_state_file.unlink(
        missing_ok=True
    )

    print(
        "COMPLETE:",
        job_id,
        terminal_reason,
    )

    return 0


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=sorted(CONDITIONS),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        default=(
            "publication_expansion/results/"
            "scratch"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--pilot-mode",
        action="store_true",
        help=(
            "Enable explicitly non-scientific reduced-run controls."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=TRAIN_FRACTION,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
    )

    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser


def main():
    args = build_parser().parse_args()
    raise SystemExit(run_job(args))


if __name__ == "__main__":
    main()

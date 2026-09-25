"""Independent pretrained publication experiment runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from seqeval.metrics import f1_score as seqeval_f1

from publication_expansion.src.anchor import (
    batch_fingerprint,
    capture_rng_state,
    gradient_fingerprint,
    model_fingerprint,
    object_sha256,
    optimizer_non_lr_fingerprint,
    restore_rng_state,
)
from publication_expansion.src.pretrained_data_adapter import (
    FIXED_SPLIT_SEED,
    load_sources,
    prepare_cifar10_dataset,
    prepare_sequence_dataset,
    prepare_wnut_dataset,
)
from publication_expansion.src.pretrained_model_adapter import (
    build_image_classifier,
    build_pretrained_adamw,
    build_sequence_classifier,
    build_token_classifier,
    load_deit_processor,
    load_distilbert_tokenizer,
)
from publication_expansion.src.scratch_data_contract import (
    make_dataloader,
)
from publication_expansion.src.transition_executor import (
    transactional_step,
)


SEEDS = {42, 123, 456, 789, 2026}

DATASETS = {
    "ag_news",
    "dbpedia_14",
    "wnut_17",
    "cifar10",
}

CONDITIONS = {
    "safe_correct_order",
    "late_schedule_no_guard",
    "single_bad_then_correct",
    "update_guard_failfast",
    "update_guard_rescue",
    "executed_lr_assertion",
}

FULL_TRAINING_CONDITIONS = {
    "safe_correct_order",
    "late_schedule_no_guard",
    "single_bad_then_correct",
    "update_guard_rescue",
}

GUARDED_CONDITIONS = {
    "update_guard_failfast",
    "update_guard_rescue",
}

NUM_EPOCHS = 3
GRAD_ACCUMULATION = 1
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
TRACE_WINDOW = 300
GUARD_WINDOW = 100

TASK_CONFIG = {
    "ag_news": {
        "task": "sequence",
        "peak_lr": 2e-5,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "primary_metric": "macro_f1",
    },
    "dbpedia_14": {
        "task": "sequence",
        "peak_lr": 2e-5,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "primary_metric": "macro_f1",
    },
    "wnut_17": {
        "task": "token",
        "peak_lr": 2e-5,
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "primary_metric": "entity_f1",
    },
    "cifar10": {
        "task": "image",
        "peak_lr": 5e-5,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "primary_metric": "accuracy",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def atomic_torch_save(
    path: Path,
    obj,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(obj, tmp)
    os.replace(tmp, path)


def linear_warmup_decay_lr(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Frozen one-based linear warmup + linear decay."""

    if step < 1:
        raise ValueError(
            "step must be one-based"
        )

    if warmup_steps < 1:
        raise ValueError(
            "warmup_steps must be >= 1"
        )

    if total_steps < warmup_steps:
        raise ValueError(
            "total_steps must be >= warmup_steps"
        )

    if step <= warmup_steps:
        return float(
            peak_lr
            * step
            / warmup_steps
        )

    if total_steps == warmup_steps:
        return 0.0

    progress = (
        step - warmup_steps
    ) / (
        total_steps - warmup_steps
    )

    return float(
        peak_lr
        * max(
            0.0,
            1.0 - progress,
        )
    )


def transition_spec(
    *,
    condition: str,
    step: int,
    intended_lr: float,
    threshold: Optional[float],
):
    if condition == "safe_correct_order":
        return {
            "mode": "safe",
            "threshold": None,
            "action":
                "schedule_before_step",
        }

    if condition == "late_schedule_no_guard":
        return {
            "mode": "late_no_guard",
            "threshold": None,
            "action":
                "schedule_after_step",
        }

    if condition == "single_bad_then_correct":
        if step == 1:
            return {
                "mode": "late_no_guard",
                "threshold": None,
                "action":
                    "single_bad_first_step",
            }

        return {
            "mode": "safe",
            "threshold": None,
            "action":
                "schedule_before_step_after_bad_first",
        }

    if condition == "update_guard_failfast":
        if step <= GUARD_WINDOW:
            return {
                "mode": "failfast",
                "threshold": threshold,
                "action":
                    "late_schedule_guard_monitor",
            }

        return {
            "mode": "late_no_guard",
            "threshold": None,
            "action":
                "schedule_after_step",
        }

    if condition == "update_guard_rescue":
        if step <= GUARD_WINDOW:
            return {
                "mode": "rescue",
                "threshold": threshold,
                "action":
                    "late_schedule_guard_monitor",
            }

        return {
            "mode": "late_no_guard",
            "threshold": None,
            "action":
                "schedule_after_step",
        }

    if condition == "executed_lr_assertion":
        return {
            "mode": "assertion",
            "threshold": None,
            "action":
                "assert_executed_lr",
        }

    raise ValueError(
        f"unknown condition: {condition}"
    )


def uniform_group_lr(optimizer) -> float:
    values = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    first = values[0]

    if not all(
        math.isclose(
            value,
            first,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for value in values
    ):
        raise RuntimeError(
            "optimizer parameter groups have "
            "different learning rates"
        )

    return first


def macro_f1(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
) -> float:
    values = []

    for c in range(num_classes):
        tp = np.sum(
            (labels == c)
            & (preds == c)
        )
        fp = np.sum(
            (labels != c)
            & (preds == c)
        )
        fn = np.sum(
            (labels == c)
            & (preds != c)
        )

        denom = 2 * tp + fp + fn

        values.append(
            0.0
            if denom == 0
            else float(
                2.0 * tp / denom
            )
        )

    return float(np.mean(values))


def prepare_task(dataset: str):
    cfg = TASK_CONFIG[dataset]

    if cfg["task"] == "sequence":
        tokenizer = (
            load_distilbert_tokenizer()
        )

        prepared = (
            prepare_sequence_dataset(
                dataset_name=dataset,
                tokenizer=tokenizer,
            )
        )

        model = build_sequence_classifier(
            num_labels=prepared[
                "num_labels"
            ],
        )

    elif cfg["task"] == "token":
        tokenizer = (
            load_distilbert_tokenizer()
        )

        prepared = prepare_wnut_dataset(
            tokenizer=tokenizer
        )

        model = build_token_classifier(
            num_labels=prepared[
                "num_labels"
            ],
            id2label=prepared[
                "id2label"
            ],
            label2id=prepared[
                "label2id"
            ],
        )

    elif cfg["task"] == "image":
        processor = load_deit_processor()

        prepared = (
            prepare_cifar10_dataset(
                processor=processor
            )
        )

        model = build_image_classifier(
            num_labels=prepared[
                "num_labels"
            ],
            id2label=prepared[
                "id2label"
            ],
            label2id=prepared[
                "label2id"
            ],
        )

    else:
        raise ValueError(
            "unknown task type"
        )

    return prepared, model


def forward_batch(
    *,
    task: str,
    model,
    batch,
    device,
):
    labels = batch["labels"].to(
        device,
        non_blocking=True,
    )

    if task == "sequence":
        output = model(
            input_ids=batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            ),
            attention_mask=batch[
                "attention_mask"
            ].to(
                device,
                non_blocking=True,
            ),
        )

        logits = output.logits

        loss = F.cross_entropy(
            logits,
            labels,
        )

    elif task == "token":
        output = model(
            input_ids=batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            ),
            attention_mask=batch[
                "attention_mask"
            ].to(
                device,
                non_blocking=True,
            ),
        )

        logits = output.logits

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.shape[-1],
            ),
            labels.reshape(-1),
            ignore_index=-100,
        )

    elif task == "image":
        output = model(
            pixel_values=batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True,
            )
        )

        logits = output.logits

        loss = F.cross_entropy(
            logits,
            labels,
        )

    else:
        raise ValueError(
            f"unsupported task: {task}"
        )

    return logits, labels, loss


def evaluate(
    *,
    task: str,
    model,
    loader,
    device,
    num_labels: int,
    id2label=None,
):
    model.eval()

    losses = []

    if task in {
        "sequence",
        "image",
    }:
        all_labels = []
        all_preds = []

    else:
        true_sequences = []
        pred_sequences = []
        active_correct = 0
        active_total = 0

    with torch.no_grad():
        for batch in loader:
            logits, labels, loss = (
                forward_batch(
                    task=task,
                    model=model,
                    batch=batch,
                    device=device,
                )
            )

            losses.append(
                float(loss.item())
            )

            preds = logits.argmax(
                dim=-1
            )

            if task in {
                "sequence",
                "image",
            }:
                all_labels.append(
                    labels.cpu().numpy()
                )
                all_preds.append(
                    preds.cpu().numpy()
                )

            else:
                labels_cpu = (
                    labels.cpu().numpy()
                )

                preds_cpu = (
                    preds.cpu().numpy()
                )

                for y_true, y_pred in zip(
                    labels_cpu,
                    preds_cpu,
                ):
                    true_seq = []
                    pred_seq = []

                    for true_id, pred_id in zip(
                        y_true,
                        y_pred,
                    ):
                        if true_id == -100:
                            continue

                        true_seq.append(
                            id2label[
                                int(true_id)
                            ]
                        )

                        pred_seq.append(
                            id2label[
                                int(pred_id)
                            ]
                        )

                        active_total += 1

                        if pred_id == true_id:
                            active_correct += 1

                    true_sequences.append(
                        true_seq
                    )
                    pred_sequences.append(
                        pred_seq
                    )

    result = {
        "loss": float(
            np.mean(losses)
        ),
    }

    if task in {
        "sequence",
        "image",
    }:
        labels = np.concatenate(
            all_labels
        )
        preds = np.concatenate(
            all_preds
        )

        result["accuracy"] = float(
            np.mean(labels == preds)
        )

        if task == "sequence":
            result["macro_f1"] = (
                macro_f1(
                    labels,
                    preds,
                    num_labels,
                )
            )

    else:
        result["entity_f1"] = float(
            seqeval_f1(
                true_sequences,
                pred_sequences,
            )
        )

        result["token_accuracy"] = float(
            active_correct
            / max(1, active_total)
        )

    return result


def primary_metric_value(
    dataset: str,
    metrics: Dict[str, float],
) -> float:
    key = TASK_CONFIG[
        dataset
    ]["primary_metric"]

    return float(metrics[key])


def run_job(args) -> int:
    if args.dataset not in DATASETS:
        raise ValueError(
            "unsupported dataset"
        )

    if args.seed not in SEEDS:
        raise ValueError(
            "seed is not in frozen seed set"
        )

    if args.condition not in CONDITIONS:
        raise ValueError(
            "unsupported condition"
        )

    if (
        args.condition
        in GUARDED_CONDITIONS
        and args.threshold is None
    ):
        raise ValueError(
            "guarded condition requires "
            "--threshold"
        )

    cfg = TASK_CONFIG[
        args.dataset
    ]

    task = cfg["task"]

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
        f"{args.dataset}"
        f"__seed{args.seed}"
        f"__{args.condition}"
    )

    result_dir = (
        Path(args.output_root)
        / args.dataset
        / f"seed{args.seed}"
        / args.condition
    )

    result_file = (
        result_dir / "result.json"
    )

    if (
        result_file.exists()
        and not args.force
    ):
        existing = json.loads(
            result_file.read_text()
        )

        if existing.get(
            "job_complete"
        ):
            print(
                "SKIP completed job:",
                job_id,
            )
            return 0

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(args.seed)

    prepared, model = prepare_task(
        args.dataset
    )

    model = model.to(device)

    pin = device.type == "cuda"

    train_loader = make_dataloader(
        prepared["train"],
        batch_size=cfg[
            "train_batch_size"
        ],
        shuffle=True,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    val_loader = make_dataloader(
        prepared["validation"],
        batch_size=cfg[
            "eval_batch_size"
        ],
        shuffle=False,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    test_loader = make_dataloader(
        prepared["test"],
        batch_size=cfg[
            "eval_batch_size"
        ],
        shuffle=False,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    total_steps = (
        len(train_loader)
        * NUM_EPOCHS
    )

    warmup_steps = max(
        1,
        math.ceil(
            total_steps
            * WARMUP_RATIO
        ),
    )

    peak_lr = float(
        cfg["peak_lr"]
    )

    # All mismatch conditions begin at the peak LR.
    # Safe correct-order also begins there, but installs
    # P(1) before the first optimizer transition.
    optimizer = build_pretrained_adamw(
        model,
        initial_lr=peak_lr,
        weight_decay=WEIGHT_DECAY,
    )

    sources = load_sources()

    manifest = {
        "job_id": job_id,
        "dataset": args.dataset,
        "task": task,
        "seed": args.seed,
        "condition": args.condition,
        "threshold": args.threshold,
        "split_seed":
            FIXED_SPLIT_SEED,
        "epochs": NUM_EPOCHS,
        "gradient_accumulation":
            GRAD_ACCUMULATION,
        "gradient_clip":
            GRAD_CLIP,
        "weight_decay":
            WEIGHT_DECAY,
        "adamw_betas":
            [0.9, 0.999],
        "adamw_eps": 1e-8,
        "peak_lr": peak_lr,
        "warmup_ratio":
            WARMUP_RATIO,
        "warmup_steps":
            warmup_steps,
        "total_optimizer_steps":
            total_steps,
        "trace_window_steps":
            TRACE_WINDOW,
        "guard_window_steps":
            GUARD_WINDOW,
        "primary_metric":
            cfg["primary_metric"],
        "train_batch_size":
            cfg["train_batch_size"],
        "eval_batch_size":
            cfg["eval_batch_size"],
        "device": str(device),
        "sources":
            sources,
    }

    atomic_json(
        result_dir
        / "job_manifest.json",
        manifest,
    )

    checkpoint_file = (
        result_dir
        / "resume_checkpoint.pt"
    )

    best_state_file = (
        result_dir
        / "best_state.pt"
    )

    final_state_file = (
        result_dir
        / "final_state.pt"
    )

    start_epoch = 0
    global_step = 0
    best_primary = (
        -float("inf")
    )
    best_epoch = None
    history = []
    traces = []
    anchor_evidence = None
    terminal_reason = None
    assertion_record = None

    if (
        checkpoint_file.exists()
        and not args.force
    ):
        print(
            "RESUME:",
            checkpoint_file,
        )

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

        if (
            train_loader.generator
            is not None
        ):
            train_loader.generator.set_state(
                ckpt[
                    "train_loader_generator_state"
                ]
            )

        start_epoch = (
            ckpt["next_epoch"]
        )
        global_step = (
            ckpt["global_step"]
        )
        best_primary = (
            ckpt["best_primary"]
        )
        best_epoch = (
            ckpt["best_epoch"]
        )
        history = ckpt["history"]
        traces = ckpt["traces"]
        anchor_evidence = (
            ckpt[
                "anchor_evidence"
            ]
        )

    print(
        "JOB:",
        job_id,
        "| DEVICE:",
        device,
        "| STEPS:",
        total_steps,
        "| WARMUP:",
        warmup_steps,
    )

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):
        model.train()

        epoch_losses = []

        for batch in train_loader:
            step = global_step + 1

            optimizer.zero_grad(
                set_to_none=True
            )

            entry_lr = uniform_group_lr(
                optimizer
            )

            if step == 1:
                pre_model_hash = (
                    model_fingerprint(
                        model
                    )
                )

                pre_optimizer_hash = (
                    optimizer_non_lr_fingerprint(
                        optimizer
                    )
                )

                first_batch_hash = (
                    batch_fingerprint(
                        batch
                    )
                )

                rng_hash = (
                    object_sha256(
                        capture_rng_state()
                    )
                )

            logits, labels, loss = (
                forward_batch(
                    task=task,
                    model=model,
                    batch=batch,
                    device=device,
                )
            )

            if not torch.isfinite(
                loss
            ):
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
                        pre_optimizer_hash,
                    "first_batch":
                        first_batch_hash,
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

            intended_lr = (
                linear_warmup_decay_lr(
                    step,
                    peak_lr=peak_lr,
                    warmup_steps=(
                        warmup_steps
                    ),
                    total_steps=(
                        total_steps
                    ),
                )
            )

            spec = transition_spec(
                condition=args.condition,
                step=step,
                intended_lr=intended_lr,
                threshold=args.threshold,
            )

            if spec["mode"] == "assertion":
                passed = all(
                    math.isclose(
                        float(
                            group["lr"]
                        ),
                        intended_lr,
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    )
                    for group
                    in optimizer.param_groups
                )

                assertion_record = {
                    "step": int(step),
                    "entry_lr":
                        float(entry_lr),
                    "intended_lr":
                        float(
                            intended_lr
                        ),
                    "passed":
                        bool(passed),
                }

                if not passed:
                    terminal_reason = (
                        "executed_lr_assertion_failed"
                    )
                    break

                raise RuntimeError(
                    "executed_lr_assertion unexpectedly "
                    "passed; this condition is defined "
                    "to audit the late-order mismatch"
                )

            transition = (
                transactional_step(
                    model=model,
                    optimizer=optimizer,
                    intended_lr=(
                        intended_lr
                    ),
                    threshold=spec[
                        "threshold"
                    ],
                    mode=spec["mode"],
                )
            )

            global_step = step

            action = spec["action"]

            if transition.guard_triggered:
                if transition.replayed:
                    action = (
                        "rescue_replay"
                    )
                elif transition.stopped:
                    action = (
                        "failfast_stop"
                    )

            if step <= TRACE_WINDOW:
                traces.append(
                    {
                        "step":
                            int(step),
                        "entry_lr":
                            float(entry_lr),
                        "scheduler_or_guard_action":
                            action,
                        "intended_lr":
                            float(
                                intended_lr
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

            if transition.stopped:
                terminal_reason = (
                    "guard_failfast"
                )
                break

        if terminal_reason is not None:
            break

        val_metrics = evaluate(
            task=task,
            model=model,
            loader=val_loader,
            device=device,
            num_labels=prepared[
                "num_labels"
            ],
            id2label=prepared.get(
                "id2label"
            ),
        )

        primary = (
            primary_metric_value(
                args.dataset,
                val_metrics,
            )
        )

        history.append(
            {
                "epoch":
                    epoch + 1,
                "global_step":
                    global_step,
                "train_loss":
                    float(
                        np.mean(
                            epoch_losses
                        )
                    ),
                "validation":
                    val_metrics,
                "validation_primary":
                    primary,
            }
        )

        print(
            f"epoch={epoch + 1} "
            f"step={global_step} "
            f"train_loss="
            f"{np.mean(epoch_losses):.4f} "
            f"val_primary="
            f"{primary:.4f}"
        )

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch + 1

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

            atomic_torch_save(
                best_state_file,
                best_state,
            )

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
            "best_primary":
                best_primary,
            "best_epoch":
                best_epoch,
            "history":
                history,
            "traces":
                traces,
            "anchor_evidence":
                anchor_evidence,
        }

        atomic_torch_save(
            checkpoint_file,
            checkpoint,
        )

    if terminal_reason is None:
        terminal_reason = (
            "completed_3_epochs"
        )

    test_metrics = None

    if (
        args.condition
        in FULL_TRAINING_CONDITIONS
        and terminal_reason
        == "completed_3_epochs"
    ):
        final_state = {
            key:
                value.detach()
                .cpu()
                .clone()
            for key, value
            in model.state_dict().items()
        }

        atomic_torch_save(
            final_state_file,
            final_state,
        )

        if not best_state_file.exists():
            raise RuntimeError(
                "best validation state "
                "was not created"
            )

        best_state = torch.load(
            best_state_file,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            best_state
        )

        test_metrics = evaluate(
            task=task,
            model=model,
            loader=test_loader,
            device=device,
            num_labels=prepared[
                "num_labels"
            ],
            id2label=prepared.get(
                "id2label"
            ),
        )

    result = {
        "job_complete": True,
        "job_id": job_id,
        "dataset": args.dataset,
        "task": task,
        "seed": args.seed,
        "condition":
            args.condition,
        "threshold":
            args.threshold,
        "terminal_reason":
            terminal_reason,
        "global_steps":
            global_step,
        "best_epoch":
            best_epoch,
        "best_validation_primary":
            (
                None
                if best_epoch is None
                else float(
                    best_primary
                )
            ),
        "primary_metric":
            cfg[
                "primary_metric"
            ],
        "test_metrics":
            test_metrics,
        "anchor_evidence":
            anchor_evidence,
        "assertion_record":
            assertion_record,
        "trace_file":
            "trace.json",
        "history_file":
            "history.json",
        "best_state_file":
            (
                "best_state.pt"
                if best_state_file.exists()
                else None
            ),
        "final_state_file":
            (
                "final_state.pt"
                if final_state_file.exists()
                else None
            ),
    }

    atomic_json(
        result_dir / "trace.json",
        traces,
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
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--condition",
        required=True,
        choices=sorted(
            CONDITIONS
        ),
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
            "pretrained"
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
        "--force",
        action="store_true",
    )

    return parser


def main():
    args = (
        build_parser()
        .parse_args()
    )

    raise SystemExit(
        run_job(args)
    )


if __name__ == "__main__":
    main()

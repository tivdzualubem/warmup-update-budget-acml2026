"""
DBPedia Warmup Safety Gate experiment with calibrated update-ratio threshold.

Threshold rule:
- Run stable_with_guard and fixed_order_with_guard first.
- Compute max early update/parameter ratio from only those safe runs.
- Set threshold = 10 × max_safe_update_ratio.
- Manifest records threshold_source = calibrated_10x_safe_max.
"""

import os
import sys
import json
import copy
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from data.data_loader_scratch import load_text_dataset, get_dataloaders
from models.transformer import TransformerClassifier, create_model_configs
from utils.metrics_scratch import (
    expected_calibration_error,
    multiclass_nll,
    multiclass_brier,
    mean_ci95,
)

RUN_MODE = "full"      # quick test: seed 42, 1 epoch, 1% training data
# package default: full run     # full final run: seeds 42,123,456, 30 epochs, full data

DATASET_NAME = "dbpedia_14"
RUN_TAG = "dbpedia_warmup_safety_gate_v2_calibrated"

if RUN_MODE == "quick":
    SEEDS = [42]
    MODEL_NAMES = ["medium-4", "large-6"]
    NUM_EPOCHS = 1
    TRAIN_FRACTION = 0.01
elif RUN_MODE == "full":
    SEEDS = [42, 123, 456]
    MODEL_NAMES = ["medium-4", "large-6"]
    NUM_EPOCHS = 30
    TRAIN_FRACTION = 1.0
else:
    raise ValueError("RUN_MODE must be 'quick' or 'full'")

CALIBRATION_CONDITIONS = [
    "stable_with_guard",
    "fixed_order_with_guard",
]

POST_CALIBRATION_CONDITIONS = [
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "single_bad_first_step_then_stable",
]

CONDITIONS = [
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "stable_with_guard",
    "fixed_order_with_guard",
    "single_bad_first_step_then_stable",
]

MAX_LENGTH = 128
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5

BASE_LR = 1e-4
LEGACY_INITIAL_LR = 1.0
WARMUP_STEPS = 8000
TOTAL_STEPS = 50000

GUARD_WINDOW_STEPS = 100
DIAGNOSTIC_STEPS = 300

THRESHOLD_MULTIPLIER = 10.0
THRESHOLD_SOURCE = "calibrated_10x_safe_max"

RESULT_ROOT = PROJECT_ROOT / f"results/warmup_safety_gate_{DATASET_NAME}_{RUN_TAG}"
PER_SEED_DIR = RESULT_ROOT / "per_seed"
TRACE_DIR = RESULT_ROOT / "scheduler_traces"
GUARD_DIR = RESULT_ROOT / "guard_events"
EARLY_DIR = RESULT_ROOT / "early_logs"

for d in [RESULT_ROOT, PER_SEED_DIR, TRACE_DIR, GUARD_DIR, EARLY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_lr(step):
    if step < WARMUP_STEPS:
        scale = step / max(1, WARMUP_STEPS)
    else:
        progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        progress = min(max(progress, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return BASE_LR * scale


def legacy_lr(model, step):
    step = max(1, step)
    d_model = model.d_model
    return (d_model ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def clone_trainable_params(model):
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def compute_update_ratios(model, before_params):
    total_update_sq = 0.0
    total_param_sq = 0.0
    layer_update_sq = {}
    layer_param_sq = {}

    for name, p in model.named_parameters():
        if not p.requires_grad or name not in before_params:
            continue

        before = before_params[name].to(p.device)
        after = p.detach()

        upd_sq = torch.sum((after - before) ** 2).item()
        par_sq = torch.sum(before ** 2).item()

        total_update_sq += upd_sq
        total_param_sq += par_sq

        if "embedding" in name:
            group = "embedding"
        elif "classifier" in name:
            group = "classifier"
        elif "encoder_layers" in name:
            parts = name.split(".")
            group = f"layer_{parts[1]}" if len(parts) > 1 else "encoder"
        else:
            group = "other"

        layer_update_sq[group] = layer_update_sq.get(group, 0.0) + upd_sq
        layer_param_sq[group] = layer_param_sq.get(group, 0.0) + par_sq

    global_ratio = np.sqrt(total_update_sq) / (np.sqrt(total_param_sq) + 1e-12)

    layer_ratios = {
        k: np.sqrt(layer_update_sq[k]) / (np.sqrt(layer_param_sq[k]) + 1e-12)
        for k in layer_update_sq
    }

    return float(global_ratio), layer_ratios


def evaluate_full(model, loader, device, num_classes):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    acc = correct / total
    ece = expected_calibration_error(labels, probs)
    nll = multiclass_nll(labels, probs)
    brier = multiclass_brier(labels, probs, num_classes)

    return {
        "loss": float(total_loss / len(loader)),
        "acc": float(acc),
        "ece": float(ece),
        "nll": float(nll),
        "brier": float(brier),
    }


def compute_lr_for_condition(condition, model, current_step, lr_before):
    next_step = current_step + 1

    if condition in ["buggy_no_guard", "buggy_failfast_guard", "buggy_rescue_guard"]:
        lr_used = lr_before
        lr_after = legacy_lr(model, next_step)
        return lr_used, lr_after, "schedule_after_step"

    if condition == "fixed_order_with_guard":
        lr_used = legacy_lr(model, next_step)
        lr_after = lr_used
        return lr_used, lr_after, "schedule_before_step"

    if condition == "stable_with_guard":
        lr_used = stable_lr(next_step)
        lr_after = lr_used
        return lr_used, lr_after, "schedule_before_step"

    if condition == "single_bad_first_step_then_stable":
        if current_step == 0:
            lr_used = LEGACY_INITIAL_LR
        else:
            lr_used = stable_lr(next_step)
        lr_after = stable_lr(next_step)
        return lr_used, lr_after, "single_bad_then_stable"

    raise ValueError(f"Unknown condition: {condition}")


def run_one(model_name, condition, seed, threshold, device, calibration_phase=False):
    print("\n" + "=" * 100)
    print(f"RUN: model={model_name} | condition={condition} | seed={seed}")
    print("=" * 100)

    set_seed(seed)

    train_dataset, val_dataset, test_dataset, vocab, num_classes = load_text_dataset(
        DATASET_NAME,
        max_length=MAX_LENGTH,
        split_seed=seed,
        train_fraction=TRAIN_FRACTION,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
    )

    cfg = [c for c in create_model_configs() if c["name"] == model_name][0]

    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=num_classes,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=MAX_LENGTH,
    ).to(device)

    num_parameters = model.count_parameters()

    init_lr = BASE_LR if condition == "stable_with_guard" else LEGACY_INITIAL_LR

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    criterion = nn.CrossEntropyLoss()

    scheduler_trace = []
    guard_events = []
    early_logs = []

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    stopped_by_guard = False
    rescued_once = False
    first_guard_trigger_step = None
    current_step = 0
    max_early_update_ratio = 0.0

    for epoch in range(NUM_EPOCHS):
        if stopped_by_guard:
            break

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batches_seen = 0

        for batch in train_loader:
            batches_seen += 1

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            before_params = clone_trainable_params(model)
            model_state_before = copy.deepcopy(model.state_dict())
            optim_state_before = copy.deepcopy(optimizer.state_dict())

            lr_before = float(optimizer.param_groups[0]["lr"])
            lr_used, lr_after, schedule_order = compute_lr_for_condition(
                condition, model, current_step, lr_before
            )

            if schedule_order in ["schedule_before_step", "single_bad_then_stable"]:
                set_optimizer_lr(optimizer, lr_used)

            optimizer.step()

            unsafe_update_ratio, unsafe_layer_ratios = compute_update_ratios(model, before_params)

            current_step += 1

            if schedule_order == "schedule_after_step":
                set_optimizer_lr(optimizer, lr_after)

            final_update_ratio = unsafe_update_ratio
            final_layer_ratios = unsafe_layer_ratios
            guard_action = "none"

            if (
                not calibration_phase
                and current_step <= GUARD_WINDOW_STEPS
                and condition in ["buggy_failfast_guard", "buggy_rescue_guard"]
            ):
                if unsafe_update_ratio > threshold:
                    first_guard_trigger_step = first_guard_trigger_step or current_step

                    if condition == "buggy_failfast_guard":
                        guard_action = "failfast_stop"
                        stopped_by_guard = True

                    elif condition == "buggy_rescue_guard":
                        guard_action = "rescue_correct_step"
                        rescued_once = True

                        model.load_state_dict(model_state_before)
                        optimizer.load_state_dict(optim_state_before)

                        safe_lr = legacy_lr(model, current_step)
                        set_optimizer_lr(optimizer, safe_lr)

                        optimizer.step()

                        final_update_ratio, final_layer_ratios = compute_update_ratios(model, before_params)
                        lr_used = safe_lr
                        lr_after = safe_lr

                    guard_events.append({
                        "dataset_name": DATASET_NAME,
                        "model_name": model_name,
                        "condition": condition,
                        "seed": int(seed),
                        "step": int(current_step),
                        "action": guard_action,
                        "threshold": float(threshold),
                        "threshold_source": THRESHOLD_SOURCE,
                        "observed_update_ratio": float(unsafe_update_ratio),
                        "final_update_ratio": float(final_update_ratio),
                    })

            max_early_update_ratio = max(max_early_update_ratio, final_update_ratio)

            scheduler_trace.append({
                "step": int(current_step),
                "lr_before_step": float(lr_before),
                "lr_used": float(lr_used),
                "lr_after_step": float(lr_after),
                "condition": condition,
            })

            if current_step <= DIAGNOSTIC_STEPS:
                probs = torch.softmax(logits.detach(), dim=1)
                early_logs.append({
                    "step": int(current_step),
                    "loss": float(loss.item()),
                    "lr_before_step": float(lr_before),
                    "lr_used": float(lr_used),
                    "lr_after_step": float(lr_after),
                    "update_to_param_ratio": float(final_update_ratio),
                    "unsafe_update_to_param_ratio": float(unsafe_update_ratio),
                    "mean_confidence": float(probs.max(dim=1)[0].mean().item()),
                    "layer_update_to_param_ratio": final_layer_ratios,
                })

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if stopped_by_guard:
                break

        train_loss = total_loss / max(1, batches_seen)
        train_acc = correct / max(1, total)

        val_eval = evaluate_full(model, val_loader, device, num_classes)
        val_loss = val_eval["loss"]
        val_acc = val_eval["acc"]

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"stopped_by_guard={stopped_by_guard}"
        )

        if val_loss < best_val_loss and not stopped_by_guard:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None and not stopped_by_guard:
        model.load_state_dict(best_state)

    train_eval = evaluate_full(model, train_loader, device, num_classes)
    test_eval = evaluate_full(model, test_loader, device, num_classes)

    generalization_gap = train_eval["acc"] - test_eval["acc"]

    file_stub = f"{DATASET_NAME}_{model_name}_{condition}_seed{seed}"

    scheduler_trace_file = TRACE_DIR / f"{file_stub}_scheduler_trace.json"
    guard_events_file = GUARD_DIR / f"{file_stub}_guard_events.json"
    early_logs_file = EARLY_DIR / f"{file_stub}_early_logs.json"

    save_json(scheduler_trace, scheduler_trace_file)
    save_json(guard_events, guard_events_file)
    save_json(early_logs, early_logs_file)

    result = {
        "dataset_name": DATASET_NAME,
        "run_tag": RUN_TAG,
        "run_mode": RUN_MODE,
        "model_name": model_name,
        "condition": condition,
        "seed": int(seed),
        "num_parameters": int(num_parameters),
        "train_fraction": float(TRAIN_FRACTION),
        "num_epochs_config": int(NUM_EPOCHS),
        "threshold": None if threshold is None else float(threshold),
        "threshold_source": THRESHOLD_SOURCE,
        "threshold_multiplier": float(THRESHOLD_MULTIPLIER),
        "threshold_calibration_conditions": CALIBRATION_CONDITIONS,
        "calibration_phase": bool(calibration_phase),
        "test_acc": float(test_eval["acc"]),
        "test_ece": float(test_eval["ece"]),
        "test_nll": float(test_eval["nll"]),
        "test_brier": float(test_eval["brier"]),
        "train_acc": float(train_eval["acc"]),
        "generalization_gap_01": float(generalization_gap),
        "guard_trigger_count": int(len(guard_events)),
        "stopped_by_guard": bool(stopped_by_guard),
        "rescued_once": bool(rescued_once),
        "first_guard_trigger_step": first_guard_trigger_step,
        "max_early_update_ratio": float(max_early_update_ratio),
        "scheduler_trace_file": str(scheduler_trace_file.relative_to(RESULT_ROOT)),
        "guard_events_file": str(guard_events_file.relative_to(RESULT_ROOT)),
        "early_logs_file": str(early_logs_file.relative_to(RESULT_ROOT)),
    }

    per_seed_file = PER_SEED_DIR / f"{file_stub}.json"
    save_json(result, per_seed_file)

    print(
        f"FINAL | test_acc={result['test_acc']:.4f} | "
        f"ece={result['test_ece']:.4f} | "
        f"nll={result['test_nll']:.4f} | "
        f"max_update={result['max_early_update_ratio']:.3e} | "
        f"guard_triggers={result['guard_trigger_count']} | "
        f"stopped={result['stopped_by_guard']} | "
        f"rescued={result['rescued_once']}"
    )

    return result


def calibrate_threshold(device):
    print("\n" + "=" * 100)
    print("CALIBRATING THRESHOLD FROM SAFE RUNS ONLY")
    print("=" * 100)

    calibration_results = []

    for model_name in MODEL_NAMES:
        for condition in CALIBRATION_CONDITIONS:
            for seed in SEEDS:
                result = run_one(
                    model_name=model_name,
                    condition=condition,
                    seed=seed,
                    threshold=None,
                    device=device,
                    calibration_phase=True,
                )
                calibration_results.append(result)

                pd.DataFrame(calibration_results).to_csv(
                    RESULT_ROOT / "threshold_calibration_safe_runs.csv",
                    index=False,
                )

    safe_max = max(r["max_early_update_ratio"] for r in calibration_results)
    threshold = THRESHOLD_MULTIPLIER * safe_max

    calibration_record = {
        "threshold": float(threshold),
        "threshold_source": THRESHOLD_SOURCE,
        "threshold_multiplier": float(THRESHOLD_MULTIPLIER),
        "safe_max_update_ratio": float(safe_max),
        "threshold_calibration_conditions": CALIBRATION_CONDITIONS,
        "calibration_models": MODEL_NAMES,
        "calibration_seeds": SEEDS,
        "calibration_rule": "threshold = 10 * max(max_early_update_ratio over stable_with_guard and fixed_order_with_guard)",
    }

    save_json(calibration_record, RESULT_ROOT / "threshold_calibration_record.json")

    print("\nCALIBRATION COMPLETE")
    print(json.dumps(calibration_record, indent=2))

    return threshold, calibration_record, calibration_results


def summarize_results(results, threshold):
    rows = []

    for (model_name, condition), sub_df in pd.DataFrame(results).groupby(["model_name", "condition"]):
        rows.append({
            "dataset_name": DATASET_NAME,
            "model_name": model_name,
            "condition": condition,
            "num_parameters": int(sub_df["num_parameters"].iloc[0]),
            "seeds": sorted(sub_df["seed"].astype(int).tolist()),
            "threshold": float(threshold),
            "threshold_source": THRESHOLD_SOURCE,
            "threshold_multiplier": float(THRESHOLD_MULTIPLIER),
            "threshold_calibration_conditions": CALIBRATION_CONDITIONS,
            "guard_trigger_rate": float((sub_df["guard_trigger_count"] > 0).mean()),
            "stopped_rate": float(sub_df["stopped_by_guard"].mean()),
            "rescued_rate": float(sub_df["rescued_once"].mean()),
            "first_guard_trigger_step_mean": float(sub_df["first_guard_trigger_step"].dropna().mean()) if sub_df["first_guard_trigger_step"].notna().any() else None,
            "max_early_update_ratio_mean": float(sub_df["max_early_update_ratio"].mean()),
            "test_acc_stats": mean_ci95(sub_df["test_acc"].values),
            "ece_stats": mean_ci95(sub_df["test_ece"].values),
            "nll_stats": mean_ci95(sub_df["test_nll"].values),
            "brier_stats": mean_ci95(sub_df["test_brier"].values),
            "gap_stats": mean_ci95(sub_df["generalization_gap_01"].values),
        })

    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    threshold, calibration_record, calibration_results = calibrate_threshold(device)

    manifest = {
        "dataset_name": DATASET_NAME,
        "run_tag": RUN_TAG,
        "run_mode": RUN_MODE,
        "models": MODEL_NAMES,
        "conditions": CONDITIONS,
        "calibration_conditions": CALIBRATION_CONDITIONS,
        "post_calibration_conditions": POST_CALIBRATION_CONDITIONS,
        "seeds": SEEDS,
        "train_fraction": TRAIN_FRACTION,
        "base_lr": BASE_LR,
        "legacy_initial_lr": LEGACY_INITIAL_LR,
        "warmup_steps": WARMUP_STEPS,
        "total_steps": TOTAL_STEPS,
        "guard_window_steps": GUARD_WINDOW_STEPS,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "threshold": float(threshold),
        "threshold_source": THRESHOLD_SOURCE,
        "threshold_multiplier": float(THRESHOLD_MULTIPLIER),
        "safe_max_update_ratio": float(calibration_record["safe_max_update_ratio"]),
        "threshold_calibration_conditions": CALIBRATION_CONDITIONS,
        "threshold_calibration_rule": calibration_record["calibration_rule"],
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "device": device,
    }

    save_json(manifest, RESULT_ROOT / "run_manifest.json")

    print("\n" + "=" * 100)
    print("DBPEDIA WARMUP SAFETY GATE MANIFEST")
    print("=" * 100)
    for k, v in manifest.items():
        print(f"{k:<35}: {v}")

    results = list(calibration_results)

    pd.DataFrame(results).to_csv(
        RESULT_ROOT / "warmup_safety_gate_per_seed.csv",
        index=False,
    )

    save_json(
        summarize_results(results, threshold),
        RESULT_ROOT / "warmup_safety_gate_summary.json",
    )

    for model_name in MODEL_NAMES:
        for condition in POST_CALIBRATION_CONDITIONS:
            for seed in SEEDS:
                result = run_one(
                    model_name=model_name,
                    condition=condition,
                    seed=seed,
                    threshold=threshold,
                    device=device,
                    calibration_phase=False,
                )

                results.append(result)

                pd.DataFrame(results).to_csv(
                    RESULT_ROOT / "warmup_safety_gate_per_seed.csv",
                    index=False,
                )

                save_json(
                    summarize_results(results, threshold),
                    RESULT_ROOT / "warmup_safety_gate_summary.json",
                )

    final_summary = summarize_results(results, threshold)

    print("\nSUMMARY")
    print(json.dumps(final_summary, indent=2))

    print("\nSaved result root:", RESULT_ROOT)


if __name__ == "__main__":
    main()

import os
import sys
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader_scratch import load_text_dataset, get_dataloaders
from models.transformer import TransformerClassifier, create_model_configs

# ======================== CHANGE ONLY THIS BLOCK ========================
DATASET_NAME = "dbpedia_14"
RUN_TAG = "dbpedia_earlydiag_plus_v1"
SEEDS = [42, 123, 456]

MODEL_NAMES = ["medium-4", "large-6"]
CONDITIONS = ["stable_schedule", "legacy_warmup_fixed_order", "legacy_buggy_warmup"]

TRAIN_FRACTION = 1.0
MAX_LENGTH = 128
BATCH_SIZE = 32

DIAGNOSTIC_STEPS = 300
MAX_EPOCHS_FOR_DIAGNOSTIC = 3
REP_EVAL_EVERY = 10
# ======================================================================


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_condition_settings(config, condition_name):
    if condition_name == "stable_schedule":
        if config["n_layers"] >= 4:
            return {
                "learning_rate": 1e-4,
                "scheduler_mode": "cosine",
                "warmup_steps": 8000,
                "legacy_initial_lr": 1.0
            }
        else:
            return {
                "learning_rate": 5e-4,
                "scheduler_mode": "cosine",
                "warmup_steps": 4000,
                "legacy_initial_lr": 1.0
            }

    if condition_name == "legacy_warmup_fixed_order":
        return {
            "learning_rate": 1e-4,
            "scheduler_mode": "legacy_warmup_fixed_order",
            "warmup_steps": 4000,
            "legacy_initial_lr": 1.0
        }

    if condition_name == "legacy_buggy_warmup":
        return {
            "learning_rate": 1e-4,
            "scheduler_mode": "legacy_buggy_warmup",
            "warmup_steps": 4000,
            "legacy_initial_lr": 1.0
        }

    raise ValueError(f"Unknown condition_name: {condition_name}")


def make_optimizer(model, settings):
    init_lr = settings["legacy_initial_lr"] if settings["scheduler_mode"] in [
        "legacy_buggy_warmup", "legacy_warmup_fixed_order"
    ] else settings["learning_rate"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    return optimizer


def get_lr_for_mode(current_step, model, settings, total_steps):
    mode = settings["scheduler_mode"]
    lr = settings["learning_rate"]
    warmup_steps = settings["warmup_steps"]

    if mode == "cosine":
        if current_step < warmup_steps:
            scale = current_step / max(1, warmup_steps)
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr * scale

    if mode in ["legacy_buggy_warmup", "legacy_warmup_fixed_order"]:
        step = max(1, current_step)
        d_model = model.d_model
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    raise ValueError(f"Unknown scheduler mode: {mode}")


def set_optimizer_lr(optimizer, lr_value):
    for pg in optimizer.param_groups:
        pg["lr"] = lr_value


def grad_norm_l2(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def param_norm_l2(model):
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += p.detach().pow(2).sum().item()
    return math.sqrt(total)


def clone_named_params(model):
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def update_norm_l2(model, before_params):
    total = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            diff = p.detach() - before_params[name]
            total += diff.pow(2).sum().item()
    return math.sqrt(total)


def layer_group_name(param_name):
    if param_name.startswith("embedding"):
        return "embedding"
    if param_name.startswith("pos_encoding"):
        return "pos_encoding"
    if "layers." in param_name:
        parts = param_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                return f"layer_{parts[i+1]}"
    if "encoder_layers." in param_name:
        parts = param_name.split(".")
        for i, part in enumerate(parts):
            if part == "encoder_layers" and i + 1 < len(parts):
                return f"layer_{parts[i+1]}"
    if param_name.startswith("classifier"):
        return "classifier"
    if param_name.startswith("fc") or param_name.startswith("head"):
        return "classifier"
    return "other"


def layerwise_grad_norms(model):
    sums = defaultdict(float)
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = layer_group_name(name)
            sums[g] += p.grad.detach().pow(2).sum().item()
    return {k: float(math.sqrt(v)) for k, v in sums.items()}


def layerwise_param_norms(model):
    sums = defaultdict(float)
    for name, p in model.named_parameters():
        if p.requires_grad:
            g = layer_group_name(name)
            sums[g] += p.detach().pow(2).sum().item()
    return {k: float(math.sqrt(v)) for k, v in sums.items()}


def layerwise_update_ratios(model, before_params):
    upd = defaultdict(float)
    par = defaultdict(float)
    for name, p in model.named_parameters():
        if p.requires_grad:
            g = layer_group_name(name)
            diff = p.detach() - before_params[name]
            upd[g] += diff.pow(2).sum().item()
            par[g] += before_params[name].pow(2).sum().item()

    out = {}
    for g in upd:
        upd_norm = math.sqrt(upd[g])
        par_norm = math.sqrt(par[g])
        out[g] = float(upd_norm / max(par_norm, 1e-12))
    return out


@torch.no_grad()
def get_repr(model, input_ids):
    model.eval()
    reps = model.encode(input_ids)
    model.train()
    return reps.detach().clone()


@torch.no_grad()
def representation_drift(model, probe_input_ids, base_repr):
    cur = get_repr(model, probe_input_ids)
    drift_l2 = (cur - base_repr).norm(dim=1).mean().item()

    cur_n = cur / cur.norm(dim=1, keepdim=True).clamp_min(1e-12)
    base_n = base_repr / base_repr.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (cur_n * base_n).sum(dim=1).mean().item()

    return float(drift_l2), float(cos)


def build_window_summary(logs, step_limit):
    sub = logs[:min(step_limit, len(logs))]
    if len(sub) == 0:
        return {}

    layer_names = sorted({
        layer_name
        for x in sub
        for layer_name in x["layer_update_to_param_ratio"].keys()
    })

    layer_max = {}
    for layer_name in layer_names:
        vals = [
            x["layer_update_to_param_ratio"].get(layer_name, np.nan)
            for x in sub
        ]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            layer_max[layer_name] = float(max(vals))

    repr_rows = [x for x in sub if x["repr_drift_l2"] is not None]

    emb_vals = [x["layer_update_to_param_ratio"].get("embedding", np.nan) for x in sub]
    emb_vals = [v for v in emb_vals if not np.isnan(v)]

    cls_vals = [x["layer_update_to_param_ratio"].get("classifier", np.nan) for x in sub]
    cls_vals = [v for v in cls_vals if not np.isnan(v)]

    return {
        "steps_included": int(len(sub)),
        "loss_step1": float(sub[0]["loss"]),
        "max_loss": float(max(x["loss"] for x in sub)),
        "max_grad_norm": float(max(x["grad_norm"] for x in sub)),
        "max_update_ratio": float(max(x["update_to_param_ratio"] for x in sub)),
        "max_embedding_update_ratio": float(max(emb_vals)) if emb_vals else np.nan,
        "max_classifier_update_ratio": float(max(cls_vals)) if cls_vals else np.nan,
        "mean_confidence_first10_within_window": float(np.mean([x["mean_confidence"] for x in sub[:min(10, len(sub))]])),
        "last_repr_drift_l2": float(repr_rows[-1]["repr_drift_l2"]) if repr_rows else None,
        "last_repr_cos_to_init": float(repr_rows[-1]["repr_cos_to_init"]) if repr_rows else None,
        "last_lr_before_step": float(sub[-1]["lr_before_step"]),
        "last_lr_used": float(sub[-1]["lr_used"]),
        "last_lr_after_step": float(sub[-1]["lr_after_step"]),
    }


def run_one(config, condition_name, seed, results_root):
    run_name = f"{config['name']}_{condition_name}_seed{seed}"
    out_path = os.path.join(results_root, "per_run", run_name + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"Loading saved diagnostic run: {run_name}")
        with open(out_path, "r") as f:
            return json.load(f)

    set_seed(seed)

    train_dataset, val_dataset, test_dataset, vocab, num_classes = load_text_dataset(
        dataset_name=DATASET_NAME,
        max_length=MAX_LENGTH,
        split_seed=42,
        val_ratio=0.15,
        train_fraction=TRAIN_FRACTION
    )
    train_loader, _, _ = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=num_classes,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        max_seq_len=MAX_LENGTH,
        dropout=0.1,
        pad_idx=0
    ).to(device)

    settings = get_condition_settings(config, condition_name)
    optimizer = make_optimizer(model, settings)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * MAX_EPOCHS_FOR_DIAGNOSTIC
    logs = []
    global_step = 0

    model.train()

    probe_batch = next(iter(train_loader))
    probe_input_ids = probe_batch["input_ids"].to(device)
    base_repr = get_repr(model, probe_input_ids)

    for epoch in range(MAX_EPOCHS_FOR_DIAGNOSTIC):
        for batch_idx, batch in enumerate(train_loader, start=1):
            if global_step >= DIAGNOSTIC_STEPS:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs.detach(), dim=1)
            mean_conf = probs.max(dim=1).values.mean().item()
            batch_acc = (outputs.detach().argmax(dim=1) == labels).float().mean().item()
            logit_norm = outputs.detach().norm(dim=1).mean().item()

            loss.backward()

            grad_norm = grad_norm_l2(model)
            layer_grad = layerwise_grad_norms(model)
            param_norm_before = param_norm_l2(model)
            layer_param_before = layerwise_param_norms(model)
            params_before = clone_named_params(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            lr_before = optimizer.param_groups[0]["lr"]

            if settings["scheduler_mode"] == "legacy_buggy_warmup":
                lr_used = lr_before
                optimizer.step()
                global_step += 1
                lr_after = get_lr_for_mode(global_step, model, settings, total_steps)
                set_optimizer_lr(optimizer, lr_after)
            else:
                global_step += 1
                lr_used = get_lr_for_mode(global_step, model, settings, total_steps)
                set_optimizer_lr(optimizer, lr_used)
                optimizer.step()
                lr_after = optimizer.param_groups[0]["lr"]

            update_norm = update_norm_l2(model, params_before)
            update_ratio = update_norm / max(param_norm_before, 1e-12)
            layer_update_ratio = layerwise_update_ratios(model, params_before)

            rep_drift_l2 = None
            rep_cos_to_init = None
            if global_step == 1 or global_step % REP_EVAL_EVERY == 0:
                rep_drift_l2, rep_cos_to_init = representation_drift(model, probe_input_ids, base_repr)

            logs.append({
                "step": global_step,
                "epoch": epoch + 1,
                "batch_in_epoch": batch_idx,
                "lr_before_step": float(lr_before),
                "lr_used": float(lr_used),
                "lr_after_step": float(lr_after),
                "loss": float(loss.item()),
                "batch_acc": float(batch_acc),
                "mean_confidence": float(mean_conf),
                "logit_norm": float(logit_norm),
                "grad_norm": float(grad_norm),
                "param_norm_before": float(param_norm_before),
                "update_norm": float(update_norm),
                "update_to_param_ratio": float(update_ratio),
                "layer_grad_norms": layer_grad,
                "layer_param_norms_before": layer_param_before,
                "layer_update_to_param_ratio": layer_update_ratio,
                "repr_drift_l2": rep_drift_l2,
                "repr_cos_to_init": rep_cos_to_init,
            })

        if global_step >= DIAGNOSTIC_STEPS:
            break

    summary = {
        "dataset_name": DATASET_NAME,
        "run_tag": RUN_TAG,
        "model_name": config["name"],
        "condition": condition_name,
        "seed": seed,
        "diagnostic_steps_requested": DIAGNOSTIC_STEPS,
        "diagnostic_steps_collected": len(logs),
        "logs": logs,
        "early_summary_50": build_window_summary(logs, 50),
        "early_summary_100": build_window_summary(logs, 100),
        "early_summary_300": build_window_summary(logs, 300),
    }

    save_json(summary, out_path)
    print(f"Saved diagnostic run: {out_path}")
    return summary


def main():
    results_root = f"transformer_project/results/early_diagnostics_plus_{DATASET_NAME}_{RUN_TAG}"
    os.makedirs(results_root, exist_ok=True)

    manifest = {
        "dataset_name": DATASET_NAME,
        "run_tag": RUN_TAG,
        "seeds": SEEDS,
        "model_names": MODEL_NAMES,
        "conditions": CONDITIONS,
        "train_fraction": TRAIN_FRACTION,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "max_epochs_for_diagnostic": MAX_EPOCHS_FOR_DIAGNOSTIC,
        "representation_eval_every": REP_EVAL_EVERY,
    }
    save_json(manifest, os.path.join(results_root, "run_manifest.json"))

    configs = [cfg for cfg in create_model_configs() if cfg["name"] in MODEL_NAMES]
    all_runs = []

    print("=" * 60)
    print("EARLY DIAGNOSTICS PLUS RUN")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Results: {results_root}")

    for cfg in configs:
        for condition in CONDITIONS:
            print(f"\n{'='*60}")
            print(f"MODEL={cfg['name']} | CONDITION={condition}")
            print("="*60)
            for seed in SEEDS:
                run = run_one(cfg, condition, seed, results_root)
                all_runs.append(run)

    save_json(all_runs, os.path.join(results_root, "all_runs.json"))
    print(f"\nSaved all diagnostics to: {results_root}")


if __name__ == "__main__":
    main()

import os
import sys
import json
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader_scratch import load_text_dataset, get_dataloaders
from models.transformer import TransformerClassifier, create_model_configs
from utils.trainer_scratch import ScratchTrainer
from utils.metrics_scratch import (
    expected_calibration_error,
    multiclass_nll,
    multiclass_brier,
    temperature_scaling,
    mean_ci95
)

# ======================== CHANGE ONLY THIS BLOCK ========================
DATASET_NAME = "dbpedia_14"
RUN_TAG = "dbpedia_initial_lr_sweep_v1"
SEEDS = [42, 123, 456]

MODEL_NAMES = ["medium-4", "large-6"]
INITIAL_LR_VALUES = [1.0, 0.3, 0.1, 0.03, 0.01]

TRAIN_FRACTION = 1.0
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
# ======================================================================


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_results_root():
    root = f"transformer_project/results/initial_lr_sweep_{DATASET_NAME}_{RUN_TAG}"
    os.makedirs(root, exist_ok=True)
    return root


def get_buggy_sweep_settings(initial_lr):
    return {
        "learning_rate": 1e-4,
        "scheduler_mode": "legacy_buggy_warmup",
        "warmup_steps": 4000,
        "legacy_initial_lr": float(initial_lr),
    }


def lr_tag(x):
    return str(x).replace(".", "p")


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result_block(result):
    print("\n" + "-" * 60)
    print(
        f"model={result['model_name']} | seed={result['seed']} | "
        f"initial_lr={result['initial_lr']} | test_acc={result['test_acc']:.4f} | "
        f"ece={result['test_ece']:.4f} | nll={result['test_nll']:.4f}"
    )
    print(
        f"gap={result['generalization_gap_01']:.4f} | "
        f"scheduler_trace_steps={result['scheduler_trace_steps']}"
    )
    print("-" * 60)


def run_single_seed(config, seed, initial_lr, results_root):
    run_name = f"{config['name']}_legacy_buggy_initlr_{lr_tag(initial_lr)}_seed{seed}"
    seed_json = os.path.join(results_root, "per_seed", run_name + ".json")
    trace_json = os.path.join(results_root, "scheduler_traces", run_name + "_scheduler_trace.json")
    checkpoint_dir = os.path.join(results_root, "checkpoints", run_name)

    os.makedirs(os.path.dirname(seed_json), exist_ok=True)
    os.makedirs(os.path.dirname(trace_json), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(seed_json):
        print(f"Loading saved result: {run_name}")
        result = load_json(seed_json)
        print_result_block(result)
        return result

    set_seed(seed)

    train_dataset, val_dataset, test_dataset, vocab, num_classes = load_text_dataset(
        dataset_name=DATASET_NAME,
        max_length=MAX_LENGTH,
        split_seed=42,
        val_ratio=0.15,
        train_fraction=TRAIN_FRACTION
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE
    )

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
    )

    settings = get_buggy_sweep_settings(initial_lr)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS

    trainer = ScratchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=settings["learning_rate"],
        scheduler_mode=settings["scheduler_mode"],
        warmup_steps=settings["warmup_steps"],
        total_steps=total_steps,
        legacy_initial_lr=settings["legacy_initial_lr"],
        checkpoint_dir=checkpoint_dir
    )

    trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        resume=True
    )

    scheduler_trace = trainer.scheduler_trace
    save_json(scheduler_trace, trace_json)

    _, train_acc, _, _, _, _ = trainer.evaluate(train_loader, return_predictions=True)
    _, val_acc, _, _, val_logits, val_labels = trainer.evaluate(val_loader, return_predictions=True)
    _, test_acc, _, test_probs, test_logits, test_labels = trainer.evaluate(test_loader, return_predictions=True)

    val_logits = np.array(val_logits)
    val_labels = np.array(val_labels)
    test_logits = np.array(test_logits)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)

    train_error_01 = 1.0 - train_acc
    test_error_01 = 1.0 - test_acc
    gap_01 = test_error_01 - train_error_01

    ece_before = expected_calibration_error(test_labels, test_probs)
    nll_before = multiclass_nll(test_labels, test_probs)
    brier_before = multiclass_brier(test_labels, test_probs, num_classes)

    temperature = temperature_scaling(val_labels, val_logits)
    scaled_test_logits = test_logits / temperature
    exp_logits = np.exp(scaled_test_logits - np.max(scaled_test_logits, axis=1, keepdims=True))
    scaled_test_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    ece_after = expected_calibration_error(test_labels, scaled_test_probs)
    nll_after = multiclass_nll(test_labels, scaled_test_probs)
    brier_after = multiclass_brier(test_labels, scaled_test_probs, num_classes)

    result = {
        "dataset_name": DATASET_NAME,
        "model_name": config["name"],
        "seed": seed,
        "condition": "legacy_buggy_warmup",
        "initial_lr": float(initial_lr),
        "train_fraction": TRAIN_FRACTION,
        "num_parameters": model.count_parameters(),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "train_error_01": float(train_error_01),
        "test_error_01": float(test_error_01),
        "generalization_gap_01": float(gap_01),
        "test_ece": float(ece_before),
        "test_ece_scaled": float(ece_after),
        "test_nll": float(nll_before),
        "test_nll_scaled": float(nll_after),
        "test_brier": float(brier_before),
        "test_brier_scaled": float(brier_after),
        "temperature": float(temperature),
        "scheduler_trace_file": os.path.relpath(trace_json, results_root),
        "scheduler_trace_steps": len(scheduler_trace),
    }

    print_result_block(result)
    save_json(result, seed_json)
    print(f"Saved: {seed_json}")
    print(f"Saved: {trace_json}")

    return result


def aggregate_seed_results(results):
    return {
        "dataset_name": results[0]["dataset_name"],
        "model_name": results[0]["model_name"],
        "condition": results[0]["condition"],
        "initial_lr": results[0]["initial_lr"],
        "num_parameters": results[0]["num_parameters"],
        "seeds": [r["seed"] for r in results],
        "test_acc_stats": mean_ci95([r["test_acc"] for r in results]),
        "gap_stats": mean_ci95([r["generalization_gap_01"] for r in results]),
        "ece_stats": mean_ci95([r["test_ece"] for r in results]),
        "ece_scaled_stats": mean_ci95([r["test_ece_scaled"] for r in results]),
        "nll_stats": mean_ci95([r["test_nll"] for r in results]),
        "nll_scaled_stats": mean_ci95([r["test_nll_scaled"] for r in results]),
        "brier_stats": mean_ci95([r["test_brier"] for r in results]),
        "brier_scaled_stats": mean_ci95([r["test_brier_scaled"] for r in results]),
        "scheduler_trace_steps_stats": mean_ci95([r["scheduler_trace_steps"] for r in results]),
        "per_seed_results": results
    }


def main():
    results_root = make_results_root()

    manifest = {
        "dataset_name": DATASET_NAME,
        "run_tag": RUN_TAG,
        "seeds": SEEDS,
        "model_names": MODEL_NAMES,
        "initial_lr_values": INITIAL_LR_VALUES,
        "condition": "legacy_buggy_warmup",
        "train_fraction": TRAIN_FRACTION,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE
    }
    save_json(manifest, os.path.join(results_root, "run_manifest.json"))

    print_header("INITIAL-LR CAUSAL SWEEP")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Results: {results_root}")

    configs = [cfg for cfg in create_model_configs() if cfg["name"] in MODEL_NAMES]
    all_summaries = []

    for cfg in configs:
        model_block = {"model_name": cfg["name"], "initial_lr_blocks": []}

        for initial_lr in INITIAL_LR_VALUES:
            print_header(f"MODEL={cfg['name']} | initial_lr={initial_lr}")
            results = []
            for seed in SEEDS:
                results.append(run_single_seed(cfg, seed, initial_lr, results_root))

            summary = aggregate_seed_results(results)
            summary_path = os.path.join(
                results_root,
                "summaries",
                f"{cfg['name']}_initlr_{lr_tag(initial_lr)}_summary.json"
            )
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            save_json(summary, summary_path)
            model_block["initial_lr_blocks"].append(summary)

        all_summaries.append(model_block)

    save_json(all_summaries, os.path.join(results_root, "initial_lr_sweep_summary.json"))
    print(f"\nSaved all sweep outputs to: {results_root}")


if __name__ == "__main__":
    main()

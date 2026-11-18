"""
Pretrained baseline experiment runner.
Supports AG News and DBPedia.
"""

import os
import json
import random
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from data.data_loader_pretrained import load_pretrained_dataset
from utils.trainer_pretrained import PretrainedTrainer
from utils.metrics_pretrained import (
    expected_calibration_error,
    multiclass_nll,
    multiclass_brier,
    temperature_scaling
)


# ---------------- change only these ----------------
DATASET_NAME = "dbpedia_14"          # "ag_news" or "dbpedia_14"
MODEL_NAME = "distilbert-base-uncased"
SEEDS = [42, 123, 456]
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
RUN_TAG = "main"                  # change only if you want a separate fresh run
# ---------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(results, filepath):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def model_alias(name: str):
    return name.replace("/", "_")


def train_single_seed(seed: int, results_dir: str):
    seed_json = os.path.join(results_dir, f"distilbert_seed{seed}.json")
    checkpoint_dir = os.path.join(results_dir, "checkpoints", f"seed_{seed}")

    if os.path.exists(seed_json):
        print(f"Seed {seed} already finished. Loading saved result.")
        return load_json(seed_json)

    set_seed(seed)

    train_loader, val_loader, test_loader, tokenizer, num_classes = load_pretrained_dataset(
        dataset_name=DATASET_NAME,
        tokenizer_name=MODEL_NAME,
        max_length=MAX_LENGTH,
        split_seed=42,
        val_ratio=0.15,
        batch_size=BATCH_SIZE
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes
    )

    trainer = PretrainedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        checkpoint_dir=checkpoint_dir
    )

    trainer.train(early_stopping_patience=2, resume=True)

    train_loss, train_acc, _, _, _, _ = trainer.evaluate(train_loader, return_predictions=True)
    val_loss, val_acc, _, _, val_logits, val_labels = trainer.evaluate(val_loader, return_predictions=True)
    test_loss, test_acc, _, test_probs, test_logits, test_labels = trainer.evaluate(test_loader, return_predictions=True)

    val_logits = np.array(val_logits)
    val_labels = np.array(val_labels)
    test_logits = np.array(test_logits)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)

    train_error_01 = 1.0 - train_acc
    test_error_01 = 1.0 - test_acc
    gen_gap_01 = test_error_01 - train_error_01

    test_ece = expected_calibration_error(test_labels, test_probs)
    test_nll = multiclass_nll(test_labels, test_probs)
    test_brier = multiclass_brier(test_labels, test_probs, num_classes)

    _, temperature = temperature_scaling(val_labels, val_logits)
    scaled_test_logits = test_logits / temperature
    exp_logits = np.exp(scaled_test_logits - np.max(scaled_test_logits, axis=1, keepdims=True))
    scaled_test_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    test_ece_scaled = expected_calibration_error(test_labels, scaled_test_probs)
    test_nll_scaled = multiclass_nll(test_labels, scaled_test_probs)
    test_brier_scaled = multiclass_brier(test_labels, scaled_test_probs, num_classes)

    result = {
        "dataset_name": DATASET_NAME,
        "model_name": MODEL_NAME,
        "seed": seed,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "train_error_01": float(train_error_01),
        "test_error_01": float(test_error_01),
        "generalization_gap_01": float(gen_gap_01),
        "test_ece": float(test_ece),
        "test_ece_scaled": float(test_ece_scaled),
        "test_nll": float(test_nll),
        "test_nll_scaled": float(test_nll_scaled),
        "test_brier": float(test_brier),
        "test_brier_scaled": float(test_brier_scaled),
        "temperature": float(temperature),
        "num_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    }

    save_results(result, seed_json)
    return result


def main():
    alias = model_alias(MODEL_NAME)
    results_dir = f"transformer_project/results/pretrained_{DATASET_NAME}_{alias}_{RUN_TAG}"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("PRETRAINED BASELINE RUN")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Seeds  : {SEEDS}")
    print(f"Folder : {results_dir}")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print('='*60)
        result = train_single_seed(seed, results_dir)
        all_results.append(result)

    summary = {
        "dataset_name": DATASET_NAME,
        "model_name": MODEL_NAME,
        "seeds": SEEDS,
        "num_parameters": all_results[0]["num_parameters"],
        "test_acc": float(np.mean([r["test_acc"] for r in all_results])),
        "test_acc_std": float(np.std([r["test_acc"] for r in all_results])),
        "generalization_gap_01": float(np.mean([r["generalization_gap_01"] for r in all_results])),
        "generalization_gap_01_std": float(np.std([r["generalization_gap_01"] for r in all_results])),
        "test_ece": float(np.mean([r["test_ece"] for r in all_results])),
        "test_ece_std": float(np.std([r["test_ece"] for r in all_results])),
        "test_ece_scaled": float(np.mean([r["test_ece_scaled"] for r in all_results])),
        "test_nll": float(np.mean([r["test_nll"] for r in all_results])),
        "test_nll_scaled": float(np.mean([r["test_nll_scaled"] for r in all_results])),
        "test_brier": float(np.mean([r["test_brier"] for r in all_results])),
        "test_brier_scaled": float(np.mean([r["test_brier_scaled"] for r in all_results])),
        "temperature": float(np.mean([r["temperature"] for r in all_results])),
        "per_seed_results": all_results
    }

    save_results(summary, os.path.join(results_dir, "summary.json"))

    print(f"\n{'='*60}")
    print("MEAN RESULTS")
    print('='*60)
    print(f"Test Acc: {summary['test_acc']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"Gap     : {summary['generalization_gap_01']:.4f} ± {summary['generalization_gap_01_std']:.4f}")
    print(f"ECE     : {summary['test_ece']:.4f} → {summary['test_ece_scaled']:.4f}")
    print(f"NLL     : {summary['test_nll']:.4f} → {summary['test_nll_scaled']:.4f}")
    print(f"Brier   : {summary['test_brier']:.4f} → {summary['test_brier_scaled']:.4f}")
    print(f"T       : {summary['temperature']:.4f}")
    print(f"RESULTS SAVED TO: {results_dir}")


if __name__ == "__main__":
    main()

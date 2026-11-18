import os
import sys
import json
import random
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader_scratch import (
    load_text_dataset,
    get_dataloaders,
    build_shifted_test_loader
)
from models.transformer import TransformerClassifier, create_model_configs
from utils.trainer_scratch import ScratchTrainer
from utils.metrics_scratch import (
    compute_model_norms,
    compute_rademacher_bound,
    compute_pacbayes_bound,
    compute_margin_bound,
    expected_calibration_error,
    multiclass_nll,
    multiclass_brier,
    temperature_scaling,
    mean_ci95
)

# ======================== CHANGE ONLY THIS BLOCK ========================
DATASET_NAME = "ag_news"
RUN_TAG = "full_agnews_v1"
SEEDS = [42, 123, 456]

RUN_BASELINES = True
RUN_STABLE_SWEEP = True
RUN_ABLATION = True
RUN_DATA_SIZE_SWEEP = True
RUN_SHIFT_EVAL = True

MODEL_NAMES_STABLE = ["tiny-1", "small-1", "small-4", "medium-4", "large-6"]
MODEL_NAMES_ABLATION = ["medium-4", "large-6"]
ABLATION_CONDITIONS = ["stable_schedule", "legacy_warmup_fixed_order", "legacy_buggy_warmup"]

MODEL_NAMES_DATA_SIZE = ["tiny-1", "medium-4"]
TRAIN_FRACTIONS = [0.5, 0.2, 0.1]

SHIFT_CONFIGS = [
    {"name": "truncation", "max_tokens": 32},
    {"name": "unk_corruption", "rate": 0.2, "seed": 42}
]

MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
# ======================================================================


def save_results(results, filepath):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_results_root():
    root = f"transformer_project/results/scratch_empirical_{DATASET_NAME}_{RUN_TAG}"
    os.makedirs(root, exist_ok=True)
    return root


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


def print_summary_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result_block(result):
    print("\n" + "-" * 60)
    print(
        f"RESULT SUMMARY | model={result['model_name']} | seed={result['seed']} | "
        f"condition={result['condition']} | frac={result['train_fraction']:.2f}"
    )
    print(
        f"train_acc={result['train_acc']:.4f} | "
        f"val_acc={result['val_acc']:.4f} | "
        f"test_acc={result['test_acc']:.4f} | "
        f"gap={result['generalization_gap_01']:.4f}"
    )
    print(
        f"ece={result['test_ece']:.4f} -> {result['test_ece_scaled']:.4f} | "
        f"nll={result['test_nll']:.4f} -> {result['test_nll_scaled']:.4f} | "
        f"brier={result['test_brier']:.4f} -> {result['test_brier_scaled']:.4f} | "
        f"T={result['temperature']:.4f}"
    )
    print(
        f"rad={result['rademacher_bound']:.4e} | "
        f"pac={result['pacbayes_bound']:.4f} | "
        f"margin={result['margin_bound']:.4f}"
    )
    print(
        f"scheduler_trace_steps={result.get('scheduler_trace_steps', 0)}"
    )

    if result["shift_results"]:
        print("Shift results:")
        for shift_name, vals in result["shift_results"].items():
            print(
                f"  {shift_name}: "
                f"acc={vals['test_acc']:.4f} | "
                f"ece={vals['test_ece']:.4f} | "
                f"nll={vals['test_nll']:.4f} | "
                f"brier={vals['test_brier']:.4f}"
            )
    print("-" * 60)


def print_aggregate_block(summary, label):
    test_stats = summary["test_acc_stats"]
    gap_stats = summary["gap_stats"]
    ece_stats = summary["ece_stats"]
    ece_scaled_stats = summary["ece_scaled_stats"]

    print("\n" + "-" * 60)
    print(f"{label}")
    print(
        f"test_acc={test_stats['mean']:.4f} ± {test_stats['std']:.4f} "
        f"[{test_stats['ci95_low']:.4f}, {test_stats['ci95_high']:.4f}]"
    )
    print(
        f"gap={gap_stats['mean']:.4f} ± {gap_stats['std']:.4f} "
        f"[{gap_stats['ci95_low']:.4f}, {gap_stats['ci95_high']:.4f}]"
    )
    print(
        f"ece={ece_stats['mean']:.4f} -> {ece_scaled_stats['mean']:.4f}"
    )

    if "shift_summary" in summary:
        print("Shift summary:")
        for shift_name, shift_vals in summary["shift_summary"].items():
            acc_stats = shift_vals["test_acc_stats"]
            print(
                f"  {shift_name}: "
                f"acc={acc_stats['mean']:.4f} ± {acc_stats['std']:.4f} "
                f"[{acc_stats['ci95_low']:.4f}, {acc_stats['ci95_high']:.4f}]"
            )
    print("-" * 60)


def train_baseline_once(results_root):
    out_json = os.path.join(results_root, "baseline_results.json")
    if os.path.exists(out_json):
        print(f"Loading saved baselines: {out_json}")
        results = load_results(out_json)
        print_summary_header("BASELINES")
        for r in results:
            print(
                f"{r['model_name']}: "
                f"train_acc={r['train_acc']:.4f} | "
                f"test_acc={r['test_acc']:.4f} | "
                f"gap={r['generalization_gap_01']:.4f} | "
                f"ece={r['test_ece']:.4f} | "
                f"nll={r['test_nll']:.4f} | "
                f"brier={r['test_brier']:.4f}"
            )
        return results

    train_dataset, val_dataset, test_dataset, vocab, num_classes = load_text_dataset(
        dataset_name=DATASET_NAME,
        max_length=MAX_LENGTH,
        split_seed=42,
        val_ratio=0.15,
        train_fraction=1.0
    )

    train_texts = train_dataset.texts
    train_labels = train_dataset.labels
    test_texts = test_dataset.texts
    test_labels = test_dataset.labels

    results = []

    for name in ["bow", "tfidf"]:
        if name == "bow":
            vectorizer = CountVectorizer(max_features=10000, stop_words="english")
        else:
            vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, train_labels)

        train_acc = clf.score(X_train, train_labels)
        test_acc = clf.score(X_test, test_labels)

        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        test_probs = clf.predict_proba(X_test)

        train_error = 1.0 - accuracy_score(train_labels, train_preds)
        test_error = 1.0 - accuracy_score(test_labels, test_preds)
        gap = test_error - train_error

        result = {
            "model_name": f"{name}_baseline",
            "num_parameters": int(X_train.shape[1] * len(np.unique(train_labels))),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "train_error_01": float(train_error),
            "test_error_01": float(test_error),
            "generalization_gap_01": float(gap),
            "test_ece": float(expected_calibration_error(np.array(test_labels), test_probs)),
            "test_nll": float(multiclass_nll(np.array(test_labels), test_probs)),
            "test_brier": float(multiclass_brier(np.array(test_labels), test_probs, num_classes))
        }
        results.append(result)

    save_results(results, out_json)

    print_summary_header("BASELINES")
    for r in results:
        print(
            f"{r['model_name']}: "
            f"train_acc={r['train_acc']:.4f} | "
            f"test_acc={r['test_acc']:.4f} | "
            f"gap={r['generalization_gap_01']:.4f} | "
            f"ece={r['test_ece']:.4f} | "
            f"nll={r['test_nll']:.4f} | "
            f"brier={r['test_brier']:.4f}"
        )
    print(f"Saved: {out_json}")

    return results


def run_single_seed(config, seed, condition_name, train_fraction, results_root):
    run_name = f"{config['name']}_{condition_name}_frac{int(train_fraction*100):03d}_seed{seed}"
    seed_json = os.path.join(results_root, "per_seed", run_name + ".json")
    trace_json = os.path.join(results_root, "scheduler_traces", run_name + "_scheduler_trace.json")
    checkpoint_dir = os.path.join(results_root, "checkpoints", run_name)

    os.makedirs(os.path.dirname(seed_json), exist_ok=True)
    os.makedirs(os.path.dirname(trace_json), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(seed_json):
        print(f"Loading saved result: {run_name}")
        result = load_results(seed_json)
        print_result_block(result)
        return result

    set_seed(seed)

    train_dataset, val_dataset, test_dataset, vocab, num_classes = load_text_dataset(
        dataset_name=DATASET_NAME,
        max_length=MAX_LENGTH,
        split_seed=42,
        val_ratio=0.15,
        train_fraction=train_fraction
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

    settings = get_condition_settings(config, condition_name)
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
    save_results(scheduler_trace, trace_json)

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
    gap_01 = test_error_01 - train_error_01

    norms = compute_model_norms(model, exclude_embedding=True)
    rad_bound = compute_rademacher_bound(norms, len(train_loader.dataset), config["d_model"])
    pac_info = compute_pacbayes_bound(
        model, train_loader, device=trainer.device, sigma=0.01, delta=0.05, posterior_samples=3
    )
    margin_info = compute_margin_bound(model, train_loader, device=trainer.device, gamma=0.1)

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

    shift_results = {}
    if RUN_SHIFT_EVAL and condition_name == "stable_schedule":
        for shift_cfg in SHIFT_CONFIGS:
            shift_name = shift_cfg["name"]
            shift_loader = build_shifted_test_loader(test_dataset, shift_cfg, batch_size=BATCH_SIZE)
            _, shift_acc, _, shift_probs, shift_logits, shift_labels = trainer.evaluate(
                shift_loader, return_predictions=True
            )

            shift_probs = np.array(shift_probs)
            shift_logits = np.array(shift_logits)
            shift_labels = np.array(shift_labels)

            shift_results[shift_name] = {
                "test_acc": float(shift_acc),
                "test_error_01": float(1.0 - shift_acc),
                "test_ece": float(expected_calibration_error(shift_labels, shift_probs)),
                "test_nll": float(multiclass_nll(shift_labels, shift_probs)),
                "test_brier": float(multiclass_brier(shift_labels, shift_probs, num_classes))
            }

    result = {
        "dataset_name": DATASET_NAME,
        "model_name": config["name"],
        "seed": seed,
        "condition": condition_name,
        "train_fraction": train_fraction,
        "num_parameters": model.count_parameters(),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "train_error_01": float(train_error_01),
        "test_error_01": float(test_error_01),
        "generalization_gap_01": float(gap_01),
        "rademacher_bound": float(rad_bound),
        "log_rademacher_bound": float(np.log10(max(rad_bound, 1e-10))),
        "pacbayes_bound": float(pac_info["bound"]),
        "margin_bound": float(margin_info["bound"]),
        "test_ece": float(ece_before),
        "test_ece_scaled": float(ece_after),
        "test_nll": float(nll_before),
        "test_nll_scaled": float(nll_after),
        "test_brier": float(brier_before),
        "test_brier_scaled": float(brier_after),
        "temperature": float(temperature),
        "shift_results": shift_results,
        "scheduler_trace_file": os.path.relpath(trace_json, results_root),
        "scheduler_trace_steps": len(scheduler_trace),
    }

    print_result_block(result)
    save_results(result, seed_json)
    print(f"Saved: {seed_json}")
    print(f"Saved: {trace_json}")

    return result


def aggregate_seed_results(results):
    summary = {
        "dataset_name": results[0]["dataset_name"],
        "model_name": results[0]["model_name"],
        "condition": results[0]["condition"],
        "train_fraction": results[0]["train_fraction"],
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
        "rademacher_stats": mean_ci95([r["rademacher_bound"] for r in results]),
        "pacbayes_stats": mean_ci95([r["pacbayes_bound"] for r in results]),
        "margin_stats": mean_ci95([r["margin_bound"] for r in results]),
        "temperature_stats": mean_ci95([r["temperature"] for r in results]),
        "scheduler_trace_steps_stats": mean_ci95([r["scheduler_trace_steps"] for r in results]),
        "per_seed_results": results
    }

    if results[0].get("shift_results"):
        shift_names = results[0]["shift_results"].keys()
        if len(shift_names) > 0:
            summary["shift_summary"] = {}
            for shift_name in shift_names:
                summary["shift_summary"][shift_name] = {
                    "test_acc_stats": mean_ci95([r["shift_results"][shift_name]["test_acc"] for r in results]),
                    "ece_stats": mean_ci95([r["shift_results"][shift_name]["test_ece"] for r in results]),
                    "nll_stats": mean_ci95([r["shift_results"][shift_name]["test_nll"] for r in results]),
                    "brier_stats": mean_ci95([r["shift_results"][shift_name]["test_brier"] for r in results]),
                }

    return summary


def run_stable_sweep(results_root):
    configs = [cfg for cfg in create_model_configs() if cfg["name"] in MODEL_NAMES_STABLE]
    out_dir = os.path.join(results_root, "stable_sweep")
    os.makedirs(out_dir, exist_ok=True)

    all_model_summaries = []

    for cfg in configs:
        results = []
        print_summary_header(f"STABLE SWEEP: {cfg['name']}")

        for seed in SEEDS:
            results.append(run_single_seed(cfg, seed, "stable_schedule", 1.0, results_root))

        summary = aggregate_seed_results(results)
        save_results(summary, os.path.join(out_dir, f"{cfg['name']}_stable_summary.json"))
        print_aggregate_block(summary, f"STABLE SUMMARY | model={cfg['name']}")
        all_model_summaries.append(summary)

    save_results(all_model_summaries, os.path.join(out_dir, "stable_sweep_summary.json"))
    print(f"Saved: {os.path.join(out_dir, 'stable_sweep_summary.json')}")


def run_ablation(results_root):
    configs = [cfg for cfg in create_model_configs() if cfg["name"] in MODEL_NAMES_ABLATION]
    out_dir = os.path.join(results_root, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    all_ablation = []

    for cfg in configs:
        model_block = {"model_name": cfg["name"], "conditions": []}

        for condition_name in ABLATION_CONDITIONS:
            print_summary_header(f"ABLATION: {cfg['name']} | {condition_name}")

            results = []
            for seed in SEEDS:
                results.append(run_single_seed(cfg, seed, condition_name, 1.0, results_root))

            summary = aggregate_seed_results(results)
            save_results(summary, os.path.join(out_dir, f"{cfg['name']}_{condition_name}_summary.json"))
            print_aggregate_block(summary, f"ABLATION SUMMARY | model={cfg['name']} | condition={condition_name}")
            model_block["conditions"].append(summary)

        all_ablation.append(model_block)

    save_results(all_ablation, os.path.join(out_dir, "ablation_summary.json"))
    print(f"Saved: {os.path.join(out_dir, 'ablation_summary.json')}")


def run_data_size_sweep(results_root):
    configs = [cfg for cfg in create_model_configs() if cfg["name"] in MODEL_NAMES_DATA_SIZE]
    out_dir = os.path.join(results_root, "data_size_sweep")
    os.makedirs(out_dir, exist_ok=True)

    all_sweep = []

    for cfg in configs:
        model_block = {"model_name": cfg["name"], "fractions": []}

        for frac in TRAIN_FRACTIONS:
            print_summary_header(f"DATA SIZE SWEEP: {cfg['name']} | fraction={frac}")

            results = []
            for seed in SEEDS:
                results.append(run_single_seed(cfg, seed, "stable_schedule", frac, results_root))

            summary = aggregate_seed_results(results)
            save_results(summary, os.path.join(out_dir, f"{cfg['name']}_frac{int(frac*100):03d}_summary.json"))
            print_aggregate_block(summary, f"DATA SIZE SUMMARY | model={cfg['name']} | fraction={frac}")
            model_block["fractions"].append(summary)

        all_sweep.append(model_block)

    save_results(all_sweep, os.path.join(out_dir, "data_size_sweep_summary.json"))
    print(f"Saved: {os.path.join(out_dir, 'data_size_sweep_summary.json')}")


def main():
    results_root = make_results_root()

    manifest = {
        "dataset_name": DATASET_NAME,
        "run_tag": RUN_TAG,
        "seeds": SEEDS,
        "run_baselines": RUN_BASELINES,
        "run_stable_sweep": RUN_STABLE_SWEEP,
        "run_ablation": RUN_ABLATION,
        "run_data_size_sweep": RUN_DATA_SIZE_SWEEP,
        "run_shift_eval": RUN_SHIFT_EVAL,
        "model_names_stable": MODEL_NAMES_STABLE,
        "model_names_ablation": MODEL_NAMES_ABLATION,
        "ablation_conditions": ABLATION_CONDITIONS,
        "model_names_data_size": MODEL_NAMES_DATA_SIZE,
        "train_fractions": TRAIN_FRACTIONS,
        "shift_configs": SHIFT_CONFIGS,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE
    }
    save_results(manifest, os.path.join(results_root, "run_manifest.json"))

    print("=" * 60)
    print("SCRATCH EMPIRICAL PIPELINE")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Results: {results_root}")
    print(f"Saved: {os.path.join(results_root, 'run_manifest.json')}")

    if RUN_BASELINES:
        train_baseline_once(results_root)

    if RUN_STABLE_SWEEP:
        run_stable_sweep(results_root)

    if RUN_ABLATION:
        run_ablation(results_root)

    if RUN_DATA_SIZE_SWEEP:
        run_data_size_sweep(results_root)

    print_summary_header(f"ALL REQUESTED RUNS SAVED UNDER: {results_root}")


if __name__ == "__main__":
    main()

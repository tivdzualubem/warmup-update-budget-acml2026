#!/usr/bin/env python3

import csv
import math
from pathlib import Path


WARMUP_STEPS = 8000
TOTAL_STEPS = 50000
BASE_LR = 1e-4
LEGACY_INITIAL_LR = 1.0

MODELS = {
    "medium-4": 128,
    "large-6": 256,
}

CONDITIONS = [
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "stable_with_guard",
    "fixed_order_with_guard",
    "single_bad_first_step_then_stable",
]


def stable_lr(step):
    if step < WARMUP_STEPS:
        scale = step / WARMUP_STEPS
    else:
        progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        progress = min(max(progress, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return BASE_LR * scale


def legacy_lr(d_model, step):
    step = max(1, step)
    return (d_model ** -0.5) * min(
        step ** -0.5,
        step * (WARMUP_STEPS ** -1.5),
    )


def trace_condition(model_name, d_model, condition):
    rows = []

    entry_lr = (
        BASE_LR
        if condition == "stable_with_guard"
        else LEGACY_INITIAL_LR
    )

    for step in (1, 2):
        if condition == "buggy_failfast_guard" and step == 2:
            rows.append({
                "model": model_name,
                "condition": condition,
                "step": step,
                "entry_lr": "N/A",
                "scheduler_guard_action": "run already stopped after step-1 fail-fast detection",
                "candidate_lr_consumed": "N/A",
                "committed_lr_consumed": "N/A",
                "actual_exit_lr": "N/A",
            })
            break

        if condition == "buggy_no_guard":
            candidate_lr = entry_lr
            committed_lr = candidate_lr
            exit_lr = legacy_lr(d_model, step)
            action = "candidate step first; install legacy schedule after step"

        elif condition == "buggy_failfast_guard":
            candidate_lr = entry_lr
            committed_lr = candidate_lr
            exit_lr = legacy_lr(d_model, step)
            action = (
                "candidate step first; install legacy schedule; "
                "guard detects unsafe committed candidate and stops"
            )

        elif condition == "buggy_rescue_guard":
            candidate_lr = entry_lr

            if step == 1:
                committed_lr = legacy_lr(d_model, step)
                exit_lr = committed_lr
                action = (
                    "candidate step first; guard detects unsafe candidate; "
                    "restore model+optimizer; install corrected legacy LR; replay"
                )
            else:
                committed_lr = candidate_lr
                exit_lr = legacy_lr(d_model, step)
                action = (
                    "candidate step first; no guard trigger in evaluated runs; "
                    "install next legacy LR after step"
                )

        elif condition == "stable_with_guard":
            candidate_lr = stable_lr(step)
            committed_lr = candidate_lr
            exit_lr = candidate_lr
            action = "install stable LR before candidate step; guard does not trigger"

        elif condition == "fixed_order_with_guard":
            candidate_lr = legacy_lr(d_model, step)
            committed_lr = candidate_lr
            exit_lr = candidate_lr
            action = "install corrected legacy LR before candidate step; guard does not trigger"

        elif condition == "single_bad_first_step_then_stable":
            if step == 1:
                candidate_lr = LEGACY_INITIAL_LR
                committed_lr = candidate_lr
                exit_lr = LEGACY_INITIAL_LR
                action = (
                    "deliberately commit one initial-LR transition; "
                    "stable LR is computed but not installed after the step"
                )
            else:
                candidate_lr = stable_lr(step)
                committed_lr = candidate_lr
                exit_lr = candidate_lr
                action = "install stable LR before candidate step"

        else:
            raise ValueError(condition)

        rows.append({
            "model": model_name,
            "condition": condition,
            "step": step,
            "entry_lr": entry_lr,
            "scheduler_guard_action": action,
            "candidate_lr_consumed": candidate_lr,
            "committed_lr_consumed": committed_lr,
            "actual_exit_lr": exit_lr,
        })

        entry_lr = exit_lr

    return rows


def main():
    output = Path(
        "results/rebuttal_2026/scheduler_trace/two_step_scheduler_trace.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name, d_model in MODELS.items():
        for condition in CONDITIONS:
            rows.extend(
                trace_condition(
                    model_name,
                    d_model,
                    condition,
                )
            )

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "condition",
                "step",
                "entry_lr",
                "scheduler_guard_action",
                "candidate_lr_consumed",
                "committed_lr_consumed",
                "actual_exit_lr",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 110)
    print("CORRECTED TWO-STEP SCHEDULER TRACE")
    print("=" * 110)

    for row in rows:
        print(
            f"{row['model']:<8} "
            f"{row['condition']:<34} "
            f"step={row['step']} | "
            f"entry={row['entry_lr']} | "
            f"candidate={row['candidate_lr_consumed']} | "
            f"committed={row['committed_lr_consumed']} | "
            f"exit={row['actual_exit_lr']}"
        )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

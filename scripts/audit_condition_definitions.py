#!/usr/bin/env python3

import csv
import math
from pathlib import Path

BASE_LR = 1e-4
LEGACY_INITIAL_LR = 1.0
WARMUP_STEPS = 8000
TOTAL_STEPS = 50000

MODELS = {
    "medium-4": 128,
    "large-6": 256,
}


def stable_lr(t):
    if t < WARMUP_STEPS:
        return BASE_LR * (t / WARMUP_STEPS)

    progress = (t - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def legacy_lr(d_model, t):
    t = max(1, t)
    return (d_model ** -0.5) * min(
        t ** -0.5,
        t * (WARMUP_STEPS ** -1.5),
    )


def main():
    out_dir = Path("results/rebuttal_2026/condition_definitions")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for model, d_model in MODELS.items():
        stable1 = stable_lr(1)
        legacy1 = legacy_lr(d_model, 1)

        definitions = [
            {
                "condition": "buggy_no_guard",
                "role": "Legacy/no-guard; member of clean order-only causal pair",
                "schedule_family": "legacy Transformer",
                "timing": "install alpha_legacy(t) after candidate transition",
                "initial_optimizer_lr": LEGACY_INITIAL_LR,
                "step1_candidate_lr": LEGACY_INITIAL_LR,
                "step1_committed_lr": LEGACY_INITIAL_LR,
            },
            {
                "condition": "buggy_failfast_guard",
                "role": "Legacy schedule with post-candidate fail-fast diagnostic",
                "schedule_family": "legacy Transformer",
                "timing": "candidate first; detect unsafe update after transition; stop",
                "initial_optimizer_lr": LEGACY_INITIAL_LR,
                "step1_candidate_lr": LEGACY_INITIAL_LR,
                "step1_committed_lr": LEGACY_INITIAL_LR,
            },
            {
                "condition": "buggy_rescue_guard",
                "role": "Legacy schedule with rollback-and-replay rescue",
                "schedule_family": "legacy Transformer",
                "timing": "candidate first; on unsafe update restore state and replay with alpha_legacy(t)",
                "initial_optimizer_lr": LEGACY_INITIAL_LR,
                "step1_candidate_lr": LEGACY_INITIAL_LR,
                "step1_committed_lr": legacy1,
            },
            {
                "condition": "stable_with_guard",
                "role": "Independent safe reference; not the order-only control",
                "schedule_family": "stable linear-warmup/cosine",
                "timing": "install alpha_stable(t) before candidate transition",
                "initial_optimizer_lr": BASE_LR,
                "step1_candidate_lr": stable1,
                "step1_committed_lr": stable1,
            },
            {
                "condition": "fixed_order_with_guard",
                "role": "Corrected member of clean order-only causal pair",
                "schedule_family": "legacy Transformer",
                "timing": "install alpha_legacy(t) before candidate transition",
                "initial_optimizer_lr": LEGACY_INITIAL_LR,
                "step1_candidate_lr": legacy1,
                "step1_committed_lr": legacy1,
            },
            {
                "condition": "single_bad_first_step_then_stable",
                "role": "Single-impulse ablation",
                "schedule_family": "one initial-LR transition, then stable schedule",
                "timing": "commit step 1 at initial LR; install alpha_stable(t) before later transitions",
                "initial_optimizer_lr": LEGACY_INITIAL_LR,
                "step1_candidate_lr": LEGACY_INITIAL_LR,
                "step1_committed_lr": LEGACY_INITIAL_LR,
            },
        ]

        for row in definitions:
            rows.append({
                "model": model,
                "d_model": d_model,
                **row,
                "alpha_stable_step1": stable1,
                "alpha_legacy_step1": legacy1,
            })

    out_csv = out_dir / "condition_lr_documentation.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("CORRECTED CONDITION / LR DOCUMENTATION")
    print("=" * 100)

    for model, d_model in MODELS.items():
        print(f"\n{model} (d_model={d_model})")
        print(f"  stable step-1 LR : {stable_lr(1):.15g}")
        print(f"  legacy step-1 LR : {legacy_lr(d_model, 1):.15g}")
        print(f"  legacy step-2 LR : {legacy_lr(d_model, 2):.15g}")

    print("\nClean order-only causal pair:")
    print("  buggy_no_guard  <->  fixed_order_with_guard")
    print("  same legacy formula; schedule installed after vs before optimizer transition")

    print("\nStable role:")
    print("  independent safe reference using a different schedule family")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

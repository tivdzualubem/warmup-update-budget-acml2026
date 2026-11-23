"""Execute the archived scratch step protocol on a three-coordinate CPU fixture.

Run: python examples/first_two_steps.py --output evidence/verification/first_two_steps.json

This imports the frozen runner's step policy, scheduler and Adam constructor;
optimizer hooks record rates at actual Adam.step calls. It loads no dataset or
checkpoint and performs no research training. Float64 makes tiny displacements
readable; the archived research runs used fp32. The fixture's threshold is a
demonstration value, not an estimated or recalibrated research threshold.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/warmup-update-budget-acml2026"
sys.path.insert(0, str(SOURCE))
from publication_expansion.src.scratch_job_runner import (  # noqa: E402
    initial_lr_for_condition, step_semantics,
)
from publication_expansion.src.scratch_model_adapter import (  # noqa: E402
    build_scratch_adam, get_scratch_model_config,
)
from publication_expansion.src.scheduler_semantics import stable_lr  # noqa: E402
from publication_expansion.src.transition_executor import transactional_step  # noqa: E402

CONDITIONS = (
    "stable_with_guard", "single_bad_first_step_then_stable",
    "fixed_order_with_guard", "buggy_no_guard",
    "buggy_rescue_guard", "buggy_failfast_guard",
)
GRADIENTS = ([0.25, -0.5, 0.0], [-0.4, 0.1, 0.2])
DEMONSTRATION_THRESHOLD = 0.01


class ThreeCoordinates(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))


def state(model, optimizer):
    """Return readable parameter and Adam state, without pickle or tensors."""
    current = optimizer.state.get(model.weight, {})
    return {
        "weights": model.weight.detach().tolist(),
        "step": float(current["step"].item()) if "step" in current else 0.,
        "exp_avg": current["exp_avg"].tolist() if "exp_avg" in current else None,
        "exp_avg_sq": current["exp_avg_sq"].tolist() if "exp_avg_sq" in current else None,
    }


def instrument(model, optimizer):
    """Observe actual optimizer calls, including rejected candidates and replays."""
    calls = []

    def before(opt, args, kwargs):
        calls.append({"lr_consumed": [float(g["lr"]) for g in opt.param_groups],
                      "before": state(model, opt)})

    def after(opt, args, kwargs):
        calls[-1]["after"] = state(model, opt)

    optimizer.register_step_pre_hook(before)
    optimizer.register_step_post_hook(after)
    return calls


def trace_condition(model_name, condition):
    width = get_scratch_model_config(model_name)["d_model"]
    model = ThreeCoordinates()
    optimizer = build_scratch_adam(model, initial_lr=initial_lr_for_condition(condition))
    calls = instrument(model, optimizer)
    rows = []
    for step, gradient in enumerate(GRADIENTS, start=1):
        model.weight.grad = torch.tensor(gradient, dtype=torch.float64)
        entry = [float(g["lr"]) for g in optimizer.param_groups]
        policy = step_semantics(condition=condition, step=step, d_model=width,
                                threshold=DEMONSTRATION_THRESHOLD)
        count_before = len(calls)
        result = transactional_step(model=model, optimizer=optimizer,
                                    intended_lr=policy["executor_intended_lr"],
                                    threshold=policy["threshold"], mode=policy["mode"])
        rows.append({"model_configuration": model_name, "d_model": width,
                     "condition": condition, "transition": step, "entry_lr": entry,
                     "prescribed_lr": policy["trace_intended_lr"], "action": policy["action"],
                     "optimizer_calls": calls[count_before:], "result": asdict(result),
                     "retained_state": state(model, optimizer)})
        if result.stopped:
            break
    return rows


def assertion_example():
    model = ThreeCoordinates()
    optimizer = build_scratch_adam(model, initial_lr=1.)
    model.weight.grad = torch.tensor(GRADIENTS[0], dtype=torch.float64)
    calls = instrument(model, optimizer)
    before = state(model, optimizer)
    result = transactional_step(model=model, optimizer=optimizer, intended_lr=stable_lr(1),
                                threshold=None, mode="lr_assertion")
    return {"before": before, "after": state(model, optimizer), "optimizer_calls": calls,
            "actual_consumed_lr": None, "archived_result": asdict(result),
            "note": "The archived candidate_lr field contains the rejected entry LR; no step consumed it."}


def zero_lr_example():
    model = ThreeCoordinates()
    optimizer = build_scratch_adam(model, initial_lr=1e-4)
    for gradient in GRADIENTS:
        model.weight.grad = torch.tensor(gradient, dtype=torch.float64)
        optimizer.step()
    before = state(model, optimizer)
    # This is the schedule's boundary value, not a recovered 49,999-step state.
    model.weight.grad = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    optimizer.param_groups[0]["lr"] = stable_lr(50000)
    calls = instrument(model, optimizer)
    optimizer.step()
    return {"schedule_index": 50000, "before": before, "after": state(model, optimizer),
            "optimizer_calls": calls, "note": "Three toy updates total; no long prefix was run."}


def generate_report():
    sources = ["publication_expansion/src/scratch_job_runner.py",
               "publication_expansion/src/scratch_model_adapter.py",
               "publication_expansion/src/scheduler_semantics.py",
               "publication_expansion/src/transition_executor.py",
               "publication_expansion/configs/hyperparameters_v1.json",
               "src/transformer_project/models/transformer.py"]
    configuration = json.loads((SOURCE/"publication_expansion/configs/hyperparameters_v1.json").read_text())
    return {"scope": "Fixed-gradient CPU float64 example; not research training or original-device reproduction",
            "python": platform.python_version(), "torch": torch.__version__,
            "frozen_scratch_configuration": configuration["scratch_controlled"],
            "model_widths": {m: get_scratch_model_config(m)["d_model"] for m in ("medium-4", "large-6")},
            "fixture_threshold": DEMONSTRATION_THRESHOLD,
            "source_sha256": {p: hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in sources},
            "traces": [row for model in ("medium-4", "large-6")
                       for condition in CONDITIONS for row in trace_condition(model, condition)],
            "lr_assertion": assertion_example(), "zero_lr": zero_lr_example()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = generate_report()
    for row in report["traces"]:
        result = row["result"]
        print(f'{row["model_configuration"]:8} {row["condition"]:34} t={row["transition"]} '
              f'entry={row["entry_lr"]} candidate={result["candidate_lr_by_group"]} '
              f'committed={result["committed_lr_by_group"]} exit={result["actual_exit_lr_by_group"]} '
              f'calls={len(row["optimizer_calls"])} retained_counter={row["retained_state"]["step"]}')
    print("LR assertion: zero optimizer calls; zero LR: unchanged weights, changed moments/counter.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

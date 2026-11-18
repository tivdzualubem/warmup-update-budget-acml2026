"""Pure scheduler and condition semantics for the publication expansion.

No model, optimizer, dataset, or device dependency is allowed here.
Optimizer steps use one-based indexing throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


SCRATCH_WARMUP_STEPS = 8000
SCRATCH_TOTAL_STEPS = 50000
SCRATCH_BASE_LR = 1e-4
LEGACY_INITIAL_LR = 1.0


def _require_step(t: int) -> None:
    if not isinstance(t, int) or isinstance(t, bool) or t < 1:
        raise ValueError("optimizer step t must be an integer >= 1")


def legacy_transformer_lr(
    t: int,
    d_model: int,
    warmup_steps: int = SCRATCH_WARMUP_STEPS,
) -> float:
    """Original one-based Transformer schedule L(t)."""
    _require_step(t)

    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")

    return (d_model ** -0.5) * min(
        t ** -0.5,
        t * (warmup_steps ** -1.5),
    )


def stable_lr(
    t: int,
    base_lr: float = SCRATCH_BASE_LR,
    warmup_steps: int = SCRATCH_WARMUP_STEPS,
    total_steps: int = SCRATCH_TOTAL_STEPS,
) -> float:
    """Stable linear-warmup/cosine schedule S(t)."""
    _require_step(t)

    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps")

    if t <= warmup_steps:
        return base_lr * t / warmup_steps

    if t >= total_steps:
        return 0.0

    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def pretrained_warmup_steps(
    total_steps: int,
    warmup_ratio: float = 0.10,
) -> int:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if not (0.0 < warmup_ratio < 1.0):
        raise ValueError("warmup_ratio must lie strictly between 0 and 1")

    warmup_steps = math.ceil(total_steps * warmup_ratio)

    if warmup_steps >= total_steps:
        raise ValueError(
            "warmup_steps must be smaller than total_steps"
        )

    return warmup_steps


def pretrained_linear_lr(
    t: int,
    peak_lr: float,
    total_steps: int,
    warmup_ratio: float = 0.10,
) -> float:
    """One-based linear-warmup/linear-decay schedule P(t)."""
    _require_step(t)

    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")

    warmup_steps = pretrained_warmup_steps(
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
    )

    if t <= warmup_steps:
        return peak_lr * t / warmup_steps

    if t >= total_steps:
        return 0.0

    return peak_lr * (total_steps - t) / (
        total_steps - warmup_steps
    )


@dataclass(frozen=True)
class TransitionPlan:
    """Expected LR/state semantics for one optimizer transition."""

    condition: str
    step: int
    entry_lr: float
    intended_lr: float
    install_before_step: bool
    candidate_lr_consumed: Optional[float]
    guard_mode: str
    abort_before_update: bool = False


@dataclass(frozen=True)
class TransitionResolution:
    """Expected committed semantics after candidate evaluation."""

    committed_lr_consumed: Optional[float]
    actual_exit_lr: float
    stop: bool
    rollback: bool
    replay: bool


SCRATCH_CONDITIONS = {
    "stable_with_guard",
    "fixed_order_with_guard",
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "single_bad_first_step_then_stable",
}

PRETRAINED_CONDITIONS = {
    "safe_correct_order",
    "late_schedule_no_guard",
    "single_bad_then_correct",
    "update_guard_failfast",
    "update_guard_rescue",
    "executed_lr_assertion",
}


def scratch_transition_plan(
    *,
    condition: str,
    t: int,
    entry_lr: float,
    d_model: int,
) -> TransitionPlan:
    """Return the frozen scratch condition semantics for step t."""
    _require_step(t)

    if condition not in SCRATCH_CONDITIONS:
        raise ValueError(f"unknown scratch condition: {condition}")

    legacy = legacy_transformer_lr(t, d_model)
    stable = stable_lr(t)

    if condition == "stable_with_guard":
        return TransitionPlan(
            condition, t, entry_lr, stable,
            True, stable, "monitor_only"
        )

    if condition == "fixed_order_with_guard":
        return TransitionPlan(
            condition, t, entry_lr, legacy,
            True, legacy, "monitor_only"
        )

    if condition == "buggy_no_guard":
        return TransitionPlan(
            condition, t, entry_lr, legacy,
            False, entry_lr, "none"
        )

    if condition == "buggy_failfast_guard":
        return TransitionPlan(
            condition, t, entry_lr, legacy,
            False, entry_lr, "failfast"
        )

    if condition == "buggy_rescue_guard":
        return TransitionPlan(
            condition, t, entry_lr, legacy,
            False, entry_lr, "rescue"
        )

    # single_bad_first_step_then_stable
    if t == 1:
        return TransitionPlan(
            condition, t, entry_lr, stable,
            False, entry_lr, "none"
        )

    return TransitionPlan(
        condition, t, entry_lr, stable,
        True, stable, "none"
    )


def pretrained_transition_plan(
    *,
    condition: str,
    t: int,
    entry_lr: float,
    peak_lr: float,
    total_steps: int,
    warmup_ratio: float = 0.10,
) -> TransitionPlan:
    """Return the frozen pretrained condition semantics for step t."""
    _require_step(t)

    if condition not in PRETRAINED_CONDITIONS:
        raise ValueError(
            f"unknown pretrained condition: {condition}"
        )

    intended = pretrained_linear_lr(
        t=t,
        peak_lr=peak_lr,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
    )

    if condition == "safe_correct_order":
        return TransitionPlan(
            condition, t, entry_lr, intended,
            True, intended, "none"
        )

    if condition == "late_schedule_no_guard":
        return TransitionPlan(
            condition, t, entry_lr, intended,
            False, entry_lr, "none"
        )

    if condition == "single_bad_then_correct":
        if t == 1:
            return TransitionPlan(
                condition, t, entry_lr, intended,
                False, entry_lr, "none"
            )

        return TransitionPlan(
            condition, t, entry_lr, intended,
            True, intended, "none"
        )

    if condition == "update_guard_failfast":
        return TransitionPlan(
            condition, t, entry_lr, intended,
            False, entry_lr, "failfast"
        )

    if condition == "update_guard_rescue":
        return TransitionPlan(
            condition, t, entry_lr, intended,
            False, entry_lr, "rescue"
        )

    # executed_lr_assertion
    mismatch = not math.isclose(
        entry_lr,
        intended,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )

    return TransitionPlan(
        condition=condition,
        step=t,
        entry_lr=entry_lr,
        intended_lr=intended,
        install_before_step=False,
        candidate_lr_consumed=None if mismatch else entry_lr,
        guard_mode="pre_update_assertion",
        abort_before_update=mismatch,
    )


def resolve_transition(
    plan: TransitionPlan,
    *,
    guard_triggered: bool = False,
) -> TransitionResolution:
    """Resolve candidate -> committed semantics.

    This function models LR/state ordering only. Actual model and optimizer
    rollback/replay will be implemented separately and tested against this
    contract.
    """
    if plan.abort_before_update:
        return TransitionResolution(
            committed_lr_consumed=None,
            actual_exit_lr=plan.entry_lr,
            stop=True,
            rollback=False,
            replay=False,
        )

    if plan.guard_mode == "failfast" and guard_triggered:
        return TransitionResolution(
            committed_lr_consumed=plan.candidate_lr_consumed,
            actual_exit_lr=plan.intended_lr,
            stop=True,
            rollback=False,
            replay=False,
        )

    if plan.guard_mode == "rescue" and guard_triggered:
        return TransitionResolution(
            committed_lr_consumed=plan.intended_lr,
            actual_exit_lr=plan.intended_lr,
            stop=False,
            rollback=True,
            replay=True,
        )

    if plan.install_before_step:
        exit_lr = plan.intended_lr
    elif (
        plan.condition
        in {
            "single_bad_first_step_then_stable",
            "single_bad_then_correct",
        }
        and plan.step == 1
    ):
        # Frozen correction: step 1 genuinely exits at the entry LR.
        exit_lr = plan.entry_lr
    else:
        # Late schedule installation occurs after candidate transition.
        exit_lr = plan.intended_lr

    return TransitionResolution(
        committed_lr_consumed=plan.candidate_lr_consumed,
        actual_exit_lr=exit_lr,
        stop=False,
        rollback=False,
        replay=False,
    )

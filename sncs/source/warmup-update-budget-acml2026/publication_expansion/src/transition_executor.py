"""Transactional optimizer-step utilities.

This module implements candidate-update measurement, fail-fast behavior,
rollback/replay rescue, and pre-update LR assertions independently of any
dataset or model architecture.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclass
class StepSnapshot:
    model_state: dict
    optimizer_state: dict
    gradients: Dict[str, Optional[torch.Tensor]]


@dataclass
class StepResult:
    candidate_lr_by_group: List[float]
    committed_lr_by_group: Optional[List[float]]
    actual_exit_lr_by_group: List[float]
    candidate_update_ratio: Optional[float]
    committed_update_ratio: Optional[float]
    guard_triggered: bool
    stopped: bool
    rolled_back: bool
    replayed: bool


def clone_trainable_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def clone_gradients(model: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    return {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def restore_gradients(
    model: nn.Module,
    gradients: Dict[str, Optional[torch.Tensor]],
) -> None:
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        grad = gradients[name]
        p.grad = None if grad is None else grad.to(
            device=p.device,
            dtype=p.dtype,
        ).clone()


def snapshot_step_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> StepSnapshot:
    return StepSnapshot(
        model_state=copy.deepcopy(model.state_dict()),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        gradients=clone_gradients(model),
    )


def restore_step_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    snapshot: StepSnapshot,
) -> None:
    model.load_state_dict(snapshot.model_state)
    optimizer.load_state_dict(snapshot.optimizer_state)
    restore_gradients(model, snapshot.gradients)


def optimizer_lrs(
    optimizer: torch.optim.Optimizer,
) -> List[float]:
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def set_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def assert_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    intended_lr: float,
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> bool:
    """Return True iff every parameter-group LR matches intended_lr."""
    return all(
        math.isclose(
            float(group["lr"]),
            float(intended_lr),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        for group in optimizer.param_groups
    )


def update_ratio(
    model: nn.Module,
    before: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> float:
    update_sq = 0.0
    param_sq = 0.0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        prior = before[name].to(
            device=p.device,
            dtype=p.dtype,
        )

        update_sq += torch.sum(
            (p.detach() - prior) ** 2
        ).item()

        param_sq += torch.sum(
            prior ** 2
        ).item()

    return math.sqrt(update_sq) / (
        math.sqrt(param_sq) + eps
    )


def transactional_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    intended_lr: float,
    threshold: Optional[float],
    mode: str,
) -> StepResult:
    """Execute one frozen transition protocol.

    Modes
    -----
    safe:
        Install intended LR before optimizer.step.

    late_no_guard:
        Candidate consumes entry LR. Intended LR is installed after step.

    failfast:
        Candidate consumes entry LR. Intended LR is installed after step.
        If candidate ratio exceeds threshold, stop without rollback.

    rescue:
        Candidate consumes entry LR. If ratio exceeds threshold, restore
        model + optimizer + exact clipped gradients, install intended LR,
        and replay once.

    lr_assertion:
        Check every parameter-group LR before optimizer.step. On mismatch,
        stop before any parameter mutation.
    """

    allowed = {
        "safe",
        "late_no_guard",
        "failfast",
        "rescue",
        "lr_assertion",
    }

    if mode not in allowed:
        raise ValueError(f"unknown mode: {mode}")

    if mode in {"failfast", "rescue"} and threshold is None:
        raise ValueError(
            f"threshold is required for mode={mode}"
        )

    before = clone_trainable_params(model)
    entry_lrs = optimizer_lrs(optimizer)

    if mode == "lr_assertion":
        if not assert_optimizer_lr(
            optimizer,
            intended_lr,
        ):
            return StepResult(
                candidate_lr_by_group=entry_lrs,
                committed_lr_by_group=None,
                actual_exit_lr_by_group=entry_lrs,
                candidate_update_ratio=None,
                committed_update_ratio=None,
                guard_triggered=True,
                stopped=True,
                rolled_back=False,
                replayed=False,
            )

        optimizer.step()

        ratio = update_ratio(model, before)

        return StepResult(
            candidate_lr_by_group=entry_lrs,
            committed_lr_by_group=entry_lrs,
            actual_exit_lr_by_group=optimizer_lrs(optimizer),
            candidate_update_ratio=ratio,
            committed_update_ratio=ratio,
            guard_triggered=False,
            stopped=False,
            rolled_back=False,
            replayed=False,
        )

    if mode == "safe":
        set_optimizer_lrs(
            optimizer,
            intended_lr,
        )

        consumed = optimizer_lrs(optimizer)
        optimizer.step()
        ratio = update_ratio(model, before)

        return StepResult(
            candidate_lr_by_group=consumed,
            committed_lr_by_group=consumed,
            actual_exit_lr_by_group=optimizer_lrs(optimizer),
            candidate_update_ratio=ratio,
            committed_update_ratio=ratio,
            guard_triggered=False,
            stopped=False,
            rolled_back=False,
            replayed=False,
        )

    # Candidate transition for late/failfast/rescue.
    snapshot = snapshot_step_state(
        model,
        optimizer,
    )

    candidate_lrs = optimizer_lrs(optimizer)
    optimizer.step()

    candidate_ratio = update_ratio(
        model,
        before,
    )

    # Late scheduler installation.
    set_optimizer_lrs(
        optimizer,
        intended_lr,
    )

    if mode == "late_no_guard":
        return StepResult(
            candidate_lr_by_group=candidate_lrs,
            committed_lr_by_group=candidate_lrs,
            actual_exit_lr_by_group=optimizer_lrs(optimizer),
            candidate_update_ratio=candidate_ratio,
            committed_update_ratio=candidate_ratio,
            guard_triggered=False,
            stopped=False,
            rolled_back=False,
            replayed=False,
        )

    triggered = candidate_ratio > float(threshold)

    if mode == "failfast":
        return StepResult(
            candidate_lr_by_group=candidate_lrs,
            committed_lr_by_group=candidate_lrs,
            actual_exit_lr_by_group=optimizer_lrs(optimizer),
            candidate_update_ratio=candidate_ratio,
            committed_update_ratio=candidate_ratio,
            guard_triggered=triggered,
            stopped=triggered,
            rolled_back=False,
            replayed=False,
        )

    # Rescue.
    if not triggered:
        return StepResult(
            candidate_lr_by_group=candidate_lrs,
            committed_lr_by_group=candidate_lrs,
            actual_exit_lr_by_group=optimizer_lrs(optimizer),
            candidate_update_ratio=candidate_ratio,
            committed_update_ratio=candidate_ratio,
            guard_triggered=False,
            stopped=False,
            rolled_back=False,
            replayed=False,
        )

    restore_step_state(
        model,
        optimizer,
        snapshot,
    )

    # Exact pre-candidate gradients have now been restored.
    set_optimizer_lrs(
        optimizer,
        intended_lr,
    )

    replay_lrs = optimizer_lrs(optimizer)
    optimizer.step()

    committed_ratio = update_ratio(
        model,
        before,
    )

    return StepResult(
        candidate_lr_by_group=candidate_lrs,
        committed_lr_by_group=replay_lrs,
        actual_exit_lr_by_group=optimizer_lrs(optimizer),
        candidate_update_ratio=candidate_ratio,
        committed_update_ratio=committed_ratio,
        guard_triggered=True,
        stopped=False,
        rolled_back=True,
        replayed=True,
    )

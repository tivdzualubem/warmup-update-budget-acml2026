"""Version 2 finite-state executor with at most one revalidated replay.

This is a new execution contract, not the executor used for archived runs.
It reuses one previously computed gradient. It does not rerun a forward pass,
provide stochastic-optimizer RNG rollback, or certify CUDA/AMP/distributed use.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from typing import Optional
import torch

EXECUTOR_VERSION = 'finite_state_revalidated_replay_v2'

@dataclass
class StepSnapshot:
    model_state: dict
    optimizer_state: dict
    gradients: dict

@dataclass
class StepResult:
    candidate_lr_by_group: list[float]
    committed_lr_by_group: Optional[list[float]]
    actual_exit_lr_by_group: list[float]
    candidate_update_ratio: Optional[float]
    committed_update_ratio: Optional[float]
    guard_triggered: bool
    stopped: bool
    rolled_back: bool
    replayed: bool
    failure_reason: Optional[str] = None
    replay_update_ratio: Optional[float] = None
    optimizer_step_calls: int = 0
    committed_transitions: int = 0

def all_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True

def state_is_finite(model, optimizer):
    return all_finite(model.state_dict()) and all_finite(optimizer.state_dict())

def gradients_are_finite(model):
    return all(p.grad is None or all_finite(p.grad) for p in model.parameters())

def snapshot_step_state(model, optimizer):
    return StepSnapshot(copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
                        {n: None if p.grad is None else p.grad.detach().clone()
                         for n,p in model.named_parameters()})

def snapshot_is_finite(snapshot):
    return all_finite(snapshot.model_state) and all_finite(snapshot.optimizer_state) and all_finite(snapshot.gradients)

def restore_step_state(model, optimizer, snapshot):
    if not snapshot_is_finite(snapshot):
        raise ValueError('Refusing to restore a nonfinite snapshot.')
    model.load_state_dict(snapshot.model_state)
    optimizer.load_state_dict(snapshot.optimizer_state)
    for name,p in model.named_parameters():
        grad = snapshot.gradients[name]
        p.grad = None if grad is None else grad.to(p).clone()

def optimizer_lrs(optimizer):
    return [float(group['lr']) for group in optimizer.param_groups]

def set_optimizer_lrs(optimizer, value):
    for group in optimizer.param_groups:
        group['lr'] = float(value)

def update_ratio(model, snapshot):
    update_sq, param_sq = 0.0, 0.0
    for name,p in model.named_parameters():
        if p.requires_grad:
            prior = snapshot.model_state[name].to(p.device, dtype=torch.float64)
            delta = p.detach().to(torch.float64) - prior
            update_sq += float(torch.sum(delta * delta).item())
            param_sq += float(torch.sum(prior * prior).item())
    return math.sqrt(update_sq) / (math.sqrt(param_sq) + 1e-12)

def transactional_step(*, model, optimizer, intended_lr, threshold, mode, step=1,
                       guard_window=100, verified_snapshot=None):
    """Commit finite candidates, or replay once and enforce the same ceiling.

    Threshold enforcement is confined to steps 1..guard_window. Finite-state
    validation is always active. A rejected replay has no committed LR/ratio.
    A finite fail-fast violation is rolled back in v2 (unlike archived fail-fast).
    An invalid entry state needs an earlier verified snapshot to be restored;
    otherwise it stops before mutation and explicitly reports that limitation.
    """
    if mode not in {'safe','late_no_guard','failfast','rescue','lr_assertion'}:
        raise ValueError('Unknown execution mode.')
    if step < 1 or guard_window < 1:
        raise ValueError('Step and guard window must be positive.')
    if not math.isfinite(float(intended_lr)) or intended_lr < 0:
        raise ValueError('Intended LR must be finite and nonnegative.')
    guarded = mode in {'failfast','rescue'} and step <= guard_window
    if guarded and (threshold is None or not math.isfinite(float(threshold)) or threshold < 0):
        raise ValueError('An active guard requires a finite nonnegative threshold.')
    if verified_snapshot is not None and not snapshot_is_finite(verified_snapshot):
        raise ValueError('The supplied recovery snapshot is not finite.')
    entry = optimizer_lrs(optimizer)
    candidate = entry[:]
    candidate_ratio = None
    calls = 0
    replayed = False
    replay_ratio = None

    def failure(reason, snapshot=None, triggered=True):
        if snapshot is not None:
            restore_step_state(model,optimizer,snapshot)
        return StepResult(candidate if calls else [],None,optimizer_lrs(optimizer),candidate_ratio,None,
                          triggered,True,snapshot is not None,replayed,reason,
                          replay_ratio,calls,0)

    if not state_is_finite(model,optimizer):
        return failure('nonfinite_entry_state' if verified_snapshot else 'nonfinite_entry_state_no_verified_snapshot', verified_snapshot)
    if not gradients_are_finite(model):
        # Without an earlier snapshot the current model/optimizer are verified
        # finite, but the invalid gradient must not be saved and replayed.
        if verified_snapshot is None:
            clean = snapshot_step_state(model,optimizer)
            clean.gradients = {n:None for n in clean.gradients}
        else:
            clean = verified_snapshot
        return failure('nonfinite_gradient', clean)
    snapshot = snapshot_step_state(model,optimizer)
    if mode == 'lr_assertion' and not all(math.isclose(v,intended_lr,rel_tol=1e-12,abs_tol=1e-15) for v in entry):
        return failure('lr_mismatch_before_update', triggered=True)
    if mode == 'safe':
        set_optimizer_lrs(optimizer,intended_lr)
    candidate = optimizer_lrs(optimizer)
    try:
        calls += 1
        optimizer.step()
    except Exception:
        restore_step_state(model,optimizer,snapshot)
        raise
    if not state_is_finite(model,optimizer):
        return failure('nonfinite_candidate_state',snapshot)
    candidate_ratio = update_ratio(model,snapshot)
    if not math.isfinite(candidate_ratio):
        candidate_ratio = None
        return failure('nonfinite_candidate_ratio',snapshot)
    if mode in {'late_no_guard','failfast','rescue'}:
        set_optimizer_lrs(optimizer,intended_lr)
    triggered = bool(guarded and candidate_ratio > threshold)
    if not triggered:
        return StepResult(candidate,candidate,optimizer_lrs(optimizer),candidate_ratio,
                          candidate_ratio,False,False,False,False,None,None,calls,1)
    if mode == 'failfast':
        return failure('candidate_exceeds_threshold',snapshot)
    restore_step_state(model,optimizer,snapshot)
    set_optimizer_lrs(optimizer,intended_lr)
    replay_lr = optimizer_lrs(optimizer)
    replayed = True
    try:
        calls += 1
        optimizer.step()
    except Exception:
        restore_step_state(model,optimizer,snapshot)
        raise
    if not state_is_finite(model,optimizer):
        return failure('nonfinite_replay_state',snapshot)
    replay_ratio = update_ratio(model,snapshot)
    if not math.isfinite(replay_ratio):
        replay_ratio = None
        return failure('nonfinite_replay_ratio',snapshot)
    if replay_ratio > threshold:
        return failure('replay_exceeds_threshold',snapshot)
    return StepResult(candidate,replay_lr,optimizer_lrs(optimizer),candidate_ratio,
                      replay_ratio,True,False,True,True,None,replay_ratio,calls,1)

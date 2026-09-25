import copy
import unittest

import torch
from torch import nn

from publication_expansion.src.transition_executor import (
    clone_trainable_params,
    optimizer_lrs,
    transactional_step,
)


def make_model():
    torch.manual_seed(1234)

    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.Tanh(),
        nn.Linear(5, 2),
    ).double()

    return model


def make_adam(model, lr):
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )


def install_fixed_gradients(model):
    for i, p in enumerate(model.parameters()):
        values = torch.arange(
            1,
            p.numel() + 1,
            dtype=p.dtype,
        ).reshape_as(p)

        p.grad = (
            (i + 1) * 0.001 * values
        ).clone()


def clone_optimizer_state(optimizer):
    return copy.deepcopy(
        optimizer.state_dict()
    )


def assert_models_equal(testcase, a, b):
    a_state = a.state_dict()
    b_state = b.state_dict()

    testcase.assertEqual(
        set(a_state),
        set(b_state),
    )

    for key in a_state:
        testcase.assertTrue(
            torch.equal(
                a_state[key],
                b_state[key],
            ),
            msg=f"model mismatch at {key}",
        )


def assert_optimizer_states_equal(testcase, a, b):
    a_state = a.state_dict()
    b_state = b.state_dict()

    testcase.assertEqual(
        len(a_state["param_groups"]),
        len(b_state["param_groups"]),
    )

    for ga, gb in zip(
        a_state["param_groups"],
        b_state["param_groups"],
    ):
        testcase.assertEqual(
            ga.keys(),
            gb.keys(),
        )

        for key in ga:
            if key == "params":
                testcase.assertEqual(
                    ga[key],
                    gb[key],
                )
            else:
                testcase.assertEqual(
                    ga[key],
                    gb[key],
                )

    testcase.assertEqual(
        a_state["state"].keys(),
        b_state["state"].keys(),
    )

    for pid in a_state["state"]:
        sa = a_state["state"][pid]
        sb = b_state["state"][pid]

        testcase.assertEqual(
            sa.keys(),
            sb.keys(),
        )

        for key in sa:
            va = sa[key]
            vb = sb[key]

            if torch.is_tensor(va):
                testcase.assertTrue(
                    torch.equal(va, vb),
                    msg=f"optimizer tensor mismatch: {pid}/{key}",
                )
            else:
                testcase.assertEqual(va, vb)


class TransitionExecutorTests(unittest.TestCase):
    def test_safe_step_consumes_intended_lr(self):
        model = make_model()
        install_fixed_gradients(model)

        optimizer = make_adam(
            model,
            lr=1.0,
        )

        result = transactional_step(
            model=model,
            optimizer=optimizer,
            intended_lr=1e-4,
            threshold=None,
            mode="safe",
        )

        self.assertEqual(
            result.candidate_lr_by_group,
            [1e-4],
        )
        self.assertEqual(
            result.committed_lr_by_group,
            [1e-4],
        )
        self.assertEqual(
            result.actual_exit_lr_by_group,
            [1e-4],
        )

    def test_late_step_consumes_entry_lr(self):
        model = make_model()
        install_fixed_gradients(model)

        optimizer = make_adam(
            model,
            lr=1.0,
        )

        result = transactional_step(
            model=model,
            optimizer=optimizer,
            intended_lr=1e-4,
            threshold=None,
            mode="late_no_guard",
        )

        self.assertEqual(
            result.candidate_lr_by_group,
            [1.0],
        )
        self.assertEqual(
            result.committed_lr_by_group,
            [1.0],
        )
        self.assertEqual(
            result.actual_exit_lr_by_group,
            [1e-4],
        )

    def test_failfast_leaves_candidate_committed(self):
        model = make_model()
        install_fixed_gradients(model)

        optimizer = make_adam(
            model,
            lr=1.0,
        )

        before = clone_trainable_params(model)

        result = transactional_step(
            model=model,
            optimizer=optimizer,
            intended_lr=1e-4,
            threshold=0.0,
            mode="failfast",
        )

        self.assertTrue(result.guard_triggered)
        self.assertTrue(result.stopped)
        self.assertFalse(result.rolled_back)
        self.assertFalse(result.replayed)

        changed = any(
            not torch.equal(
                p.detach(),
                before[name],
            )
            for name, p in model.named_parameters()
            if p.requires_grad
        )

        self.assertTrue(changed)

    def test_rescue_exactly_matches_clean_safe_step(self):
        # Branch from exactly the same initial model.
        base = make_model()

        safe_model = copy.deepcopy(base)
        rescue_model = copy.deepcopy(base)

        install_fixed_gradients(safe_model)
        install_fixed_gradients(rescue_model)

        safe_optimizer = make_adam(
            safe_model,
            lr=1.0,
        )

        rescue_optimizer = make_adam(
            rescue_model,
            lr=1.0,
        )

        safe_result = transactional_step(
            model=safe_model,
            optimizer=safe_optimizer,
            intended_lr=1e-4,
            threshold=None,
            mode="safe",
        )

        rescue_result = transactional_step(
            model=rescue_model,
            optimizer=rescue_optimizer,
            intended_lr=1e-4,
            threshold=0.0,
            mode="rescue",
        )

        self.assertTrue(
            rescue_result.guard_triggered
        )
        self.assertTrue(
            rescue_result.rolled_back
        )
        self.assertTrue(
            rescue_result.replayed
        )

        assert_models_equal(
            self,
            safe_model,
            rescue_model,
        )

        assert_optimizer_states_equal(
            self,
            safe_optimizer,
            rescue_optimizer,
        )

        self.assertEqual(
            safe_result.committed_lr_by_group,
            rescue_result.committed_lr_by_group,
        )

        self.assertEqual(
            safe_result.actual_exit_lr_by_group,
            rescue_result.actual_exit_lr_by_group,
        )

        self.assertEqual(
            safe_result.committed_update_ratio,
            rescue_result.committed_update_ratio,
        )

    def test_rescue_restores_same_gradients_for_replay(self):
        base = make_model()
        model = copy.deepcopy(base)

        install_fixed_gradients(model)

        original_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        optimizer = make_adam(
            model,
            lr=1.0,
        )

        transactional_step(
            model=model,
            optimizer=optimizer,
            intended_lr=1e-4,
            threshold=0.0,
            mode="rescue",
        )

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            self.assertTrue(
                torch.equal(
                    p.grad,
                    original_grads[name],
                ),
                msg=f"gradient mismatch after rescue: {name}",
            )

    def test_lr_assertion_aborts_without_mutation(self):
        model = make_model()
        install_fixed_gradients(model)

        optimizer = make_adam(
            model,
            lr=1.0,
        )

        before_model = copy.deepcopy(
            model.state_dict()
        )

        before_optimizer = clone_optimizer_state(
            optimizer
        )

        result = transactional_step(
            model=model,
            optimizer=optimizer,
            intended_lr=1e-4,
            threshold=None,
            mode="lr_assertion",
        )

        self.assertTrue(result.guard_triggered)
        self.assertTrue(result.stopped)
        self.assertIsNone(
            result.committed_lr_by_group
        )

        for key, value in model.state_dict().items():
            self.assertTrue(
                torch.equal(
                    value,
                    before_model[key],
                )
            )

        self.assertEqual(
            optimizer.state_dict(),
            before_optimizer,
        )

    def test_lr_assertion_accepts_matching_lr(self):
        model = make_model()
        install_fixed_gradients(model)

        optimizer = make_adam(
            model,
            lr=1e-4,
        )

        result = transactional_step(
            model=model,
            optimizer=optimizer,
            intended_lr=1e-4,
            threshold=None,
            mode="lr_assertion",
        )

        self.assertFalse(result.guard_triggered)
        self.assertFalse(result.stopped)
        self.assertEqual(
            result.committed_lr_by_group,
            [1e-4],
        )

    def test_all_parameter_groups_are_checked(self):
        model = make_model()
        install_fixed_gradients(model)

        params = list(model.parameters())

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": params[:2],
                    "lr": 1e-4,
                },
                {
                    "params": params[2:],
                    "lr": 1e-3,
                },
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        before = copy.deepcopy(
            model.state_dict()
        )

        result = transactional_step(
            model=model,
            optimizer=optimizer,
            intended_lr=1e-4,
            threshold=None,
            mode="lr_assertion",
        )

        self.assertTrue(result.stopped)

        for key, value in model.state_dict().items():
            self.assertTrue(
                torch.equal(
                    value,
                    before[key],
                )
            )

        self.assertEqual(
            optimizer_lrs(optimizer),
            [1e-4, 1e-3],
        )


if __name__ == "__main__":
    unittest.main()

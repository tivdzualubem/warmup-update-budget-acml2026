import unittest

import numpy as np
import torch

from publication_expansion.src.pretrained_job_runner import (
    GUARD_WINDOW,
    linear_warmup_decay_lr,
    macro_f1,
    primary_metric_value,
    transition_spec,
    uniform_group_lr,
)


class PretrainedRunnerTests(unittest.TestCase):
    def test_linear_schedule_boundaries(self):
        peak = 2e-5
        warmup = 100
        total = 1000

        self.assertAlmostEqual(
            linear_warmup_decay_lr(
                1,
                peak_lr=peak,
                warmup_steps=warmup,
                total_steps=total,
            ),
            peak / 100,
        )

        self.assertAlmostEqual(
            linear_warmup_decay_lr(
                100,
                peak_lr=peak,
                warmup_steps=warmup,
                total_steps=total,
            ),
            peak,
        )

        self.assertAlmostEqual(
            linear_warmup_decay_lr(
                1000,
                peak_lr=peak,
                warmup_steps=warmup,
                total_steps=total,
            ),
            0.0,
        )

    def test_late_and_safe_have_same_intended_target(self):
        intended = 1e-7

        safe = transition_spec(
            condition="safe_correct_order",
            step=1,
            intended_lr=intended,
            threshold=None,
        )

        late = transition_spec(
            condition="late_schedule_no_guard",
            step=1,
            intended_lr=intended,
            threshold=None,
        )

        self.assertEqual(
            safe["mode"],
            "safe",
        )

        self.assertEqual(
            late["mode"],
            "late_no_guard",
        )

    def test_single_bad_only_step_one(self):
        first = transition_spec(
            condition="single_bad_then_correct",
            step=1,
            intended_lr=1e-7,
            threshold=None,
        )

        second = transition_spec(
            condition="single_bad_then_correct",
            step=2,
            intended_lr=2e-7,
            threshold=None,
        )

        self.assertEqual(
            first["mode"],
            "late_no_guard",
        )

        self.assertEqual(
            second["mode"],
            "safe",
        )

    def test_guard_window(self):
        early = transition_spec(
            condition="update_guard_rescue",
            step=GUARD_WINDOW,
            intended_lr=1e-7,
            threshold=1e-4,
        )

        late = transition_spec(
            condition="update_guard_rescue",
            step=GUARD_WINDOW + 1,
            intended_lr=1e-7,
            threshold=1e-4,
        )

        self.assertEqual(
            early["mode"],
            "rescue",
        )

        self.assertEqual(
            late["mode"],
            "late_no_guard",
        )

    def test_assertion_condition_is_explicit(self):
        spec = transition_spec(
            condition="executed_lr_assertion",
            step=1,
            intended_lr=1e-7,
            threshold=None,
        )

        self.assertEqual(
            spec["mode"],
            "assertion",
        )

    def test_macro_f1(self):
        labels = np.array(
            [0, 0, 1, 1]
        )

        preds = np.array(
            [0, 1, 1, 1]
        )

        score = macro_f1(
            labels,
            preds,
            2,
        )

        expected = (
            (2 / 3)
            + (4 / 5)
        ) / 2

        self.assertAlmostEqual(
            score,
            expected,
        )

    def test_primary_metrics(self):
        self.assertEqual(
            primary_metric_value(
                "ag_news",
                {
                    "macro_f1": 0.8,
                    "accuracy": 0.9,
                },
            ),
            0.8,
        )

        self.assertEqual(
            primary_metric_value(
                "wnut_17",
                {
                    "entity_f1": 0.5,
                },
            ),
            0.5,
        )

        self.assertEqual(
            primary_metric_value(
                "cifar10",
                {
                    "accuracy": 0.95,
                },
            ),
            0.95,
        )

    def test_uniform_lr_requires_matching_groups(self):
        p1 = torch.nn.Parameter(
            torch.tensor([1.0])
        )
        p2 = torch.nn.Parameter(
            torch.tensor([2.0])
        )

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p1],
                    "lr": 2e-5,
                },
                {
                    "params": [p2],
                    "lr": 2e-5,
                },
            ]
        )

        self.assertEqual(
            uniform_group_lr(
                optimizer
            ),
            2e-5,
        )

        optimizer.param_groups[
            1
        ]["lr"] = 3e-5

        with self.assertRaises(
            RuntimeError
        ):
            uniform_group_lr(
                optimizer
            )


if __name__ == "__main__":
    unittest.main()

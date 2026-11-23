"""Behavioral checks of actual archived optimizer calls on the CPU fixture."""
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.first_two_steps import (generate_report, GRADIENTS, ThreeCoordinates, build_scratch_adam)
from publication_expansion.src.scheduler_semantics import stable_lr, legacy_transformer_lr


class TestActualFirstTwoSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = generate_report()

    def rows(self, model, condition):
        return [r for r in self.report["traces"]
                if r["model_configuration"] == model and r["condition"] == condition]

    def test_frozen_constructor_and_schedule_agree_with_configuration(self):
        cfg = self.report["frozen_scratch_configuration"]
        options = build_scratch_adam(ThreeCoordinates(), initial_lr=1.).param_groups[0]
        self.assertEqual(options["betas"], (cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]))
        self.assertEqual(options["eps"], cfg["optimizer"]["epsilon"])
        self.assertEqual(options["weight_decay"], 0.)
        self.assertEqual(cfg["stable_schedule"]["warmup_steps"], 8000)
        self.assertEqual(cfg["stable_schedule"]["total_steps"], 50000)
        self.assertEqual(self.report["model_widths"], {"medium-4":128, "large-6":256})
        self.assertAlmostEqual(stable_lr(1), 1.25e-8, places=22)
        self.assertEqual(stable_lr(8000), 1e-4)
        self.assertAlmostEqual(stable_lr(29000), 5e-5, places=18)
        self.assertAlmostEqual(legacy_transformer_lr(1,128), 1.2352647110032732e-7, places=20)

    def test_rates_at_actual_optimizer_calls(self):
        for model, width in (("medium-4", 128), ("large-6", 256)):
            s1, s2 = stable_lr(1), stable_lr(2)
            l1, l2 = legacy_transformer_lr(1, width), legacy_transformer_lr(2, width)
            expected = {"stable_with_guard": [[s1], [s2]],
                        "single_bad_first_step_then_stable": [[1.], [s2]],
                        "fixed_order_with_guard": [[l1], [l2]],
                        "buggy_no_guard": [[1.], [l1]],
                        "buggy_rescue_guard": [[1., l1], [l1]],
                        "buggy_failfast_guard": [[1.]]}
            for condition, steps in expected.items():
                rows = self.rows(model, condition)
                self.assertEqual(len(rows), len(steps))
                for row, rates in zip(rows, steps):
                    self.assertEqual(row["d_model"], width)
                    self.assertEqual([c["lr_consumed"][0] for c in row["optimizer_calls"]], rates)
                    self.assertEqual(row["result"]["candidate_lr_by_group"], [rates[0]])
                    self.assertEqual(row["result"]["committed_lr_by_group"], [rates[-1]])

    def test_first_adam_formula_and_moments_independent_of_rate(self):
        gradient = torch.tensor(GRADIENTS[0], dtype=torch.float64)
        for row in self.report["traces"]:
            if row["transition"] != 1:
                continue
            for call in row["optimizer_calls"]:
                lr = call["lr_consumed"][0]
                before = torch.tensor(call["before"]["weights"], dtype=torch.float64)
                after = torch.tensor(call["after"]["weights"], dtype=torch.float64)
                expected = before - lr * gradient / (gradient.abs() + 1e-9)
                torch.testing.assert_close(after, expected, rtol=0, atol=5e-16)
                first = torch.tensor(call["after"]["exp_avg"], dtype=torch.float64)
                second = torch.tensor(call["after"]["exp_avg_sq"], dtype=torch.float64)
                torch.testing.assert_close(first, (1-.9)*gradient, rtol=1e-14, atol=1e-16)
                torch.testing.assert_close(second, (1-.98)*gradient.square(), rtol=1e-14, atol=1e-16)
                self.assertLessEqual(float(torch.linalg.vector_norm(after-before)), lr*math.sqrt(3)+1e-15)

    def test_rescue_restores_counter_and_matches_clean_first_transition(self):
        for model in ("medium-4", "large-6"):
            rescued = self.rows(model, "buggy_rescue_guard")
            clean = self.rows(model, "fixed_order_with_guard")
            self.assertEqual(rescued[0]["retained_state"], clean[0]["retained_state"])
            self.assertEqual(rescued[0]["retained_state"]["step"], 1.)
            self.assertEqual(rescued[0]["optimizer_calls"][1]["before"]["step"], 0.)
            self.assertEqual(rescued[1]["retained_state"]["step"], 2.)
            # Repair at t=1 does not remove the later schedule lag.
            self.assertNotEqual(rescued[1]["retained_state"]["weights"], clean[1]["retained_state"]["weights"])

    def test_actual_exit_lr_is_distinct_from_target_for_isolated_fault(self):
        row = self.rows("medium-4", "single_bad_first_step_then_stable")[0]
        self.assertEqual(row["prescribed_lr"], stable_lr(1))
        self.assertEqual(row["result"]["actual_exit_lr_by_group"], [1.])
        late = self.rows("medium-4", "buggy_no_guard")[0]
        self.assertNotEqual(late["result"]["committed_lr_by_group"], late["result"]["actual_exit_lr_by_group"])

    def test_archived_failfast_retains_mutation_and_assertion_prevents_it(self):
        failed = self.rows("medium-4", "buggy_failfast_guard")[0]
        self.assertTrue(failed["result"]["stopped"])
        self.assertFalse(failed["result"]["rolled_back"])
        self.assertNotEqual(failed["optimizer_calls"][0]["before"]["weights"], failed["retained_state"]["weights"])
        asserted = self.report["lr_assertion"]
        self.assertEqual(asserted["optimizer_calls"], [])
        self.assertEqual(asserted["before"], asserted["after"])
        self.assertIsNone(asserted["actual_consumed_lr"])

    def test_zero_lr_changes_optimizer_state_without_parameter_training(self):
        zero = self.report["zero_lr"]
        self.assertEqual(zero["optimizer_calls"][0]["lr_consumed"], [0.])
        self.assertEqual(zero["before"]["weights"], zero["after"]["weights"])
        self.assertEqual(zero["after"]["step"], zero["before"]["step"]+1)
        self.assertNotEqual(zero["before"]["exp_avg"], zero["after"]["exp_avg"])
        self.assertNotEqual(zero["before"]["exp_avg_sq"], zero["after"]["exp_avg_sq"])
        self.assertGreater(stable_lr(49999), 0.)
        self.assertEqual(stable_lr(50000), 0.)


if __name__ == "__main__":
    unittest.main()

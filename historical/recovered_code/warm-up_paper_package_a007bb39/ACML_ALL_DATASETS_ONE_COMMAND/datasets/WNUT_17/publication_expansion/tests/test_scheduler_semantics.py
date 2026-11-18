import math
import unittest

from publication_expansion.src.scheduler_semantics import (
    LEGACY_INITIAL_LR,
    legacy_transformer_lr,
    pretrained_linear_lr,
    pretrained_transition_plan,
    pretrained_warmup_steps,
    resolve_transition,
    scratch_transition_plan,
    stable_lr,
)


class ScheduleTests(unittest.TestCase):
    def test_one_based_indexing_rejects_zero(self):
        with self.assertRaises(ValueError):
            legacy_transformer_lr(0, 128)

        with self.assertRaises(ValueError):
            stable_lr(0)

    def test_legacy_step1_medium4(self):
        self.assertTrue(
            math.isclose(
                legacy_transformer_lr(1, 128),
                1.2352647110032733e-07,
                rel_tol=1e-14,
            )
        )

    def test_legacy_step1_large6(self):
        self.assertTrue(
            math.isclose(
                legacy_transformer_lr(1, 256),
                8.734640537108554e-08,
                rel_tol=1e-14,
            )
        )

    def test_stable_step1(self):
        self.assertTrue(
            math.isclose(
                stable_lr(1),
                1.25e-08,
                rel_tol=1e-14,
            )
        )

    def test_pretrained_warmup_uses_ceiling(self):
        self.assertEqual(
            pretrained_warmup_steps(1001, 0.10),
            101,
        )

    def test_pretrained_schedule_hits_peak_at_warmup_end(self):
        total = 1000
        warmup = pretrained_warmup_steps(total, 0.10)
        peak = 2e-5

        self.assertTrue(
            math.isclose(
                pretrained_linear_lr(
                    warmup, peak, total, 0.10
                ),
                peak,
                rel_tol=1e-14,
            )
        )

    def test_pretrained_schedule_zero_at_final_step(self):
        self.assertEqual(
            pretrained_linear_lr(
                1000, 2e-5, 1000, 0.10
            ),
            0.0,
        )


class ScratchConditionTests(unittest.TestCase):
    def test_clean_order_pair_same_formula_different_timing(self):
        buggy = scratch_transition_plan(
            condition="buggy_no_guard",
            t=1,
            entry_lr=LEGACY_INITIAL_LR,
            d_model=128,
        )

        fixed = scratch_transition_plan(
            condition="fixed_order_with_guard",
            t=1,
            entry_lr=LEGACY_INITIAL_LR,
            d_model=128,
        )

        self.assertEqual(
            buggy.intended_lr,
            fixed.intended_lr,
        )
        self.assertFalse(buggy.install_before_step)
        self.assertTrue(fixed.install_before_step)
        self.assertEqual(
            buggy.candidate_lr_consumed,
            1.0,
        )
        self.assertEqual(
            fixed.candidate_lr_consumed,
            fixed.intended_lr,
        )

    def test_single_bad_step1_actual_exit_is_one(self):
        plan = scratch_transition_plan(
            condition="single_bad_first_step_then_stable",
            t=1,
            entry_lr=1.0,
            d_model=128,
        )

        result = resolve_transition(plan)

        self.assertEqual(plan.candidate_lr_consumed, 1.0)
        self.assertEqual(result.committed_lr_consumed, 1.0)
        self.assertEqual(result.actual_exit_lr, 1.0)

    def test_single_bad_step2_installs_stable_before_step(self):
        plan = scratch_transition_plan(
            condition="single_bad_first_step_then_stable",
            t=2,
            entry_lr=1.0,
            d_model=128,
        )

        self.assertTrue(plan.install_before_step)
        self.assertEqual(
            plan.candidate_lr_consumed,
            stable_lr(2),
        )

    def test_failfast_commits_bad_candidate_then_stops(self):
        plan = scratch_transition_plan(
            condition="buggy_failfast_guard",
            t=1,
            entry_lr=1.0,
            d_model=128,
        )

        result = resolve_transition(
            plan,
            guard_triggered=True,
        )

        self.assertEqual(
            result.committed_lr_consumed,
            1.0,
        )
        self.assertTrue(result.stop)
        self.assertFalse(result.rollback)
        self.assertFalse(result.replay)

    def test_rescue_replaces_candidate_with_legacy_replay(self):
        plan = scratch_transition_plan(
            condition="buggy_rescue_guard",
            t=1,
            entry_lr=1.0,
            d_model=128,
        )

        result = resolve_transition(
            plan,
            guard_triggered=True,
        )

        self.assertEqual(
            result.committed_lr_consumed,
            plan.intended_lr,
        )
        self.assertFalse(result.stop)
        self.assertTrue(result.rollback)
        self.assertTrue(result.replay)


class PretrainedConditionTests(unittest.TestCase):
    def test_natural_late_first_step_consumes_peak(self):
        peak = 2e-5

        plan = pretrained_transition_plan(
            condition="late_schedule_no_guard",
            t=1,
            entry_lr=peak,
            peak_lr=peak,
            total_steps=1000,
        )

        self.assertEqual(
            plan.candidate_lr_consumed,
            peak,
        )
        self.assertLess(plan.intended_lr, peak)

    def test_single_bad_pretrained_step1_exits_at_peak(self):
        peak = 2e-5

        plan = pretrained_transition_plan(
            condition="single_bad_then_correct",
            t=1,
            entry_lr=peak,
            peak_lr=peak,
            total_steps=1000,
        )

        result = resolve_transition(plan)

        self.assertEqual(
            result.actual_exit_lr,
            peak,
        )

    def test_executed_lr_assertion_aborts_before_bad_update(self):
        peak = 2e-5

        plan = pretrained_transition_plan(
            condition="executed_lr_assertion",
            t=1,
            entry_lr=peak,
            peak_lr=peak,
            total_steps=1000,
        )

        result = resolve_transition(plan)

        self.assertTrue(plan.abort_before_update)
        self.assertIsNone(plan.candidate_lr_consumed)
        self.assertIsNone(result.committed_lr_consumed)
        self.assertTrue(result.stop)
        self.assertFalse(result.rollback)
        self.assertFalse(result.replay)

    def test_executed_lr_assertion_allows_matching_lr(self):
        peak = 2e-5
        intended = pretrained_linear_lr(
            1,
            peak,
            1000,
        )

        plan = pretrained_transition_plan(
            condition="executed_lr_assertion",
            t=1,
            entry_lr=intended,
            peak_lr=peak,
            total_steps=1000,
        )

        self.assertFalse(plan.abort_before_update)
        self.assertEqual(
            plan.candidate_lr_consumed,
            intended,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from publication_expansion.src.scratch_job_runner import (
    initial_lr_for_condition,
    step_semantics,
)


class ScratchJobRunnerTests(unittest.TestCase):
    def test_clean_order_pair_uses_same_legacy_target(self):
        buggy = step_semantics(
            condition="buggy_no_guard",
            step=1,
            d_model=128,
            threshold=None,
        )

        fixed = step_semantics(
            condition="fixed_order_with_guard",
            step=1,
            d_model=128,
            threshold=None,
        )

        self.assertEqual(
            buggy["trace_intended_lr"],
            fixed["trace_intended_lr"],
        )

        self.assertEqual(
            buggy["mode"],
            "late_no_guard",
        )

        self.assertEqual(
            fixed["mode"],
            "safe",
        )

    def test_single_bad_step1_retains_entry_lr_semantics(self):
        spec = step_semantics(
            condition=(
                "single_bad_first_step_then_stable"
            ),
            step=1,
            d_model=128,
            threshold=None,
        )

        self.assertEqual(
            spec["mode"],
            "late_no_guard",
        )

        self.assertEqual(
            spec["executor_intended_lr"],
            1.0,
        )

        self.assertNotEqual(
            spec["trace_intended_lr"],
            1.0,
        )

    def test_guard_is_only_active_first_100_steps(self):
        early = step_semantics(
            condition="buggy_rescue_guard",
            step=100,
            d_model=128,
            threshold=1e-4,
        )

        late = step_semantics(
            condition="buggy_rescue_guard",
            step=101,
            d_model=128,
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

    def test_initial_lr_policy(self):
        self.assertEqual(
            initial_lr_for_condition(
                "stable_with_guard"
            ),
            1e-4,
        )

        self.assertEqual(
            initial_lr_for_condition(
                "fixed_order_with_guard"
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

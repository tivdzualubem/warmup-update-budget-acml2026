import unittest

import torch

from publication_expansion.src.pretrained_severity import (
    MULTIPLIERS,
    clone_gradients,
    restore_gradients,
    severity_candidates,
    update_ratio,
)


class PretrainedSeverityTests(
    unittest.TestCase
):
    def test_candidate_grid(self):
        candidates = (
            severity_candidates(
                intended_lr=1e-6,
                peak_lr=5e-4,
            )
        )

        self.assertEqual(
            [
                item["multiplier"]
                for item
                in candidates[:-1]
            ],
            list(MULTIPLIERS),
        )

        self.assertAlmostEqual(
            candidates[-1][
                "multiplier"
            ],
            500.0,
            places=12,
        )

        self.assertTrue(
            candidates[-1][
                "natural_late_schedule"
            ]
        )

    def test_gradient_restore_exact(self):
        model = torch.nn.Linear(
            3,
            2,
        )

        x = torch.ones(
            1,
            3,
        )

        model(x).sum().backward()

        original = clone_gradients(
            model
        )

        for parameter in (
            model.parameters()
        ):
            parameter.grad = None

        restore_gradients(
            model,
            original,
        )

        restored = clone_gradients(
            model
        )

        for name in original:
            self.assertTrue(
                torch.equal(
                    original[name],
                    restored[name],
                )
            )

    def test_update_ratio_zero_before_step(self):
        model = torch.nn.Linear(
            3,
            2,
        )

        before = {
            name:
                parameter.detach()
                .clone()
            for name, parameter
            in model.named_parameters()
        }

        self.assertEqual(
            update_ratio(
                model,
                before,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

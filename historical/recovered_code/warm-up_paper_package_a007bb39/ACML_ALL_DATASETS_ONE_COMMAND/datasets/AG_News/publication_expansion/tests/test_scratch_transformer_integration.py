import math
import unittest

import torch
import torch.nn.functional as F

from publication_expansion.src.anchor import (
    capture_anchor,
    gradient_fingerprint,
    model_fingerprint,
    object_sha256,
    optimizer_non_lr_fingerprint,
    restore_anchor,
)
from publication_expansion.src.scheduler_semantics import (
    legacy_transformer_lr,
)
from publication_expansion.src.scratch_model_adapter import (
    build_scratch_adam,
    build_scratch_model,
    get_scratch_model_config,
)
from publication_expansion.src.transition_executor import (
    transactional_step,
)


class ScratchTransformerIntegrationTests(unittest.TestCase):
    def test_publication_model_configs_are_exact(self):
        medium = get_scratch_model_config("medium-4")
        large = get_scratch_model_config("large-6")

        self.assertEqual(
            medium,
            {
                "name": "medium-4",
                "d_model": 128,
                "n_layers": 4,
                "n_heads": 8,
                "d_ff": 512,
            },
        )

        self.assertEqual(
            large,
            {
                "name": "large-6",
                "d_model": 256,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 1024,
            },
        )

    def test_real_transformer_order_only_step(self):
        torch.manual_seed(2026)

        model = build_scratch_model(
            model_name="medium-4",
            vocab_size=128,
            num_classes=4,
            max_seq_len=16,
        )

        optimizer = build_scratch_adam(
            model,
            initial_lr=1.0,
        )

        batch = {
            "input_ids": torch.tensor(
                [
                    [
                        2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 0, 0, 0, 0,
                    ],
                    [
                        14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 0, 0, 0,
                    ],
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [1, 3],
                dtype=torch.long,
            ),
        }

        anchor = capture_anchor(
            model=model,
            optimizer=optimizer,
            batch=batch,
            metadata={
                "dataset": "synthetic_smoke",
                "model": "medium-4",
                "seed": 2026,
            },
        )

        intended_lr = legacy_transformer_lr(
            1,
            d_model=128,
        )

        def run_branch(mode):
            restore_anchor(
                model=model,
                optimizer=optimizer,
                anchor=anchor,
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)

            pre_model = model_fingerprint(model)
            pre_optimizer_non_lr = (
                optimizer_non_lr_fingerprint(
                    optimizer
                )
            )

            logits = model(
                batch["input_ids"]
            )

            loss = F.cross_entropy(
                logits,
                batch["labels"],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            record = {
                "pre_model": pre_model,
                "pre_optimizer_non_lr":
                    pre_optimizer_non_lr,
                "logits":
                    object_sha256(
                        logits.detach()
                    ),
                "loss":
                    object_sha256(
                        loss.detach()
                    ),
                "gradient":
                    gradient_fingerprint(model),
            }

            result = transactional_step(
                model=model,
                optimizer=optimizer,
                intended_lr=intended_lr,
                threshold=None,
                mode=mode,
            )

            record["result"] = result
            record["post_model"] = (
                model_fingerprint(model)
            )

            return record

        buggy = run_branch("late_no_guard")
        fixed = run_branch("safe")

        # Before optimizer semantics diverge, everything must match.
        for key in (
            "pre_model",
            "pre_optimizer_non_lr",
            "logits",
            "loss",
            "gradient",
        ):
            self.assertEqual(
                buggy[key],
                fixed[key],
            )

        # Same intended schedule value.
        self.assertTrue(
            math.isclose(
                intended_lr,
                1.2352647110032733e-7,
                rel_tol=0.0,
                abs_tol=1e-18,
            )
        )

        # Buggy branch consumes optimizer entry LR = 1.0.
        self.assertTrue(
            math.isclose(
                buggy["result"]
                .candidate_lr_by_group[0],
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )

        self.assertTrue(
            math.isclose(
                buggy["result"]
                .committed_lr_by_group[0],
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )

        # Fixed branch consumes the intended step-1 LR.
        self.assertTrue(
            math.isclose(
                fixed["result"]
                .committed_lr_by_group[0],
                intended_lr,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        )

        # After the transition both optimizers store the same target LR.
        self.assertTrue(
            math.isclose(
                buggy["result"]
                .actual_exit_lr_by_group[0],
                intended_lr,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        )

        self.assertTrue(
            math.isclose(
                fixed["result"]
                .actual_exit_lr_by_group[0],
                intended_lr,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        )

        # The consumed LR difference must produce a dramatically
        # different realized first update.
        self.assertGreater(
            buggy["result"].committed_update_ratio,
            fixed["result"].committed_update_ratio
            * 100000.0,
        )

        self.assertNotEqual(
            buggy["post_model"],
            fixed["post_model"],
        )


if __name__ == "__main__":
    unittest.main()

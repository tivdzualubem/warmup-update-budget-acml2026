import copy
import random
import unittest

import numpy as np
import torch
from torch import nn

from publication_expansion.src.anchor import (
    anchor_fingerprints,
    batch_fingerprint,
    capture_anchor,
    capture_rng_state,
    gradient_fingerprint,
    model_fingerprint,
    object_sha256,
    optimizer_fingerprint,
    optimizer_non_lr_fingerprint,
    restore_anchor,
    restore_rng_state,
)


class TinyDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def make_model():
    torch.manual_seed(444)
    return TinyDropoutModel().double()


def make_optimizer(model, lr=1.0):
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )


def fixed_batch():
    x = torch.tensor(
        [
            [0.1, 0.2, -0.3, 0.4],
            [0.5, -0.2, 0.7, 0.1],
            [-0.4, 0.8, 0.2, -0.1],
        ],
        dtype=torch.float64,
    )

    y = torch.tensor(
        [0, 2, 1],
        dtype=torch.long,
    )

    return {
        "inputs": x,
        "labels": y,
    }


class AnchorTests(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    def test_hash_detects_tensor_mutation(self):
        x = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float64,
        )

        h1 = object_sha256(x)

        x[1] += 1.0

        h2 = object_sha256(x)

        self.assertNotEqual(h1, h2)

    def test_hash_supports_scalar_tensors(self):
        x = torch.tensor(
            3.25,
            dtype=torch.float64,
        )

        h1 = object_sha256(x)
        h2 = object_sha256(x.clone())

        self.assertEqual(h1, h2)

        x += 1.0

        self.assertNotEqual(
            h1,
            object_sha256(x),
        )

    def test_optimizer_non_lr_hash_ignores_only_lr(self):
        model = make_model()
        optimizer = make_optimizer(model, lr=1.0)

        full_before = optimizer_fingerprint(
            optimizer
        )
        non_lr_before = optimizer_non_lr_fingerprint(
            optimizer
        )

        optimizer.param_groups[0]["lr"] = 1e-4

        full_after = optimizer_fingerprint(
            optimizer
        )
        non_lr_after = optimizer_non_lr_fingerprint(
            optimizer
        )

        self.assertNotEqual(
            full_before,
            full_after,
        )

        self.assertEqual(
            non_lr_before,
            non_lr_after,
        )

    def test_rng_restore_is_exact(self):
        state = capture_rng_state()

        py1 = [random.random() for _ in range(4)]
        np1 = np.random.random(4)
        torch1 = torch.rand(4)

        restore_rng_state(state)

        py2 = [random.random() for _ in range(4)]
        np2 = np.random.random(4)
        torch2 = torch.rand(4)

        self.assertEqual(py1, py2)

        self.assertTrue(
            np.array_equal(np1, np2)
        )

        self.assertTrue(
            torch.equal(torch1, torch2)
        )

    def test_anchor_restores_model_optimizer_and_rng(self):
        model = make_model()
        optimizer = make_optimizer(model)

        # Create non-empty Adam moment/step state first.
        for p in model.parameters():
            p.grad = torch.full_like(
                p,
                0.01,
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch = fixed_batch()

        anchor = capture_anchor(
            model=model,
            optimizer=optimizer,
            batch=batch,
            metadata={
                "seed": 123,
                "split_seed": 42,
            },
        )

        expected_anchor_hashes = anchor_fingerprints(
            anchor
        )

        expected_model = model_fingerprint(
            model
        )

        expected_optimizer = optimizer_fingerprint(
            optimizer
        )

        expected_rng = copy.deepcopy(
            anchor["rng_state"]
        )

        # Deliberately mutate everything.
        for p in model.parameters():
            p.data.add_(
                torch.randn_like(p)
            )
            p.grad = torch.ones_like(p)

        optimizer.param_groups[0]["lr"] = 0.123

        random.random()
        np.random.random()
        torch.rand(5)

        restore_anchor(
            model=model,
            optimizer=optimizer,
            anchor=anchor,
        )

        self.assertEqual(
            model_fingerprint(model),
            expected_model,
        )

        self.assertEqual(
            optimizer_fingerprint(optimizer),
            expected_optimizer,
        )

        restored_anchor = capture_anchor(
            model=model,
            optimizer=optimizer,
            batch=batch,
            metadata=anchor["metadata"],
        )

        # RNG state after restore must equal the saved state.
        restored_anchor["rng_state"] = capture_rng_state()

        self.assertEqual(
            object_sha256(
                restored_anchor["rng_state"]
            ),
            object_sha256(expected_rng),
        )

        self.assertEqual(
            anchor_fingerprints(anchor),
            expected_anchor_hashes,
        )

    def test_two_branches_match_before_lr_divergence(self):
        base_model = make_model()
        base_optimizer = make_optimizer(
            base_model,
            lr=1.0,
        )

        batch = fixed_batch()

        anchor = capture_anchor(
            model=base_model,
            optimizer=base_optimizer,
            batch=batch,
            metadata={
                "dataset": "synthetic",
                "model": "tiny_dropout",
                "seed": 123,
                "split_seed": 42,
            },
        )

        branch_records = []

        for _ in range(2):
            model = make_model()
            optimizer = make_optimizer(
                model,
                lr=1.0,
            )

            restore_anchor(
                model=model,
                optimizer=optimizer,
                anchor=anchor,
            )

            model.train()

            x = anchor["batch"]["inputs"].clone()
            y = anchor["batch"]["labels"].clone()

            record = {
                "model":
                    model_fingerprint(model),
                "optimizer":
                    optimizer_fingerprint(optimizer),
                "optimizer_non_lr":
                    optimizer_non_lr_fingerprint(
                        optimizer
                    ),
                "batch":
                    batch_fingerprint(
                        {
                            "inputs": x,
                            "labels": y,
                        }
                    ),
                "rng_before_forward":
                    object_sha256(
                        capture_rng_state()
                    ),
            }

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits,
                y,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            record["logits"] = object_sha256(
                logits.detach()
            )

            record["loss"] = object_sha256(
                loss.detach()
            )

            record["clipped_gradient"] = (
                gradient_fingerprint(model)
            )

            branch_records.append(record)

        self.assertEqual(
            branch_records[0],
            branch_records[1],
        )


if __name__ == "__main__":
    unittest.main()

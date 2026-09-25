import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from publication_expansion.src.scratch_data_contract import (
    FIXED_SPLIT_SEED,
    indices_sha256,
    make_dataloader,
    make_split_indices,
    split_manifest,
)


class ScratchSplitTests(unittest.TestCase):
    def test_split_is_deterministic(self):
        a = make_split_indices(
            1000,
            split_seed=42,
            val_ratio=0.15,
        )

        b = make_split_indices(
            1000,
            split_seed=42,
            val_ratio=0.15,
        )

        self.assertTrue(
            np.array_equal(a.train, b.train)
        )

        self.assertTrue(
            np.array_equal(
                a.validation,
                b.validation,
            )
        )

    def test_split_seed_changes_split(self):
        a = make_split_indices(
            1000,
            split_seed=42,
        )

        b = make_split_indices(
            1000,
            split_seed=123,
        )

        self.assertFalse(
            np.array_equal(a.train, b.train)
        )

    def test_expected_split_sizes(self):
        split = make_split_indices(
            1000,
            split_seed=42,
            val_ratio=0.15,
        )

        self.assertEqual(
            len(split.validation),
            150,
        )

        self.assertEqual(
            len(split.train),
            850,
        )

    def test_train_and_validation_are_disjoint(self):
        split = make_split_indices(
            1000,
            split_seed=42,
        )

        overlap = set(split.train).intersection(
            set(split.validation)
        )

        self.assertEqual(len(overlap), 0)

    def test_split_covers_all_examples(self):
        split = make_split_indices(
            1000,
            split_seed=42,
        )

        combined = np.concatenate(
            [
                split.train,
                split.validation,
            ]
        )

        self.assertEqual(
            set(combined.tolist()),
            set(range(1000)),
        )

    def test_fraction_sampling_is_reproducible(self):
        a = make_split_indices(
            1000,
            split_seed=42,
            train_fraction=0.25,
        )

        b = make_split_indices(
            1000,
            split_seed=42,
            train_fraction=0.25,
        )

        self.assertTrue(
            np.array_equal(a.train, b.train)
        )

    def test_manifest_hash_detects_order_change(self):
        split = make_split_indices(
            100,
            split_seed=42,
        )

        original = indices_sha256(
            split.train
        )

        reversed_hash = indices_sha256(
            split.train[::-1]
        )

        self.assertNotEqual(
            original,
            reversed_hash,
        )

    def test_manifest_records_fixed_split_seed(self):
        split = make_split_indices(
            100,
            split_seed=FIXED_SPLIT_SEED,
        )

        manifest = split_manifest(
            split,
            split_seed=FIXED_SPLIT_SEED,
            val_ratio=0.15,
            train_fraction=1.0,
        )

        self.assertEqual(
            manifest["split_seed"],
            42,
        )


class LoaderOrderTests(unittest.TestCase):
    @staticmethod
    def dataset():
        values = torch.arange(
            40,
            dtype=torch.long,
        )

        return TensorDataset(values)

    @staticmethod
    def flatten_order(loader):
        result = []

        for batch in loader:
            values = batch[0]
            result.extend(
                values.tolist()
            )

        return result

    def test_same_experiment_seed_same_shuffle(self):
        a = make_dataloader(
            self.dataset(),
            batch_size=5,
            shuffle=True,
            experiment_seed=123,
            num_workers=0,
        )

        b = make_dataloader(
            self.dataset(),
            batch_size=5,
            shuffle=True,
            experiment_seed=123,
            num_workers=0,
        )

        self.assertEqual(
            self.flatten_order(a),
            self.flatten_order(b),
        )

    def test_different_experiment_seed_changes_shuffle(self):
        a = make_dataloader(
            self.dataset(),
            batch_size=5,
            shuffle=True,
            experiment_seed=123,
            num_workers=0,
        )

        b = make_dataloader(
            self.dataset(),
            batch_size=5,
            shuffle=True,
            experiment_seed=456,
            num_workers=0,
        )

        self.assertNotEqual(
            self.flatten_order(a),
            self.flatten_order(b),
        )

    def test_eval_order_does_not_shuffle(self):
        loader = make_dataloader(
            self.dataset(),
            batch_size=5,
            shuffle=False,
            experiment_seed=123,
            num_workers=0,
        )

        self.assertEqual(
            self.flatten_order(loader),
            list(range(40)),
        )

    def test_condition_recreation_preserves_first_batch(self):
        a = make_dataloader(
            self.dataset(),
            batch_size=8,
            shuffle=True,
            experiment_seed=2026,
            num_workers=0,
        )

        b = make_dataloader(
            self.dataset(),
            batch_size=8,
            shuffle=True,
            experiment_seed=2026,
            num_workers=0,
        )

        first_a = next(iter(a))[0]
        first_b = next(iter(b))[0]

        self.assertTrue(
            torch.equal(
                first_a,
                first_b,
            )
        )


if __name__ == "__main__":
    unittest.main()

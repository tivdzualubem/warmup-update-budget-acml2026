import unittest
from unittest.mock import patch

import numpy as np

import publication_expansion.src.pretrained_data_adapter as data


class FakeTokenizerOutput(dict):
    def __init__(self, word_ids_by_batch):
        super().__init__(
            {
                "input_ids": [
                    [1, 2, 3, 4, 0]
                    for _ in word_ids_by_batch
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 0]
                    for _ in word_ids_by_batch
                ],
            }
        )

        self._word_ids = (
            word_ids_by_batch
        )

    def word_ids(self, batch_index):
        return self._word_ids[
            batch_index
        ]


class FakeWnutTokenizer:
    def __call__(self, tokens, **kwargs):
        self.last_kwargs = kwargs

        return FakeTokenizerOutput(
            [
                [
                    None,
                    0,
                    0,
                    1,
                    None,
                ]
                for _ in tokens
            ]
        )


class FakeSequenceTokenizer:
    def __call__(self, texts, **kwargs):
        self.texts = texts
        self.kwargs = kwargs

        return {
            "input_ids": [
                [1, 2, 0]
                for _ in texts
            ],
            "attention_mask": [
                [1, 1, 0]
                for _ in texts
            ],
        }


class PretrainedDataAdapterTests(
    unittest.TestCase
):
    def test_frozen_revision_forwarded(self):
        src = data.dataset_source(
            "wnut_17"
        )

        with patch.object(
            data,
            "load_dataset",
            return_value={"ok": True},
        ) as loader:
            result = (
                data.load_frozen_dataset(
                    "wnut_17"
                )
            )

        loader.assert_called_once_with(
            src["repository"],
            revision=src["revision"],
        )

        self.assertEqual(
            result,
            {"ok": True},
        )

    def test_sequence_field_contract(self):
        tokenizer = (
            FakeSequenceTokenizer()
        )

        encoded = (
            data.tokenize_sequence_batch(
                {
                    "content": [
                        "alpha",
                        "beta",
                    ],
                    "label": [3, 7],
                },
                tokenizer=tokenizer,
                dataset_name="dbpedia_14",
                max_length=128,
            )
        )

        self.assertEqual(
            tokenizer.texts,
            ["alpha", "beta"],
        )

        self.assertTrue(
            tokenizer.kwargs[
                "truncation"
            ]
        )

        self.assertEqual(
            tokenizer.kwargs["padding"],
            "max_length",
        )

        self.assertEqual(
            tokenizer.kwargs[
                "max_length"
            ],
            128,
        )

        self.assertEqual(
            encoded["labels"],
            [3, 7],
        )

    def test_first_subtoken_alignment(self):
        aligned = (
            data.align_first_subtoken_labels(
                word_ids=[
                    None,
                    0,
                    0,
                    1,
                    2,
                    2,
                    None,
                ],
                word_labels=[
                    5,
                    7,
                    9,
                ],
            )
        )

        self.assertEqual(
            aligned,
            [
                -100,
                5,
                -100,
                7,
                9,
                -100,
                -100,
            ],
        )

    def test_wnut_tokenization_contract(self):
        tokenizer = FakeWnutTokenizer()

        encoded = data.tokenize_wnut_batch(
            {
                "tokens": [
                    ["NewYork", "rocks"]
                ],
                "ner_tags": [
                    [7, 0]
                ],
            },
            tokenizer=tokenizer,
            max_length=5,
        )

        self.assertTrue(
            tokenizer.last_kwargs[
                "is_split_into_words"
            ]
        )

        self.assertEqual(
            encoded["labels"][0],
            [
                -100,
                7,
                -100,
                0,
                -100,
            ],
        )

    def test_cifar_split_is_deterministic_balanced_and_disjoint(self):
        labels = np.repeat(
            np.arange(10),
            5000,
        )

        a = (
            data.make_cifar_stratified_split(
                labels,
                validation_size=5000,
                split_seed=42,
            )
        )

        b = (
            data.make_cifar_stratified_split(
                labels,
                validation_size=5000,
                split_seed=42,
            )
        )

        self.assertEqual(a, b)

        self.assertEqual(
            len(a.train_indices),
            45000,
        )

        self.assertEqual(
            len(a.validation_indices),
            5000,
        )

        self.assertTrue(
            set(a.train_indices).isdisjoint(
                a.validation_indices
            )
        )

        val_labels = labels[
            a.validation_indices
        ]

        counts = np.bincount(
            val_labels,
            minlength=10,
        )

        self.assertEqual(
            counts.tolist(),
            [500] * 10,
        )

    def test_cifar_split_seed_changes_selection(self):
        labels = np.repeat(
            np.arange(10),
            5000,
        )

        a = (
            data.make_cifar_stratified_split(
                labels,
                split_seed=42,
            )
        )

        b = (
            data.make_cifar_stratified_split(
                labels,
                split_seed=123,
            )
        )

        self.assertNotEqual(
            a.validation_indices,
            b.validation_indices,
        )


if __name__ == "__main__":
    unittest.main()

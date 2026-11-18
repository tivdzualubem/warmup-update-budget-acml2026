import unittest

import numpy as np
import torch

from publication_expansion.src.scratch_dataset_adapter import (
    ScratchTextDataset,
    build_vocab,
    get_dataset_spec,
    prepare_scratch_dataset_from_examples,
    text_to_indices,
)


class VocabularyTests(unittest.TestCase):
    def test_special_tokens_have_frozen_indices(self):
        vocab = build_vocab(
            ["alpha alpha beta beta gamma"]
        )

        self.assertEqual(vocab["<PAD>"], 0)
        self.assertEqual(vocab["<UNK>"], 1)

    def test_min_frequency_is_two(self):
        vocab = build_vocab(
            ["alpha alpha beta"]
        )

        self.assertIn("alpha", vocab)
        self.assertNotIn("beta", vocab)

    def test_lowercase_whitespace_tokenization(self):
        vocab = build_vocab(
            ["Hello hello WORLD world"]
        )

        ids = text_to_indices(
            "HELLO WORLD",
            vocab,
            max_length=4,
        )

        self.assertEqual(
            ids[:2],
            [vocab["hello"], vocab["world"]],
        )

    def test_unknown_and_padding(self):
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "known": 2,
        }

        ids = text_to_indices(
            "known missing",
            vocab,
            max_length=4,
        )

        self.assertEqual(
            ids,
            [2, 1, 0, 0],
        )

    def test_truncation(self):
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "a": 2,
            "b": 3,
            "c": 4,
        }

        self.assertEqual(
            text_to_indices(
                "a b c",
                vocab,
                max_length=2,
            ),
            [2, 3],
        )


class DatasetSpecTests(unittest.TestCase):
    def test_ag_news_spec(self):
        spec = get_dataset_spec("ag_news")

        self.assertEqual(
            spec["hf_name"],
            "fancyzhx/ag_news",
        )
        self.assertEqual(spec["text_field"], "text")
        self.assertEqual(spec["num_classes"], 4)

    def test_dbpedia_spec(self):
        spec = get_dataset_spec("dbpedia_14")

        self.assertEqual(
            spec["hf_name"],
            "fancyzhx/dbpedia_14",
        )
        self.assertEqual(
            spec["text_field"],
            "content",
        )
        self.assertEqual(spec["num_classes"], 14)


class PreparedDatasetTests(unittest.TestCase):
    @staticmethod
    def make_examples(n=20):
        texts = [
            f"common common unique{i}"
            for i in range(n)
        ]

        labels = [
            i % 4
            for i in range(n)
        ]

        return texts, labels

    def test_fixed_split_repeats_exactly(self):
        texts, labels = self.make_examples()

        a = prepare_scratch_dataset_from_examples(
            dataset_name="ag_news",
            train_texts_full=texts,
            train_labels_full=labels,
            test_texts=["test common"],
            test_labels=[0],
            split_seed=42,
            val_ratio=0.20,
        )

        b = prepare_scratch_dataset_from_examples(
            dataset_name="ag_news",
            train_texts_full=texts,
            train_labels_full=labels,
            test_texts=["test common"],
            test_labels=[0],
            split_seed=42,
            val_ratio=0.20,
        )

        self.assertEqual(
            a.manifest["split"],
            b.manifest["split"],
        )

        self.assertEqual(
            a.train_dataset.texts,
            b.train_dataset.texts,
        )

    def test_vocab_uses_training_subset_only(self):
        texts = [
            "common common trainword trainword",
            "common common valonly valonly",
            "common common third third",
            "common common fourth fourth",
        ]

        labels = [0, 1, 2, 3]

        prepared = prepare_scratch_dataset_from_examples(
            dataset_name="ag_news",
            train_texts_full=texts,
            train_labels_full=labels,
            test_texts=["testonly testonly"],
            test_labels=[0],
            split_seed=42,
            val_ratio=0.25,
        )

        validation_words = set(
            " ".join(
                prepared.validation_dataset.texts
            ).lower().split()
        )

        exclusive_validation_words = (
            validation_words
            - set(
                " ".join(
                    prepared.train_dataset.texts
                ).lower().split()
            )
        )

        for word in exclusive_validation_words:
            self.assertNotIn(
                word,
                prepared.vocab,
            )

        self.assertNotIn(
            "testonly",
            prepared.vocab,
        )

    def test_official_test_is_not_resplit(self):
        texts, labels = self.make_examples()

        test_texts = [
            "official test zero",
            "official test one",
        ]

        test_labels = [0, 1]

        prepared = prepare_scratch_dataset_from_examples(
            dataset_name="ag_news",
            train_texts_full=texts,
            train_labels_full=labels,
            test_texts=test_texts,
            test_labels=test_labels,
            split_seed=42,
        )

        self.assertEqual(
            prepared.test_dataset.texts,
            test_texts,
        )

        self.assertEqual(
            prepared.test_dataset.labels,
            test_labels,
        )

    def test_dataset_item_contract(self):
        dataset = ScratchTextDataset(
            ["hello hello"],
            [2],
            {
                "<PAD>": 0,
                "<UNK>": 1,
                "hello": 2,
            },
            max_length=4,
        )

        item = dataset[0]

        self.assertEqual(
            item["input_ids"].dtype,
            torch.long,
        )

        self.assertEqual(
            item["labels"].dtype,
            torch.long,
        )

        self.assertEqual(
            item["input_ids"].tolist(),
            [2, 2, 0, 0],
        )

        self.assertEqual(
            item["labels"].item(),
            2,
        )

    def test_manifest_records_split_policy(self):
        texts, labels = self.make_examples()

        prepared = prepare_scratch_dataset_from_examples(
            dataset_name="ag_news",
            train_texts_full=texts,
            train_labels_full=labels,
            test_texts=["common test"],
            test_labels=[0],
            split_seed=42,
            val_ratio=0.15,
        )

        self.assertEqual(
            prepared.manifest["split"][
                "split_seed"
            ],
            42,
        )

        self.assertEqual(
            prepared.manifest["vocab_policy"][
                "source"
            ],
            "final_training_subset_only",
        )


if __name__ == "__main__":
    unittest.main()


class RevisionPinTests(unittest.TestCase):
    def test_real_loader_forwards_explicit_revision(self):
        from unittest.mock import patch

        from publication_expansion.src.scratch_dataset_adapter import (
            load_scratch_hf_dataset,
        )

        class FakeSplit(list):
            _fingerprint = "fake-fingerprint"

        train = FakeSplit(
            [
                {
                    "text": f"common common example{i}",
                    "label": i % 4,
                }
                for i in range(20)
            ]
        )

        test = FakeSplit(
            [
                {
                    "text": "common common test",
                    "label": 0,
                }
            ]
        )

        fake_raw = {
            "train": train,
            "test": test,
        }

        revision = (
            "eb185aade064a813bc0b7f42de02595523103ca4"
        )

        with patch(
            "datasets.load_dataset",
            return_value=fake_raw,
        ) as mocked:
            prepared = load_scratch_hf_dataset(
                "ag_news",
                split_seed=42,
                val_ratio=0.20,
                revision=revision,
            )

        mocked.assert_called_once_with(
            "fancyzhx/ag_news",
            revision=revision,
        )

        self.assertEqual(
            prepared.manifest[
                "hf_revision_requested"
            ],
            revision,
        )

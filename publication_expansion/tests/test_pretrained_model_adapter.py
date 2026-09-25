import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import publication_expansion.src.pretrained_model_adapter as adapter


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)
        self.norm = nn.LayerNorm(4)


class PretrainedModelAdapterTests(unittest.TestCase):
    def test_sources_are_frozen(self):
        sources = adapter.load_sources()

        self.assertEqual(
            sources["status"],
            "frozen",
        )

        self.assertIn(
            "distilbert_base_uncased",
            sources["models"],
        )

        self.assertIn(
            "deit_tiny_patch16_224",
            sources["models"],
        )

    def test_adamw_decay_contract(self):
        model = DummyModel()

        optimizer = (
            adapter.build_pretrained_adamw(
                model,
                initial_lr=2e-5,
            )
        )

        self.assertEqual(
            optimizer.defaults["betas"],
            (0.9, 0.999),
        )

        self.assertEqual(
            optimizer.defaults["eps"],
            1e-8,
        )

        groups = optimizer.param_groups

        self.assertEqual(
            sorted(
                group["weight_decay"]
                for group in groups
            ),
            [0.0, 0.01],
        )

        decay_ids = {
            id(p)
            for group in groups
            if group["weight_decay"] == 0.01
            for p in group["params"]
        }

        no_decay_ids = {
            id(p)
            for group in groups
            if group["weight_decay"] == 0.0
            for p in group["params"]
        }

        self.assertIn(
            id(model.linear.weight),
            decay_ids,
        )

        self.assertIn(
            id(model.linear.bias),
            no_decay_ids,
        )

        self.assertIn(
            id(model.norm.weight),
            no_decay_ids,
        )

        self.assertIn(
            id(model.norm.bias),
            no_decay_ids,
        )

    def test_exact_revisions_forwarded(self):
        distilbert = adapter.model_source(
            "distilbert"
        )

        deit = adapter.model_source(
            "deit_tiny"
        )

        with patch.object(
            adapter.AutoTokenizer,
            "from_pretrained",
        ) as tokenizer:
            adapter.load_distilbert_tokenizer()

            tokenizer.assert_called_once_with(
                distilbert["repository"],
                revision=distilbert["revision"],
                use_fast=True,
            )

        with patch.object(
            adapter.AutoModelForSequenceClassification,
            "from_pretrained",
        ) as sequence:
            adapter.build_sequence_classifier(
                num_labels=4
            )

            _, kwargs = sequence.call_args

            self.assertEqual(
                kwargs["revision"],
                distilbert["revision"],
            )

            self.assertEqual(
                kwargs["num_labels"],
                4,
            )

        with patch.object(
            adapter.AutoModelForImageClassification,
            "from_pretrained",
        ) as image:
            adapter.build_image_classifier(
                num_labels=10,
                id2label={
                    i: str(i)
                    for i in range(10)
                },
                label2id={
                    str(i): i
                    for i in range(10)
                },
            )

            _, kwargs = image.call_args

            self.assertEqual(
                kwargs["revision"],
                deit["revision"],
            )

            self.assertTrue(
                kwargs[
                    "ignore_mismatched_sizes"
                ]
            )


if __name__ == "__main__":
    unittest.main()

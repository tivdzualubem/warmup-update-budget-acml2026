import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from datasets import Dataset

from publication_expansion.src.kaggle_pilot import (
    CALIBRATION_SEEDS,
    DATASETS,
    TRAIN_FRACTIONS,
    build_pilot_thresholds,
    calibration_jobs,
    diagnostic_jobs,
    main_jobs,
    severity_jobs,
)
from publication_expansion.src.pretrained_data_adapter import (
    deterministic_dataset_fraction,
)
from publication_expansion.src.pretrained_job_runner import (
    NUM_EPOCHS,
    resolve_run_controls,
)


class KagglePilotTests(
    unittest.TestCase
):
    def test_production_defaults_remain_locked(self):
        controls = resolve_run_controls(
            pilot_mode=False,
            train_fraction=1.0,
            epochs=NUM_EPOCHS,
            max_optimizer_steps=None,
        )

        self.assertFalse(
            controls["pilot_mode"]
        )

        with self.assertRaises(
            ValueError
        ):
            resolve_run_controls(
                pilot_mode=False,
                train_fraction=0.25,
                epochs=NUM_EPOCHS,
                max_optimizer_steps=None,
            )

    def test_deterministic_fraction(self):
        dataset = Dataset.from_dict(
            {
                "x":
                    list(
                        range(100)
                    )
            }
        )

        a = deterministic_dataset_fraction(
            dataset,
            fraction=0.25,
            seed=42,
        )

        b = deterministic_dataset_fraction(
            dataset,
            fraction=0.25,
            seed=42,
        )

        c = deterministic_dataset_fraction(
            dataset,
            fraction=0.25,
            seed=123,
        )

        self.assertEqual(
            len(a),
            25,
        )

        self.assertEqual(
            a["x"],
            b["x"],
        )

        self.assertNotEqual(
            a["x"],
            c["x"],
        )

    def test_pilot_job_counts(self):
        self.assertEqual(
            len(calibration_jobs()),
            16,
        )

        self.assertEqual(
            len(severity_jobs()),
            4,
        )

        fake = {
            "records": [
                {
                    "dataset":
                        dataset,
                    "threshold":
                        1e-4,
                }
                for dataset
                in DATASETS
            ]
        }

        self.assertEqual(
            len(main_jobs(fake)),
            16,
        )

        self.assertEqual(
            len(
                diagnostic_jobs(
                    fake
                )
            ),
            8,
        )

    def test_only_dbpedia_is_reduced(self):
        self.assertEqual(
            TRAIN_FRACTIONS[
                "dbpedia_14"
            ],
            0.25,
        )

        self.assertEqual(
            TRAIN_FRACTIONS[
                "ag_news"
            ],
            1.0,
        )

        self.assertEqual(
            TRAIN_FRACTIONS[
                "wnut_17"
            ],
            1.0,
        )

        self.assertEqual(
            TRAIN_FRACTIONS[
                "cifar10"
            ],
            1.0,
        )

    def test_threshold_uses_only_four_nonheldout_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for dataset in DATASETS:
                for seed in CALIBRATION_SEEDS:
                    directory = (
                        root
                        / dataset
                        / f"seed{seed}"
                        / "safe_correct_order"
                    )

                    directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    (
                        directory
                        / "result.json"
                    ).write_text(
                        json.dumps(
                            {
                                "pilot_mode":
                                    True,
                                "terminal_reason":
                                    "pilot_max_steps",
                            }
                        )
                    )

                    ratio = (
                        seed
                        / 1_000_000
                    )

                    (
                        directory
                        / "trace.json"
                    ).write_text(
                        json.dumps(
                            [
                                {
                                    "step":
                                        step,
                                    "committed_update_ratio":
                                        ratio,
                                }
                                for step
                                in range(
                                    1,
                                    101,
                                )
                            ]
                        )
                    )

            manifest = (
                build_pilot_thresholds(
                    root
                )
            )

            self.assertEqual(
                len(
                    manifest[
                        "records"
                    ]
                ),
                4,
            )

            ag = [
                record
                for record
                in manifest[
                    "records"
                ]
                if record[
                    "dataset"
                ]
                == "ag_news"
            ][0]

            self.assertEqual(
                ag[
                    "calibration_seeds"
                ],
                list(
                    CALIBRATION_SEEDS
                ),
            )

            self.assertAlmostEqual(
                ag["threshold"],
                10
                * 2026
                / 1_000_000,
            )


if __name__ == "__main__":
    unittest.main()

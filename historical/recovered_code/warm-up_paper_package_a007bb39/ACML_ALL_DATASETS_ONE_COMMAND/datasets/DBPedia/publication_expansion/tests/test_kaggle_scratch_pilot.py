import json
import tempfile
import unittest
from pathlib import Path

from publication_expansion.src.kaggle_scratch_pilot import (
    CALIBRATION_SEEDS,
    DATASETS,
    SAFE_CONDITIONS,
    TRAIN_FRACTIONS,
    build_thresholds,
    calibration_jobs,
    diagnostic_jobs,
    main_jobs,
)
from publication_expansion.src.scratch_job_runner import (
    NUM_EPOCHS,
    TRAIN_FRACTION,
    resolve_scratch_run_controls,
)


class KaggleScratchPilotTests(
    unittest.TestCase
):
    def test_production_defaults_locked(self):
        controls = (
            resolve_scratch_run_controls(
                pilot_mode=False,
                train_fraction=TRAIN_FRACTION,
                epochs=NUM_EPOCHS,
                max_optimizer_steps=None,
            )
        )

        self.assertFalse(
            controls["pilot_mode"]
        )

        with self.assertRaises(
            ValueError
        ):
            resolve_scratch_run_controls(
                pilot_mode=False,
                train_fraction=0.25,
                epochs=3,
                max_optimizer_steps=100,
            )

    def test_job_counts(self):
        self.assertEqual(
            len(calibration_jobs()),
            16,
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
            10,
        )

        self.assertEqual(
            len(
                diagnostic_jobs(
                    fake
                )
            ),
            2,
        )

        self.assertEqual(
            16 + 10 + 2,
            28,
        )

    def test_only_dbpedia_reduced(self):
        self.assertEqual(
            TRAIN_FRACTIONS[
                "ag_news"
            ],
            1.0,
        )

        self.assertEqual(
            TRAIN_FRACTIONS[
                "dbpedia_14"
            ],
            0.25,
        )

    def test_threshold_uses_both_safe_conditions_and_four_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for dataset in DATASETS:
                for seed in CALIBRATION_SEEDS:
                    for condition_index, condition in enumerate(
                        SAFE_CONDITIONS
                    ):
                        directory = (
                            root
                            / dataset
                            / "medium-4"
                            / f"seed{seed}"
                            / condition
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
                            + condition_index
                            / 10_000
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

            manifest = build_thresholds(
                root
            )

            self.assertEqual(
                len(
                    manifest[
                        "records"
                    ]
                ),
                2,
            )

            ag = [
                x
                for x in manifest[
                    "records"
                ]
                if x["dataset"]
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

            self.assertEqual(
                ag[
                    "safe_conditions"
                ],
                list(
                    SAFE_CONDITIONS
                ),
            )


if __name__ == "__main__":
    unittest.main()

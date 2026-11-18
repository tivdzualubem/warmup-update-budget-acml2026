import json
import tempfile
import unittest
from pathlib import Path

from publication_expansion.src.scratch_matrix import (
    SAFE_CONDITIONS,
    build_jobs,
    compute_loso_thresholds,
    threshold_lookup,
)


class ScratchMatrixTests(unittest.TestCase):
    def test_phase_job_counts(self):
        self.assertEqual(
            len(build_jobs("safe")),
            40,
        )

        self.assertEqual(
            len(build_jobs("nonguarded")),
            40,
        )

    def _make_safe_results(
        self,
        root: Path,
        *,
        mismatch_anchor=False,
    ):
        seeds = (42, 123, 456)

        for seed in seeds:
            for index, condition in enumerate(
                SAFE_CONDITIONS
            ):
                directory = (
                    root
                    / "ag_news"
                    / "medium-4"
                    / f"seed{seed}"
                    / condition
                )

                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                anchor = {
                    "model_state": "same",
                    "optimizer_non_lr": "same",
                    "first_batch": "same",
                    "rng_before_forward": "same",
                    "logits": "same",
                    "loss": "same",
                    "clipped_gradient": "same",
                }

                if (
                    mismatch_anchor
                    and seed == 42
                    and index == 1
                ):
                    anchor["logits"] = "different"

                (directory / "result.json").write_text(
                    json.dumps(
                        {
                            "job_complete": True,
                            "anchor_evidence": anchor,
                        }
                    )
                )

                ratio = seed / 1_000_000.0

                trace = [
                    {
                        "step": step,
                        "committed_update_ratio":
                            ratio,
                    }
                    for step in range(1, 101)
                ]

                (directory / "trace.json").write_text(
                    json.dumps(trace)
                )

        return seeds

    def test_loso_excludes_heldout_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            seeds = self._make_safe_results(
                root
            )

            manifest = compute_loso_thresholds(
                root,
                kappa=10.0,
                datasets=("ag_news",),
                models=("medium-4",),
                seeds=seeds,
            )

            # Held-out 456: calibration max is seed 123.
            threshold = threshold_lookup(
                manifest,
                dataset="ag_news",
                model="medium-4",
                seed=456,
            )

            self.assertAlmostEqual(
                threshold,
                10.0 * 123 / 1_000_000.0,
            )

    def test_anchor_mismatch_aborts_thresholding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            seeds = self._make_safe_results(
                root,
                mismatch_anchor=True,
            )

            with self.assertRaises(
                RuntimeError
            ):
                compute_loso_thresholds(
                    root,
                    datasets=("ag_news",),
                    models=("medium-4",),
                    seeds=seeds,
                )

    def test_guarded_matrix_uses_seed_specific_threshold(self):
        records = []

        for dataset in (
            "ag_news",
            "dbpedia_14",
        ):
            for model in (
                "medium-4",
                "large-6",
            ):
                for seed in (
                    42,
                    123,
                    456,
                    789,
                    2026,
                ):
                    records.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "heldout_seed": seed,
                            "threshold":
                                seed / 1000.0,
                        }
                    )

        manifest = {
            "records": records
        }

        jobs = build_jobs(
            "guarded",
            threshold_manifest=manifest,
        )

        self.assertEqual(len(jobs), 40)

        target = [
            job
            for job in jobs
            if (
                job["dataset"] == "ag_news"
                and job["model"] == "medium-4"
                and job["seed"] == 789
                and job["condition"]
                == "buggy_rescue_guard"
            )
        ]

        self.assertEqual(
            len(target),
            1,
        )

        self.assertEqual(
            target[0]["threshold"],
            0.789,
        )

    def test_nominal_matrix_is_120_jobs(self):
        safe = len(build_jobs("safe"))
        nonguarded = len(
            build_jobs("nonguarded")
        )

        fake = {
            "records": [
                {
                    "dataset": dataset,
                    "model": model,
                    "heldout_seed": seed,
                    "threshold": 1e-4,
                }
                for dataset in (
                    "ag_news",
                    "dbpedia_14",
                )
                for model in (
                    "medium-4",
                    "large-6",
                )
                for seed in (
                    42,
                    123,
                    456,
                    789,
                    2026,
                )
            ]
        }

        guarded = len(
            build_jobs(
                "guarded",
                threshold_manifest=fake,
            )
        )

        self.assertEqual(
            safe + nonguarded + guarded,
            120,
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from publication_expansion.src.pretrained_matrix import (
    KAPPA_GRID,
    SAFE_CONDITION,
    audit_matched_anchors,
    build_jobs,
    compute_loso_thresholds,
    threshold_lookup,
)


class PretrainedMatrixTests(
    unittest.TestCase
):
    def _write_safe_runs(
        self,
        root: Path,
        *,
        seeds=(42, 123, 456),
    ):
        for seed in seeds:
            directory = (
                root
                / "ag_news"
                / f"seed{seed}"
                / SAFE_CONDITION
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
                        "job_complete":
                            True,
                        "terminal_reason":
                            "completed_3_epochs",
                        "anchor_evidence":
                            {
                                "x":
                                    f"seed-{seed}"
                            },
                    }
                )
            )

            ratio = (
                seed / 1_000_000
            )

            trace = [
                {
                    "step": step,
                    "committed_update_ratio":
                        ratio,
                }
                for step
                in range(1, 101)
            ]

            (
                directory
                / "trace.json"
            ).write_text(
                json.dumps(trace)
            )

    def test_phase_counts(self):
        self.assertEqual(
            len(build_jobs("safe")),
            20,
        )

        self.assertEqual(
            len(
                build_jobs(
                    "nonguarded"
                )
            ),
            40,
        )

    def test_loso_excludes_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            seeds = (
                42,
                123,
                456,
            )

            self._write_safe_runs(
                root,
                seeds=seeds,
            )

            manifest = (
                compute_loso_thresholds(
                    root,
                    datasets=(
                        "ag_news",
                    ),
                    seeds=seeds,
                )
            )

            threshold = (
                threshold_lookup(
                    manifest,
                    dataset="ag_news",
                    seed=456,
                )
            )

            self.assertAlmostEqual(
                threshold,
                10
                * 123
                / 1_000_000,
            )

    def test_sensitivity_grid_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            seeds = (
                42,
                123,
                456,
            )

            self._write_safe_runs(
                root,
                seeds=seeds,
            )

            manifest = (
                compute_loso_thresholds(
                    root,
                    datasets=(
                        "ag_news",
                    ),
                    seeds=seeds,
                )
            )

            record = manifest[
                "records"
            ][0]

            self.assertEqual(
                set(
                    record[
                        "sensitivity_thresholds"
                    ]
                ),
                {
                    str(k)
                    for k
                    in KAPPA_GRID
                },
            )

    def test_full_nominal_matrix_is_120(self):
        fake = {
            "records": [
                {
                    "dataset":
                        dataset,
                    "heldout_seed":
                        seed,
                    "primary_threshold":
                        1e-4,
                }
                for dataset in (
                    "ag_news",
                    "dbpedia_14",
                    "wnut_17",
                    "cifar10",
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

        total = (
            len(build_jobs("safe"))
            + len(
                build_jobs(
                    "nonguarded"
                )
            )
            + len(
                build_jobs(
                    "rescue",
                    threshold_manifest=fake,
                )
            )
            + len(
                build_jobs(
                    "diagnostic",
                    threshold_manifest=fake,
                )
            )
        )

        self.assertEqual(
            total,
            120,
        )

    def test_anchor_audit_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for condition, value in (
                (
                    "safe_correct_order",
                    "same",
                ),
                (
                    "late_schedule_no_guard",
                    "different",
                ),
            ):
                directory = (
                    root
                    / "ag_news"
                    / "seed42"
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
                            "anchor_evidence":
                                {
                                    "logits":
                                        value
                                }
                        }
                    )
                )

            with self.assertRaises(
                RuntimeError
            ):
                audit_matched_anchors(
                    root,
                    datasets=(
                        "ag_news",
                    ),
                    seeds=(42,),
                    conditions=(
                        "safe_correct_order",
                        "late_schedule_no_guard",
                    ),
                )


if __name__ == "__main__":
    unittest.main()

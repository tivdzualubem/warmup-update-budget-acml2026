"""Real reduced Kaggle pilot using the production pretrained runner.

Pilot outputs are explicitly NON-SCIENTIFIC. They validate the complete
pipeline and provide early empirical signals before the full 5-seed run.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DATASETS = (
    "ag_news",
    "dbpedia_14",
    "wnut_17",
    "cifar10",
)

HELDOUT_SEED = 42

CALIBRATION_SEEDS = (
    123,
    456,
    789,
    2026,
)

EPOCHS = 3
CALIBRATION_STEPS = 100
KAPPA = 10.0

TRAIN_FRACTIONS = {
    "ag_news": 1.0,
    "dbpedia_14": 0.25,
    "wnut_17": 1.0,
    "cifar10": 1.0,
}

FULL_CONDITIONS = (
    "safe_correct_order",
    "late_schedule_no_guard",
    "single_bad_then_correct",
    "update_guard_rescue",
)

DIAGNOSTIC_CONDITIONS = (
    "update_guard_failfast",
    "executed_lr_assertion",
)


def job_dir(
    root: Path,
    dataset: str,
    seed: int,
    condition: str,
) -> Path:
    return (
        root
        / dataset
        / f"seed{seed}"
        / condition
    )


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"required pilot artifact missing: {path}"
        )

    return json.loads(
        path.read_text()
    )


def atomic_json(
    path: Path,
    obj,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    os.replace(tmp, path)


def base_job(
    *,
    dataset: str,
    seed: int,
    condition: str,
    threshold=None,
    calibration=False,
):
    return {
        "dataset": dataset,
        "seed": int(seed),
        "condition": condition,
        "threshold": threshold,
        "train_fraction":
            TRAIN_FRACTIONS[dataset],
        "epochs": EPOCHS,
        "max_optimizer_steps":
            (
                CALIBRATION_STEPS
                if calibration
                else None
            ),
        "calibration":
            bool(calibration),
    }


def calibration_jobs():
    return [
        base_job(
            dataset=dataset,
            seed=seed,
            condition="safe_correct_order",
            calibration=True,
        )
        for dataset in DATASETS
        for seed in CALIBRATION_SEEDS
    ]


def build_pilot_thresholds(
    output_root: Path,
):
    records = []

    for dataset in DATASETS:
        ratios = []
        sources = []

        for seed in CALIBRATION_SEEDS:
            directory = job_dir(
                output_root,
                dataset,
                seed,
                "safe_correct_order",
            )

            result = load_json(
                directory / "result.json"
            )

            if not result.get(
                "pilot_mode"
            ):
                raise RuntimeError(
                    "calibration artifact is not marked pilot_mode"
                )

            if (
                result.get(
                    "terminal_reason"
                )
                != "pilot_max_steps"
            ):
                raise RuntimeError(
                    "calibration run did not terminate "
                    "at pilot_max_steps: "
                    f"{dataset}/seed{seed}"
                )

            trace = load_json(
                directory / "trace.json"
            )

            early = [
                row
                for row in trace
                if (
                    1
                    <= int(row["step"])
                    <= CALIBRATION_STEPS
                )
            ]

            steps = {
                int(row["step"])
                for row in early
            }

            expected = set(
                range(
                    1,
                    CALIBRATION_STEPS + 1,
                )
            )

            if steps != expected:
                raise RuntimeError(
                    "calibration trace missing steps 1..100: "
                    f"{dataset}/seed{seed}"
                )

            ratios.extend(
                float(
                    row[
                        "committed_update_ratio"
                    ]
                )
                for row in early
            )

            sources.append(
                {
                    "seed": int(seed),
                    "condition":
                        "safe_correct_order",
                }
            )

        safe_max = max(ratios)

        records.append(
            {
                "dataset":
                    dataset,
                "heldout_seed":
                    HELDOUT_SEED,
                "calibration_seeds":
                    list(CALIBRATION_SEEDS),
                "calibration_steps":
                    CALIBRATION_STEPS,
                "train_fraction":
                    TRAIN_FRACTIONS[
                        dataset
                    ],
                "safe_max_committed_update_ratio":
                    float(safe_max),
                "kappa":
                    KAPPA,
                "threshold":
                    float(
                        KAPPA
                        * safe_max
                    ),
                "source_jobs":
                    sources,
            }
        )

    return {
        "schema_version": 1,
        "scientific_results": False,
        "purpose":
            "held-out-seed Kaggle pilot guard calibration",
        "heldout_seed":
            HELDOUT_SEED,
        "kappa":
            KAPPA,
        "records":
            records,
    }


def threshold_lookup(
    manifest: Dict,
    dataset: str,
) -> float:
    records = [
        record
        for record in manifest[
            "records"
        ]
        if record["dataset"]
        == dataset
    ]

    if len(records) != 1:
        raise RuntimeError(
            f"expected one pilot threshold for {dataset}"
        )

    return float(
        records[0]["threshold"]
    )


def main_jobs(
    threshold_manifest: Dict,
):
    jobs = []

    for dataset in DATASETS:
        threshold = threshold_lookup(
            threshold_manifest,
            dataset,
        )

        for condition in FULL_CONDITIONS:
            jobs.append(
                base_job(
                    dataset=dataset,
                    seed=HELDOUT_SEED,
                    condition=condition,
                    threshold=(
                        threshold
                        if condition
                        == "update_guard_rescue"
                        else None
                    ),
                )
            )

    return jobs


def diagnostic_jobs(
    threshold_manifest: Dict,
):
    jobs = []

    for dataset in DATASETS:
        threshold = threshold_lookup(
            threshold_manifest,
            dataset,
        )

        for condition in DIAGNOSTIC_CONDITIONS:
            jobs.append(
                base_job(
                    dataset=dataset,
                    seed=HELDOUT_SEED,
                    condition=condition,
                    threshold=(
                        threshold
                        if condition
                        == "update_guard_failfast"
                        else None
                    ),
                )
            )

    return jobs


def severity_jobs():
    return [
        {
            "dataset": dataset,
            "seed": HELDOUT_SEED,
        }
        for dataset in DATASETS
    ]


def pretrained_command(
    job: Dict,
    *,
    output_root: Path,
    device: str,
    num_workers: int,
):
    script = (
        Path(__file__).resolve()
        .parents[1]
        / "scripts"
        / "run_pretrained_job.py"
    )

    command = [
        sys.executable,
        str(script),
        "--dataset",
        job["dataset"],
        "--seed",
        str(job["seed"]),
        "--condition",
        job["condition"],
        "--output-root",
        str(output_root),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
        "--pilot-mode",
        "--train-fraction",
        str(job["train_fraction"]),
        "--epochs",
        str(job["epochs"]),
    ]

    if (
        job["max_optimizer_steps"]
        is not None
    ):
        command.extend(
            [
                "--max-optimizer-steps",
                str(
                    job[
                        "max_optimizer_steps"
                    ]
                ),
            ]
        )

    if job["threshold"] is not None:
        command.extend(
            [
                "--threshold",
                repr(
                    float(
                        job["threshold"]
                    )
                ),
            ]
        )

    return command


def severity_command(
    job: Dict,
    *,
    output_root: Path,
    device: str,
    num_workers: int,
):
    script = (
        Path(__file__).resolve()
        .parents[1]
        / "scripts"
        / "run_pretrained_severity.py"
    )

    return [
        sys.executable,
        str(script),
        "--dataset",
        job["dataset"],
        "--seed",
        str(job["seed"]),
        "--output-root",
        str(output_root),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]


def run_commands(
    jobs: List[Dict],
    *,
    output_root: Path,
    device: str,
    num_workers: int,
    execute: bool,
):
    for index, job in enumerate(
        jobs,
        start=1,
    ):
        command = pretrained_command(
            job,
            output_root=output_root,
            device=device,
            num_workers=num_workers,
        )

        print(
            f"[{index}/{len(jobs)}] "
            + shlex.join(command),
            flush=True,
        )

        if execute:
            subprocess.run(
                command,
                check=True,
            )


def run_severity(
    *,
    output_root: Path,
    device: str,
    num_workers: int,
    execute: bool,
):
    jobs = severity_jobs()

    for index, job in enumerate(
        jobs,
        start=1,
    ):
        command = severity_command(
            job,
            output_root=output_root,
            device=device,
            num_workers=num_workers,
        )

        print(
            f"[severity {index}/{len(jobs)}] "
            + shlex.join(command),
            flush=True,
        )

        if execute:
            subprocess.run(
                command,
                check=True,
            )


def summarize(
    output_root: Path,
    severity_root: Path,
):
    rows = []

    for dataset in DATASETS:
        for condition in (
            *FULL_CONDITIONS,
            *DIAGNOSTIC_CONDITIONS,
        ):
            directory = job_dir(
                output_root,
                dataset,
                HELDOUT_SEED,
                condition,
            )

            result = load_json(
                directory / "result.json"
            )

            trace = load_json(
                directory / "trace.json"
            )

            first = (
                trace[0]
                if trace
                else None
            )

            test_metrics = (
                result.get(
                    "test_metrics"
                )
                or {}
            )

            rows.append(
                {
                    "dataset":
                        dataset,
                    "condition":
                        condition,
                    "terminal_reason":
                        result.get(
                            "terminal_reason"
                        ),
                    "global_steps":
                        result.get(
                            "global_steps"
                        ),
                    "best_epoch":
                        result.get(
                            "best_epoch"
                        ),
                    "best_validation_primary":
                        result.get(
                            "best_validation_primary"
                        ),
                    "test_accuracy":
                        test_metrics.get(
                            "accuracy"
                        ),
                    "test_macro_f1":
                        test_metrics.get(
                            "macro_f1"
                        ),
                    "test_entity_f1":
                        test_metrics.get(
                            "entity_f1"
                        ),
                    "first_entry_lr":
                        (
                            first.get(
                                "entry_lr"
                            )
                            if first
                            else None
                        ),
                    "first_intended_lr":
                        (
                            first.get(
                                "intended_lr"
                            )
                            if first
                            else None
                        ),
                    "first_candidate_lr":
                        (
                            first.get(
                                "candidate_lr_consumed"
                            )
                            if first
                            else None
                        ),
                    "first_committed_ratio":
                        (
                            first.get(
                                "committed_update_ratio"
                            )
                            if first
                            else None
                        ),
                    "guard_triggered_step1":
                        (
                            first.get(
                                "guard_triggered"
                            )
                            if first
                            else None
                        ),
                }
            )

    csv_path = (
        output_root
        / "pilot_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    severity = {}

    for dataset in DATASETS:
        path = (
            severity_root
            / dataset
            / f"seed{HELDOUT_SEED}"
            / "result.json"
        )

        severity[dataset] = (
            load_json(path)
        )

    summary = {
        "schema_version": 1,
        "scientific_results": False,
        "purpose":
            "real reduced Kaggle pilot",
        "heldout_seed":
            HELDOUT_SEED,
        "train_fractions":
            TRAIN_FRACTIONS,
        "epochs":
            EPOCHS,
        "rows":
            rows,
        "severity":
            severity,
    }

    atomic_json(
        output_root
        / "pilot_summary.json",
        summary,
    )

    return summary

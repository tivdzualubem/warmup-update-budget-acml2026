"""Pretrained publication matrix and LOSO guard calibration."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DATASETS = (
    "ag_news",
    "dbpedia_14",
    "wnut_17",
    "cifar10",
)

SEEDS = (42, 123, 456, 789, 2026)

SAFE_CONDITION = "safe_correct_order"

NONGUARDED_CONDITIONS = (
    "late_schedule_no_guard",
    "single_bad_then_correct",
)

RESCUE_CONDITION = "update_guard_rescue"

DIAGNOSTIC_CONDITIONS = (
    "update_guard_failfast",
    "executed_lr_assertion",
)

PRIMARY_KAPPA = 10.0

KAPPA_GRID = (
    1.1,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
)

CALIBRATION_WINDOW = 100


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
            f"required file missing: {path}"
        )

    return json.loads(
        path.read_text()
    )


def compute_loso_thresholds(
    output_root: Path,
    *,
    datasets: Iterable[str] = DATASETS,
    seeds: Iterable[int] = SEEDS,
    primary_kappa: float = PRIMARY_KAPPA,
    calibration_window: int = CALIBRATION_WINDOW,
) -> Dict[str, object]:
    datasets = tuple(datasets)
    seeds = tuple(seeds)

    if primary_kappa <= 0:
        raise ValueError(
            "primary_kappa must be > 0"
        )

    records: List[Dict[str, object]] = []

    for dataset in datasets:
        for heldout_seed in seeds:
            calibration_seeds = [
                seed
                for seed in seeds
                if seed != heldout_seed
            ]

            ratios: List[float] = []
            sources = []

            for seed in calibration_seeds:
                directory = job_dir(
                    output_root,
                    dataset,
                    seed,
                    SAFE_CONDITION,
                )

                result = load_json(
                    directory / "result.json"
                )

                if not result.get(
                    "job_complete"
                ):
                    raise RuntimeError(
                        "safe calibration job incomplete: "
                        f"{dataset}/seed{seed}"
                    )

                if (
                    result.get("terminal_reason")
                    != "completed_3_epochs"
                ):
                    raise RuntimeError(
                        "safe calibration job did not "
                        "complete normally: "
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
                        <= calibration_window
                    )
                ]

                steps = {
                    int(row["step"])
                    for row in early
                }

                expected = set(
                    range(
                        1,
                        calibration_window + 1,
                    )
                )

                if steps != expected:
                    raise RuntimeError(
                        "safe trace does not contain "
                        f"exact steps 1..{calibration_window}: "
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
                            SAFE_CONDITION,
                    }
                )

            safe_max = max(ratios)

            sensitivity = {
                str(kappa):
                    float(kappa * safe_max)
                for kappa in KAPPA_GRID
            }

            records.append(
                {
                    "dataset":
                        dataset,
                    "heldout_seed":
                        int(heldout_seed),
                    "calibration_seeds":
                        calibration_seeds,
                    "safe_condition":
                        SAFE_CONDITION,
                    "calibration_window_steps":
                        calibration_window,
                    "safe_max_committed_update_ratio":
                        float(safe_max),
                    "primary_kappa":
                        float(primary_kappa),
                    "primary_threshold":
                        float(
                            primary_kappa
                            * safe_max
                        ),
                    "sensitivity_thresholds":
                        sensitivity,
                    "source_jobs":
                        sources,
                }
            )

    return {
        "schema_version": 1,
        "method":
            "leave_one_seed_out_safe_max",
        "ratio_field":
            "committed_update_ratio",
        "safe_condition":
            SAFE_CONDITION,
        "primary_kappa":
            float(primary_kappa),
        "kappa_grid":
            list(KAPPA_GRID),
        "calibration_window_steps":
            calibration_window,
        "records":
            records,
    }


def threshold_lookup(
    manifest: Dict[str, object],
    *,
    dataset: str,
    seed: int,
) -> float:
    matches = [
        record
        for record
        in manifest["records"]
        if (
            record["dataset"]
            == dataset
            and int(
                record["heldout_seed"]
            )
            == seed
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one threshold for "
            f"{dataset}/seed{seed}"
        )

    return float(
        matches[0][
            "primary_threshold"
        ]
    )


def audit_matched_anchors(
    output_root: Path,
    *,
    datasets: Iterable[str] = DATASETS,
    seeds: Iterable[int] = SEEDS,
    conditions: Iterable[str] = (
        "safe_correct_order",
        "late_schedule_no_guard",
        "single_bad_then_correct",
        "update_guard_failfast",
        "update_guard_rescue",
        "executed_lr_assertion",
    ),
) -> Dict[str, object]:
    """
    Compare all available pre-transition anchor evidence.

    Missing jobs are skipped so this can be run after any phase.
    """

    checks = []

    for dataset in datasets:
        for seed in seeds:
            anchors = {}

            for condition in conditions:
                path = (
                    job_dir(
                        output_root,
                        dataset,
                        seed,
                        condition,
                    )
                    / "result.json"
                )

                if not path.exists():
                    continue

                result = load_json(path)

                anchor = result.get(
                    "anchor_evidence"
                )

                if anchor:
                    anchors[condition] = anchor

            if len(anchors) < 2:
                continue

            reference_condition = sorted(
                anchors
            )[0]

            reference = anchors[
                reference_condition
            ]

            for condition, anchor in (
                anchors.items()
            ):
                if anchor != reference:
                    raise RuntimeError(
                        "matched-anchor violation: "
                        f"{dataset}/seed{seed}: "
                        f"{reference_condition} vs "
                        f"{condition}"
                    )

            checks.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "conditions":
                        sorted(anchors),
                    "matched": True,
                }
            )

    return {
        "schema_version": 1,
        "all_matched": True,
        "checks": checks,
    }


def build_jobs(
    phase: str,
    *,
    threshold_manifest:
        Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if phase == "safe":
        conditions = (
            SAFE_CONDITION,
        )

    elif phase == "nonguarded":
        conditions = (
            NONGUARDED_CONDITIONS
        )

    elif phase == "rescue":
        conditions = (
            RESCUE_CONDITION,
        )

    elif phase == "diagnostic":
        conditions = (
            DIAGNOSTIC_CONDITIONS
        )

    else:
        raise ValueError(
            "phase must be safe, nonguarded, "
            "rescue, or diagnostic"
        )

    jobs = []

    for dataset in DATASETS:
        for seed in SEEDS:
            for condition in conditions:
                threshold = None

                if condition in {
                    "update_guard_failfast",
                    "update_guard_rescue",
                }:
                    if (
                        threshold_manifest
                        is None
                    ):
                        raise ValueError(
                            "guarded job requires "
                            "threshold manifest"
                        )

                    threshold = (
                        threshold_lookup(
                            threshold_manifest,
                            dataset=dataset,
                            seed=seed,
                        )
                    )

                jobs.append(
                    {
                        "dataset":
                            dataset,
                        "seed":
                            seed,
                        "condition":
                            condition,
                        "threshold":
                            threshold,
                    }
                )

    return jobs


def command_for_job(
    job: Dict[str, object],
    *,
    output_root: str,
    device: str,
    num_workers: int,
) -> List[str]:
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
        str(job["dataset"]),
        "--seed",
        str(job["seed"]),
        "--condition",
        str(job["condition"]),
        "--output-root",
        output_root,
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]

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


def run_matrix(
    jobs: List[Dict[str, object]],
    *,
    output_root: str,
    device: str,
    num_workers: int,
    execute: bool,
) -> None:
    for index, job in enumerate(
        jobs,
        start=1,
    ):
        command = command_for_job(
            job,
            output_root=output_root,
            device=device,
            num_workers=num_workers,
        )

        print(
            f"[{index}/{len(jobs)}] "
            + shlex.join(command)
        )

        if execute:
            subprocess.run(
                command,
                check=True,
            )

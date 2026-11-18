"""Real reduced scratch pilot for Kaggle.

Uses the production scratch runner.
Outputs are NON-SCIENTIFIC pilot evidence.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DATASETS = (
    "ag_news",
    "dbpedia_14",
)

MODEL = "medium-4"

HELDOUT_SEED = 42

CALIBRATION_SEEDS = (
    123,
    456,
    789,
    2026,
)

SAFE_CONDITIONS = (
    "stable_with_guard",
    "fixed_order_with_guard",
)

MAIN_CONDITIONS = (
    "stable_with_guard",
    "fixed_order_with_guard",
    "buggy_no_guard",
    "single_bad_first_step_then_stable",
    "buggy_rescue_guard",
)

DIAGNOSTIC_CONDITIONS = (
    "buggy_failfast_guard",
)

TRAIN_FRACTIONS = {
    "ag_news": 1.0,
    "dbpedia_14": 0.25,
}

EPOCHS = 3
CALIBRATION_STEPS = 100
KAPPA = 10.0


def job_dir(
    root: Path,
    dataset: str,
    seed: int,
    condition: str,
):
    return (
        root
        / dataset
        / MODEL
        / f"seed{seed}"
        / condition
    )


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"required scratch pilot artifact missing: {path}"
        )

    return json.loads(
        path.read_text()
    )


def atomic_json(
    path: Path,
    obj,
):
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


def make_job(
    *,
    dataset,
    seed,
    condition,
    threshold=None,
    calibration=False,
):
    return {
        "dataset": dataset,
        "model": MODEL,
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
        make_job(
            dataset=dataset,
            seed=seed,
            condition=condition,
            calibration=True,
        )
        for dataset in DATASETS
        for seed in CALIBRATION_SEEDS
        for condition in SAFE_CONDITIONS
    ]


def build_thresholds(
    output_root: Path,
):
    records = []

    for dataset in DATASETS:
        ratios = []
        sources = []

        for seed in CALIBRATION_SEEDS:
            for condition in SAFE_CONDITIONS:
                directory = job_dir(
                    output_root,
                    dataset,
                    seed,
                    condition,
                )

                result = load_json(
                    directory
                    / "result.json"
                )

                if not result.get(
                    "pilot_mode"
                ):
                    raise RuntimeError(
                        "scratch calibration artifact "
                        "is not marked pilot_mode"
                    )

                if (
                    result.get(
                        "terminal_reason"
                    )
                    != "pilot_max_steps"
                ):
                    raise RuntimeError(
                        "scratch calibration did not "
                        "stop at 100 steps: "
                        f"{dataset}/seed{seed}/{condition}"
                    )

                trace = load_json(
                    directory
                    / "trace.json"
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
                        CALIBRATION_STEPS
                        + 1,
                    )
                )

                if steps != expected:
                    raise RuntimeError(
                        "scratch calibration trace "
                        "missing exact steps 1..100: "
                        f"{dataset}/seed{seed}/{condition}"
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
                        "seed":
                            int(seed),
                        "condition":
                            condition,
                    }
                )

        safe_max = max(ratios)

        records.append(
            {
                "dataset":
                    dataset,
                "model":
                    MODEL,
                "heldout_seed":
                    HELDOUT_SEED,
                "calibration_seeds":
                    list(
                        CALIBRATION_SEEDS
                    ),
                "safe_conditions":
                    list(
                        SAFE_CONDITIONS
                    ),
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
            "scratch Kaggle pilot LOSO calibration",
        "heldout_seed":
            HELDOUT_SEED,
        "records":
            records,
    }


def threshold_lookup(
    manifest,
    dataset,
):
    matches = [
        record
        for record
        in manifest["records"]
        if record["dataset"]
        == dataset
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"expected one scratch threshold for {dataset}"
        )

    return float(
        matches[0]["threshold"]
    )


def main_jobs(
    threshold_manifest,
):
    jobs = []

    for dataset in DATASETS:
        threshold = threshold_lookup(
            threshold_manifest,
            dataset,
        )

        for condition in MAIN_CONDITIONS:
            jobs.append(
                make_job(
                    dataset=dataset,
                    seed=HELDOUT_SEED,
                    condition=condition,
                    threshold=(
                        threshold
                        if condition
                        == "buggy_rescue_guard"
                        else None
                    ),
                )
            )

    return jobs


def diagnostic_jobs(
    threshold_manifest,
):
    jobs = []

    for dataset in DATASETS:
        threshold = threshold_lookup(
            threshold_manifest,
            dataset,
        )

        jobs.append(
            make_job(
                dataset=dataset,
                seed=HELDOUT_SEED,
                condition=(
                    "buggy_failfast_guard"
                ),
                threshold=threshold,
            )
        )

    return jobs


def command_for_job(
    job,
    *,
    output_root,
    device,
    num_workers,
):
    script = (
        Path(__file__).resolve()
        .parents[1]
        / "scripts"
        / "run_scratch_job.py"
    )

    command = [
        sys.executable,
        str(script),
        "--dataset",
        job["dataset"],
        "--model",
        job["model"],
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
        str(
            job[
                "train_fraction"
            ]
        ),
        "--epochs",
        str(job["epochs"]),
    ]

    if (
        job[
            "max_optimizer_steps"
        ]
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


def run_jobs(
    jobs,
    *,
    output_root,
    device,
    num_workers,
    execute,
):
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
            f"[scratch {index}/{len(jobs)}] "
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
):
    rows = []

    for dataset in DATASETS:
        for condition in (
            *MAIN_CONDITIONS,
            *DIAGNOSTIC_CONDITIONS,
        ):
            directory = job_dir(
                output_root,
                dataset,
                HELDOUT_SEED,
                condition,
            )

            result = load_json(
                directory
                / "result.json"
            )

            trace = load_json(
                directory
                / "trace.json"
            )

            first = (
                trace[0]
                if trace
                else None
            )

            metrics = (
                result.get(
                    "test_metrics"
                )
                or {}
            )

            rows.append(
                {
                    "dataset":
                        dataset,
                    "model":
                        MODEL,
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
                    "test_accuracy":
                        metrics.get(
                            "accuracy"
                        ),
                    "test_macro_f1":
                        metrics.get(
                            "macro_f1"
                        ),
                    "test_ece":
                        metrics.get(
                            "ece"
                        ),
                    "generalization_gap_accuracy":
                        result.get(
                            "generalization_gap_accuracy"
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
                    "first_committed_update_ratio":
                        (
                            first.get(
                                "committed_update_ratio"
                            )
                            if first
                            else None
                        ),
                    "first_guard_triggered":
                        (
                            first.get(
                                "guard_triggered"
                            )
                            if first
                            else None
                        ),
                }
            )

    path = (
        output_root
        / "scratch_pilot_summary.csv"
    )

    with path.open(
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

    summary = {
        "schema_version": 1,
        "scientific_results": False,
        "purpose":
            "real reduced scratch Kaggle pilot",
        "heldout_seed":
            HELDOUT_SEED,
        "model":
            MODEL,
        "epochs":
            EPOCHS,
        "train_fractions":
            TRAIN_FRACTIONS,
        "rows":
            rows,
    }

    atomic_json(
        output_root
        / "scratch_pilot_summary.json",
        summary,
    )

    return summary

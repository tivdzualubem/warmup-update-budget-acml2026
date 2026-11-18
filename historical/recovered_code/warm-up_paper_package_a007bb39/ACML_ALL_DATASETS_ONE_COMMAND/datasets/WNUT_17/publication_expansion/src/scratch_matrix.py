"""Scratch publication matrix and LOSO threshold utilities."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DATASETS = ("ag_news", "dbpedia_14")
MODELS = ("medium-4", "large-6")
SEEDS = (42, 123, 456, 789, 2026)

SAFE_CONDITIONS = (
    "stable_with_guard",
    "fixed_order_with_guard",
)

NONGUARDED_CONDITIONS = (
    "buggy_no_guard",
    "single_bad_first_step_then_stable",
)

GUARDED_CONDITIONS = (
    "buggy_failfast_guard",
    "buggy_rescue_guard",
)

PRIMARY_KAPPA = 10.0
CALIBRATION_WINDOW = 100


def _job_dir(
    output_root: Path,
    dataset: str,
    model: str,
    seed: int,
    condition: str,
) -> Path:
    return (
        output_root
        / dataset
        / model
        / f"seed{seed}"
        / condition
    )


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"required result missing: {path}"
        )

    return json.loads(path.read_text())


def _safe_anchor_audit(
    output_root: Path,
    *,
    datasets: Iterable[str],
    models: Iterable[str],
    seeds: Iterable[int],
) -> None:
    """Verify the two safe conditions start identically."""

    for dataset in datasets:
        for model in models:
            for seed in seeds:
                anchors = []

                for condition in SAFE_CONDITIONS:
                    result = _load_json(
                        _job_dir(
                            output_root,
                            dataset,
                            model,
                            seed,
                            condition,
                        )
                        / "result.json"
                    )

                    if not result.get("job_complete"):
                        raise RuntimeError(
                            "safe calibration job is incomplete: "
                            f"{dataset}/{model}/seed{seed}/"
                            f"{condition}"
                        )

                    anchor = result.get(
                        "anchor_evidence"
                    )

                    if not anchor:
                        raise RuntimeError(
                            "missing anchor evidence: "
                            f"{dataset}/{model}/seed{seed}/"
                            f"{condition}"
                        )

                    anchors.append(anchor)

                if anchors[0] != anchors[1]:
                    raise RuntimeError(
                        "matched-anchor violation between "
                        "stable_with_guard and "
                        "fixed_order_with_guard for "
                        f"{dataset}/{model}/seed{seed}"
                    )


def compute_loso_thresholds(
    output_root: Path,
    *,
    kappa: float = PRIMARY_KAPPA,
    calibration_window: int = CALIBRATION_WINDOW,
    datasets: Iterable[str] = DATASETS,
    models: Iterable[str] = MODELS,
    seeds: Iterable[int] = SEEDS,
) -> Dict[str, object]:
    datasets = tuple(datasets)
    models = tuple(models)
    seeds = tuple(seeds)

    if kappa <= 0:
        raise ValueError("kappa must be > 0")

    _safe_anchor_audit(
        output_root,
        datasets=datasets,
        models=models,
        seeds=seeds,
    )

    records: List[Dict[str, object]] = []

    for dataset in datasets:
        for model in models:
            for heldout_seed in seeds:
                calibration_seeds = [
                    seed
                    for seed in seeds
                    if seed != heldout_seed
                ]

                ratios = []
                source_jobs = []

                for seed in calibration_seeds:
                    for condition in SAFE_CONDITIONS:
                        directory = _job_dir(
                            output_root,
                            dataset,
                            model,
                            seed,
                            condition,
                        )

                        trace = _load_json(
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

                        if len(early) < calibration_window:
                            raise RuntimeError(
                                "safe trace does not contain "
                                f"steps 1..{calibration_window}: "
                                f"{dataset}/{model}/seed{seed}/"
                                f"{condition}"
                            )

                        ratios.extend(
                            float(
                                row[
                                    "committed_update_ratio"
                                ]
                            )
                            for row in early
                        )

                        source_jobs.append(
                            {
                                "seed": int(seed),
                                "condition": condition,
                            }
                        )

                safe_max = max(ratios)
                threshold = kappa * safe_max

                records.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "heldout_seed":
                            int(heldout_seed),
                        "calibration_seeds":
                            calibration_seeds,
                        "safe_conditions":
                            list(SAFE_CONDITIONS),
                        "calibration_window_steps":
                            calibration_window,
                        "safe_max_committed_update_ratio":
                            float(safe_max),
                        "kappa": float(kappa),
                        "threshold":
                            float(threshold),
                        "source_jobs":
                            source_jobs,
                    }
                )

    return {
        "schema_version": 1,
        "method":
            "leave_one_seed_out_safe_max",
        "ratio_field":
            "committed_update_ratio",
        "kappa": float(kappa),
        "calibration_window_steps":
            calibration_window,
        "safe_conditions":
            list(SAFE_CONDITIONS),
        "records": records,
    }


def threshold_lookup(
    threshold_manifest: Dict[str, object],
    *,
    dataset: str,
    model: str,
    seed: int,
) -> float:
    matches = [
        record
        for record
        in threshold_manifest["records"]
        if (
            record["dataset"] == dataset
            and record["model"] == model
            and int(record["heldout_seed"]) == seed
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one LOSO threshold for "
            f"{dataset}/{model}/seed{seed}"
        )

    return float(matches[0]["threshold"])


def build_jobs(
    phase: str,
    *,
    threshold_manifest:
        Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if phase == "safe":
        conditions = SAFE_CONDITIONS
    elif phase == "nonguarded":
        conditions = NONGUARDED_CONDITIONS
    elif phase == "guarded":
        conditions = GUARDED_CONDITIONS
    else:
        raise ValueError(
            "phase must be safe, nonguarded, or guarded"
        )

    jobs = []

    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                for condition in conditions:
                    threshold = None

                    if condition in GUARDED_CONDITIONS:
                        if threshold_manifest is None:
                            raise ValueError(
                                "guarded phase requires "
                                "threshold manifest"
                            )

                        threshold = threshold_lookup(
                            threshold_manifest,
                            dataset=dataset,
                            model=model,
                            seed=seed,
                        )

                    jobs.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "threshold": threshold,
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
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_scratch_job.py"
    )

    command = [
        sys.executable,
        str(script),
        "--dataset",
        str(job["dataset"]),
        "--model",
        str(job["model"]),
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
                repr(float(job["threshold"])),
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
    for index, job in enumerate(jobs, start=1):
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

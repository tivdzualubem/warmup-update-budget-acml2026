#!/usr/bin/env python3
"""Audit warmup guard thresholds from archived ACML 2026 experiment artifacts.

The script reads the raw AG News and DBpedia artifact ZIP files directly and
recomputes guard-threshold statistics from the archived early-step logs. It
never modifies the archives and does not require model training or a GPU.

Outputs
-------
- threshold_window_audit.csv
- published_vs_window_thresholds.csv
- kappa_sensitivity.csv
- leave_one_seed_out.csv
- model_transfer.csv
- cross_dataset_transfer.csv
- audit_summary.json

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

SAFE_CONDITIONS = ("stable_with_guard", "fixed_order_with_guard")
UNSAFE_CONDITIONS = (
    "buggy_no_guard",
    "buggy_failfast_guard",
    "buggy_rescue_guard",
    "single_bad_first_step_then_stable",
)
DEFAULT_KAPPAS = (1.1, 1.25, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0)

FILE_RE = re.compile(
    r"(?P<dataset>ag_news|dbpedia_14)_"
    r"(?P<model>medium-4|large-6)_"
    r"(?P<condition>stable_with_guard|fixed_order_with_guard|buggy_no_guard|"
    r"buggy_failfast_guard|buggy_rescue_guard|single_bad_first_step_then_stable)_"
    r"seed(?P<seed>\d+)_early_logs\.json$"
)


@dataclass(frozen=True)
class RunAudit:
    dataset: str
    model: str
    condition: str
    seed: int
    max_ratio_window: float
    step_of_max: int
    first_step_ratio: float
    first_step_unsafe_ratio: float
    source_member: str

    @property
    def is_safe(self) -> bool:
        return self.condition in SAFE_CONDITIONS

    @property
    def is_unsafe(self) -> bool:
        return self.condition in UNSAFE_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and stress-test ACML warmup guard thresholds from raw ZIP artifacts."
    )
    parser.add_argument("--agnews-zip", type=Path, required=True, help="Path to acml_raw_agnews.zip")
    parser.add_argument(
        "--dbpedia-core-zip", type=Path, required=True, help="Path to acml_raw_dbpedia_core.zip"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/rebuttal/guard_threshold_audit")
    )
    parser.add_argument("--window", type=int, default=100, help="Guard window in optimizer steps")
    parser.add_argument(
        "--kappa",
        type=float,
        nargs="*",
        default=list(DEFAULT_KAPPAS),
        help="Kappa values for threshold sensitivity analysis",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.agnews_zip, args.dbpedia_core_zip):
        if not path.is_file():
            raise FileNotFoundError(f"Artifact ZIP not found: {path}")
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Not a valid ZIP archive: {path}")
    if args.window < 1:
        raise ValueError("--window must be >= 1")
    if not args.kappa or any((not math.isfinite(k) or k <= 0) for k in args.kappa):
        raise ValueError("All --kappa values must be positive finite numbers")


def read_json_member(archive: zipfile.ZipFile, member: str):
    with archive.open(member) as handle:
        return json.load(handle)


def find_unique_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZIP member ending with {suffix!r}; found {len(matches)}")
    return matches[0]


def audit_archive(zip_path: Path, expected_dataset: str, window: int) -> Tuple[List[RunAudit], Mapping[str, object]]:
    audits: List[RunAudit] = []

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("_early_logs.json")]
        for member in sorted(members):
            match = FILE_RE.search(member)
            if match is None:
                continue
            if match.group("dataset") != expected_dataset:
                continue

            records = read_json_member(archive, member)
            if not isinstance(records, list) or not records:
                raise RuntimeError(f"Early log is empty or malformed: {member}")

            by_step = {int(row["step"]): row for row in records}
            if 1 not in by_step:
                raise RuntimeError(f"Step 1 missing from early log: {member}")

            window_rows = [row for row in records if 1 <= int(row["step"]) <= window]
            if not window_rows:
                raise RuntimeError(f"No records within steps 1..{window}: {member}")

            max_row = max(window_rows, key=lambda row: float(row["update_to_param_ratio"]))
            first = by_step[1]
            audits.append(
                RunAudit(
                    dataset=expected_dataset,
                    model=match.group("model"),
                    condition=match.group("condition"),
                    seed=int(match.group("seed")),
                    max_ratio_window=float(max_row["update_to_param_ratio"]),
                    step_of_max=int(max_row["step"]),
                    first_step_ratio=float(first["update_to_param_ratio"]),
                    first_step_unsafe_ratio=float(
                        first.get("unsafe_update_to_param_ratio", first["update_to_param_ratio"])
                    ),
                    source_member=member,
                )
            )

        calibration_member = find_unique_member(archive, "threshold_calibration_record.json")
        calibration_record = read_json_member(archive, calibration_member)

    expected_runs = len(SAFE_CONDITIONS + UNSAFE_CONDITIONS) * 2 * 3
    if len(audits) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} audited runs for {expected_dataset}; found {len(audits)}. "
            "Check artifact completeness or update the filename parser."
        )

    return audits, calibration_record


def safe_runs(runs: Iterable[RunAudit]) -> List[RunAudit]:
    return [run for run in runs if run.is_safe]


def unsafe_runs(runs: Iterable[RunAudit]) -> List[RunAudit]:
    return [run for run in runs if run.is_unsafe]


def calibration_max(runs: Sequence[RunAudit]) -> RunAudit:
    if not runs:
        raise ValueError("Calibration set is empty")
    return max(runs, key=lambda run: run.max_ratio_window)


def evaluate_threshold(
    threshold: float, safe_eval: Sequence[RunAudit], unsafe_eval: Sequence[RunAudit]
) -> Dict[str, object]:
    safe_fp = [run for run in safe_eval if run.max_ratio_window > threshold]
    unsafe_detected = [run for run in unsafe_eval if run.first_step_unsafe_ratio > threshold]
    return {
        "safe_total": len(safe_eval),
        "safe_false_positives": len(safe_fp),
        "safe_false_positive_rate": len(safe_fp) / len(safe_eval) if safe_eval else None,
        "unsafe_total": len(unsafe_eval),
        "unsafe_detected": len(unsafe_detected),
        "unsafe_detection_rate": len(unsafe_detected) / len(unsafe_eval) if unsafe_eval else None,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ag_runs, ag_published = audit_archive(args.agnews_zip, "ag_news", args.window)
    db_runs, db_published = audit_archive(args.dbpedia_core_zip, "dbpedia_14", args.window)
    datasets = {"ag_news": ag_runs, "dbpedia_14": db_runs}
    published = {"ag_news": ag_published, "dbpedia_14": db_published}

    audit_rows = []
    for dataset, runs in datasets.items():
        for run in sorted(runs, key=lambda r: (r.model, r.condition, r.seed)):
            audit_rows.append(asdict(run) | {"is_safe": run.is_safe, "is_unsafe": run.is_unsafe})
    write_csv(
        args.output_dir / "threshold_window_audit.csv",
        audit_rows,
        [
            "dataset",
            "model",
            "condition",
            "seed",
            "is_safe",
            "is_unsafe",
            "max_ratio_window",
            "step_of_max",
            "first_step_ratio",
            "first_step_unsafe_ratio",
            "source_member",
        ],
    )

    threshold_rows = []
    dataset_window_thresholds: Dict[str, float] = {}
    for dataset, runs in datasets.items():
        max_run = calibration_max(safe_runs(runs))
        corrected_threshold = 10.0 * max_run.max_ratio_window
        dataset_window_thresholds[dataset] = corrected_threshold
        record = published[dataset]
        threshold_rows.append(
            {
                "dataset": dataset,
                "window_steps": args.window,
                "window_safe_max": max_run.max_ratio_window,
                "window_safe_max_model": max_run.model,
                "window_safe_max_condition": max_run.condition,
                "window_safe_max_seed": max_run.seed,
                "window_safe_max_step": max_run.step_of_max,
                "window_threshold_kappa10": corrected_threshold,
                "archived_safe_max": record.get("safe_max_update_ratio"),
                "archived_threshold": record.get("threshold"),
                "archived_threshold_multiplier": record.get("threshold_multiplier"),
            }
        )
    write_csv(
        args.output_dir / "published_vs_window_thresholds.csv",
        threshold_rows,
        [
            "dataset",
            "window_steps",
            "window_safe_max",
            "window_safe_max_model",
            "window_safe_max_condition",
            "window_safe_max_seed",
            "window_safe_max_step",
            "window_threshold_kappa10",
            "archived_safe_max",
            "archived_threshold",
            "archived_threshold_multiplier",
        ],
    )

    kappa_rows = []
    for dataset, runs in datasets.items():
        calibration = calibration_max(safe_runs(runs)).max_ratio_window
        for kappa in sorted(set(args.kappa)):
            threshold = kappa * calibration
            metrics = evaluate_threshold(threshold, safe_runs(runs), unsafe_runs(runs))
            kappa_rows.append(
                {
                    "dataset": dataset,
                    "window_steps": args.window,
                    "kappa": kappa,
                    "calibration_safe_max": calibration,
                    "threshold": threshold,
                    **metrics,
                }
            )
    write_csv(
        args.output_dir / "kappa_sensitivity.csv",
        kappa_rows,
        [
            "dataset",
            "window_steps",
            "kappa",
            "calibration_safe_max",
            "threshold",
            "safe_total",
            "safe_false_positives",
            "safe_false_positive_rate",
            "unsafe_total",
            "unsafe_detected",
            "unsafe_detection_rate",
        ],
    )

    loso_rows = []
    for dataset, runs in datasets.items():
        seeds = sorted({run.seed for run in runs})
        for held_out in seeds:
            calibration_runs = [run for run in safe_runs(runs) if run.seed != held_out]
            safe_eval = [run for run in safe_runs(runs) if run.seed == held_out]
            unsafe_eval = [run for run in unsafe_runs(runs) if run.seed == held_out]
            max_run = calibration_max(calibration_runs)
            threshold = 10.0 * max_run.max_ratio_window
            metrics = evaluate_threshold(threshold, safe_eval, unsafe_eval)
            loso_rows.append(
                {
                    "dataset": dataset,
                    "held_out_seed": held_out,
                    "kappa": 10.0,
                    "calibration_safe_max": max_run.max_ratio_window,
                    "calibration_source_model": max_run.model,
                    "calibration_source_condition": max_run.condition,
                    "calibration_source_seed": max_run.seed,
                    "threshold": threshold,
                    **metrics,
                }
            )
    write_csv(
        args.output_dir / "leave_one_seed_out.csv",
        loso_rows,
        [
            "dataset",
            "held_out_seed",
            "kappa",
            "calibration_safe_max",
            "calibration_source_model",
            "calibration_source_condition",
            "calibration_source_seed",
            "threshold",
            "safe_total",
            "safe_false_positives",
            "safe_false_positive_rate",
            "unsafe_total",
            "unsafe_detected",
            "unsafe_detection_rate",
        ],
    )

    model_transfer_rows = []
    for dataset, runs in datasets.items():
        models = sorted({run.model for run in runs})
        for source_model in models:
            for target_model in models:
                if source_model == target_model:
                    continue
                calibration_runs = [run for run in safe_runs(runs) if run.model == source_model]
                safe_eval = [run for run in safe_runs(runs) if run.model == target_model]
                unsafe_eval = [run for run in unsafe_runs(runs) if run.model == target_model]
                max_run = calibration_max(calibration_runs)
                threshold = 10.0 * max_run.max_ratio_window
                metrics = evaluate_threshold(threshold, safe_eval, unsafe_eval)
                model_transfer_rows.append(
                    {
                        "dataset": dataset,
                        "source_model": source_model,
                        "target_model": target_model,
                        "kappa": 10.0,
                        "calibration_safe_max": max_run.max_ratio_window,
                        "threshold": threshold,
                        **metrics,
                    }
                )
    write_csv(
        args.output_dir / "model_transfer.csv",
        model_transfer_rows,
        [
            "dataset",
            "source_model",
            "target_model",
            "kappa",
            "calibration_safe_max",
            "threshold",
            "safe_total",
            "safe_false_positives",
            "safe_false_positive_rate",
            "unsafe_total",
            "unsafe_detected",
            "unsafe_detection_rate",
        ],
    )

    cross_dataset_rows = []
    names = tuple(datasets)
    for source_dataset in names:
        for target_dataset in names:
            if source_dataset == target_dataset:
                continue
            source_runs = datasets[source_dataset]
            target_runs = datasets[target_dataset]
            max_run = calibration_max(safe_runs(source_runs))
            threshold = 10.0 * max_run.max_ratio_window
            metrics = evaluate_threshold(threshold, safe_runs(target_runs), unsafe_runs(target_runs))
            cross_dataset_rows.append(
                {
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "kappa": 10.0,
                    "calibration_safe_max": max_run.max_ratio_window,
                    "threshold": threshold,
                    **metrics,
                }
            )
    write_csv(
        args.output_dir / "cross_dataset_transfer.csv",
        cross_dataset_rows,
        [
            "source_dataset",
            "target_dataset",
            "kappa",
            "calibration_safe_max",
            "threshold",
            "safe_total",
            "safe_false_positives",
            "safe_false_positive_rate",
            "unsafe_total",
            "unsafe_detected",
            "unsafe_detection_rate",
        ],
    )

    summary = {
        "guard_window_steps": args.window,
        "safe_conditions": list(SAFE_CONDITIONS),
        "unsafe_conditions": list(UNSAFE_CONDITIONS),
        "datasets": {},
        "outputs": [
            "threshold_window_audit.csv",
            "published_vs_window_thresholds.csv",
            "kappa_sensitivity.csv",
            "leave_one_seed_out.csv",
            "model_transfer.csv",
            "cross_dataset_transfer.csv",
        ],
    }
    for dataset, runs in datasets.items():
        max_run = calibration_max(safe_runs(runs))
        unsafe_min = min(run.first_step_unsafe_ratio for run in unsafe_runs(runs))
        unsafe_max = max(run.first_step_unsafe_ratio for run in unsafe_runs(runs))
        summary["datasets"][dataset] = {
            "safe_run_count": len(safe_runs(runs)),
            "unsafe_run_count": len(unsafe_runs(runs)),
            "first_100_safe_max": max_run.max_ratio_window,
            "first_100_safe_max_source": {
                "model": max_run.model,
                "condition": max_run.condition,
                "seed": max_run.seed,
                "step": max_run.step_of_max,
            },
            "first_100_threshold_kappa10": 10.0 * max_run.max_ratio_window,
            "unsafe_first_step_ratio_min": unsafe_min,
            "unsafe_first_step_ratio_max": unsafe_max,
            "archived_safe_max": published[dataset].get("safe_max_update_ratio"),
            "archived_threshold": published[dataset].get("threshold"),
        }

    summary_path = args.output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Guard threshold audit complete")
    print(f"Output directory: {args.output_dir.resolve()}")
    for dataset, data in summary["datasets"].items():
        print(
            f"{dataset}: safe max steps 1-{args.window} = {data['first_100_safe_max']:.12g}; "
            f"kappa=10 threshold = {data['first_100_threshold_kappa10']:.12g}; "
            f"unsafe step-1 range = [{data['unsafe_first_step_ratio_min']:.6g}, "
            f"{data['unsafe_first_step_ratio_max']:.6g}]"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

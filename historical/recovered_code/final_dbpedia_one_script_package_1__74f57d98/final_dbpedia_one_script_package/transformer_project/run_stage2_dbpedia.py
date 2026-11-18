import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_DIR.parent

RUNS = [
    {
        "name": "full_scratch",
        "script": "run_empirical_scratch.py",
        "expected_output": "transformer_project/results/scratch_empirical_dbpedia_14_full_dbpedia_v1/"
    },
    {
        "name": "ablation_shift",
        "script": "run_ablation_shift_subset.py",
        "expected_output": "transformer_project/results/scratch_empirical_dbpedia_14_dbpedia_ablation_shift_v1/"
    },
    {
        "name": "early_diagnostics_plus",
        "script": "run_early_diagnostics_plus.py",
        "expected_output": "transformer_project/results/early_diagnostics_plus_dbpedia_14_dbpedia_earlydiag_plus_v1/"
    },
    {
        "name": "initial_lr_sweep",
        "script": "run_initial_lr_sweep.py",
        "expected_output": "transformer_project/results/initial_lr_sweep_dbpedia_14_dbpedia_initial_lr_sweep_v1/"
    },
]

manifest = {
    "entry_command": "python transformer_project/run_stage2_dbpedia.py",
    "dataset": "dbpedia_14",
    "run_order": RUNS,
}

manifest_path = PROJECT_DIR / "stage2_dbpedia_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("=" * 70)
print("DBPEDIA STAGE-2 DRIVER")
print("=" * 70)
print(f"Manifest written to: {manifest_path}")
print("Run order:")
for i, run in enumerate(RUNS, 1):
    print(f"{i}. {run['name']} -> {run['script']}")
    print(f"   expected output: {run['expected_output']}")

for i, run in enumerate(RUNS, 1):
    script_path = PROJECT_DIR / run["script"]
    print("\n" + "=" * 70)
    print(f"[{i}/{len(RUNS)}] RUNNING: {run['name']}")
    print(f"Script: {script_path}")
    print("=" * 70)

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=str(PACKAGE_ROOT)
    )

print("\n" + "=" * 70)
print("ALL DBPEDIA STAGE-2 RUNS FINISHED")
print("=" * 70)
print("Expected output folders:")
for run in RUNS:
    print("-", run["expected_output"])

import sys
import json
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_DIR.parent

RUNS = [
    {
        "name": "warmup_safety_gate_calibrated",
        "script": "run_warmup_safety_gate_dbpedia.py",
        "expected_output": "transformer_project/results/warmup_safety_gate_dbpedia_14_dbpedia_warmup_safety_gate_v2_calibrated/"
    }
]

manifest = {
    "entry_command": "python transformer_project/run_stage2_dbpedia_warmup_safety_gate_calibrated.py",
    "dataset": "dbpedia_14",
    "run_mode": "full",
    "models": ["medium-4", "large-6"],
    "seeds": [42, 123, 456],
    "conditions": [
        "buggy_no_guard",
        "buggy_failfast_guard",
        "buggy_rescue_guard",
        "stable_with_guard",
        "fixed_order_with_guard",
        "single_bad_first_step_then_stable"
    ],
    "threshold_source": "calibrated_10x_safe_max",
    "threshold_rule": "threshold = 10 * max safe early update ratio from stable_with_guard and fixed_order_with_guard",
    "expected_output": RUNS[0]["expected_output"]
}

manifest_path = PROJECT_DIR / "stage2_dbpedia_warmup_safety_gate_calibrated_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("=" * 80)
print("DBPEDIA CALIBRATED WARMUP SAFETY GATE DRIVER")
print("=" * 80)
print("Manifest:", manifest_path)

for run in RUNS:
    script_path = PROJECT_DIR / run["script"]
    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=str(PACKAGE_ROOT)
    )

print("\nFinished DBPedia calibrated warmup safety gate run.")
print("Expected output:", RUNS[0]["expected_output"])

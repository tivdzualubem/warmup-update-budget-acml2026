"""Standalone Kaggle readiness check. No installs, training, or dataset download.

Paste this entire file into one Python notebook cell, or run it as a script.
It makes one public HTTPS request for dataset metadata and writes a small report.
It does not read credentials, private files, or other notebooks.
"""
from pathlib import Path
import importlib.metadata
import json
import platform
import shutil
import time
import urllib.request

DATASET = "fancyzhx/dbpedia_14"
REVISION = "9abd46cf7fc8b4c64290f26993c540b92aa145ac"


def check():
    started = time.perf_counter()
    output = Path("/kaggle/working")
    if not output.is_dir():
        output = Path.cwd()
    report = {
        "purpose": "Preliminary runtime compatibility check for the ACML peak-first pair",
        "research_training_run": False,
        "packages_installed_by_this_script": False,
        "full_dataset_downloaded": False,
        "python": platform.python_version(),
        "platform": platform.system(),
        "kaggle_working_directory_present": Path("/kaggle/working").is_dir(),
        "packages": {},
        "checks_to_resolve": [],
    }
    for name in ("torch", "numpy", "datasets", "huggingface-hub", "pyarrow"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
            report["checks_to_resolve"].append("Library missing: " + name)

    disk = shutil.disk_usage(output)
    report["disk"] = {"free_GiB": round(disk.free / 2**30, 2),
                      "total_GiB": round(disk.total / 2**30, 2)}
    report["system_RAM"] = {"available_GiB": None, "total_GiB": None}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
        report["system_RAM"] = {
            "available_GiB": round(values.get("MemAvailable", 0) / 2**30, 2),
            "total_GiB": round(values.get("MemTotal", 0) / 2**30, 2),
            "scope": "OS-reported RAM; may differ from a notebook/container's enforced limit",
        }

    report["cuda"] = {"available": False, "small_tensor_check_passed": False}
    try:
        import torch
        report["cuda"]["torch_build"] = str(torch.__version__)
        report["cuda"]["cuda_build"] = torch.version.cuda
        report["cuda"]["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda"]["devices"] = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["cuda"]["devices"].append({
                    "index": index, "name": props.name,
                    "total_VRAM_GiB": round(props.total_memory / 2**30, 2),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                })
            # One GPU only, as in the research protocol. No model or optimizer.
            with torch.cuda.device(0), torch.no_grad():
                free, total = torch.cuda.mem_get_info()
                report["cuda"]["device_0_free_VRAM_GiB"] = round(free / 2**30, 2)
                a = torch.ones((128, 128), device="cuda:0", dtype=torch.float32)
                b = a @ a
                torch.cuda.synchronize(0)
                report["cuda"]["small_tensor_check_passed"] = bool(torch.all(b == 128).item())
                del a, b
            if not report["cuda"]["small_tensor_check_passed"]:
                report["checks_to_resolve"].append("Basic GPU arithmetic check failed")
        else:
            report["checks_to_resolve"].append("No CUDA GPU available in this notebook session")
    except Exception as error:
        report["cuda"]["error"] = type(error).__name__ + ": " + str(error)[:240]
        report["checks_to_resolve"].append("Torch/GPU check could not complete")

    # NumPy must actually import; a distribution version alone is insufficient.
    try:
        import numpy as np
        report["numpy_import_passed"] = bool(np.array([1, 2], dtype=np.int64).sum() == 3)
    except Exception as error:
        report["numpy_import_passed"] = False
        report["numpy_import_error"] = type(error).__name__ + ": " + str(error)[:240]
        report["checks_to_resolve"].append("NumPy import check failed")

    report["dataset_access"] = {"dataset": DATASET, "revision": REVISION,
                                "metadata_reachable": False, "revision_matches": False}
    url = "https://huggingface.co/api/datasets/" + DATASET + "/revision/" + REVISION
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ACML-runtime-check/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            metadata = json.loads(response.read(2_000_000))
        report["dataset_access"]["metadata_reachable"] = True
        report["dataset_access"]["revision_matches"] = metadata.get("sha") == REVISION
        if metadata.get("sha") != REVISION:
            report["checks_to_resolve"].append("Dataset metadata did not confirm the pinned revision")
    except Exception as error:
        report["dataset_access"]["error"] = type(error).__name__ + ": " + str(error)[:240]
        report["checks_to_resolve"].append("Public dataset metadata request failed; check Internet availability")

    report["session_limits"] = {
        "remaining_GPU_quota": "Not determined by this script; check Kaggle's displayed quota",
        "session_time_limit": "Not determined by this script; check the current session allowance",
    }
    report["not_yet_verified"] = [
        "Import/compatibility of the complete dataset-loading stack",
        "Full pinned dataset download, preprocessing and split identity",
        "Actual-model GPU memory, real-data resume correctness and training throughput",
        "Whether the fixed pair fits the available session time, quota and storage",
    ]
    report["status"] = ("CHECKS_NEED_ATTENTION" if report["checks_to_resolve"]
                        else "INITIAL_CHECKS_PASSED_REAL_MODEL_PREFLIGHT_STILL_REQUIRED")
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    path = output / "ACML_Kaggle_Check_Report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nReport saved to:", path)
    print("Send this report back. No experiment has been started.")
    return report


if __name__ == "__main__":
    check()

"""Run the fixed ACML clean/peak-first pair after the verified Kaggle preflight.

THIS STARTS THE RESEARCH EXPERIMENT. Run one cell in the SAME Kaggle notebook.
No installations, changed settings, extra seeds, recovery extension or new guard.
Re-running this launcher never starts a second pair. It resumes a valid interrupted
pair, or packages an already completed pair. Preserve all files between sessions.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

PREFLIGHT_ROOT = Path("/kaggle/working/ACML_preflight_20261112_191929_5acfd7")
RUN_ROOT = Path("/kaggle/working/ACML_peak_pair_20261112_190032_158001")
EXPECTED_IDENTITY = {
    "protocol": "ea6bfb615e48e63055f85577ee570f82b0517b99e00e7a762e54de07fd962be6",
    "code": "2f6991b0811855a82b51db14515b9a38353cdb304b79890e2184a65f6c18a474",
    "data": "ff41c9d40d42ee5992f4faa6e101200fde6a32ac99da4e6853bdb344713dfbf0",
    "environment": "fa6fe07d5dfa3d4deef98c30659ecd1c30daea4254cf8cbc5b1b181723f1734f",
}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preflight():
    package = PREFLIGHT_ROOT / "ACML_Peak_First_Pair"
    directory = PREFLIGHT_ROOT / "real_data_check"
    report = json.loads((directory / "preflight.json").read_text())
    if report.get("status") != "passed" or not report.get("resume_probe_exact") or not report.get("first_transition_anchors_match"):
        raise ValueError("The required real-data/device checks have not passed.")
    if report["identity"] != EXPECTED_IDENTITY:
        raise ValueError("This is not the preflight report already reviewed. Preserve files and return the new report.")
    for key, name in (("protocol", "protocol"), ("code", "source_hashes"), ("data", "data_manifest"), ("environment", "environment")):
        if json_digest(json.loads((directory / (name + ".json")).read_text())) != EXPECTED_IDENTITY[key]:
            raise ValueError("Changed preflight record: " + name)
    hashes = json.loads((directory / "source_hashes.json").read_text())
    for name, expected in hashes.items():
        path = package / name
        path.resolve().relative_to(package.resolve())
        if file_digest(path) != expected:
            raise ValueError("Checked source file changed: " + name)
    if json_digest(json.loads((package / "protocol.json").read_text())) != EXPECTED_IDENTITY["protocol"]:
        raise ValueError("Fixed protocol changed. Do not alter the horizon or seed after observing results.")
    return package, directory / "preflight.json"


def tail_records(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def run_mode(run):
    if (run / "result.json").exists():
        result = json.loads((run / "result.json").read_text())
        if result.get("status") != "completed" or result.get("identity") != EXPECTED_IDENTITY:
            raise ValueError("Existing result has an unexpected status or identity.")
        if result.get("fixed_positive_lr_steps_per_arm") != 49999 or result.get("zero_lr_steps_per_arm") != 0:
            raise ValueError("Existing result does not match the fixed positive-LR budget.")
        return "completed"
    if not run.exists():
        return "new"
    for log in sorted((run / "logs").glob("segment_*.jsonl")):
        if any(row.get("event") == "failure" for row in tail_records(log)):
            raise RuntimeError("A recorded run failure needs diagnosis. Return the reports; do not silently retry or replace the seed.")
    checkpoint = run / "checkpoints/latest.pt"
    if checkpoint.exists() and checkpoint.with_suffix(".sha256.json").exists():
        return "resume"
    raise RuntimeError("An incomplete run folder exists without a verified latest checkpoint. Preserve it and return its logs.")


def progress(run, minutes):
    logs = list((run / "logs").glob("segment_*.jsonl"))
    if not logs:
        return f"{minutes:.1f} minutes: loading data and verifying the environment."
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    rows = tail_records(latest)
    steps = [r["step"] for r in rows if isinstance(r.get("step"), int)]
    if steps:
        step = max(steps)
        return f"{minutes:.1f} minutes: paired transition {step:,}/49,999 ({100*step/49999:.1f}%)."
    return f"{minutes:.1f} minutes: initializing/evaluating the pair."


def build_returns(package, run, launch_directory):
    groups = {"RESULTS": [], "PREDICTIONS": [], "STATES": []}
    for path in sorted(run.rglob("*")) if run.exists() else []:
        if not path.is_file() or path.suffix == ".tmp":
            continue
        rel = path.relative_to(run)
        group = "STATES" if path.suffix == ".pt" else "PREDICTIONS" if rel.parts[0] == "predictions" else "RESULTS"
        groups[group].append((path, "run/" + rel.as_posix()))
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "CHECKSUMS.json":
            groups["RESULTS"].append((path, "code/" + path.relative_to(package).as_posix()))
    for path in sorted(launch_directory.glob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            groups["RESULTS"].append((path, "launcher/" + path.name))
    for name in ("preflight.json", "protocol.json", "environment.json", "source_hashes.json", "data_manifest.json"):
        path = PREFLIGHT_ROOT / "real_data_check" / name
        groups["RESULTS"].append((path, "preflight/" + name))
    for name in ("execution_settings.json", "ACML_Kaggle_Preflight_Report.json", "01_cpu_checks.log", "02_real_data_check.log"):
        path = PREFLIGHT_ROOT / name
        if path.exists():
            groups["RESULTS"].append((path, "preflight/" + name))
    manifest = {name: {"bytes": path.stat().st_size, "sha256": file_digest(path), "bundle": group}
                for group, files in groups.items() for path, name in files}
    manifest_path = launch_directory / "ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    returned = []
    for group, files in groups.items():
        if not files:
            continue
        output = RUN_ROOT / ("ACML_Peak_First_" + group + ".zip")
        temp = output.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            archive.write(manifest_path, "ARTIFACT_MANIFEST.json")
            for path, name in files:
                archive.write(path, name)
        os.replace(temp, output)
        returned.append({"file": str(output), "bytes": output.stat().st_size, "sha256": file_digest(output)})
    (RUN_ROOT / "RETURN_ARCHIVES.json").write_text(json.dumps(returned, indent=2) + "\n")
    return returned


def main():
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this script in the same Kaggle notebook as the passed preflight.")
    import fcntl
    lock_path = Path(str(RUN_ROOT) + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This pair is already running. Do not start another copy.")
        package, preflight_path = verify_preflight()
        RUN_ROOT.mkdir(exist_ok=True)
        run = RUN_ROOT / "run"
        launch_directory = RUN_ROOT / "launcher"
        launch_directory.mkdir(exist_ok=True)
        attempt = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        status = {"attempt": attempt, "preflight_identity": EXPECTED_IDENTITY, "research_process_launched": False}
        try:
            mode = run_mode(run)
            status["mode"] = mode
            if mode != "completed":
                if shutil.disk_usage(RUN_ROOT).free < 4 * 2**30:
                    raise RuntimeError("Less than 4 GiB disk space remains for retained evidence and return archives. Preserve all files and report this.")
                overrides = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
                             "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
                             "HF_HOME": "/kaggle/working/acml_hf_cache", "CUDA_VISIBLE_DEVICES": "0"}
                checked_overrides = json.loads((PREFLIGHT_ROOT / "execution_settings.json").read_text())["research_and_preflight_environment_overrides"]
                if overrides != checked_overrides:
                    raise ValueError("Execution settings differ from the checked preflight.")
                environment = dict(os.environ, **overrides)
                command = [sys.executable, "-u", "-B", "peak_pair.py", "run", "--device", "cuda:0",
                           "--output", str(run), "--preflight", str(preflight_path)]
                if mode == "resume":
                    command.append("--resume")
                status["command"] = command
                status["environment_overrides"] = overrides
                status["research_process_launched"] = True
                log = launch_directory / (attempt + ".log")
                print("Starting" if mode == "new" else "Resuming", "the fixed clean/peak-first pair on one GPU.", flush=True)
                print("49,999 positive-LR transitions per arm. No extra recovery training.", flush=True)
                print("Timing projection: about 102 minutes for training/validation, plus other costs. Allow extra time.", flush=True)
                started = time.monotonic()
                with log.open("w", encoding="utf-8") as output:
                    # The child retains the same lock even if the notebook parent
                    # disappears, preventing a second launcher from duplicating it.
                    process = subprocess.Popen(command, cwd=package, env=environment, stdout=output,
                                               stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    try:
                        while True:
                            try:
                                code = process.wait(timeout=60)
                                break
                            except subprocess.TimeoutExpired:
                                print(progress(run, (time.monotonic()-started)/60), flush=True)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                status["elapsed_minutes"] = (time.monotonic()-started)/60
                status["exit_code"] = code
                if code:
                    print("\n".join(log.read_text(errors="replace").splitlines()[-40:]), flush=True)
                    raise RuntimeError("The run stopped with an error. Preserve all output and return the RESULTS archive.")
            if run_mode(run) != "completed":
                raise RuntimeError("No verified completion record exists.")
            status["status"] = "COMPLETED"
            print("Both arms completed the fixed budget. Packaging the retained evidence.", flush=True)
        except BaseException as error:
            status["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "STOPPED"
            status["error_type"] = type(error).__name__
            status["error"] = str(error)
            (launch_directory / (attempt + "_traceback.txt")).write_text(traceback.format_exc(), encoding="utf-8")
            print(status["status"] + ": " + str(error), flush=True)
        finally:
            (launch_directory / (attempt + ".json")).write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            archives = build_returns(package, run, launch_directory)
            print("\nSTATUS:", status["status"], flush=True)
            for archive in archives:
                print(archive["file"], "  ", round(archive["bytes"]/2**20, 1), "MiB", flush=True)
            print("Send RESULTS and PREDICTIONS back for analysis. Also download and preserve STATES; it contains the model/Adam/RNG evidence.", flush=True)
            print("Keep the complete run folder, preflight folder and dataset cache. No files have been deleted.", flush=True)
        return status


if __name__ == "__main__":
    main()

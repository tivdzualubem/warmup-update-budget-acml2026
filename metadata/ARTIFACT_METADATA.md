# Reproducibility Metadata

## Artifact repository

- Repository tag: `acml2026-artifacts-v1`
- Release revision: resolved by the repository tag `acml2026-artifacts-v1`
- Original training-code commit: not available
- Frozen source version: source files included in this artifact repository

## Execution environment

- Python: `3.12.3`
- Python build: `main, Mar 23 2026, 19:04:32`
- Compiler: `GCC 13.3.0`
- Python executable: `python3` from a dedicated virtual environment (absolute path omitted for anonymity)
- PyTorch version: `2.10.0+cu130`
- PyTorch CUDA build: `13.0`
- cuDNN: `91501`
- CUDA available: `True`
- CUDA device count: `1`

## GPU

- Model: `NVIDIA GeForce RTX 5090`
- Memory: `31.36 GiB`
- Compute capability: `12.0`

## Operating system

- Distribution: `Ubuntu 24.04.3 LTS`
- Kernel: `6.17.0-29-generic`
- Architecture: `x86_64`

## CPU and memory

- CPU: `Intel(R) Core(TM) i9-14900KF`
- System RAM: `62 GiB`
- Swap: `8.0 GiB`

## Experiment launch commands

- AG News: `python transformer_project/run_stage2_agnews_warmup_safety_gate_calibrated.py`
- DBpedia: `python transformer_project/run_stage2_dbpedia_warmup_safety_gate_calibrated.py`

## Data construction

- Validation data was created only from the official training split.
- Validation fraction: `15%`
- Split seed: `42`
- Vocabulary was constructed only from the remaining training subset.
- Minimum token frequency: `2`
- Maximum vocabulary size: `30,000`

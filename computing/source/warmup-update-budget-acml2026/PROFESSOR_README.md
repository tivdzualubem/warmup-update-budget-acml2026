# Full publication expansion: reproducible professor execution handoff

**Project:** Warmup Is an Update Budget: Scheduler Semantics and First-Step Adam Impulses  
**Supervisor:** Professor Bader Rasheed  
**Frozen source commit:** `bfcbc0bae87d6969fcd4d416cdc457de52c92048` (`publication-expansion`)

## Read this first

The accompanying Kaggle `ACML_Reduced_Validation.ipynb` is **only an end-to-end integration validation**. Its tiny real-data subsets, one seed and shortened scratch training do **not** establish scientific accuracy, instability rates, calibration or effect sizes. Never merge its `validation_runtime_results/` into the publication results directory. The **full** commands below use the unmodified frozen job runners, normal datasets, split seed 42, all five experiment seeds and original hyperparameters. No pretrained or scratch results from the notebook substitute for a full matrix job.

The complete planned full expansion is **240 condition × seed × environment jobs**: 120 scratch (2 datasets × 2 models × 5 seeds × 6 conditions); 120 pretrained (4 datasets × 5 seeds × 6 conditions). Some conditions terminate diagnostically; there are 180 nominal full-training runs plus 60 diagnostic early-stop conditions. This is a substantial accelerator workload. The code ships **without** downloaded Hugging Face checkpoints/data or optimizer weights; the professor machine downloads pinned revisions once online, then can reuse cached files. `Hugging Face` Internet/cache access is required initially.

## Environment

Use a fresh compatible Python 3.12 CUDA environment, e.g. with a PyTorch build appropriate for the GPUs. Known Kaggle runtime tested: Python 3.12.13; torch 2.10.0+cu128; torchvision 0.25.0+cu128; transformers 5.17.0; datasets 5.0.0; numpy 2.0.2; Pillow 12.3.0; scikit-learn 1.9.1; scipy 1.18.1; seqeval 1.2.2. Install missing dependencies in an isolated environment. Hardware may need different PyTorch builds. The pinned model/dataset commits and split/optimizer manifests are in `publication_expansion/configs/` and `publication_expansion/manifests/`. Do not silently change hyperparameters or torch/transformers semantics.

From the unpacked repository root:

```bash
python -m pip install 'datasets==5.0.0' 'transformers==5.17.0' 'Pillow==12.3.0' 'scikit-learn==1.9.1' 'scipy==1.18.1' 'setuptools_scm==7.1.0'
python -m pip install --no-build-isolation 'seqeval==1.2.2'
python -m unittest discover -s publication_expansion/tests -t . -q
```

Ensure `torch.cuda.is_available()` reports true on GPU machines. Check package availability and test results before running expensive jobs. We observed 100 passing unit tests on Kaggle in the original frozen code. A new runtime must reproduce the tests.

## Full study order (restartable, phase-by-phase)

All commands below launch actual full runs and save raw `result.json`, `trace.json`, `history.json`, `job_manifest.json`, statuses and optimizer/model state/checkpoints as implemented by the original runners. They resume completed jobs and can resume unfinished jobs at epoch boundaries. Change `--gpus 0 1` to the professor's available GPU IDs (e.g., only `--gpus 0`). Two concurrent model processes each need their own usable GPU RAM. For planning, omit `--execute` to list exact proposed commands without executing.

```bash
# First: scratch SAFE; this creates unbiased independent seed references.
python -u professor_full_matrix.py --phase scratch-safe --gpus 0 1 --execute
python -u professor_full_matrix.py --phase scratch-thresholds --execute

# Then scratch causal and guarded conditions; threshold manifest created above.
python -u professor_full_matrix.py --phase scratch-nonguarded --gpus 0 1 --execute
python -u professor_full_matrix.py --phase scratch-guarded --gpus 0 1 --execute

# Pretrained reference across all 5 seeds, 3 full epochs each.
python -u professor_full_matrix.py --phase pretrained-safe --gpus 0 1 --execute
python -u professor_full_matrix.py --phase pretrained-thresholds --execute

# Pretrained causal, rescue, and diagnostic comparisons.
python -u professor_full_matrix.py --phase pretrained-nonguarded --gpus 0 1 --execute
python -u professor_full_matrix.py --phase pretrained-rescue --gpus 0 1 --execute
python -u professor_full_matrix.py --phase pretrained-diagnostic --gpus 0 1 --execute

# Full first-transition severity grid (7 intentional LR multipliers + natural mismatch).
python -u professor_full_matrix.py --phase pretrained-severity --gpus 0 1 --execute

# Check cross-condition matched first-batch/model/gradient anchors.
python -u professor_full_matrix.py --phase pretrained-anchor-audit --execute
```

The LOSO (leave-one-seed-out) threshold builders require safe reference results from *all five* seeds, and calibrate only from **other-seed safe jobs**. They must finish **before** running guarded conditions. The output `loso_thresholds.json` is the auditable source of the held-out calibration thresholds; do not use the Kaggle pilot thresholds or a fallback/default for the publication run. First-step/early traces capture the measured intended, candidate/consumed and committed LRs and update ratios, with the dense early window recorded through step 300. Build sensitivity analyses/figures from the original traces, not quick-validation CSVs.

## Results and review

- Scratch: `publication_expansion/results/scratch/{dataset}/{medium-4|large-6}/seed{seed}/{condition}/`
- Pretrained: `publication_expansion/results/pretrained/{dataset}/seed{seed}/{condition}/`
- LOSO calibration files: `publication_expansion/results/{scratch|pretrained}/loso_thresholds.json`
- Independent command logs: `publication_expansion/results/full_run_logs/`
- Frozen reference: `publication_expansion/manifests/experiment_matrix_v1.json`

Review all five seed-paired results and traces for actual optimizer LR, safe maxima, threshold sources, finite losses, first-batch/model/gradient anchors, precision and scheduler state. Diagnostic stops are expected in some conditions; an update guard is not guaranteed to trigger on every pretrained task (the earlier WNUT-17 pilot did **not** trigger at κ=10). Make no claims of universal detection. Validate all missing/failed jobs before statistical summaries.

## Packaging/provenance

The Kaggle validation notebook creates the full professor bundle **only if the 48-job reduced rehearsal passes**. It includes the exact frozen source checkout (not a recreated sketch), this runbook, `professor_full_matrix.py`, frozen configuration manifests, reduced-validation audit and compact logs. It excludes heavyweight model weights and does not claim that full runs have been done. Preserve the tarball SHA-256 recorded in the output and share this README and archive with the professor.

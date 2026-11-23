# Configured-peak first-step pair

Prepared 12 November 2026. This is the separate implementation of the small matched comparison requested in the professor's 8 November feedback. **No research run or real-data/device preflight has been performed.** The existing manuscript remains on hold.

The fixed configuration is DBpedia, medium-4, seed 42 and 49,999 positive-LR transitions per arm. We reuse the base configuration and prefix budget from the professor's earlier specified setup, applying his 8 November priority: the clean first LR is 1.25e-8 and the perturbed first LR is 1e-4, with identical subsequent scheduled rates. This adaptation implements the requested small comparison; it does not imply that the November message repeated every configuration value. No further scope-approval request is required. This is a new pair from initialization; the additional 50,000-step-per-arm recovery extension remains excluded.

## What is ready

The runner has explicit paired batches, separate optimizer and random states, first-transition identity checks, consumed-rate assertions, fixed stopping, sample-weighted uncapped logit CE, retained predictions and complete paired checkpoints. Synthetic tests cover interruption/resume, dropout, the actual medium-4 architecture, uneven evaluation batches, nonfinite failures, checkpoint corruption and a full small simulated run. See `evidence/cpu_test_report.json` and `evidence/cpu_test_log.txt` for the tested version and results.

These tests do not certify the real dataset pipeline, the execution GPU, numerical behavior on that device, throughput or the research outcome. Real data are not bundled. The six files under `frozen_source/` are unchanged historical adapters; `FROZEN_SOURCE_HASHES.json` records them. New paired sampling uses an explicit permutation/cursor and does not reproduce the old DataLoader's implicit RNG consumption. Historical results and selections are untouched.

## Read and check locally

Commands below are run from this extracted package folder. Plain `python peak_pair.py` only describes the proposal and does not import Torch or download data.

```text
python -B peak_pair.py describe
python -B check_cpu.py
```

The synthetic checks used Python 3.12.14, Torch 2.8.0+cpu and NumPy 2.5.4. `requirements-cpu-tests.txt` records those package versions. Use a separate environment; do not replace the existing research environment blindly. For actual GPU execution, install the Torch build appropriate to that machine. `requirements-data.txt` supplies the preserved adapter's dataset-library versions as a starting point, not a certified new host lock. The preflight records every installed distribution and the actual device/software settings; its identity must match the research run and any resume.

## Execution on the professor's training machine

The following commands implement the requested execution checks and small comparison. They have not been executed on the professor's machine from this workspace. The real-data preflight may download the pinned official dataset, performs discarded short training probes, evaluates the validation split, writes full diagnostic states and measures cost. Use a fresh output directory. Test data are loaded and hashed for provenance but are not evaluated during preflight.

```text
python -B check_cpu.py
python -B peak_pair.py preflight --device cuda:0 --output outputs/preflight_01
```

Read `outputs/preflight_01/preflight.json` before proceeding. It must say `passed`, with an exact uninterrupted/resumed match on the actual device and dataset. Check the measured cost and available disk/memory. If deterministic execution fails, preserve the failure; do not silently disable determinism. CUDA memory figures are whole-process peaks while an arm runs, with both models resident, and must not be added as independent per-model requirements. CPU peak RAM is not measured. Preflight time projections are approximate and exclude some evaluation, output and checkpoint costs.

After the checks pass and the measured resource requirement fits the execution machine, proceed with the fixed pair:

```text
python -B peak_pair.py run --device cuda:0 --output outputs/pair_01 --preflight outputs/preflight_01/preflight.json
```

If interrupted, keep the entire output folder. Resume with the same source, configuration, data and environment:

```text
python -B peak_pair.py run --device cuda:0 --output outputs/pair_01 --preflight outputs/preflight_01/preflight.json --resume
```

Do not overwrite another run, change a seed or extend the horizon after seeing outcomes. A failed arm invalidates the in-memory pair; only the last complete verified paired checkpoint is resumable. A checkpoint checksum failure stops execution. An interruption between replacing a checkpoint and its checksum may need explicit recovery from a retained earlier pair; preserve both files and investigate rather than bypassing the check. Load checkpoint files only from this trusted run.

## What to return after a completed pair

Keep the complete `outputs/pair_01/` and `outputs/preflight_01/` folders, plus this exact source package. There is no automatic deletion or upload. If checkpoints make a ZIP too large, retain them in accessible storage and return their locations and hashes alongside the small records; do not discard them.

| Output | Purpose |
|:--|:--|
| `protocol.json`, `source_hashes.json`, `environment.json` | Frozen design, executed source identities and installed environment |
| `data_manifest.json`, `vocabulary.json`, `split_indices.npz` | Pinned dataset identity, text/label digests, vocabulary and actual population indices |
| `checkpoints/` | Both arms' model/Adam/RNG states, sampler permutation/cursor, counters and hashes; retained at 0, 1, 8,000 and 49,999 plus latest |
| `predictions/` | Logits, labels and official indices at every fixed validation/training-subset evaluation, plus fixed-final test |
| `evaluation_history.json` | Absolute CE and accuracy for both arms at every prespecified evaluation |
| `online_training_history.json` | Example-weighted online CE over successive 100-step blocks, including the final shorter block |
| `logs/segment_*.jsonl` | Every consumed LR, batch identity, early/checkpoint displacement, timing, evaluation, checkpoint and failure events |
| `result.json` | Fixed-final primary CE contrast, secondary accuracy contrast, absolute metrics, positive-LR counts and recorded cost |

The training monitor is a fixed 8,192-example subset, not full-training accuracy. Online CE is a moving-training-state statistic, distinct from checkpoint evaluation. No validation-best selection or early stopping is used. Test metrics appear only after both final training states exist.

`committed_trajectory_cost` describes the retained trajectory. `recorded_attempt_cost` additionally counts logged repeated work after an interruption. Abruptly killed segments and partially completed failed steps can leave unmeasured cost; the report exposes these gaps. Checkpoint write time is shared by the pair. A saved state's cumulative checkpoint time excludes its own write; use the segment logs for recorded checkpoint totals. Prediction-file bytes are separated by arm; paired checkpoint file bytes are shared and not double-counted. Timing includes instrumentation, finite checks and selected displacement measurements; it is not a claim about an uninstrumented baseline's speed.

## Reading the result

The primary endpoint is validation CE of peak-first minus clean at the fixed final step. Show both curves, both absolute endpoints and the complete positive-LR budget. A single pair cannot establish equivalence or between-seed uncertainty. An endpoint deficit is limited to this configuration and horizon, not permanent damage. A shrinking gap caused by deterioration of the clean arm is not evidence of recovery by the perturbed arm. Preserve a small or null-looking effect.

The next manuscript update follows the returned real results and their checks. The professor selects the venue separately. The full recovery extension, broader benchmarks and a new guard remain deferred.

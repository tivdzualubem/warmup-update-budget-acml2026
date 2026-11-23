
# Warmup Is an Update Budget - ACML 2026 Reproducibility Artifacts

This repository contains the anonymous source code, experiment configurations, paper-ready results, analysis notebooks, conference PDFs, execution metadata, audit records, and integrity checks associated with the ACML 2026 submission:

**Warmup Is an Update Budget: Scheduler Semantics and First-Step Adam Impulses**

## Repository structure

* `src/transformer_project/`: canonical experiment implementation and calibrated run scripts.
* `configs/`: experiment and artifact configuration files.
* `results/ag_news/`: AG News paper-ready tables, summaries, figures, and supporting outputs.
* `results/dbpedia/`: DBpedia paper-ready tables, summaries, figures, and supporting outputs.
* `notebooks/`: cleaned AG News and DBpedia analysis notebooks.
* `paper/`: current anonymous main paper and supplementary material.
* `metadata/`: run manifests, execution-environment metadata, and remaining metadata checklist.
* `audit/`: inventory, evidence mapping, and completeness records.
* `checksums/`: SHA-256 integrity checks.
* `scripts/`: repository and artifact utility scripts.
* `artifacts/`: assembled, extracted, and provenance-preserving artifact materials used to prepare the release archives.

## Central experimental design

The calibrated core contains 72 expected experimental combinations:

* 2 datasets: AG News and DBpedia
* 2 model sizes: medium-4 and large-6
* 6 conditions: stable, corrected, legacy/no guard, fail-fast, rescue, and single-bad
* 3 random seeds: 42, 123, and 456

Fail-fast runs terminate after detecting an unsafe update and must not be interpreted as fully trained classifiers.

## Environment setup

From the repository root:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The recorded execution environment is documented in:

```
metadata/ARTIFACT_METADATA.md
```

Any metadata still awaiting confirmation is listed in:

```
metadata/REQUIRED_METADATA_TO_FILL.txt
```

## Running the calibrated experiments

Run the experiment entry points from the `src` directory:

```
cd src

python transformer_project/run_stage2_agnews_warmup_safety_gate_calibrated.py

python transformer_project/run_stage2_dbpedia_warmup_safety_gate_calibrated.py
```

The experiments are computationally intensive. The archived results and paper-ready summaries are provided so that the reported findings can be inspected without rerunning every training configuration.

## Data construction

For both datasets:

* Official train and test splits were used.
* The validation set was created only from the official training split.
* Validation fraction: 15%.
* Split seed: 42.
* Vocabulary construction used only the remaining training subset.
* Minimum token frequency: 2.
* Maximum vocabulary size: 30,000.

## Evidence traceability

The mapping between the paper claims, tables, figures, and archived evidence is documented in:

```
audit/EVIDENCE_MAP.md
```

Primary paper-ready outputs are located under:

```
results/ag_news/paper_ready/
results/dbpedia/paper_ready/
```

Run manifests are located under:

```
metadata/
src/transformer_project/
```

## Integrity verification

Verify tracked artifact checksums from the repository root with:

```
sha256sum -c checksums/SHA256SUMS.txt
```

## Release archives

The complete artifact release is divided into five archives:

* `acml_artifacts_core.zip`
* `acml_raw_agnews.zip`
* `acml_raw_dbpedia_core.zip`
* `acml_raw_dbpedia_scheduler_medium4.zip`
* `acml_raw_dbpedia_scheduler_large6.zip`

The split release keeps the core repository compact while preserving the complete raw scheduler traces and calibrated run outputs.

## Reproducibility policy

Reported values must be traceable to archived calibrated outputs or derived paper-ready tables. Missing environment or provenance information must remain explicitly marked as pending rather than reconstructed by assumption.

This repository and its documents are prepared for anonymous review. Personal identities and machine-specific local paths are intentionally excluded.

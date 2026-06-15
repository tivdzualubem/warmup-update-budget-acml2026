# Warmup as a Realized Update Budget — ACML 2026 Artifact Package

This package collects the calibrated Warmup Safety Gate evidence used by the ACML 2026 anonymous submission and its supplement.

## Contents

- `raw/calibrated_runs/`: extracted calibrated AG News and DBpedia runs, including per-seed outputs, scheduler traces, guard events, early diagnostics, manifests, requirements, and source code.
- `raw/dbpedia_pretrained_context/`: contextual pretrained DBpedia package; not part of the central scratch-training causal claim.
- `paper_outputs/`: paper-ready tables, paired-seed summaries, diagnostic figures, and documentation tables.
- `notebooks/`: AG News and DBpedia analysis notebooks.
- `conference_pdfs/`: current anonymous main paper and supplement.
- `source_archives/`: original uploaded ZIP packages retained unchanged for provenance.
- `audit/`: inventory, evidence map, and completeness checks.
- `metadata/`: environment and repository metadata. Missing fields are listed explicitly and must not be guessed.
- `checksums/`: SHA-256 checksums for integrity verification.

## Central experimental design

The calibrated core contains 72 expected run combinations:

- 2 datasets: AG News, DBpedia
- 2 models: medium-4, large-6
- 6 conditions: stable, corrected, legacy/no guard, fail-fast, rescue, single-bad
- 3 seeds: 42, 123, 456

Fail-fast runs stop after detection and should not be interpreted as trained classifiers.

## Reproducibility policy

All reported values must be traceable to the raw calibrated outputs or derived paper tables. Missing hardware, software, or commit metadata must be filled from the original execution environment rather than reconstructed by assumption.

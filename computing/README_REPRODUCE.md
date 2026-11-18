# Computing revision responding to the 13 November feedback

This revision includes the completed configured-peak pair requested in the 8 November feedback, independently checked against all returned logs, predictions and retained full states. It preserves the title and all historical selections. The selected venue is Computing (Springer). The main manuscript uses its official SVJour3 smallextended template. All three author affiliations and institutional emails, corresponding-author details and the user-confirmed competing-interest declaration are included.

The title and archived checkpoint selections are unchanged. Main and supplement are in manuscript/, as Markdown, complete LaTeX and compiled PDFs. references.bib supplies the bibliography; figures/ contains the images and make_figures.py, make_submission_figures.py and make_peak_figures.py and make_computing_figures.py regenerate them. Historical evidence limits remain explicit in the supplement and ARTIFACT_AVAILABILITY.md.

## Versioned repository

https://github.com/tivdzualubem/warmup-update-budget-acml2026/tree/computing-submission-2026-11-18

The `computing/` directory in that version is the current reproducibility package. The repository root also preserves the earlier calibrated, rebuttal and publication-expansion code. See the repository RESEARCH_INDEX.md for the cohort map and external evidence boundaries. From a Git checkout, change into `computing/` before using the commands below.

## Reproduction commands

Use Python 3.12 with the exact local versions in requirements-analysis.txt. PyTorch is CPU-only for these numerical fixtures. From this package root:

```text
python reproduce_revision.py --analysis-only
```

This rebuilds saved-record analyses, figures, audits and Markdown, then runs all 51 tests. It downloads no datasets/models and launches no research training. It retains the supplied PDFs. The full build additionally requires Pandoc 3.6.1 and Tectonic 0.17.0 on PATH (or executable paths in PANDOC and TECTONIC):

```text
python reproduce_revision.py
```

Tectonic's first compilation may fetch standard TeX packages/fonts. Use its populated cache for offline compilation. No Windows system font is required; the journal class uses standard TeX fonts. The rendered PDFs were generated from the delivered .tex and .bib sources, not from a separate PDF-only renderer.

To compile the delivered LaTeX directly without Python or Pandoc, change into manuscript/ and run:

```text
tectonic --untrusted Research_Manuscript.tex
tectonic --untrusted All_Seed_Results_and_Provenance.tex
```

The sources also use a pdfLaTeX-compatible preamble: pdflatex on the main, bibtex on Research_Manuscript, then pdflatex twice; run pdflatex twice on the supplement. The locally tested compiler is Tectonic 0.17.0; no independent pdfLaTeX build is claimed. Keep ../figures/ and references.bib beside the expected relative paths. A complete TeX installation needs the packages named in the source preamble. PDF binary digests can differ with engine/version/timestamps; compare scholarly text, numeric outputs and rendered pages as well.

Individual checks:

```text
python audit_execution_provenance.py
python audit_launcher_hotfix.py
python examples/first_two_steps.py --output evidence/verification/first_two_steps.json
python run_submission_checks.py
```

## Evidence

The original historical data/ inputs and source/ remain frozen; the new pair is added separately under data/peak_first and peak_pair_runner/. Hash checks preserve 107 original source/data/test/contrast files and 20 original scientific CSVs. The independent record audit checks 240 scientific records,1,573 history rows,55,535 transition rows and 277,675 LR fields. All 40 initialization groups match seven fingerprints. The 49 returned design JSON files, including five negative completion flags, match the original pre-execution launcher; later job records provide execution evidence separately.

The supplied partial AG News console log has 635 matching epoch rows and 40 matching completion reasons. The recovered 11 October hotfix matches its Telegram digest and addresses both observed launcher errors. Its pure string transformation is checked in memory against all four original launchers, without executing its main function or training. Reconstructed wrappers are labeled as such. Original-host application logs/executed bytes remain unverified.

All historical cohorts, five WNUT misses, small pretrained effects, archived selection metrics and positive/zero-LR budgets remain. Historical candidate-epoch weights/validation outputs and historical complete boundary states remain unavailable. The new pair has full retained states and all 42 prediction files. This package reproduces saved-record analysis and CPU fixtures, not original research training. ARTIFACT_AVAILABILITY.md and EXECUTION_PROVENANCE.md state the boundaries.

evidence/verification/ contains logs, environment versions, exact previous-delivery-ZIP reproduction and current checks. Final archive reproduction and page review bind the reviewed outputs to their hashes. SHA256SUMS.txt lists delivery members. Historical reports under evidence/previous_revision/ remain dated evidence, not current completion claims.

## New pair: reproducibility and companion evidence

The main command reconstructs the complete new-pair tables from the included result archive, checks every logged step/rate/batch identity and online-loss block, and rebuilds both papers. It does not launch training. `evidence/peak_first/FULL_AUDIT.json` records the completed independent raw-logit and tensor-state audit.

For a fresh full audit, place the two unchanged companion archives beside this extracted package and run:

```text
python analyse_peak_pair.py --predictions ../ACML_Peak_First_PREDICTIONS.zip --states ../ACML_Peak_First_STATES.zip
```

The predictions and full-state archives are separate because they are approximately 109 MB and 405 MB compressed. Their exact bytes/hashes are in EXTERNAL_EVIDENCE.json. They are available and retained, not missing historical assets. Default reproduction rebuilds saved-record analyses without loading those companions; it does not pretend to recompute raw predictions/states when they are absent. The full audit uses restricted tensor deserialization and computes CE independently with NumPy log-sum-exp.

`peak_pair_runner/` contains the exact executed source and protocol, its Kaggle CPU test report and the frozen adapters. Its original README/proposed protocol are retained provenance and predate the returned outcome; PEAK_FIRST_RESULTS.md is the current result. Its research CLI requires real data/GPU and a matching preflight; do not invoke it to reproduce tables. No rerun, extra recovery extension, expanded benchmark or new guard is required for this revision.

The new run's actual environment (Python 3.13.16, Torch 2.11.0+cu130 and its full distribution list) is in evidence/peak_first/run/environment.json and the result archive. requirements-analysis.txt describes the separate CPU analysis environment. The environments are not asserted interchangeable for exact training trajectories.

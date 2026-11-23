# Learning-rate execution and update-budget research

Current article: **Separating First-Step Learning-Rate Faults from Scheduler Delay in Adam Training**.

Tivdzua Lubem Noah, Bader Rasheed, Dmitriy S. Vatolin.

## SN Computer Science version — 23 November 2026

The current version is prepared for **SN Computer Science**, with tag **sncs-submission-2026-11-23**. The journal submission is still a draft. Journal support instructed the authors to supply supplementary materials as links in the manuscript; the current PDF links directly to the versioned supplement and reproducibility package. This repository publication does not imply journal submission or acceptance.

Start with the [main manuscript](sncs/manuscript/Research_Manuscript.pdf), [supplement](sncs/manuscript/All_Seed_Results_and_Provenance.pdf), [reproduction instructions](sncs/README_REPRODUCE.md), and [research index](RESEARCH_INDEX.md).

**[Download the complete ESM_2 reproducibility ZIP](https://github.com/tivdzualubem/warmup-update-budget-acml2026/raw/refs/tags/sncs-submission-2026-11-23/submission/sncs-2026-11-23/ESM_2.zip)** (about 28 MB). All journal-facing scientific PDFs and flat source archives are under [submission/sncs-2026-11-23/](submission/sncs-2026-11-23/).

The 430 files under sncs/ match the checked ESM_2 ZIP byte-for-byte. Install the documented dependencies, then run:

```text
cd sncs
python reproduce_revision.py --analysis-only
```

This rebuilds saved-record analyses and runs the tests; it does not train models or download datasets. Full document regeneration additionally uses the documented Pandoc and Tectonic versions. The verified delivery passed 51 tests and 560 displayed-number checks, with independent source builds and final PDF review. See [verification records](verification/sncs-2026-11-23/).

This edition adds the structured abstract and journal presentation, the same-metric DBpedia endpoint clarification, explicit pretrained first-step rates/warmups, and the qualified WNUT calibration-window explanation. The title, original scientific results and historical checkpoint selections are preserved; no new experiment was run.

[SNCS_PUBLICATION.json](SNCS_PUBLICATION.json) identifies this version. The main PDF and supplement cite the SNCS tag and the reproducibility download. The 20 November download folder preserves the earlier SNCS draft; use the 23 November files for submission.

## Previous versions and evidence boundaries

The earlier **computing-submission-2026-11-18-r2** tag and computing/ directory remain unchanged. Root training code and the historical, execution and publication_expansion directories retain earlier research stages. Use sncs/ for the current paper; do not combine different experimental cohorts.

Large prediction/state companions remain available from the authors on request, identified in [EXTERNAL_EVIDENCE.json](sncs/EXTERNAL_EVIDENCE.json). Historical Drive files remain registered under historical/; public access is not asserted. Missing original candidate checkpoints and validation predictions remain explicitly unavailable. This repository does not claim complete old-chat coverage or exact reproduction of an unpreserved historical training environment.

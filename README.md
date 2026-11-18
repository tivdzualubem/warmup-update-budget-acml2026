# Learning-rate execution and update-budget research

Current article: **Separating First-Step Learning-Rate Faults from Scheduler Delay in Adam Training**.

Tivdzua Lubem Noah, Bader Rasheed, Dmitriy S. Vatolin. Prepared for submission to *Computing*; this repository does not imply journal acceptance or completed submission.

Use the version **computing-submission-2026-11-18** for the paper. Start with [the research index](RESEARCH_INDEX.md), [current manuscript](computing/manuscript/Research_Manuscript.pdf), [supplement](computing/manuscript/All_Seed_Results_and_Provenance.pdf) and [reproduction instructions](computing/README_REPRODUCE.md).

```text
cd computing
python reproduce_revision.py --analysis-only
```

Install the documented analysis dependencies first. This command reconstructs the saved-record analyses and runs 51 checks; it does not train models. Full source regeneration also requires the documented Pandoc and Tectonic versions. All raw datasets are identified by pinned public sources; they are not required for saved-record analysis.

The root training code and results are historical. The current analysis is in computing/. The older branches, tag and commits are preserved. See historical/README_before_Computing.md for the previous root README.

Large configured-peak prediction/state archives remain separate and available from the authors, with filenames and SHA-256 hashes in computing/EXTERNAL_EVIDENCE.json. Historical Drive archives are registered under historical/. Missing original candidate checkpoints and validation outputs remain explicitly unavailable; repository completeness must not be confused with an exact rerun of an unpreserved historical training environment.

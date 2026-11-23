# Research code, evidence and reproduction index

Current journal edition: **SN Computer Science**, tag **sncs-submission-2026-11-23**. Start with sncs/README_REPRODUCE.md. Scientific downloads are under submission/sncs-2026-11-23/. The computing/ snapshot and all earlier tags remain preserved.

| Research stage | Repository location | What is available |
|---|---|---|
| Original calibrated AG News/DBpedia study | src/, configs/, notebooks/, results/ | Original training code, manifests, notebooks and paper-ready results |
| Historical initial-rate sweep and early diagnostics | historical/recovered_code/ | Original one-script packages, warmup configurations, diagnostic/sweep runners and matching support files |
| Earlier DBpedia pretrained context | historical/recovered_code/ | Recovered pretrained trainer, loader, metrics, runner and configuration/environment records |
| Rebuttal audits | scripts/, results/rebuttal_2026/, REBUTTAL_2026.md | Condition definitions, LR traces, identity, thresholds, severity and assertion comparisons |
| Expanded scratch/pretrained implementations and pilots | publication_expansion/ | Data/model adapters, runners, matrix, calibration, configs, manifests and tests; historical branch preserved |
| Returned October expansion | computing/data/ACML_RESULTS_FOR_STUDENT.zip; computing/source/ | Original 240 scientific records, 20 severity diagnostics and preserved source |
| October execution provenance | computing/evidence/execution_provenance/; computing/evidence/execution_logs/ | Original source launcher ZIP, recovered hotfix, AG News log and source/result mappings |
| Completed configured-peak comparison | computing/data/peak_first/; computing/peak_pair_runner/; computing/evidence/peak_first/ | Returned results archive, exact executed code, protocol, environment, split arrays/vocabulary, traces and complete audit records |
| Kaggle execution checks | execution/kaggle/ | Delivered standalone checks/launch wrappers and both returned preflight reports |
| Current paper analysis | sncs/*.py; sncs/tests/; sncs/examples/ | All saved-record analyses, first-two-step example, optimizer/evaluation checks and figure/source generation |
| Current paper and supplementary results | sncs/manuscript/; sncs/figures/; sncs/tables/ | Main/supplement PDFs, editable Markdown/LaTeX/BibTeX, figures, all seed tables and budgets |
| Flat journal source downloads | submission/sncs-2026-11-23/ (current); submission/ (Computing) | Main and supplement source ZIPs, with no subfolders inside each ZIP |
| Larger companion evidence | computing/EXTERNAL_EVIDENCE.json | Author-retained predictions and full states, hashes and full-audit instructions |
| Historical raw archives | historical/DRIVE_RESEARCH_REGISTER.csv | Recorded Drive URLs, sizes and known hashes; access has not been changed |

## Scope and limits

This version retains the whole published Git history and adds the current research package and recovered historical code. RECOVERED_CODE_MAP.json documents the inspected code coverage; REPOSITORY_MANIFEST.json records the release files. Private correspondence, machine caches, redundant delivery ZIPs and superseded working drafts are not required reproduction inputs and are not published here.

All 51 current tests and 560 displayed numeric checks passed in a clean extraction. Historical raw prediction-level reevaluation remains limited by missing candidate-epoch assets. The new pair's 42 prediction files and five full paired states were independently audited; these large files are separately retained. Original training-host logs and locks were not fully recovered, so exact historical training reproduction is not claimed. Current saved-record analysis is reproducible from sncs/ without downloading the external large archives.

## Preserve this version

The current repository version is sncs-submission-2026-11-23. The checked PDFs cite this SNCS version and directly link to Online Resources 1 and 2. The earlier Computing scientific snapshot remains preserved at computing-submission-2026-11-18-r2. Future corrections should use a new version rather than moving this tag. The original acml2026-artifacts-v1 tag and earlier branches remain unchanged. Scientific cohort labels and historical selections must stay separate.

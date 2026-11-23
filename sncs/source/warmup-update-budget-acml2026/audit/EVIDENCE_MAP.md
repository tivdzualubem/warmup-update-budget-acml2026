# Paper-to-Evidence Map

## Main paper

| Paper item | Primary source |
|---|---|
| Shared protocol and calibrated thresholds | calibrated manifests in `raw/calibrated_runs/`; manifest audit CSVs in `paper_outputs/*/paper_ready/` |
| Condition semantics | guard semantics CSVs and calibrated run scripts |
| First-two-step LR trace | `paper_outputs/ag_news/paper_ready/ag_news_canonical_scheduler_first_two_steps.csv` |
| Guard-window update-ratio figure | AG News and DBpedia warmup safety gate summary CSVs |
| Accuracy figure | AG News and DBpedia warmup safety gate summary CSVs |
| AG News results table | `paper_outputs/ag_news/paper_ready/ag_news_warmup_safety_gate_summary_table.csv` |
| DBpedia results table | `paper_outputs/dbpedia/paper_ready/dbpedia_14_warmup_safety_gate_summary_table.csv` |
| Paired accuracy effects | paired-seed delta CSVs in both paper-ready folders |
| Guard actions | raw guard-event logs plus guard documentation/semantics CSVs |
| ECE, NLL, Brier | per-seed outputs and warmup safety gate summary CSVs |
| Single-bad causal claim | single-bad per-seed results and scheduler traces |

## Supplement

| Supplement item | Primary source |
|---|---|
| Aggregated probabilistic metrics | AG News and DBpedia summary CSVs |
| Seed-paired effect figure | paired-seed delta CSVs |
| Toy Adam scaling | toy simulation CSV and its generating script; ensure table and figure use the same version |
| Guard action table | raw guard events and guard documentation tables |
| Artifact status | this package inventory and metadata checklist |

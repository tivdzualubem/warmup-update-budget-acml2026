AG NEWS ONE-SCRIPT PACKAGE
==========================

Run command
-----------
python transformer_project/run_stage2_agnews.py

What this single script runs
----------------------------
1. Full scratch empirical pipeline
2. Ablation + shift subset
3. Enhanced early diagnostics
4. Initial-LR causal sweep

Expected output folders
-----------------------
1. transformer_project/results/scratch_empirical_ag_news_full_agnews_v1/
2. transformer_project/results/scratch_empirical_ag_news_agnews_ablation_shift_v1/
3. transformer_project/results/early_diagnostics_plus_ag_news_agnews_earlydiag_plus_v1/
4. transformer_project/results/initial_lr_sweep_ag_news_agnews_initial_lr_sweep_v1/

Included scripts
----------------
- transformer_project/run_stage2_agnews.py
- transformer_project/run_empirical_scratch.py
- transformer_project/run_ablation_shift_subset.py
- transformer_project/run_early_diagnostics_plus.py
- transformer_project/run_initial_lr_sweep.py

Included support files
----------------------
- transformer_project/data/data_loader_scratch.py
- transformer_project/models/transformer.py
- transformer_project/utils/trainer_scratch.py
- transformer_project/utils/metrics_scratch.py

Notes
-----
- This is the final AG News source-of-truth run package.
- It includes scheduler-trace saving and the initial-LR causal sweep.

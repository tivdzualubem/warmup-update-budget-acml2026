DBPEDIA ONE-SCRIPT PACKAGE
===========================

Run command
-----------
python transformer_project/run_stage2_dbpedia.py

What this single script runs
----------------------------
1. Full scratch empirical pipeline
2. Ablation + shift subset
3. Enhanced early diagnostics
4. Initial-LR causal sweep

Expected output folders
-----------------------
1. transformer_project/results/scratch_empirical_dbpedia_14_full_dbpedia_v1/
2. transformer_project/results/scratch_empirical_dbpedia_14_dbpedia_ablation_shift_v1/
3. transformer_project/results/early_diagnostics_plus_dbpedia_14_dbpedia_earlydiag_plus_v1/
4. transformer_project/results/initial_lr_sweep_dbpedia_14_dbpedia_initial_lr_sweep_v1/

Included scripts
----------------
- transformer_project/run_stage2_dbpedia.py
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
- This is the final DBPedia source-of-truth run package.
- It includes scheduler-trace saving and the initial-LR causal sweep.

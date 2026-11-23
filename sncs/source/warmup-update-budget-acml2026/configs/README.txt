DBPEDIA CALIBRATED WARMUP SAFETY GATE PACKAGE
==============================================

Run command
-----------
python transformer_project/run_stage2_dbpedia_warmup_safety_gate_calibrated.py

Dataset
-------
DBPedia-14

Models
------
- medium-4
- large-6

Seeds
-----
- 42
- 123
- 456

Conditions
----------
- buggy_no_guard
- buggy_failfast_guard
- buggy_rescue_guard
- stable_with_guard
- fixed_order_with_guard
- single_bad_first_step_then_stable

Threshold calibration
---------------------
The guard threshold is calibrated inside the run from safe conditions only:

stable_with_guard
fixed_order_with_guard

The rule is:

threshold = 10 * max safe early update ratio

The manifest records:

threshold_source = calibrated_10x_safe_max

Expected output folder
----------------------
transformer_project/results/warmup_safety_gate_dbpedia_14_dbpedia_warmup_safety_gate_v2_calibrated/

Expected files
--------------
- run_manifest.json
- threshold_calibration_record.json
- threshold_calibration_safe_runs.csv
- warmup_safety_gate_per_seed.csv
- warmup_safety_gate_summary.json
- per_seed/
- scheduler_traces/
- guard_events/
- early_logs/

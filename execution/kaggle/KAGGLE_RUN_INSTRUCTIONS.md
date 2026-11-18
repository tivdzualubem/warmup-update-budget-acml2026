# Run the checked pair on Kaggle

**This starts the actual experiment. No extra uploads, dependencies or installations are required.** The completed preflight already placed the checked code and dataset cache in your runtime.

1. Use the **same Kaggle notebook and session** used for the successful preflight. Keep **GPU T4 x2** and **Internet enabled**. Do not restart the runtime or change its libraries or accelerator.
2. Wait until no other notebook cell is executing. Do not rerun the earlier checks.
3. After the checked session restart, use `ACML_Kaggle_Run_Pair_RESTART_CHECKED.py`. Copy its entire contents into **one new Python cell**, and run that cell once. This version points to the new successful preflight folder; the experiment and output directory are unchanged.
4. Keep the session running. Progress appears about once a minute. The measured training-plus-validation projection is about 1 hour 42 minutes; allow roughly **2–3 hours** including other work. This is an estimate, not a guaranteed finish time. The study stops at 49,999 positive-LR steps per model and never adds the deferred recovery extension.
5. At the end, download the files printed by the script from `/kaggle/working/ACML_peak_pair_20261112_190032_158001/`:
   - `ACML_Peak_First_RESULTS.zip`: send back for analysis.
   - `ACML_Peak_First_PREDICTIONS.zip`: send back for independent metric checks.
   - `ACML_Peak_First_STATES.zip`: download and preserve; contains the full model, Adam and random states. It may be too large to attach here, so keep it safely without deleting the originals.

Download and preserve the outputs before ending the session. Keep the complete run and preflight folders until their evidence is checked. The script does not upload anything automatically.

If a run stops, keep the files. An infrastructure interruption can resume from the saved complete paired checkpoint by rerunning the same launcher. It cannot start a duplicate pair or repeat an already completed experiment. A recorded scientific failure, mismatching environment, changed source or missing state stops for diagnosis. Send the RESULTS ZIP/error instead of installing packages, changing the seed or starting over.

If the preflight directory is missing because the session reset, the launcher will stop. Do not substitute a different directory or previous study checkpoint.

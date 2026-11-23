# Version 2 evaluation, execution and boundary-state utilities

These modules implement a future-run contract. The original runners in source/ and all historical numerical exports remain unchanged. No new scientific jobs have used v2.

- scratch_evaluator.evaluate(model, loader, device, num_classes) computes sample-weighted logit CE; loss and nll refer to the same uncapped logit-based quantity. It preserves model mode, global RNG and the loader generator. Custom sampler/augmentation side effects must be handled by the experiment runner.
- transition_executor.transactional_step uses finite-state checks before and after updating. One rescue replay is revalidated against the active ceiling and rejected/restored if it fails. Threshold enforcement is confined to steps 1-100; numerical finite-state checks continue. On invalid entry state, an earlier verified snapshot is required for restoration; without it, the failure is explicit. A nonfinite gradient is never replayed. The new fail-fast contract restores its candidate; archived fail-fast did not.
- resumable_state captures model/Adam state, counters, groups, RNG, sampler permutation/cursor/generator, scheduler/global step, identities and precision state. The CPU test serializes and resumes a tiny dropout model with stochastic draws and a mid-permutation cursor. It does not validate the original research data loader, CUDA/AMP or distributed state.

Run the new regression tests from the package root:

```text
python -m unittest discover -s tests -p test_submission_plan_v2.py -v
```

Integration uses explicit imports from future_runs, not replacement of the frozen publication_expansion namespace. Call the new evaluator for validation selection and evaluation in a separately versioned future experiment. Pass the one-based step to transactional_step, count only committed_transitions as retained training, and log optimizer_step_calls, failure_reason, candidate/replay/committed ratios and all consumed/exit LR fields. None means no committed result; do not convert it to a successful numerical metric. Unexpected optimizer exceptions restore the snapshot and propagate rather than silently reporting success.

These are tested primitives, not a launched or certified recovery-training runner. Actual-device resume equivalence, exact data/preprocessing identity, future-run integration and a frozen endpoint protocol remain launch requirements if the optional recovery study is selected. There is no GPU-time or guard-efficiency claim.

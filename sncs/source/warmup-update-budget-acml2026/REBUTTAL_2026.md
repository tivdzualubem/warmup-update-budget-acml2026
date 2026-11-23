# ACML 2026 Rebuttal Audits

This branch contains targeted audits prepared in response to the ACML 2026 reviews of:

**Warmup Is an Update Budget: Scheduler Semantics and First-Step Adam Impulses**

The audits use the submitted experiment artifacts and the canonical training implementation. Large raw ZIP artifacts and datasets are not committed to this repository.

## Environment

Create an environment and install the repository requirements:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

For the matched-branch identity audit, the minimal dependencies are:

    pip install numpy torch datasets

## Raw artifact inputs

The following artifact ZIPs are required for the archive-based audits:

- `acml_raw_agnews.zip`
- `acml_raw_dbpedia_core.zip`

Set their containing directory:

    ARTIFACT_ROOT=/path/to/warmup_update_budget_artifacts_acml2026

## 1. Guard threshold, kappa sensitivity, and transfer audit

Addresses the threshold-window definition, kappa sensitivity, held-out-seed behavior, model transfer, and cross-dataset transfer.

    python scripts/audit_guard_thresholds.py       --agnews-zip "$ARTIFACT_ROOT/acml_raw_agnews.zip"       --dbpedia-core-zip "$ARTIFACT_ROOT/acml_raw_dbpedia_core.zip"       --output-dir results/rebuttal_2026/guard_threshold_audit       --window 100

Validated first-100-step calibration at `kappa=10`:

- AG News: safe maximum `3.0295998e-05`, threshold `3.0295998e-04`
- DBpedia-14: safe maximum `3.4723030e-05`, threshold `3.4723030e-04`

Unsafe first-step candidate ratios span approximately `5.07` to `11.00`.

Additional validated checks:

- leave-one-seed-out: zero held-out safe false positives and all unsafe cases detected
- medium-to-large and large-to-medium transfer: zero safe false positives and all unsafe cases detected
- AG News to DBpedia-14 and DBpedia-14 to AG News transfer: zero safe false positives and all unsafe cases detected
- all unsafe candidate first-step transitions remain detected throughout the tested kappa sweep

### Candidate-versus-committed guard semantics

    python scripts/audit_guard_semantics.py       --agnews-zip "$ARTIFACT_ROOT/acml_raw_agnews.zip"       --dbpedia-core-zip "$ARTIFACT_ROOT/acml_raw_dbpedia_core.zip"       --output results/rebuttal_2026/guard_threshold_audit/guard_semantics_audit.csv

The rescue condition detects the unsafe candidate transition, restores model and optimizer state, and commits a replayed safe transition. Fail-fast detects the unsafe candidate but does not undo the already executed transition.

## 2. Initial-LR first-step severity

This is an exact first-step Adam counterfactual for fixed initial parameters and clipped gradient. It does not claim equivalent downstream training accuracy.

    python scripts/audit_initial_lr_severity.py       --semantics-csv results/rebuttal_2026/guard_threshold_audit/guard_semantics_audit.csv       --threshold-summary results/rebuttal_2026/guard_threshold_audit/audit_summary.json       --output-dir results/rebuttal_2026/initial_lr_severity

Validated result:

Even an initial LR of `1e-4`, if consumed on the first optimizer transition instead of the intended warmup-scale LR, exceeds the calibrated update-ratio threshold in every evaluated dataset/model/seed reference case.

Estimated per-run threshold-crossing initial LRs are approximately:

- AG News: `2.75e-05` to `5.97e-05`
- DBpedia-14: `3.16e-05` to `6.84e-05`

## 3. Matched-branch identity audit

Addresses whether paired conditions actually begin from the same initialization, minibatch, stochastic state, forward pass, and gradient.

AG News:

    python scripts/audit_matched_branch_identity.py       --repo .       --dataset ag_news       --output results/rebuttal_2026/matched_branch_identity/ag_news_identity.json       --device cpu

DBpedia-14:

    python scripts/audit_matched_branch_identity.py       --repo .       --dataset dbpedia_14       --output results/rebuttal_2026/matched_branch_identity/dbpedia_14_identity.json       --device cpu

The diagnostic independently recreates each scheduler condition from the same seed and compares:

- initialized trainable parameters
- first minibatch
- RNG state before the first forward pass
- first logits
- first loss
- clipped first gradient

Validated result:

- AG News: `6/6` model/seed comparisons passed all identity checks
- DBpedia-14: `6/6` model/seed comparisons passed all identity checks
- Total: `12/12`

No optimizer transition is executed by this diagnostic.

## 4. Direct LR assertion versus update-ratio guard

Compares a direct pre-transition learning-rate invariant against the realized first-step update-ratio guard.

    python scripts/audit_lr_assertion_vs_update_guard.py       --artifact-root "$ARTIFACT_ROOT"       --threshold-csv results/rebuttal_2026/guard_threshold_audit/threshold_window_audit.csv       --output-dir results/rebuttal_2026/lr_assertion_vs_update_guard

Validated first-step results across both datasets:

- unsafe conditions: LR assertion flags `48/48`; update-ratio guard flags `48/48`
- safe conditions: LR assertion flags `0/24`; update-ratio guard flags `0/24`

The two mechanisms are complementary:

- a direct LR assertion is a simpler preventive check when the intended scheduler value is known;
- the update-ratio guard checks the realized parameter displacement and can therefore diagnose the effect of the transition itself.

No claim is made that the update-ratio guard is universally superior to a direct LR invariant.

## Included outputs

Validated small outputs are stored under:

    results/rebuttal_2026/
      guard_threshold_audit/
      initial_lr_severity/
      lr_assertion_vs_update_guard/
      matched_branch_identity/

Raw datasets and large artifact ZIPs are intentionally excluded.

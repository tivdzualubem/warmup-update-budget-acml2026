# Proposed configured-peak first-step comparison

Prepared 12 November 2026 in response to the 8 November feedback. **Protocol only: no research run has been launched and there are no results.** This is the next prioritized experiment, after the execution-provenance and reproduction checks. It is separate from the deferred LR=1.0 recovery extension.

## Question and recommended scope

Does consuming the configured peak learning rate at the first transition, rather than the much smaller first warmup rate, change learning progress or the fixed-budget endpoint when every subsequent scheduled rate is identical?

Recommended smallest design: **one clean/perturbed pair on scratch DBpedia medium-4, seed 42**, each starting from the same newly recorded initialization. This setting reuses an existing architecture and data pipeline, has full validation batches, and adds only two training trajectories. It is chosen for continuity and interpretability, not because it has a favorable new result. One pair is a descriptive pilot; it cannot estimate between-seed uncertainty or establish equivalence. Do not add seeds until a desired significance level is reached. Any later replication requires a separately fixed seed set and unchanged protocol.

The dataset/model, seed, budget and endpoint below are explicit **recommended defaults**, not choices specified by the professor. They must be accepted and frozen in a dated configuration before launch. The eventual execution machine and exact software environment remain to be recorded. A broader benchmark and a new guard are outside this design.

## Fixed design to freeze before launch

| Item | Recommended setting |
|:--|:--|
| Data | `fancyzhx/dbpedia_14`, revision `9abd46cf7fc8b4c64290f26993c540b92aa145ac`; official train and test |
| Development split | The frozen adapter's 15% validation split of official training data, split seed 42; 476,000 training and 84,000 validation examples |
| Preprocessing | `content` field, lowercase whitespace tokens, training-only vocabulary, minimum frequency 2, maximum 30,000 entries including PAD/UNK, maximum length 128, PAD index 0 |
| Architecture | Existing Pre-LN `medium-4`: width 128, four layers, eight attention heads, feed-forward width 512, dropout 0.1, 14 classes |
| Pairing | Experiment seed 42; clone identical initial weights/buffers and zero Adam state; identical ordered examples and RNG streams through the fixed horizon |
| Optimizer | Adam, betas (0.9, 0.98), epsilon 1e-9, weight decay 0, gradient norm clip 1.0 |
| Batch/precision | Batch 32, accumulation 1, FP32, a single device; no AMP or distributed training in this pilot |
| Stable schedule | One-based S(t): peak 1e-4, linear warmup to t=8,000, cosine decay with terminal horizon 50,000; unchanged from the archived formula |
| Clean arm | Install S(t) immediately before every committed optimizer transition |
| Perturbed arm | Consume 1e-4 at t=1; install S(t) before every transition t>=2 |
| Termination | Exactly **49,999 committed positive-LR transitions per arm**; no early stopping, no transition at or after 50,000, no extension |
| Primary endpoint | Example-weighted validation logit CE difference `CE_peak_first - CE_clean` at t=49,999; no validation-best checkpoint selection |
| Secondary endpoints | Validation accuracy difference `accuracy_clean - accuracy_peak_first` at the fixed endpoint; prespecified learning curves; fixed-final test CE and accuracy, evaluated after training |
| Guard | No displacement-triggered rescue, clipping modification or new guard; assert each arm's own intended consumed rate and stop on nonfinite loss/gradient/state |

The clean first rate is 1.25e-8. The injected 1e-4 is **8,000 times the intended first rate** but equals the configured peak; it is not the historical LR=1.0 intervention. Both arms use the same S(2) onward. Do not switch the perturbed arm to a late schedule. No assertion should incorrectly reject its deliberately authorized first rate.

The recommended horizon covers the existing stable schedule's complete positive-LR window, so a deficit at its endpoint has an explicit budget interpretation. This is 99,998 optimizer transitions across the pair, with zero extra continuation steps. It is not a new 50,000-step-per-arm recovery extension. It is half the transition count of regenerating both prefixes plus the previously proposed full recovery extension, but actual cost must be measured rather than inferred from that ratio. With this DBpedia training population, all training batches contain 32 examples: 1,599,968 examples per arm, 3,199,936 total. Report evaluation and checkpoint overhead separately.

## Pair identity and data evidence

Use the frozen adapters as the starting implementation; make the first-rate and stopping changes in a separate versioned runner. Preserve the historical code, outputs and checkpoint choices. Save actual train/validation index arrays, ordered vocabulary/tokenization metadata and dataset revision in the new run bundle rather than retaining only their digests.

Before training, reconstruct the split using the archived algorithm and compare its validation-index digest with `75cc3a5c716ae024bd95de7e7d579546bef5a63038d940c7ff5d5a66aa08f3fa`. Save the new arrays and their hashes. A matching digest verifies those arrays under the recorded hashing scheme; it does not by itself prove original tokenized-example or model-state identity. If any historical identity check fails, diagnose it before launching and describe the new study as a separately versioned replication, not continuation of an unavailable historical state.

Record model state, optimizer state excluding LR, initial minibatch, RNG state before the forward pass, logits, loss and clipped-gradient fingerprints for both arms. They must match before the first intervention. Preserve each arm's own Adam state after divergence. Do not reset moments or copy the clean arm's state into the perturbed arm.

Use deterministic data ordering with an explicit sampler permutation and cursor. Keep each arm's Python, NumPy, CPU/CUDA Torch and loader RNG states separate. Sequential execution must reseed/restore the arm's stream; interleaving arms must not consume a shared global stream accidentally. Log batch-identity fingerprints and consumed learning rates. Save and restore training RNG around evaluation. Record nondeterministic operations and their settings; equal seed labels alone are insufficient.

## Evaluation and records

Evaluate both arms at the same fixed transitions: **0, 1, 100, 1,000, 8,000, 10,000, 20,000, 30,000, 40,000 and 49,999**. These times are prespecified; display all of them and do not select a favorable horizon. Keep test data unused until the two fixed final states exist. Do not select the best time point from the curves.

Compute uncapped logit CE using a sum of per-example losses divided by the total number of examples. Use the separately versioned corrected evaluator, with uneven-batch and extreme-logit checks. Record accuracy from all examples. Do not substitute clipped-probability NLL or an unweighted average of batch means. Retain validation logits/labels at the fixed evaluation points and final model states so the metric arithmetic can be rechecked.

Report online training CE as the example-weighted aggregate for every successive 100-transition block, plus a labeled final shorter block. Also evaluate a fixed, prespecified training-monitor subset of the first 8,192 examples in the saved training-index order at every validation checkpoint; label this as subset performance, not full-training-set accuracy. Fix its identities before training. Record training and validation metrics for both arms, not just their difference.

Preserve full states at initialization, t=1, t=8,000 and t=49,999, plus a resumable latest state at a fixed cadence of 1,000 transitions. Full state includes weights/buffers, optimizer groups/moments/counters, completed transition count, schedule parameters, RNG streams, sampler permutation/cursor, configuration/source/environment identities and precision settings. Do not delete final or candidate evidence after successful completion. Retain all failure records. If a run is interrupted, resume only from verified full state and retain both log segments.

Trace every consumed LR and batch identity; retain candidate/committed displacement for steps 1-100 and the fixed evaluation checkpoints. With no rescue these two displacements coincide. Record positive-LR count, zero-LR count (expected zero), examples, wall time, evaluation time, checkpoint time, device/software versions, peak memory and saved bytes separately for each arm.

## Interpretation fixed before observing results

Report the primary CE difference and both absolute values, plus secondary accuracy and complete prespecified curves. A small difference in one pair does not prove equivalence. Do not introduce a post hoc equivalence margin or a population confidence interval from dependent checkpoints. A null-looking result narrows practical relevance of the extreme-fault finding; report it plainly.

A persistent difference is bounded by this seed, configuration, fault and 49,999 positive-LR updates; it does not establish permanent damage. An early difference that later shrinks describes convergence within the original budget, not an effect of extra recovery training. A closing gap caused by deterioration of the clean control is not evidence of the perturbed arm recovering. Show both curves and state that limitation.

Stop on nonfinite behavior, broken pair identity or failed resume checks. Preserve the failed attempt and do not silently replace its seed. Infrastructure repairs may require restarting both arms under a new recorded protocol version; such a version must be fixed before inspecting replacement outcomes. No choice of rate, horizon or seed may depend on favorable results.

## Remaining launch prerequisites

1. Complete the saved-record execution-provenance and fresh-reproduction work requested in the same feedback.
2. Have the authors accept the explicit defaults above, or record their changes before running. The professor selects the venue separately; venue selection does not determine this scientific protocol.
3. Implement and test the separate fixed-budget pair runner. The historical runner's early stopping, best-checkpoint selection and temporary-state deletion are not suitable unchanged.
4. Verify data/preprocessing identity, a two-step paired real-model smoke check, corrected evaluator and uninterrupted-versus-resumed execution on the actual training device/loader. A CPU toy fixture alone is insufficient for that certification.
5. Measure actual-configuration throughput and evaluation/checkpoint overhead on that machine, estimate the fixed pair's cost and confirm available resources before the research run. Do not inventory the recipient's unrelated GPU hardware.

None of these prerequisites or the research pair is represented as completed by this document. The full extreme-fault recovery extension remains deferred.

# Completed configured-peak comparison

The requested DBpedia medium-4 seed-42 pair completed 49,999 positive-LR transitions and 1,599,968 examples per arm. Clean consumed 1.25e-8 on its first transition; peak-first consumed 1e-4; all subsequent paired rates match. There were no zero-LR steps, early stopping, validation-best selection, failure/restart events or recovery extension.

The primary fixed-final validation CE is 0.061363440708874895 for clean and 0.06136813597258163 for peak-first, a difference of +4.695263706736008e-6. Test accuracy is 98.5985714286% versus 98.6028571429%. Both complete learning curves and the training-subset/online statistics are retained. This single-pair small effect narrows practical extrapolation from the historical extreme first-rate fault; it does not establish equivalence, universal safety or a recovery mechanism.

All 49 record/source files, 42 prediction files and five full paired checkpoint files pass the common manifest's byte-size/hash checks. The independent audit checks 49,999 batch identities, 99,998 consumed rates, all 42 evaluation events, all 1,000 online blocks, example identities and all checkpoint moments/counters/RNG/sampler states. Recomputed prediction CE differs by at most 7.99e-15; accuracies match exactly. A 1e-12 relative/zero-absolute tolerance permits last-bit cross-platform cosine rounding in independent schedule reconstruction; actual rates between arms match exactly after the first transition.

The training session took 4,969.857 seconds (82.831 minutes). Per-arm instrumented training/evaluation costs and positive-LR budgets appear in the supplement and peak_first_costs.csv. Costs are not a guard-overhead or speedup benchmark.

The full original recovery extension remains unperformed. No historical checkpoint selection or historical scientific CSV was changed.

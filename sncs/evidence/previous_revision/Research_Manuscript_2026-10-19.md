# Separating First-Step Learning-Rate Faults from Scheduler Delay in Adam Training

## Abstract

An incorrectly ordered training loop can combine an abnormal first Adam update with a persistent one-transition delay in learning-rate scheduling. We separate these components using archived five-seed experiments on AG News and DBpedia scratch Transformers. Stable training versus an isolated first update at LR=1.0 estimates the effect of an extreme injected fault under the tested stable procedure: mean test-accuracy losses are 47.19-70.99 percentage points. Two exploratory analyses of existing legacy-schedule runs compare rescue with uncorrected late scheduling to assess first-step repair, and corrected scheduling with rescue to assess the remaining delay. Rescue improves accuracy over late scheduling by 5.72-23.86 points; corrected-minus-rescue differences range from -0.061 to +0.097 point and do not establish equivalence. The arms share recorded initial states but not identical consumed schedules or realized training budgets. Stable LR becomes zero at step 50,000, so later epochs do not extend positive-LR recovery training. A separate historical three-seed severity sweep is retained. Pretrained extensions have small final-metric effects; a displacement guard detects 35/40 induced mismatches at the primary threshold, missing all five WNUT cases, while an executed-LR assertion catches 20/20 tested pretrained mismatches.

**Keywords:** learning-rate warmup; optimizer execution semantics; Adam; first-step intervention; recovery; validation-based model selection; update displacement.

## 1. Introduction

A warmup specification is not itself an optimizer guarantee. It describes a planned sequence of rates, but parameter evolution depends on the rates and optimizer state actually consumed by `optimizer.step()`. A loop that installs the scheduled rate only after the step can log an apparently plausible learning rate while the preceding update used a much larger value. The discrepancy is particularly consequential for Adam at its first transition, when bias-corrected normalization can make the parameter-space displacement scale approximately with the consumed rate despite gradient clipping.

The experiments compare an isolated abnormal first update, continued late scheduling, correctly ordered scheduling, and rollback-and-replay. These interventions affect different parts of the optimization path. Early stopping can also produce different numbers of optimizer transitions. Separating the interventions therefore requires reporting both the consumed LR sequence and the realized training budget.

Stable versus bad-then-stable measures the first-step intervention under the stable procedure. Rescue versus late legacy measures first-step repair, and corrected legacy versus rescue measures the remaining scheduler delay. The latter two comparisons are exploratory analyses of existing runs. Early stopping remains part of each recorded procedure, so these endpoint contrasts do not hold computation fixed.

The analysis combines preserved five-seed job records, complete epoch histories, first-300-step traces, historical three-seed initial-rate sweep outputs recovered from archived executed notebooks, and an executable optimizer-state rollback verification. The two historical protocols are described separately throughout.

## 2. Related work

The Transformer schedule of Vaswani et al. (2017) combines inverse-square-root decay and linear warmup. Adam (Kingma and Ba, 2015) rescales coordinates using running first and second moments; early-step instability of adaptive optimizers motivated methods such as RAdam (Liu et al., 2020). Warmup's empirical and theoretical effects have been studied in adaptive optimization and language-model training (Ma and Yarats, 2021; Kosson et al., 2024; Kalra and Barkeshli, 2024). Loss-surface and initialization explanations (Gotmare et al., 2019; Xiong et al., 2020; Huang et al., 2020) motivate caution in generalizing across model architectures.

Existing work already explains important Adam and warmup phenomena. Kalra and Barkeshli (2024) study warmup through conditioning and sharpness dynamics; Kosson et al. (2024) analyze early update sizes in GPT training; Abuduweili and Liu (2025) study initialization of Adam second moments. The present contribution is an execution-level decomposition: distinguish a first-step LR fault from subsequent scheduler delay, trace candidate versus committed transitions, and audit exact state replay and finite positive-LR exposure. The experiments address this controlled execution fault. Calibration measures (Guo et al., 2017) describe prediction quality rather than establish a mechanism of training failure.

## 3. Method

### 3.1 Executed learning rates and update displacement

For optimizer step $t$, let $\alpha_{\mathrm{entry},t}$ be the stored LR before scheduling, $\alpha_{\mathrm{target},t}$ the prescribed rate, $\alpha_{\mathrm{cand},t}$ the rate consumed by the candidate update, $\alpha_{\mathrm{commit},t}$ the rate associated with the retained update, and $\alpha_{\mathrm{exit},t}$ the stored rate after the training-step code returns. Late scheduling permits $\alpha_{\mathrm{cand},t}\ne\alpha_{\mathrm{target},t}$ even when $\alpha_{\mathrm{exit},t}=\alpha_{\mathrm{target},t}$.

With trainable parameters $\theta$, define the normalized realized update ratio

$$
 r_t=\frac{\|\theta_t-\theta_{t-1}\|_2}{\|\theta_{t-1}\|_2+\epsilon_r}. \tag{1}
$$

When a candidate is rolled back and replayed, the candidate and committed update ratios are distinct. In the first Adam transition with no weight decay, fixed clipped gradient $g$ and bias correction imply

$$
\Delta\theta_{1,i}=-\alpha_{\mathrm{commit},1}\frac{g_i}{|g_i|+\epsilon_{\mathrm{Adam}}}. \tag{2}
$$

For active coordinates above the epsilon scale, the first displacement is approximately proportional to the consumed LR. A bound $\|\Delta\theta_1\|_2\le\alpha_{\mathrm{commit},1}\sqrt{D}$ still holds for $D$ optimized coordinates; the failure is not unbounded motion, but violation of the *intended warmup-scale displacement*.

### 3.2 Stable and legacy schedules

The October scratch stable schedule ($\alpha_{\max}=10^{-4}$, $W=8000$, $T=50000$) is

$$
\alpha_t^{S}=\begin{cases}
10^{-4}t/8000,&1\le t\le8000,\\
10^{-4}\frac{1+\cos(\pi(t-8000)/(50000-8000))}{2},&8000<t<50000,\\
0,&t\ge50000.
\end{cases}\tag{3}
$$

The distinct legacy inverse-square-root schedule is

$$
\alpha_t^L=d_{\mathrm{model}}^{-1/2}\min(t^{-1/2},tW^{-3/2}).\tag{4}
$$

Legacy-family experiments begin with optimizer LR 1.0; the code-order fault executes the first step before updating that value. The stable schedule begins at optimizer LR $10^{-4}$ but installs its much smaller scheduled value before the first step. At $t=1$, $\alpha_1^S=1.25\times10^{-8}$. The isolated-step experiment executes its first transition with LR 1.0, then uses $\alpha_t^S$ for all subsequent steps. LR=1.0 is an extreme injected fault. The correct first-step comparator is **stable versus single-bad-then-stable**, under the recorded stopping and selection procedure. In a different comparison, corrected-order legacy and late-order legacy use the same $\alpha^L_t$ but different optimizer/scheduler call order.

![The zero-LR horizon is specific to the stable cosine schedule; the legacy schedule is not zero-capped.](../figures/fig4_scheduler_and_zero_boundary.png){width=95%}

### 3.3 Interventions and matched initialization

Five scratch full-training conditions are considered: stable cosine, corrected-order legacy, late-order legacy, one bad first update followed by stable cosine, and ratio-triggered rollback-and-replay following legacy scheduling. Fail-fast is evaluated as a diagnostic stop, not as a trained classifier. Rescue replaces an unsafe candidate transition by restoring the model/optimizer/gradient snapshot and replaying at the intended legacy rate. After the guard window, the rescue condition follows the legacy schedule, not the stable schedule; accordingly its performance is compared with corrected-order legacy rather than used as the isolated-step control.

All conditions within each configuration and seed share initial model weights, optimizer state excluding LR, minibatch, forward RNG, logits, loss and clipped-gradient fingerprint. The returned archive contains 120 scratch scientific records and 120 pretrained scientific records across eight model/dataset configurations and five seeds; 40 matched-initialization groups show no hash discrepancies. These hashes validate matched recorded initial conditions, not equality of all later stochastic trajectories.

### 3.4 Experimental settings and endpoint selection

Scratch runs use AG News and DBpedia, medium-4 and large-6 Pre-LN Transformers, Adam $\beta=(0.9,0.98)$, epsilon $10^{-9}$, no weight decay, gradient clipping at 1, batch size 32, sequence length 128, training fraction 1, 15% training-derived validation with a fixed split seed 42, at most 30 epochs and patience-five early stopping. The experimental seeds are 42, 123, 456, 789 and 2026. The October schedule uses an 8,000-step warmup independent of total epoch count. Scratch test metrics are recorded using the model state with minimum validation loss according to the frozen runner.

Pretrained experiments include DistilBERT sequence classification (AG News and DBpedia), DistilBERT NER (WNUT-17) and DeiT-tiny image classification (CIFAR-10), trained for three epochs with AdamW and 10% linear warmup. AG News and DBpedia pretrained checkpoints are selected by maximum validation macro-F1; WNUT-17 uses maximum validation entity F1, and CIFAR-10 uses maximum validation accuracy. Test evaluation uses the validation-selected state. Table 5 reports the corresponding task-specific primary test metric; AG News and DBpedia accuracy is retained separately as a secondary endpoint. The returned JSON metadata agrees with the extremum of the retained validation history for every completed training run. The archive contains the saved evaluations and selection histories but no model weights, so test predictions could not be independently recomputed.

### 3.5 Historical severity study (distinct protocol)

The historical initial-LR sweep used buggy legacy scheduling with **4,000 warmup steps**, three seeds (42, 123, 456), two model sizes and five initial rates $\{1,0.3,0.1,0.03,0.01\}$ per dataset (60 runs). Outcomes were extracted from preserved executed notebook tables. The notebook manifests omit warmup, but both original sweep runners explicitly pass warmup_steps=4000 to ScratchTrainer, overriding its 8000-step default. All 30 complete AG News LR traces agree with the 4,000-step schedule and peak; the archived DBpedia first-step trace export agrees with its runner. Full DBpedia trajectories were not rechecked. We recompute sample SD from the per-seed outcomes because the notebook summaries used population SD. These results are reported separately from the five-seed October study with 8,000 warmup steps. The earlier fixed-gradient counterfactual at rates down to $10^{-4}$ measures one transition, not trained-model outcomes.

### 3.6 Analysis

Comparisons are paired within dataset/model/seed. Stable minus bad-then-stable retains the first-step comparison. The existing corrected-minus-late contrast combines first-step and later ordering effects; two explicitly exploratory contrasts separate rescue-minus-late (first-step repair) and corrected-minus-rescue (remaining delay). Accuracy differences are percentage points. Descriptive 95% Student-$t$ intervals use four degrees of freedom. Five same-direction differences have exact two-sided sign-test $p=0.0625$, so direction consistency alone is not significance at 0.05; the tests are unadjusted across settings. Small differences and intervals including zero do not establish equivalence, and no equivalence margin was prespecified. The separate seed supplement displays all five trained endpoints for each October setting, with selected epochs retained in the per-job CSV; diagnostic endpoints remain n.a. Historical three-seed outcomes are analyzed separately.

## 4. Scratch results separating first-step faults and scheduler delay

**Table 1. October scratch test accuracy (%, mean ± sample SD, five seeds).** Each column denotes a separate condition.

| Dataset / architecture | Stable cosine | Single bad → stable | Correct legacy | Late legacy | Rescue → legacy |
|:--|--:|--:|--:|--:|--:|
| AG News / medium-4 | 92.15 ± 0.44 | 44.96 ± 5.91 | 92.49 ± 0.33 | 73.40 ± 8.09 | 92.39 ± 0.21 |
| AG News / large-6 | 92.37 ± 0.37 | 43.00 ± 6.78 | 92.33 ± 0.33 | 68.51 ± 7.75 | 92.37 ± 0.23 |
| DBpedia / medium-4 | 98.62 ± 0.02 | 33.09 ± 7.20 | 98.67 ± 0.03 | 92.92 ± 1.35 | 98.64 ± 0.03 |
| DBpedia / large-6 | 98.79 ± 0.04 | 27.80 ± 2.78 | 98.58 ± 0.10 | 82.14 ± 11.59 | 98.64 ± 0.04 |

**Table 2. Paired accuracy differences (mean [descriptive 95% Student-$t$ interval], percentage points).** Stable-minus-bad retains the first-step comparison. Rescue-minus-late and corrected-minus-rescue are exploratory analyses of existing runs; their realized schedules and stopping budgets differ. Full paired seeds and exact sign tests are supplied in the supplement.

| Dataset / architecture | Stable - isolated bad, pp [95% CI] | Rescue - late, pp [95% CI], exploratory | Corrected - rescue, pp [95% CI], exploratory |
|:--|--:|--:|--:|
| AG News / medium-4 | +47.19 [+39.51, +54.88] | +18.99 [+9.06, +28.92] | +0.10 [-0.52, +0.71] |
| AG News / large-6 | +49.37 [+40.90, +57.83] | +23.86 [+14.33, +33.39] | -0.04 [-0.41, +0.32] |
| DBpedia / medium-4 | +65.53 [+56.57, +74.49] | +5.72 [+4.03, +7.40] | +0.03 [-0.02, +0.08] |
| DBpedia / large-6 | +70.99 [+67.52, +74.46] | +16.50 [+2.08, +30.93] | -0.06 [-0.21, +0.09] |

### 4.1 Isolated first update with stable scheduling held fixed

![Each line connects the same seed's stable and single-bad-first-step outcomes; no corrected-legacy values enter this contrast.](../figures/fig1_isolated_first_step_paired.png){width=95%}

Across the four scratch settings, the LR-1.0 first step followed by stable scheduling reduces mean test accuracy by 47.19-70.99 percentage points. Every seed shows a loss. These losses persist under the tested procedure. Because stable LR reaches zero at transition 50,000, the results do not establish whether further positive-LR training would recover performance.

### 4.2 Legacy schedule order and rescue

![Corrected and late-order treatments use a common inverse-square-root target formula. Consumed sequences and realized stopping budgets differ.](../figures/fig2_legacy_schedule_order_paired.png){width=95%}

Corrected legacy exceeds late legacy by 5.75-23.82 points, but this combines first-step and later delay effects. The added exploratory rescue-minus-late contrast yields +18.99 [9.06, 28.92] points on AG News medium-4, +23.86 [14.33, 33.39] on AG News large-6, +5.72 [4.03, 7.40] on DBpedia medium-4, and +16.50 [2.08, 30.93] on DBpedia large-6. Only the first candidate triggers rescue in these scratch records, so this comparison assesses first-step repair while both arms retain late legacy scheduling thereafter. Its endpoint improvement is measured under the original, unequal stopping budgets, not a fixed-computation intervention. Each setting has five positive paired differences (two-sided sign-test $p=0.0625$).

The exploratory corrected-minus-rescue differences are +0.097 [-0.515, 0.710], -0.045 [-0.414, 0.325], +0.032 [-0.017, 0.080], and -0.061 [-0.213, 0.092] points in the same setting order. These assess the remaining delay after first-step repair. After the repaired first step, rescue continues to consume the previous legacy target, whereas corrected order consumes the current target; early stopping also changes total exposure. The small endpoint differences do not prove equivalence or exclude a budget-dependent delay effect.

![Exploratory existing-run decomposition: late scheduling, first-step rescue, and corrected order. All five seeds are shown; realized budgets differ.](../figures/fig9_exploratory_repair_delay.png){width=95%}

## 5. Training progression and effective optimization horizon

The October stable cosine schedule reaches an exact zero rate for all transitions $t\ge50{,}000$. Subsequent `optimizer.step()` calls may increment counters and consume computational resources but perform no learning-rate-dependent parameter updates for scratch Adam under these conditions (zero weight decay). The legacy inverse-square-root schedule does **not** share this terminal horizon. Inference about “more training epochs” therefore depends on the schedule actually assigned to the run.

**Table 3a. Training and test performance by scratch condition.** Entries are five-seed means evaluated with the minimum-validation-loss checkpoint. Diagnostic fail-fast arms have no trained-model endpoint (n.a.). These checkpoint evaluations differ from online training loss recorded while dropout and changing weights are active.

| Setting | Condition | Train CE | Train accuracy (%) | Test accuracy (%) |
|:--|:--|--:|--:|--:|
| AG News medium-4 | Stable | 0.141 | 95.24 | 92.15 |
| AG News medium-4 | Bad then stable | 207.923 | 45.75 | 44.96 |
| AG News medium-4 | Correct legacy | 0.129 | 95.71 | 92.49 |
| AG News medium-4 | Late legacy | 6.100 | 75.26 | 73.40 |
| AG News medium-4 | Rescue | 0.140 | 95.32 | 92.39 |
| AG News medium-4 | Fail-fast | n.a. | n.a. | n.a. |
| AG News large-6 | Stable | 0.119 | 95.96 | 92.37 |
| AG News large-6 | Bad then stable | 1,207.031 | 43.26 | 43.00 |
| AG News large-6 | Correct legacy | 0.132 | 95.67 | 92.33 |
| AG News large-6 | Late legacy | 198.833 | 70.57 | 68.51 |
| AG News large-6 | Rescue | 0.137 | 95.53 | 92.37 |
| AG News large-6 | Fail-fast | n.a. | n.a. | n.a. |
| DBpedia medium-4 | Stable | 0.040 | 99.01 | 98.62 |
| DBpedia medium-4 | Bad then stable | 424.333 | 33.07 | 33.09 |
| DBpedia medium-4 | Correct legacy | 0.033 | 99.13 | 98.67 |
| DBpedia medium-4 | Late legacy | 0.761 | 93.38 | 92.92 |
| DBpedia medium-4 | Rescue | 0.031 | 99.18 | 98.64 |
| DBpedia medium-4 | Fail-fast | n.a. | n.a. | n.a. |
| DBpedia large-6 | Stable | 0.023 | 99.44 | 98.79 |
| DBpedia large-6 | Bad then stable | 2,338.591 | 27.80 | 27.80 |
| DBpedia large-6 | Correct legacy | 0.037 | 99.11 | 98.58 |
| DBpedia large-6 | Late legacy | 119.595 | 82.56 | 82.14 |
| DBpedia large-6 | Rescue | 0.034 | 99.22 | 98.64 |
| DBpedia large-6 | Fail-fast | n.a. | n.a. | n.a. |

**Table 3b. Effective optimization budget by scratch condition.** Each row is a separate five-job mean. Positive-LR and zero-LR counts refer to committed transitions; extra replay calls are rejected-candidate overhead, not additional retained training steps. Early stopping makes observed budgets unequal across conditions.

| Setting | Condition | Positive-LR steps | Zero-LR steps | Extra replay calls |
|:--|:--|--:|--:|--:|
| AG News medium-4 | Stable | 33,793 | 0 | 0.0 |
| AG News medium-4 | Bad then stable | 49,999 | 8,023 | 0.0 |
| AG News medium-4 | Correct legacy | 26,779 | 0 | 0.0 |
| AG News medium-4 | Late legacy | 87,351 | 0 | 0.0 |
| AG News medium-4 | Rescue | 25,504 | 0 | 1.0 |
| AG News medium-4 | Fail-fast | 1 | 0 | 0.0 |
| AG News large-6 | Stable | 29,330 | 0 | 0.0 |
| AG News large-6 | Bad then stable | 47,013 | 7,821 | 0.0 |
| AG News large-6 | Correct legacy | 26,779 | 0 | 0.0 |
| AG News large-6 | Late legacy | 87,989 | 0 | 0.0 |
| AG News large-6 | Rescue | 26,142 | 0 | 1.0 |
| AG News large-6 | Fail-fast | 1 | 0 | 0.0 |
| DBpedia medium-4 | Stable | 49,999 | 80,901 | 0.0 |
| DBpedia medium-4 | Bad then stable | 49,999 | 80,901 | 0.0 |
| DBpedia medium-4 | Correct legacy | 124,950 | 0 | 0.0 |
| DBpedia medium-4 | Late legacy | 446,250 | 0 | 0.0 |
| DBpedia medium-4 | Rescue | 133,875 | 0 | 1.0 |
| DBpedia medium-4 | Fail-fast | 1 | 0 | 0.0 |
| DBpedia large-6 | Stable | 49,999 | 77,926 | 0.0 |
| DBpedia large-6 | Bad then stable | 49,999 | 69,001 | 0.0 |
| DBpedia large-6 | Correct legacy | 121,975 | 0 | 0.0 |
| DBpedia large-6 | Late legacy | 324,275 | 0 | 0.0 |
| DBpedia large-6 | Rescue | 130,900 | 0 | 1.0 |
| DBpedia large-6 | Fail-fast | 1 | 0 | 0.0 |

![Online training CE trajectories are shown separately from validation-selected endpoint metrics.](../figures/fig8_condition_training_loss.png){width=95%}

The frozen schedule and logged committed early-step LRs determine the positive-LR counts. All legacy transitions remain positive; stable-family transitions at and beyond 50,000 use zero LR. The per-run supplement records the selected-checkpoint step, termination step, train metrics, cumulative LR and replay overhead. Summed LR is an exposure descriptor, not a measure of equal parameter displacement or equal effective learning. Wall-clock/GPU time is absent from these returned records and is not reconstructed from step counts.

![Validation metrics are plotted against actual optimizer transition counts; the dashed boundary marks zero stable LR at 50,000 steps.](../figures/fig3_step_indexed_training_histories.png){width=97%}

![Positive-LR committed transitions and zero-LR computation are separated for each condition.](../figures/fig6_ineffective_postzero_steps.png){width=95%}

All 20 DBpedia stable-family runs cross the 50,000-transition boundary. Across their stored histories, 91 consecutive pairs of validation-loss records at epochs **after** the zero boundary have exactly equal reported values; these repeated metrics cannot be interpreted as evidence of additional learning. Nine of 20 AG News stable-family runs also cross the boundary. Fourteen of the 40 stable-family runs select a best-validation epoch whose endpoint is at or beyond step 50,000, but the checkpoint can incorporate meaningful updates from *before* the boundary within that epoch. Epoch-level histories alone do not identify the last useful intermediate checkpoint at precisely step 49,999. The recorded epochs therefore overstate the number of transitions available for parameter recovery.

## 6. Historical initial-LR severity: separate training experiment

**Table 4. Historical buggy-first-update study (%, mean ± recomputed sample SD, three seeds; 4,000-step warmup).** Rates were consumed as initial rates under the legacy buggy-order protocol.

| Dataset / architecture | LR 1.0 | LR 0.30 | LR 0.10 | LR 0.03 | LR 0.01 |
|:--|--:|--:|--:|--:|--:|
| AG News / medium-4 | 72.02 ± 11.94 | 90.43 ± 0.64 | 92.44 ± 0.05 | 92.56 ± 0.11 | 92.69 ± 0.20 |
| AG News / large-6 | 78.47 ± 1.54 | 86.48 ± 1.35 | 91.71 ± 0.47 | 92.64 ± 0.07 | 92.16 ± 0.58 |
| DBpedia / medium-4 | 89.60 ± 3.21 | 98.50 ± 0.06 | 98.68 ± 0.05 | 98.63 ± 0.03 | 98.61 ± 0.04 |
| DBpedia / large-6 | 82.80 ± 8.85 | 97.02 ± 0.19 | 98.43 ± 0.08 | 98.62 ± 0.08 | 98.59 ± 0.04 |

![Test accuracy across the historical initial-LR sweep. This is a distinct experimental protocol and not pooled with October.](../figures/fig5_historical_initial_lr_sweep.png){width=96%}

Reducing the initial rate from 1.0 generally yields markedly better trained classifiers across both datasets and depths, showing that the observed failure is severity dependent rather than an all-or-nothing property of a single code branch. Differences among the smallest rates are not monotone, and this sweep does not identify a universal safe LR. The specific numerical results pertain to the historical 4,000-step warmup and three seeds; October's 8,000-step schedules, checkpoint selection records and five-seed estimates remain analytically distinct. Historical summaries use the saved notebook outcomes; prediction-level reevaluation would require the original checkpoints.

## 7. Pretrained generalization and monitoring distinctions

**Table 5. Pretrained primary test metrics (%, mean ± sample SD, five seeds).** AG News and DBpedia use macro-F1, WNUT-17 uses entity F1, and CIFAR-10 uses accuracy. Checkpoints are selected by the corresponding validation metric. Text-classification accuracy is a secondary endpoint.

| Pretrained task | Safe order | Late order | Single bad | Guard rescue | Primary metric |
|:--|--:|--:|--:|--:|:--|
| AG News | 94.38 ± 0.10 | 94.38 ± 0.08 | 94.38 ± 0.10 | 94.40 ± 0.09 | Macro-F1 (%) |
| DBpedia | 99.29 ± 0.01 | 99.29 ± 0.01 | 99.30 ± 0.00 | 99.29 ± 0.02 | Macro-F1 (%) |
| WNUT-17 | 42.28 ± 1.56 | 42.31 ± 1.43 | 42.49 ± 1.34 | 42.31 ± 1.43 | Entity F1 (%) |
| CIFAR-10 | 96.19 ± 0.14 | 96.11 ± 0.17 | 96.17 ± 0.10 | 96.16 ± 0.18 | Accuracy (%) |

Pretrained results show a mismatch in the first transition and measurable displacement anomalies, but substantially smaller differences in final metrics than in the scratch configurations. The present experiment changes more than a binary “pretrained/not pretrained” property across studies; optimizer, model architecture, dataset, schedule and task also differ. It does not justify concluding that pretraining *caused* the smaller response.

**Table 6. Pretrained online training performance and effective budget by condition.** Entries are separate five-job means. The logs preserve online training CE, not a final train-set accuracy evaluation. n.a. denotes an arm that stopped before any complete logged epoch. Safe and single-bad-then-correct schedules reach zero on their final transition; late/rescue schedules can consume the previous positive LR at that transition. A WNUT fail-fast arm can finish training when its guard does not trigger, while remaining a diagnostic condition without a primary test endpoint.

| Task | Condition | Last online train CE | Positive-LR steps | Zero-LR steps |
|:--|:--|--:|--:|--:|
| AG News | Safe | 0.099 | 9,563 | 1 |
| AG News | Late | 0.099 | 9,564 | 0 |
| AG News | Single bad | 0.099 | 9,563 | 1 |
| AG News | Rescue | 0.099 | 9,564 | 0 |
| AG News | Fail-fast | n.a. | 1 | 0 |
| AG News | LR assertion | n.a. | 0 | 0 |
| DBpedia | Safe | 0.013 | 44,624 | 1 |
| DBpedia | Late | 0.013 | 44,625 | 0 |
| DBpedia | Single bad | 0.013 | 44,624 | 1 |
| DBpedia | Rescue | 0.013 | 44,625 | 0 |
| DBpedia | Fail-fast | n.a. | 1 | 0 |
| DBpedia | LR assertion | n.a. | 0 | 0 |
| WNUT-17 | Safe | 0.083 | 638 | 1 |
| WNUT-17 | Late | 0.082 | 639 | 0 |
| WNUT-17 | Single bad | 0.082 | 638 | 1 |
| WNUT-17 | Rescue | 0.082 | 639 | 0 |
| WNUT-17 | Fail-fast | 0.082 | 639 | 0 |
| WNUT-17 | LR assertion | n.a. | 0 | 0 |
| CIFAR-10 | Safe | 0.035 | 4,220 | 1 |
| CIFAR-10 | Late | 0.035 | 4,221 | 0 |
| CIFAR-10 | Single bad | 0.035 | 4,220 | 1 |
| CIFAR-10 | Rescue | 0.035 | 4,221 | 0 |
| CIFAR-10 | Fail-fast | n.a. | 1 | 0 |
| CIFAR-10 | LR assertion | n.a. | 0 | 0 |

### 7.1 LR assertion, displacement alarm, and downstream loss

A direct executed-LR assertion stops 20/20 deliberately misordered pretrained first steps before optimizer mutation. The prespecified first-step displacement guard at multiplier $\kappa=10$ detects 35/40 injected mismatches across all eight study configurations, missing the five WNUT-17 cases. The WNUT candidate ratios (~$3.13\times10^{-4}$) are below its held-out safe-envelope thresholds (approximately $1.5\times10^{-3}$). This is a *detection miss at the selected operating point*, not evidence that those runs experienced the same severe downstream degradation seen for scratch Transformers.

At $\kappa=2$, all 40 induced mismatches exceed the threshold. This sensitivity result is retrospective; $\kappa=10$ remains the primary operating point. Zero observed safe-window alarms in the audited records is descriptive, not a certified false-positive bound, because transitions and runs are not independent Bernoulli trials.

![Detection fraction changes with the threshold multiplier. The primary κ=10 was fixed before the expanded study.](../figures/fig7_guard_threshold_sensitivity.png){width=91%}

The concepts are distinct. An assertion checks whether the known intended LR equals the LR about to be consumed; it is a direct prevention mechanism for the call-order fault. A displacement guard judges the resulting update relative to an empirical reference envelope; its detection threshold may be too permissive in one task or may flag benign large updates in a different task. Finally, test performance reports downstream utility rather than merely measuring an optimizer event. No guard-superiority claim follows from the present records.

**Table 7. Automatically regenerated threshold sensitivity.** The primary multiplier is 10; every other row is retrospective. Held-out safe counts are transition counts, not independent Bernoulli trials or prospective false-positive certificates. Each threshold excludes the held-out seed and uses only safe steps 1-100.

| Threshold kappa | Detected / 40 | WNUT detected / 5 | Held-out safe exceedances / transitions | Status |
|--:|--:|--:|:--|:--|
| 1.1 | 40/40 | 5/5 | 0/6000 | Retrospective |
| 2 | 40/40 | 5/5 | 0/6000 | Retrospective |
| 5 | 35/40 | 0/5 | 0/6000 | Retrospective |
| 10 | 35/40 | 0/5 | 0/6000 | Primary |
| 20 | 25/40 | 0/5 | 0/6000 | Retrospective |
| 50 | 25/40 | 0/5 | 0/6000 | Retrospective |
| 100 | 25/40 | 0/5 | 0/6000 | Retrospective |
| 200 | 20/40 | 0/5 | 0/6000 | Retrospective |
| 500 | 20/40 | 0/5 | 0/6000 | Retrospective |
| 1000 | 20/40 | 0/5 | 0/6000 | Retrospective |

The displacement guard measures the realized parameter update and can trigger rollback/replay after a threshold violation. The executed-LR assertion instead checks a known LR target before the update and catches all tested pretrained call-order mismatches. These experiments do not establish a broader advantage for displacement monitoring. Runtime and memory overhead were not measured.

### 7.2 Rollback contract

Exact replay requires restoration of weights, Adam moments and step counters, parameter groups, and clipped gradients. The CPU tests compare the unchanged executor against a clean safe step using warmed Adam state. The candidate and replay reuse gradients computed before the snapshot; they do not repeat the stochastic forward pass. For this deterministic Adam step, Python, NumPy and PyTorch CPU RNG states remain unchanged. A separate test perturbs and restores those states and checks the next draws. The step snapshot itself does not contain RNG state. These checks cover fixed-gradient CPU replay, not stochastic optimizer steps, AMP, CUDA or distributed execution.

The separate analysis reference now matches the runners: pretrained single-bad step 1 exits at its installed target LR, scratch single-bad step 1 exits at LR=1.0, and guard enforcement ends after step 100. Original training code and records are unchanged. The 80 missing threshold entries are exported from result JSON fields, and sensitivity tables are regenerated from the safe traces and LOSO manifests. Source mappings accompany Tables 1-7 and S1-S12.

## 8. Discussion and limitations

The scratch results distinguish a large first-step fault from the smaller residual difference associated with scheduler delay after rescue. Early stopping changes the realized budgets, so the endpoint comparisons describe the tested procedures. The historical sweep shows a severity-dependent response under its separate protocol. Performance plateaus after stable LR reaches zero cannot establish permanent damage or recovery under additional positive-LR training.

Initial-state fingerprints agree across branches, and saved histories agree with validation-based checkpoint selection. The absence of checkpoint weights prevents independent prediction-level reevaluation. Recovery beyond the tested budget remains unmeasured and would require matched continued positive-LR training. Small corrected-versus-rescue differences do not establish equivalence.

The LR-1.0 scratch intervention is extreme, and the experiments cover a limited set of models and tasks with five seeds per setting. Guard sensitivity depends on the calibration data and threshold; WNUT accounts for all five primary-threshold misses. The pretrained comparisons vary architecture, task, optimizer and schedule together, so their smaller effects cannot be attributed to pretraining alone. Fault prevalence and guard overhead were not measured.

## 9. Conclusion

The stable-versus-bad-then-stable comparison shows a large loss from the injected LR-1.0 first step under the tested stable procedure. Exploratory rescue-versus-late and corrected-versus-rescue analyses separate first-step repair from the remaining scheduler delay. The latter differences are small but do not establish equivalence. Stable LR becomes zero at step 50,000; subsequent epochs do not test recovery with further positive-LR updates. The guard detects 35/40 mismatches, misses every WNUT case, and has unmeasured overhead. An executed-LR assertion catches all 20 tested pretrained mismatches before mutation.

## References

Abuduweili, A., and Liu, C. (2025). Revisiting the Initial Steps in Adaptive Gradient Descent Optimization. *Proceedings of Machine Learning Research* 280, 305–322.

Gotmare, A., Keskar, N. S., Xiong, C., and Socher, R. (2019). A Closer Look at Deep Learning Heuristics: Learning Rate Restarts, Warmup and Distillation. *International Conference on Learning Representations*.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of Machine Learning Research* 70, 1321–1330.

Huang, X. S., Perez, F., Ba, J., and Volkovs, M. (2020). Improving Transformer Optimization Through Better Initialization. *Proceedings of Machine Learning Research* 119, 4475–4483.

Kalra, D. S., and Barkeshli, M. (2024). Why Warmup the Learning Rate? Underlying Mechanisms and Improvements. *Advances in Neural Information Processing Systems* 37.

Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*.

Kosson, A., Messmer, B., and Jaggi, M. (2024). Analyzing and Reducing the Need for Learning Rate Warmup in GPT Training. *Advances in Neural Information Processing Systems* 37.

Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J. (2020). On the Variance of the Adaptive Learning Rate and Beyond. *International Conference on Learning Representations*.

Loshchilov, I., and Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations*.

Ma, J., and Yarats, D. (2021). On the Adequacy of Untuned Warmup for Adaptive Optimization. *Proceedings of the AAAI Conference on Artificial Intelligence* 35(10), 8828–8836.

Vaswani, A. et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems* 30.

Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. *Proceedings of Machine Learning Research* 119, 10524–10533.

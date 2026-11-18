"""Build the 22 October plan revision without changing historical numeric exports."""
from pathlib import Path
import re,shutil,json,csv
ROOT=Path(__file__).resolve().parent
M=ROOT/'manuscript';P=ROOT/'evidence/previous_revision'

def table_block(text,marker):
    start=text.index(marker);end=text.find('\n\n',text.index('\n|',start)+2)
    return text[start:end]

def build():
    baseline=P/'Research_Manuscript_2026-10-19.md'
    seed_baseline=P/'All_Seed_Results_2026-10-19.md'
    if not baseline.exists():shutil.copy2(M/'Research_Manuscript.md',baseline)
    if not seed_baseline.exists():shutil.copy2(M/'All_Seed_Results_and_Provenance.md',seed_baseline)
    original=baseline.read_text(encoding='utf-8');text=original
    supplement=seed_baseline.read_text(encoding='utf-8')
    # Transfer long supporting tables while preserving every displayed value.
    transfers=[('**Table 3a.','S13','Archived scratch training and selected test performance'),
               ('**Table 3b.','S14','Scratch positive-LR, zero-LR and replay budgets'),
               ('**Table 6.','S15','Pretrained training performance and budgets'),
               ('**Table 4.','S16','Historical initial-rate severity study'),
               ('**Table 7.','S17','Retrospective threshold sensitivity')]
    for marker,newid,title in transfers:
        block=table_block(original,marker)
        caption=block.split('** ',1)[1]
        if newid=='S13':caption='Loss values retain the archived mean-of-batch-means logit CE definition. '+caption
        if newid=='S16':caption='Values are test accuracy in percent, mean plus/minus recomputed sample SD across three seeds (42, 123, 456), with 4,000 warmup steps. '+caption
        supplement+='\n\n## Table '+newid+' '+title+'\n\n'+caption+'\n'
        text=text.replace(block,'Supporting condition-level results appear in Supplementary Table '+newid+'.')
    # Main-table numbering follows reading order after inserting the LR table.
    text=text.replace('**Table 1.','**Table 2.').replace('**Table 2. Paired','**Table 3. Paired')
    # The first replacement also produced Table 2 for the endpoint table only.
    text=text.replace('Table 2 gives','Table 3 gives')
    # Remove repetitive archived figures from the main article.
    text=re.sub(r'^!\[.*?\]\([^\n]*\)\{width=[^\n]*\}\s*$', '',text,flags=re.M)
    abstract='''An incorrectly ordered training loop can combine an abnormal first Adam update with a continuing one-transition delay in learning-rate scheduling. Archived five-seed experiments on AG News and DBpedia distinguish these interventions under the recorded stopping and validation-selection procedures. An isolated first update at LR=1.0, an extreme injected fault, reduces selected test accuracy by 47.19-70.99 percentage points relative to stable training. Exploratory rescue-minus-late legacy comparisons yield selected test gaps of 5.72-23.86 points, but larger validation gaps at shared earlier step counts reveal learning delay. We report all 95 shared-step comparisons without interpolation or carrying stopped runs forward. These conditional contrasts do not complete the missing cell of a fixed-budget factorial. Stable LR is zero from step 50,000, so subsequent epochs do not measure recovery with additional positive-LR updates. An evaluator audit identifies clipped-probability NLL and unequal weighting of partial batches; archived selections and accuracy endpoints are preserved with those caveats. A versioned evaluator and finite-state executor are tested separately. The displacement guard detects 35/40 induced mismatches, missing all five WNUT cases; an LR assertion catches 20/20 tested pretrained mismatches. The work supports an execution-contract case study under explicit budgets, without a claim of permanent damage or general guard superiority.'''
    text=re.sub(r'(?s)(## Abstract\n\n).*?(\n\n\*\*Keywords:)',lambda m:m[1]+abstract+m[2],text,count=1)
    text=text.replace('Rescue versus late legacy measures first-step repair, and corrected legacy versus rescue measures the remaining scheduler delay.',
        'Rescue versus late legacy is a conditional first-step-repair comparison, and corrected legacy versus rescue compares the remaining delay conditional on a safe first update.')
    # Primary-source descriptions are short and limited to each work's reported scope.
    related='''The Transformer schedule of Vaswani et al. (2017) combines inverse-square-root decay and linear warmup. Existing optimization work already establishes the relevance of early update size, moment initialization and training horizon. Table R1 distinguishes these results from the execution intervention examined here. Standard first-step Adam algebra and finding a loop error are not claimed as new optimization theory.

**Table R1. Related-work comparison.** The present study traces consumed LR, candidate/committed transitions and outcomes at specified observed budgets; it does not propose a new optimizer.

| Primary work | Established subject | Distinction in this study | Supporting evidence here |
|:--|:--|:--|:--|
| Kingma and Ba (2015), Adam | Bias-corrected first and second moments | Deliberately wrong consumed LR with fixed initial gradient/state | Equation 2 and paired first-moment CPU check |
| Shazeer and Stern (2018), Adafactor | Oversized adaptive updates and update clipping | Execution mismatch; no claim to originate update-size control | Candidate/committed traces and Table 1 |
| Ma and Yarats (2021) | Warmup and early adaptive-update magnitude | One injected transition versus continuing scheduler lag | Selected contrasts and shared-step histories |
| Kosson, Messmer and Jaggi (2024) | Early update metrics and GPT optimizer normalization | Scratch classifiers under a known execution fault | Classifier protocol and consumed-LR table; no GPT mechanism test |
| Kalra and Barkeshli (2024) | Conditioning/sharpness and target-LR tolerance | No causal landscape explanation is tested here | Recorded LR/displacement/loss/accuracy, without sharpness measurements |
| Abuduweili and Liu (2025) | Nonzero initialization of second moments | Initial moments remain fixed across first-step LR arms | Zero-moment assumptions and paired Adam state fixture |

Recent work further studies Adam preconditioner-driven loss spikes (Bai et al., 2026), curvature-based warmup guarantees (Alimisis, Islamov and Lucchi, 2026), and horizon-dependent warmup duration in a quadratic model (Topollai and Choromanska, 2026, preprint). Those mechanisms and scaling claims are not established by our descriptive histories. The present evidence concerns specified LR execution faults and the distinction between selected endpoints and shared logged-step outcomes. Calibration metrics (Guo et al., 2017) describe prediction quality rather than establish a cause of training failure.'''
    text=re.sub(r'(?s)(## 2\. Related work\n\n).*?(\n\n## 3\.)',lambda m:m[1]+related+m[2],text,count=1)
    text=text.replace('In the first Adam transition with no weight decay, fixed clipped gradient $g$ and bias correction imply',
        'For zero initial moments, a finite fixed clipped gradient $g$, no weight decay, standard bias correction and $\\epsilon_{\\mathrm{Adam}}>0$, the first Adam transition satisfies')
    text=text.replace('For active coordinates above the epsilon scale,',
        'The initial moments are $m_1=(1-\\beta_1)g$ and $v_1=(1-\\beta_2)g^2$. The LR does not enter these moment updates: paired arms initially differ in weights, not moments. Later gradients and moments can diverge. For every coordinate, $|\\Delta\\theta_{1,i}|\\leq\\alpha_{\\mathrm{commit},1}$. For active coordinates above the epsilon scale,')
    text=text.replace('for $D$ optimized coordinates;', 'for $D$ optimized coordinates, not model width;')
    text=text.replace(' r_t=',' r_t^{x}=').replace('\\|\\theta_t-\\theta_{t-1}\\|_2','\\|\\theta_t^{x}-\\theta_{t-1}\\|_2')
    text=text.replace('When a candidate is rolled back', 'Here $x\\in\\{\\mathrm{candidate},\\mathrm{committed}\\}$ and $\\epsilon_r=10^{-12}$, distinct from the Adam denominator constant. This displacement diagnostic does not guarantee task quality. When a candidate is rolled back')
    schedules='''**Table 1. Actual committed scratch LR sequences (one-based transition index).** The recorded rescue row applies to the observed trajectories, in which only the first candidate triggered repair.

| Condition | At transition 1 | At transitions t >= 2 |
|:--|:--|:--|
| Stable | S(1) | S(t) |
| Single bad then stable | 1.0 | S(t) |
| Corrected legacy | L(1) | L(t) |
| Late legacy | 1.0 | L(t-1) |
| Recorded rescue | Reject 1.0; commit L(1) | L(t-1) |

![Consumed LR differs at the first transition and, in the legacy conditions, at later transitions. Rescue describes the recorded first-step repair, not a guarantee for every trajectory.](../figures/fig1_execution_conditions.png){width=95%}

![Schedules derived from the frozen configuration: widths 128/256, October warmup 8,000 and stable zero boundary 50,000. Historical warmup 4,000 is separate.](../figures/fig2_actual_configuration_schedule.png){width=95%}

This fault uses manual LR assignment in a particular loop. It does not justify universally reversing framework scheduler calls; the consumed LR must be checked under the actual API and version.'''
    text=text.replace('### 3.3 Interventions and matched initialization',schedules+'\n\n### 3.3 Interventions and matched initialization')
    text=text.replace('40 matched-initialization groups show no hash discrepancies.',
        '40 matched-initialization groups show no hash discrepancies. Of the 240 scientific records, 180 contain trained test endpoints; diagnostic records are not extra trained classifiers.')
    text=text.replace('### 3.4 Experimental settings and endpoint selection',
        '''Guard thresholds use leave-one-seed-out calibration within each dataset/model configuration. For the held-out seed, the threshold is $\\kappa=10$ times the largest committed displacement ratio over safe steps 1-100 from the other four seeds: stable and corrected legacy for scratch, and safe correct order for pretrained models. The held-out seed is excluded. Displacement guard enforcement ends after step 100. Fold-specific thresholds and retrospective sensitivity appear in the supplement.

### 3.4 Experimental settings and endpoint selection''')
    caveat='''

### 3.5 Evaluation definitions and archived selection caveat

The archived scratch fields `loss` and `nll` are different quantities. `loss` is the unweighted mean of per-batch mean logit cross-entropies. A partial final batch therefore receives the weight of a full batch. The separate `nll` field is clipped-probability NLL, using a probability floor of $10^{-12}$ and capping each example's contribution at about 27.631. Neither is silently relabeled as corrected sample-weighted CE.

For future evaluation, sample-weighted logit CE is $L=(\\sum_b n_b L_b)/(\\sum_b n_b)$, computed stably from logits without a probability cap. CPU counterexamples reproduce an archived batch-mean loss of 5.347 versus sample-weighted 0.975 for a 32-plus-1 batch partition, and ordinary CE 2000 versus clipped-probability NLL 27.631 for extreme logits. These are evaluation fixtures, not new model-training results. The corrected evaluator is partition-invariant in the regression fixture.

Archived scratch checkpoint selection used the minimum archived validation batch-mean CE. Changing the evaluator could change checkpoint rankings, but the missing checkpoint weights prevent quantifying that change. All original selections and saved accuracy endpoints are preserved. Future runs use a separately versioned evaluator and cannot replace old results without explicit labeling.
'''
    text=text.replace('### 3.5 Historical severity study (distinct protocol)',caveat+'\n### 3.6 Historical severity study (distinct protocol)')
    text=text.replace('### 3.6 Analysis','### 3.7 Analysis')
    text=text.replace('two explicitly exploratory contrasts separate rescue-minus-late (first-step repair) and corrected-minus-rescue (remaining delay).',
        'two explicitly exploratory contrasts report rescue-minus-late conditional first-step repair and corrected-minus-rescue conditional remaining delay. They do not prove an additive decomposition.')
    text=text.replace('## 4. Scratch results separating first-step faults and scheduler delay',
        '''### 3.8 Procedure effects and the incomplete fixed-budget design

Unequal stopping may be an effect of the first-step intervention itself. It does not invalidate the recorded procedure comparison, but prevents interpreting that comparison as a fixed-computation effect. For a prospective legacy design, let $Y(b;\\ell;B)$ denote the outcome at common budget $B$, with first-step fault $b$ and later lag $\\ell$. Corrected legacy supplies the safe/no-lag condition, rescue the safe/lag condition, and late legacy the bad/lag condition. The bad/no-lag legacy condition is missing. The stable bad-first arm changes the later schedule family and cannot fill that cell.

Consequently, $I(B)=Y(1;1;B)-Y(1;0;B)-Y(0;1;B)+Y(0;0;B)$ is unidentified. A new fixed-budget cell cannot be combined with unmatched old endpoints to claim a complete factorial. New work must prespecify primary endpoints, multiplicity and failure handling; additional seeds must not be added until significance appears.

## 4. Scratch results under the recorded procedures''')
    # Use one complementary selected-endpoint plot instead of three repeated ones.
    text=text.replace('### 4.1 Isolated first update with stable scheduling held fixed',
        '''![All five paired selected-test differences are shown for each setting; black diamonds are means. The two legacy contrasts are exploratory, with unequal realized stopping budgets. The small corrected-minus-rescue panel has its own scale and is not an equivalence test.](../figures/fig3_selected_paired_endpoints.png){width=95%}

### 4.1 Isolated first update with stable scheduling held fixed''')
    text=text.replace('## 5. Training progression and effective optimization horizon','## 5. Shared-step validation and optimization budgets')
    shared='''

### 5.1 All shared logged-step comparisons

For each dataset, architecture and contrast, we take the intersection of logged optimizer steps across both arms and all five seeds. This yields 95 shared-step summaries and 475 paired seed differences across 12 setting/contrast groups. Every shared step is retained: there is no interpolation, no carrying stopped runs forward, and no favorable-horizon selection. These exploratory validation comparisons are separate from validation-selected test endpoints. Figures 4-5 show the underlying validation outcomes; supplementary paired intervals cover all 95 summaries.

![Validation at every step shared by both stable-family arms and all five seeds. Faint dots show individual seeds; larger markers and bars show mean and sample SD. The dashed line is the stable zero-LR boundary. Only logged observations are displayed.](../figures/fig4_shared_step_stable_validation.png){width=97%}

![Legacy validation outcomes at all steps shared within each contrast and all five seeds. Dots show seeds; markers and bars show mean and sample SD. Each contrast retains its own full shared-step intersection.](../figures/fig5_shared_step_legacy_validation.png){width=97%}

**Table 4. DBpedia shared-step illustration at transition 44,625.** This step is the illustration specified in the supplied plan, not a newly selected optimum. All 95 summaries are available. Shared-step validation differences and selected-test differences are distinct endpoints.

| Architecture | Shared step | Rescue - late validation accuracy (pp) | Rescue - late selected test accuracy (pp) |
|:--|--:|--:|--:|
| medium-4 | 44,625 | 44.293 | 5.716 |
| large-6 | 44,625 | 40.414 | 16.505 |

The larger earlier gaps show a difference in learning progress at that step; they do not contradict the smaller selected-test gaps. They do not by themselves identify a recovery mechanism. Archived loss curves retain their batch-mean definition.

### 5.2 Budgets and training performance
'''
    text=text.replace('## 5. Shared-step validation and optimization budgets','## 5. Shared-step validation and optimization budgets'+shared)
    text=text.replace('## 6. Historical initial-LR severity: separate training experiment','## 6. Historical severity under a distinct protocol')
    text=text.replace('All legacy transitions remain positive;', 'All legacy transitions remain positive;')
    text=text.replace('Wall-clock/GPU time is absent from these returned records and is not reconstructed from step counts.',
        'Wall-clock/GPU time is absent from these returned records and is not reconstructed from step counts. At zero LR, parameters stop moving while Adam moments and counters can still change. Selected-checkpoint and termination steps are separate fields in the per-run budget export.')
    text=text.replace('The guard detects', 'The guard detects')
    text=text.replace('These experiments do not establish a broader advantage for displacement monitoring.',
        'The 35/40 guard result and 20/20 assertion result concern different populations, not a paired superiority test. These experiments do not establish a broader advantage for displacement monitoring.')
    text=text.replace('Runtime and memory overhead were not measured.',
        'Runtime and memory overhead were not measured. Even the archived no-guard path performs copying/snapshots; an efficiency claim would require a truly uninstrumented baseline.')
    contract='''

### 7.3 Future finite-state and replay contract

Source inspection and tiny CPU counterexamples identify two executor limitations. A direct comparison `NaN > threshold` is false, so threshold comparison alone does not reject nonfinite updates. Replaying at the intended LR also need not enforce a displacement ceiling: a scalar Adam fixture gives a replay ratio near 0.100 with threshold 0.001. Neither failure was observed in the saved scientific traces, and these counterexamples do not explain historical training outcomes.

The versioned future executor checks finite gradients/model/optimizer state before the candidate, and finite parameters, buffers, moments, counters and ratios after updates. Numerical failure restores a verified finite snapshot and stops. A missing verified state is reported explicitly rather than described as successful repair. A nonfinite gradient is not repaired by changing the LR.

The new contract permits at most one same-gradient replay and revalidates it. A nonfinite or over-threshold replay is restored and rejected, with no committed transition. Threshold enforcement remains confined to steps 1-100; finite-state checks continue beyond that window. Regression fixtures cover steps 1, 100 and 101, warmed Adam state, and rejected replay accounting. The new fail-fast contract restores a rejected candidate; the archived fail-fast implementation retained its finite candidate before stopping. These versioned policies are not substituted into the historical evidence.

Separate full-boundary-state primitives retain RNG and sampler position. A tiny CPU dropout fixture reproduces uninterrupted execution after serialization and resume at a mid-permutation cursor. That check does not certify the original data-loader trajectory, actual training device, AMP or distributed recovery. No new research training or checkpoint-level prediction reevaluation is included.
'''
    text=text.replace('## 8. Discussion and limitations',contract+'\n## 8. Discussion and limitations')
    text=text.replace('Exploratory rescue-versus-late and corrected-versus-rescue analyses separate first-step repair from the remaining scheduler delay.',
        'Exploratory legacy comparisons report conditional first-step-repair and remaining-delay contrasts; the incomplete factorial does not identify their interaction.')
    text=text.replace('The scratch results distinguish a large first-step fault from the smaller residual difference associated with scheduler delay after rescue.',
        'The scratch results show a large first-step intervention effect under the recorded procedure. Shared-step histories reveal learning delay that selected endpoints alone conceal; the incomplete legacy factorial prevents an additive causal decomposition.')
    text=text.replace('Fault prevalence and guard overhead were not measured.',
        'Fault prevalence and guard overhead were not measured. Model sizes vary width and depth together, so depth is not isolated. The batch-mean evaluator caveat and clipped-probability NLL limit historical loss interpretation. A five-seed t interval is descriptive, and the future numerical safeguards are supported by CPU counterexamples rather than observed historical nonfinite failures.')
    text=text.replace('The latter differences are small but do not establish equivalence.',
        'Small corrected-minus-rescue differences do not establish equivalence. All shared-step histories are retained alongside selected endpoints; recovery beyond the positive-LR budget remains unmeasured.')
    # Add exact source references for newly discussed related work.
    text+='''

Alimisis, F., Islamov, R., and Lucchi, A. (2026). Why Do We Need Warm-up? A Theoretical Perspective. *Proceedings of Machine Learning Research* 306, 1909-1980. https://proceedings.mlr.press/v306/alimisis26a.html

Bai, Z. et al. (2026). Adaptive Preconditioners Trigger Loss Spikes in Adam. *Proceedings of Machine Learning Research* 306, 5566-5601. https://proceedings.mlr.press/v306/bai26o.html

Shazeer, N., and Stern, M. (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. *Proceedings of Machine Learning Research* 80, 4596-4604. https://proceedings.mlr.press/v80/shazeer18a.html

Topollai, K., and Choromanska, A. (2026). Balancing Early Performance Sacrifices with Long-Term Gains: Scaling Learning-Rate Warmup Duration Across Training Horizons. *Preprint*, arXiv:2609.33041. https://arxiv.org/abs/2609.33041
'''
    # Provide the six required primary-source links without embedding long URLs in the table.
    text=text.replace('Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*.',
        'Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980')
    text=text.replace('Ma, J., and Yarats, D. (2021). On the Adequacy of Untuned Warmup for Adaptive Optimization. *Proceedings of the AAAI Conference on Artificial Intelligence* 35(10), 8828–8836.',
        'Ma, J., and Yarats, D. (2021). On the Adequacy of Untuned Warmup for Adaptive Optimization. *Proceedings of the AAAI Conference on Artificial Intelligence* 35(10), 8828-8836. https://doi.org/10.1609/aaai.v35i10.17069')
    # Copy original schedule image without changing its configuration-derived data.
    shutil.copy2(ROOT/'figures/fig4_scheduler_and_zero_boundary.png',ROOT/'figures/fig2_actual_configuration_schedule.png')
    for src,dest in [('fig2_selected_paired_endpoints.png','fig3_selected_paired_endpoints.png'),
                     ('fig3_shared_step_stable_validation.png','fig4_shared_step_stable_validation.png'),
                     ('fig4_shared_step_legacy_validation.png','fig5_shared_step_legacy_validation.png')]:
        shutil.copy2(ROOT/'figures'/src,ROOT/'figures'/dest)
    supplement+='''

## Evaluation definitions and shared-step exports

The archived scratch `loss` is the mean of batch-mean logit cross-entropies, not sample-weighted CE. The separate `nll` is clipped-probability NLL with probability floor 1e-12. Checkpoint selection used the minimum archived validation batch-mean CE. Original selections and all saved test accuracy values remain unchanged; missing checkpoints prevent quantifying ranking changes under a corrected evaluator. The 240 scientific job records include 180 trained test endpoints; diagnostic arms have no primary trained endpoint.

All shared-step validation comparisons are exploratory. matched_step_per_seed.csv contains 475 seed pairs; matched_step_summary.csv contains 95 summaries, and matched_step_coverage.csv lists every retained and excluded logged step for the 12 setting/contrast groups. Only steps observed in both arms and all five seeds are retained. No interpolation or carrying stopped runs forward is used. Archived losses retain their original definition. The paired intervals below use five seeds and four degrees of freedom and are unadjusted across horizons/settings; they are not confirmatory discoveries.

![All 95 paired shared-step validation accuracy differences with descriptive 95% Student-t intervals, n=5. Each panel retains all shared logged steps and uses its own vertical scale.](../figures/figS1_all_shared_step_paired_intervals.png){width=98%}

## Supporting trajectories and provenance
'''
    supporting=[('fig8_condition_training_loss.png','figS2_archived_online_training_loss.png','Archived online training loss by condition. These batch-mean losses are not corrected sample-weighted evaluation CE.'),
                ('fig6_ineffective_postzero_steps.png','figS3_positive_and_zero_lr_budgets.png','Positive-LR transitions, zero-LR computation and replay overhead remain separate.'),
                ('fig5_historical_initial_lr_sweep.png','figS4_historical_initial_lr_severity.png','Historical three-seed/W4000 trained-model severity outcomes; separate from October five-seed/W8000.'),
                ('fig7_guard_threshold_sensitivity.png','figS5_retrospective_threshold_sensitivity.png','Primary kappa=10 and retrospective sensitivity; WNUT misses remain visible.'),
                ('fig1_isolated_first_step_paired.png','figS6_stable_bad_selected_endpoints.png','All five stable and isolated-bad selected endpoints.'),
                ('fig2_legacy_schedule_order_paired.png','figS7_legacy_selected_endpoints.png','All five corrected and late legacy selected endpoints; stopping budgets differ.'),
                ('fig9_exploratory_repair_delay.png','figS8_exploratory_legacy_endpoints.png','Exploratory legacy selected endpoints; conditional comparisons do not complete a factorial.')]
    for src,dest,caption in supporting:
        shutil.copy2(ROOT/'figures'/src,ROOT/'figures'/dest)
        supplement+='\n!['+caption+'](../figures/'+dest+'){width=95%}\n'
    # Final source-plan audit: keep supporting repair/provenance history out of the main text.
    helper_history='The separate analysis reference now matches the runners: pretrained single-bad step 1 exits at its installed target LR, scratch single-bad step 1 exits at LR=1.0, and guard enforcement ends after step 100. Original training code and records are unchanged. The 80 missing threshold entries are exported from result JSON fields, and sensitivity tables are regenerated from the safe traces and LOSO manifests. Source mappings accompany Tables 1-7 and S1-S12.'
    assert helper_history in text
    text=text.replace(helper_history,'')
    start=text.index('The historical initial-LR sweep used')
    end=text.index('\n\n### 3.7 Analysis',start)
    historical_detail=text[start:end]
    text=text[:start]+'''The historical initial-LR sweep used late legacy scheduling with 4,000 warmup steps, three seeds (42, 123, 456), two model sizes and five initial rates $\\{1,0.3,0.1,0.03,0.01\\}$ per dataset (60 runs). Test accuracy and sample SD are reconstructed from executed notebook endpoints. This cohort is separate from the five-seed October study with 8,000 warmup steps; runner and trace provenance appears with Supplementary Table S16.'''+text[end:]
    supplement+='\n\n## Historical configuration and analysis-reference provenance\n\n'+historical_detail+'\n\n'+helper_history.replace('Tables 1-7 and S1-S12','main and supplementary tables')+'\n'
    text=text.replace('Descriptive 95% Student-$t$ intervals use four degrees of freedom.',
        'Descriptive 95% Student-$t$ intervals use four degrees of freedom. They treat seed-level paired differences as independent and approximately normal, conditional on the fixed data, split and configuration. Five seeds cannot reliably assess normality; the intervals are unadjusted and descriptive, not simultaneous or confirmatory guarantees.')
    supplement=supplement.replace('The paired intervals below use five seeds and four degrees of freedom and are unadjusted across horizons/settings; they are not confirmatory discoveries.',
        'Paired intervals use five independent, approximately normal seed differences conditional on fixed data, split and configuration, with four degrees of freedom. Normality cannot be assessed reliably at n=5. The intervals are descriptive and unadjusted across horizons/settings, not simultaneous, confirmatory or equivalence guarantees.')
    text=text.replace('optimizer, model architecture, dataset, schedule and task also differ.',
        'fault severity, optimizer, model architecture, dataset, schedule and task also differ.')
    text=text.replace('The pretrained comparisons vary architecture, task, optimizer and schedule',
        'The pretrained comparisons vary fault severity, architecture, task, optimizer and schedule')
    text=text.replace('The WNUT candidate ratios (~$3.13\\times10^{-4}$) are below its held-out safe-envelope thresholds (approximately $1.5\\times10^{-3}$).',
        'The WNUT candidate ratios (~$3.13\\times10^{-4}$) average about 2.05 times the fold-specific safe-window maximum across the five folds, below the primary $\\kappa=10$ thresholds (approximately $1.5\\times10^{-3}$).')
    # All references remain author-year citations; sort the bibliography by author.
    pre,refs=text.split('## References',1)
    ref_blocks=[b.strip() for b in refs.strip().split('\n\n') if b.strip()]
    text=pre+'## References\n\n'+'\n\n'.join(sorted(ref_blocks,key=str.casefold))+'\n'
    supplement=supplement.replace('| Train CE |','| Archived train loss |').replace('| Last online train CE |','| Archived online train loss |')
    # References in prose must agree with moved/renumbered tables.
    text=text.replace('Tables 3b/6','Supplementary Tables S14/S15').replace('Tables 1-7 and S1-S12','main and supplementary tables')
    text=text.replace('Table 2,','Table 3,')
    text=text.replace('The archived scratch fields','The archived October scratch fields')
    # Leave the existing title exactly unchanged.
    assert text.splitlines()[0]==original.splitlines()[0]
    text=re.sub(r'\n{3,}','\n\n',text).strip()+'\n'
    (M/'Research_Manuscript.md').write_text(text,encoding='utf-8')
    (M/'All_Seed_Results_and_Provenance.md').write_text(supplement,encoding='utf-8')
    (M/'Experimental_Evidence_and_Revision_Rationale.md').write_text((ROOT/'revision_rationale_source.md').read_text(encoding='utf-8'),encoding='utf-8')
    print('Built revised manuscript and expanded scientific supplement; title unchanged.')

if __name__=='__main__':build()

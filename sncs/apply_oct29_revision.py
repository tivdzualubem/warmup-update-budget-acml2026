"""Apply the focused 29 October revisions to the reproducible prior document."""
from pathlib import Path
import re
from build_submission_revision import build as build_prior
ROOT=Path(__file__).resolve().parent
M=ROOT/'manuscript'

def section(text,start,end,body):
    a=text.index(start);b=text.index(end,a+len(start))
    return text[:a]+start+'\n\n'+body.strip()+'\n\n'+text[b:]

def build():
    build_prior()
    text=(M/'Research_Manuscript.md').read_text(encoding='utf-8')
    supplement=(M/'All_Seed_Results_and_Provenance.md').read_text(encoding='utf-8')
    title=text.splitlines()[0]
    abstract='''Intended learning-rate schedules and selected final accuracy can conceal different training failures. We trace the rates actually consumed by Adam and compare a single incorrect first update with continuing scheduler delay in archived five-seed AG News and DBpedia experiments. A first update at LR=1.0, 10,000 times the configured stable peak, followed by the prescribed stable schedule reduces selected test accuracy by 47.19-70.99 percentage points under the recorded stopping and selection procedure. Exploratory legacy comparisons retain later lag after first-step repair or remove that lag conditional on repair. All 95 shared logged-step comparisons are retained; they reveal learning delay that smaller selected-endpoint gaps conceal. Unequal stopping and a missing legacy condition prevent equal-compute or complete interaction interpretations. Stable LR is zero from transition 50,000, so later epochs do not test additional positive-LR recovery. The archived validation batch-mean CE overweights AG News's partial final batch; DBpedia validation has only full batches and is unaffected by this mechanism. Original selections are preserved because candidate-epoch assets were not recovered. A displacement guard detects 35/40 mismatches, missing all five WNUT cases; an LR assertion catches 20/20 tested pretrained mismatches in a different population. The evidence supports a bounded execution-contract case study, not permanent damage, a new optimizer or general guard superiority.'''
    text=section(text,'## Abstract','**Keywords:',abstract)
    # section() preserves the following heading; Keywords is an inline marker.
    intro='''A warmup specification describes intended learning rates, whereas an optimizer transition consumes the rates currently stored in its parameter groups. Installing a target after `optimizer.step()` can therefore produce a plausible post-step log while the preceding update used a different rate. That observation alone does not show whether a later performance deficit arises from one incorrect transition, continuing schedule misalignment, or insufficient subsequent optimization.

We distinguish those questions through recorded interventions. Stable versus bad-then-stable changes only the first consumed rate under the stable stopping-and-selection procedure. Rescue versus late legacy examines first-step repair while later lag remains. Corrected legacy versus rescue examines the remaining lag after repair. The latter comparisons are exploratory and have unequal realized budgets; they do not complete a fixed-budget factorial.

The contribution is an experimental account of what schedule logs and selected endpoints leave unresolved. Consumed-rate traces distinguish the first fault from continuing lag. Paired validation at every shared logged step exposes learning delay that selected final accuracy can obscure. Separating positive-LR transitions from zero-LR computation identifies the opportunity for further parameter recovery. The existing histories support these distinctions without establishing a recovery law or a new explanation of warmup.

The principal scratch stress test uses LR=1.0 at the first transition, 10,000 times the configured peak of $10^{-4}$; practical claims must retain that magnitude. We analyze five-seed October records separately from the historical three-seed severity sweep. Standard first-step Adam algebra explains the specified intervention, and LR assertions provide a simple prevention baseline for the known execution mistake.'''
    text=section(text,'## 1. Introduction','## 2. Related work',intro)
    related='''Warmup's role in limiting early changes is established. Gotmare et al. (2019) relate warmup to changes in later network layers and test early layer freezing. Ma and Yarats (2021) study untuned warmup for adaptive optimizers. These studies motivate controlling early updates; they do not make the harm from an oversized update a new finding here.

Kosson, Messmer and Jaggi (2024) analyze update magnitude, direction and representation change in GPT training. Their Appendix A also studies enlarged initial updates in ResNet-20/CIFAR-10 and extends training to 800 epochs. Thus, neither classifier experiments nor a persistent gap after longer training distinguishes our work by itself. Our specific comparison separates one incorrect consumed LR from continuing one-step lag, and distinguishes selected endpoints from shared-step outcomes and positive-LR budgets. We do not reproduce their mechanism analysis or infer that a general recovery law follows from our histories.

**Table R1. Prior results and the specific evidence examined here.** The distinctions describe tested interventions and endpoints; they are not claims to originate early-update control or optimizer theory.

| Primary work | What it establishes | Different question here | Evidence here |
|:--|:--|:--|:--|
| Kingma and Ba (2015), Adam | Bias-corrected adaptive moments | Which LR a fixed first gradient actually consumes | Equation 2; matched initial-gradient fixture |
| Shazeer and Stern (2018), Adafactor | Adaptive update clipping and oversized updates | Consequences of a specified execution mismatch | Candidate/committed traces; no new clipping rule |
| Gotmare et al. (2019) | Warmup, later-layer changes and freezing | One faulty transition versus continuing lag | Table 1; isolated first-step contrast |
| Ma and Yarats (2021) | Untuned warmup for adaptive updates | One changed first rate with later stable schedule retained | Table 2; separate realized budgets |
| Kosson et al. (2024), including Appendix A | Early-update harm, normalization and extended training | Consumed-rate fault versus lag; selected versus shared-step outcomes | Figures 3-5 and Table 4; recovery remains unmeasured |
| Kalra and Barkeshli (2024) | Conditioning, sharpness regimes and target-rate tolerance | Execution intervention without a landscape-mechanism test | LR and task histories; no sharpness evidence |
| Abuduweili and Liu (2025) | Nonzero second-moment initialization | First-rate intervention with matched zero initial moments | Equation 2 and initial-state checks |

Kalra and Barkeshli (2024) and Abuduweili and Liu (2025) change the interpretation or initialization of early optimizer dynamics. Our experiments hold the initial optimizer state fixed and do not identify conditioning, representation damage or moment initialization as the cause of the later deficit. Recent work on Adam loss spikes (Bai et al., 2026), warmup theory (Alimisis, Islamov and Lucchi, 2026) and horizon-dependent warmup duration (Topollai and Choromanska, 2026, preprint) further limits broad novelty claims. The empirical distinction here concerns the specified execution faults at observed budgets. Calibration metrics (Guo et al., 2017) measure prediction quality rather than establish a failure mechanism.'''
    text=section(text,'## 2. Related work','## 3. Method',related)
    text=text.replace('LR=1.0 is an extreme injected fault.',
        'LR=1.0 is an extreme injected fault: it is 10,000 times the configured peak $10^{-4}$, whereas the intended first stable rate is $1.25\\times10^{-8}$.')
    evaluation=r'''Archived scratch checkpoint selection used the minimum validation **batch-mean logit CE**. Let $L_b$ be the mean logit CE for batch $b$, containing $n_b$ examples, and let $B$ be the number of batches. The two averages are

$$
L_{\mathrm{archived}}=\frac{1}{B}\sum_b L_b,\qquad L_{\mathrm{example}}=\frac{\sum_b n_b L_b}{\sum_b n_b}.
$$

Equal batch sizes give equal example weighting.

DBpedia validation contains 84,000 examples: exactly 2,625 batches of 32. There is no partial validation batch, so this weighting mechanism cannot change DBpedia checkpoint selection. AG News validation contains 18,000 examples: 562 batches of 32 and a final batch of 16. Each final-batch example receives about twice its correct example-weighted influence. Whether this changed AG News's selected epoch remains unknown.

The separate archived `nll` field is clipped-probability NLL, with probability floor $10^{-12}$ and per-example cap about 27.631. It was not the checkpoint-selection objective. The validation-batch statement above does not automatically apply to every training or test population.

No candidate-epoch checkpoints, matching validation logits/labels or sufficient batch-loss records were recovered from the examined returned files. The `model_state` and `logits` anchor fields are hashes, not recoverable parameters or predictions. Original selections and accuracy endpoints remain unchanged. The separately versioned future evaluator uses uncapped, example-weighted logit CE; CPU extreme-logit and uneven-batch checks are reported in the supplement.'''
    # Raw literals keep TeX explicit; collapse escaped backslashes once.
    evaluation=evaluation.replace('\\\\','\\')
    text=section(text,'### 3.5 Evaluation definitions and archived selection caveat','### 3.6 Historical severity study (distinct protocol)',evaluation)
    text=text.replace('minimum validation loss according to the frozen runner.',
        'minimum archived validation logit CE according to the frozen runner and Section 3.5.')
    text=text.replace('Archived loss curves retain their batch-mean definition.',
        'Archived loss curves retain their batch-mean definition; DBpedia validation has full batches, while the AG News partial-batch caveat remains.')
    # Keep the monitoring evidence concise and secondary to the scientific contrasts.
    guard=r'''For this known scheduling mistake, a direct assertion that the impending LR equals its intended target is the simplest prevention baseline. It stops 20/20 deliberately misordered pretrained first steps before mutation. The displacement guard is a secondary diagnostic and repair demonstration: at primary $\kappa=10$ it detects 35/40 mismatches across eight configurations and misses all five WNUT-17 cases. These counts concern different populations, not a paired superiority test.

WNUT candidate ratios (~$3.13\times10^{-4}$) average about 2.05 times the fold-specific safe-window maximum, below primary thresholds near $1.5\times10^{-3}$. This is a detection miss at that operating point, not evidence of scratch-like downstream harm. At retrospective $\kappa=2$, all 40 mismatches exceed the threshold; that result does not prospectively validate a replacement multiplier. Zero alarms among dependent safe-window transitions do not certify a false-positive rate. Full thresholds and sensitivity are in Supplementary Table S17.

An LR assertion checks a specified execution invariant; displacement monitoring measures parameter motion; task metrics measure downstream utility. The current evidence does not establish general guard superiority. Runtime and memory overhead were not measured. The archived no-guard path also performs snapshots, so an efficiency claim would require a genuinely uninstrumented baseline.'''.replace('\\\\','\\')
    text=section(text,'### 7.1 LR assertion, displacement alarm, and downstream loss','### 7.2 Rollback contract',guard)
    start=text.index('### 7.2 Rollback contract');end=text.index('## 8. Discussion and limitations',start)
    old_technical=text[start:end].strip()
    replay='''### 7.2 Replay contract and validation scope

Archived rescue restores weights, Adam moments/counters, parameter groups and clipped gradients, then replays the same gradient at the intended rate. It does not repeat the stochastic forward pass. Fixed-gradient CPU fixtures verify that restoration against a clean safe step. A separate serialized CPU fixture checks RNG and sampler restoration; it does not certify the actual training device or real data loader.

The future executor checks finite gradients/state and permits at most one revalidated replay. Numerical failure or a rejected replay restores a verified finite snapshot and stops. These are separately versioned future-run corrections motivated by CPU counterexamples, not explanations of failures observed in the archived traces. Boundary tests and detailed repair behavior appear in the supplement. Actual-device resume fidelity and throughput remain unmeasured, and no new research training or checkpoint prediction reevaluation was performed.

'''
    text=text[:start]+replay+text[end:]
    discussion='''The evidence distinguishes failures that intended schedule logs alone cannot separate. Stable versus bad-then-stable changes the first consumed rate while retaining the later stable schedule. The legacy contrasts distinguish first-step repair while lag remains and remaining lag after repair. Early stopping is part of the observed procedures, so selected endpoints are not equal-compute comparisons; the missing bad-first/no-lag legacy cell prevents an interaction or additive decomposition.

Selected final accuracy also conceals learning progress. The larger DBpedia rescue-minus-late gaps at shared step 44,625 coexist with smaller selected-test gaps. All shared logged steps are reported, without interpolation or carrying stopped runs forward. At stable step 50,000 the LR becomes zero, so additional epochs do not supply further positive-LR parameter recovery. The deficit persisted under the tested procedure; permanent damage is not established.

The first-step LR=1.0 stress test is 10,000 times the configured peak. The historical severity sweep supplies a separate severity comparison with different warmup and seeds; it does not validate a practical fault rate for the October protocol. If practical relevance requires another experiment, a targeted pair could consume the configured peak $10^{-4}$ accidentally at step one and then follow the same schedule as the clean arm, with matched initialization/data streams, fixed positive-update budget and corrected evaluation. A null result would narrow the practical claim. This is a proposal, not a reported experiment; the extreme-fault recovery extension addresses a different question.

AG News checkpoint rankings under corrected validation weighting remain unknown because candidate-epoch assets were not recovered. DBpedia validation has no partial batch and is unaffected by that mechanism. The clipped-probability NLL caveat concerns a separate reported metric. Missing weights also prevent independent prediction-level reevaluation. Five-seed intervals are descriptive, small effects do not establish equivalence, and the observed tasks/configurations limit generalization.

Pretrained comparisons vary fault severity, optimizer, architecture, task and schedule together, so their smaller effects do not isolate pretraining. Width and depth also vary together. All WNUT misses remain visible; guard overhead and fault prevalence are unmeasured. CPU replay and resume fixtures do not establish actual-device or real-loader continuation fidelity. Recovery beyond the tested positive-LR budget remains unmeasured.'''
    text=section(text,'## 8. Discussion and limitations','## 9. Conclusion',discussion)
    conclusion='''The recorded experiments distinguish an incorrect first consumed LR from continuing scheduler delay and the budget available afterward. Under the stable procedure, the LR-1.0 first step, 10,000 times the configured peak, produces a large selected-test deficit. Exploratory legacy contrasts and all shared-step histories show why selected endpoints alone can understate learning delay; they do not identify a complete interaction. Epochs after stable LR reaches zero do not test further positive-LR recovery, and permanent damage is not established. Historical selections are preserved, with the unresolved AG News validation-weighting caveat and the DBpedia full-batch distinction stated. The guard's 35/40 detections and five WNUT misses remain secondary evidence; the assertion's 20/20 result comes from another population. This execution-contract case study supports conclusions about the tested interventions and observed budgets, without a general warmup mechanism, equivalence claim or completed recovery result.'''
    text=section(text,'## 9. Conclusion','## References',conclusion)
    links={
        'Gotmare, A.,':'https://arxiv.org/abs/1810.13243',
        'Kosson, A.,':'https://proceedings.neurips.cc/paper_files/paper/2024/hash/059445c2d5b3ef918079851628fef1d6-Abstract-Conference.html',
        'Kalra, D. S.,':'https://arxiv.org/abs/2406.09405',
        'Abuduweili, A.,':'https://proceedings.mlr.press/v280/abuduweili25a.html'}
    pre,refs=text.split('## References',1);blocks=[b.strip() for b in refs.split('\n\n') if b.strip()]
    for i,block in enumerate(blocks):
        for prefix,url in links.items():
            if block.startswith(prefix) and url not in block:blocks[i]=block+' '+url
    text=pre+'## References\n\n'+'\n\n'.join(blocks)+'\n'
    # Correct the broader wording in the supplement rather than relabel old values.
    old='The archived scratch `loss` is the mean of batch-mean logit cross-entropies, not sample-weighted CE. The separate `nll` is clipped-probability NLL with probability floor 1e-12. Checkpoint selection used the minimum archived validation batch-mean CE. Original selections and all saved test accuracy values remain unchanged; missing checkpoints prevent quantifying ranking changes under a corrected evaluator. The 240 scientific job records include 180 trained test endpoints; diagnostic arms have no primary trained endpoint.'
    new='''Archived scratch selection used validation batch-mean logit CE. DBpedia validation has 84,000 examples and 2,625 full batches of 32, so batch-mean and example-weighted CE have the same weighting there. AG News validation has 18,000 examples, 562 full batches and a final batch of 16; final-batch examples receive 18,000/(563 x 16), approximately 1.9982 times their correct influence. Whether that altered AG News selection is unknown. This distinction is specific to the validation populations.

The separate clipped-probability NLL uses floor 1e-12 and was not the selection objective. Original selections and test endpoints remain unchanged. Examined returns contain aggregate validation metrics and anchor hashes, not recovered candidate-epoch weights, validation logits/labels or sufficient per-batch losses. The 240 scientific records include 180 trained test endpoints; diagnostic arms have no primary trained endpoint. Dataset-specific layout and selection-effect status are exported in validation_batch_layout.csv and archived_evaluator_provenance.csv.'''
    assert old in supplement;supplement=supplement.replace(old,new)
    supplement=supplement.replace('Archived online training loss by condition. These batch-mean losses are not corrected sample-weighted evaluation CE.',
        'Archived online training loss by condition, recorded while parameters and dropout state change. These trajectories are distinct from fixed-checkpoint validation evaluation.')
    supplement+='\n\n## Evaluation fixtures and replay implementation details\n\n'
    supplement+='The extreme-logit CPU fixture gives logit CE 2000 versus clipped-probability NLL 27.631. A deliberately uneven 32-plus-1 fixture gives an archived batch-mean loss of 5.347 versus example-weighted 0.975. These fixtures demonstrate evaluator behavior, not the magnitude or checkpoint-ranking effect of the actual AG News final batch. The corrected evaluator is partition-invariant in the regression fixture.\n\n'
    supplement+=old_technical.replace('### 7.2 Rollback contract','### Archived fixed-gradient contract').replace('### 7.3 Future finite-state and replay contract','### Separately versioned finite-state contract')+'\n'
    assert text.splitlines()[0]==title
    for name,body in [('Research_Manuscript',text),('All_Seed_Results_and_Provenance',supplement)]:
        (M/(name+'.md')).write_text(re.sub(r'\n{3,}','\n\n',body).strip()+'\n',encoding='utf-8')
    print('Applied focused 29 October revisions; title, source data and selections unchanged.')

if __name__=='__main__':build()

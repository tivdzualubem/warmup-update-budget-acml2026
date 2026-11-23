"""Integrate the completed prespecified pair while retaining historical evidence."""
from pathlib import Path
import csv,json
from apply_nov08_revision import build as previous_build
ROOT=Path(__file__).resolve().parent


def build():
    previous_build()
    M=ROOT/'manuscript';main=(M/'Research_Manuscript.md').read_text();supp=(M/'All_Seed_Results_and_Provenance.md').read_text()
    title=main.splitlines()[0]
    result=json.loads((ROOT/'evidence/peak_first/run/result.json').read_text())
    audit=json.loads((ROOT/'evidence/peak_first/FULL_AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['prediction_files_independently_recomputed']==42 and audit['full_paired_checkpoint_files_checked']==5
    curves=list(csv.DictReader((ROOT/'tables/peak_first_curves.csv').open()))
    def replace(old,new):
        nonlocal main
        assert old in main,old[:100]
        main=main.replace(old,new)
    replace('A displacement guard detects 35/40 mismatches,',
        'A separate fixed-budget DBpedia pair instead consumes the configured peak $10^{-4}$ at the first step. At 49,999 positive-LR updates, its validation CE difference is $+4.70\\times10^{-6}$ and test accuracies are 98.5986% (clean) and 98.6029% (peak-first). This single pair does not reproduce the large extreme-fault deficit or establish equivalence. A displacement guard detects 35/40 mismatches,')
    replace('Standard first-step Adam algebra explains the specified intervention,',
        'A newly recorded single-pair comparison at the configured peak tests a less extreme execution mistake at a fixed budget and yields nearly identical final endpoints. Standard first-step Adam algebra explains the specified intervention,')
    replace('No candidate-epoch checkpoints, matching validation logits/labels or sufficient batch-loss records were recovered from the examined returned files.',
        'For the historical October runs, no candidate-epoch checkpoints, matching validation logits/labels or sufficient batch-loss records were recovered from the examined returned files.')
    replace('The separately versioned future evaluator uses uncapped, example-weighted logit CE;',
        'The new configured-peak pair uses the separately versioned uncapped, example-weighted logit CE evaluator;')
    method=r'''### 3.10 Fixed-budget configured-peak comparison

We separately recorded one clean/peak-first pair on DBpedia medium-4, seed 42, with the frozen split seed 42, 476,000 training examples, 84,000 validation examples and 70,000 official test examples. The training-only vocabulary, lowercase whitespace tokenization, length 128, width 128/four-layer Pre-LN model, Adam settings, batch size 32 and stable schedule follow the preserved configuration. Both FP32 arms start from identical weights, zero Adam moments and matching first-batch, pre-forward RNG, logits, loss and clipped-gradient fingerprints. They share an explicit sampler order and retain separate matched RNG streams and optimizer states. This prospective sampler implementation differs from the historical DataLoader's implicit RNG consumption; this pair is not a continuation or exact rerun of the historical trajectories.

The clean arm consumes $S(1)=1.25\times10^{-8}$ and the peak-first arm consumes $10^{-4}$ at transition one, an 8,000-fold rate difference. Both then consume exactly the same $S(t)$ and stop at 49,999 positive-LR transitions, with no early stopping, checkpoint selection, zero-LR steps, rescue or extra continuation. Each processes 1,599,968 training examples. The primary endpoint fixed before execution is example-weighted validation logit CE of peak-first minus clean at the final step; accuracy and fixed-final test metrics are secondary.

Validation and a fixed 8,192-example training subset are evaluated at transitions 0, 1, 100, 1,000, 8,000, 10,000, 20,000, 30,000, 40,000 and 49,999. Test evaluation occurs only after both final training states exist. Online training CE is retained for every 100-step block and the final 99-step block. All logits, labels, example identities and full paired states at 0, 1, 8,000 and 49,999, plus the latest state, are preserved. Evaluation restores training RNG. Execution used one Kaggle Tesla T4, Python 3.13.16 and PyTorch 2.11.0+cu130 with deterministic operations. A short real-data/device serialization test matched uninterrupted execution exactly before the research run. This is one prespecified configuration and seed, not a replicated equivalence study.

'''
    replace('## 4. Scratch results under the recorded procedures',method+'## 4. Scratch results under the recorded procedures')
    replace('Runtime and memory overhead were not measured.',
        'The guard\'s incremental runtime and memory overhead were not measured.')
    replace('Actual-device resume fidelity and throughput remain unmeasured, and no new research training or checkpoint prediction reevaluation was performed.',
        'These replay fixtures remain distinct from the new configured-peak runner, whose real-data/device resume check passed and whose actual training cost is recorded in Section 8. The new pair has no rollback intervention. Historical candidate-epoch prediction reevaluation remains unavailable.')
    new_results=r'''## 8. Configured-peak fault at a fixed positive-LR budget

The configured-peak first step produces a small fixed-final CE difference: validation CE is 0.061363441 for clean and 0.061368136 for peak-first, giving the prespecified contrast $+4.695\times10^{-6}$. Validation accuracy is 98.5024% versus 98.5107%; test accuracy is 98.5986% versus 98.6029%. The accuracy differences correspond to seven additional correct validation predictions and three additional correct test predictions for peak-first. These fixed-set counts are descriptive, not evidence of a population advantage or equivalence.

**Table 6. Fixed-final results for the new DBpedia medium-4 pair (seed 42).** Both arms receive 49,999 positive-LR transitions and zero zero-LR transitions. CE is uncapped and example-weighted; the training subset is not the full training population.

| Population | Clean CE | Peak-first CE | Clean accuracy (%) | Peak-first accuracy (%) |
|:--|--:|--:|--:|--:|
'''
    for pop,label in [('validation','Validation (primary CE)'),('test','Official test'),('train_monitor_subset','Fixed training subset')]:
        a=result['final_metrics']['clean'][pop];b=result['final_metrics']['peak_first'][pop]
        new_results+=f"| {label} | {a['logit_ce']:.9f} | {b['logit_ce']:.9f} | {100*a['accuracy']:.4f} | {100*b['accuracy']:.4f} |\n"
    new_results+=r'''
![Absolute validation CE and accuracy at every prespecified checkpoint for the single matched pair. Both arms use the same positive-LR budget; overlapping late trajectories are retained. The horizontal scale is linear near zero and logarithmic beyond 100 transitions. No seed intervals or equivalence inference are shown.](../figures/fig6_peak_first_fixed_budget.png){width=98%}

The early metrics do not support a simple damage-and-recovery narrative. Immediately after the first transition, peak-first has lower validation CE (2.872100 versus 3.051227) but lower accuracy (6.8857% versus 7.1226%). At transition 100 its accuracy is 0.3012 percentage points lower; at 1,000 it is 0.2440 points higher. All prespecified later contrasts are retained in the supplement. The first realized normalized displacements are $6.604\times10^{-8}$ and $5.066\times10^{-4}$; the saved first-step moments match while weights differ. Thus a substantially larger initial update is present without the large final task deficit seen in the extreme-fault records.

The recorded session took 82.83 minutes, including training, evaluation and checkpointing. Instrumented training time was 34.80 minutes for clean and 34.95 for peak-first; evaluation took 4.97 and 4.93 minutes. Both budgets and costs are reported separately in the supplement. These timings characterize this run, not guard overhead or an uninstrumented speed comparison.

This new within-pair comparison controls initialization, streams, schedule and fixed budget. Its tiny observed endpoint contrast narrows the practical interpretation of the historical LR=1.0 stress test. Differences in stopping, sampler implementation, hardware and software prevent treating the historical and new results as a fully controlled cross-study dose-response. A single seed and absence of a prespecified equivalence margin preclude a general safety or equivalence conclusion.

'''
    replace('## 8. Discussion and limitations',new_results+'## 9. Discussion and limitations')
    replace('## 9. Conclusion','## 10. Conclusion')
    begin=main.index('The first-step LR=1.0 stress test is 10,000 times the configured peak.',main.index('## 9. Discussion'))
    end=main.index('\n\nAG News checkpoint rankings',begin)
    main=main[:begin]+r'''The first-step LR=1.0 stress test is 10,000 times the configured peak. The historical severity sweep has different warmup and seeds, so it does not validate a practical fault rate for the October protocol. The new fixed-budget peak-first pair yields a validation CE gap of only $4.70\times10^{-6}$ and nearly identical test accuracies. It does not reproduce the large extreme-fault endpoint deficit in this configuration. This negative practical finding must accompany the stress-test result; neither establishes a universal safe rate or an equivalence bound. The full extreme-fault recovery extension addresses a different question and remains unperformed.'''+main[end:]
    replace('Missing weights also prevent independent prediction-level reevaluation.',
        'Missing historical weights prevent independent historical prediction-level reevaluation; all 42 new prediction files are retained and their metrics were independently recomputed.')
    replace('CPU replay and resume fixtures do not establish actual-device or real-loader continuation fidelity.',
        'CPU rollback fixtures do not certify historical-device replay. The new pair has a separate actual-device/real-loader resume check; it does not validate cross-device continuation or the deferred recovery extension.')
    replace('This execution-contract case study supports conclusions about the tested interventions and observed budgets,',
        'The new fixed-budget configured-peak pair has nearly identical final endpoints and narrows the practical interpretation of the extreme-fault result. This execution-contract case study supports conclusions about the tested interventions and observed budgets,')
    supp=supp.replace('No new research model training, corrected checkpoint reselection or prediction-level reevaluation is reported. The configured-peak pair is prospective; full extreme-fault recovery remains unperformed.',
        'The new fixed-budget configured-peak pair is reported separately below, including all saved predictions and full retained states. Historical checkpoint selections are unchanged; historical candidate-epoch assets remain unavailable. Full extreme-fault recovery remains unperformed.')
    supp=supp.replace('No new research training or checkpoint-level prediction reevaluation is included.',
        'These CPU fixtures are separate from the completed configured-peak pair and its actual-device/data resume check reported below; historical checkpoint-level prediction reevaluation remains unavailable.')
    supp+='''\n\n## Completed configured-peak pair: design and evidence\n\nOne DBpedia medium-4 pair, seed 42 and split seed 42, was run with the prespecified 49,999-positive-update budget. The only first-transition rate change is from 1.25e-8 to 1e-4; rates at every later transition match exactly within the pair. Adam uses betas (0.9, 0.98), epsilon 1e-9, no weight decay and gradient norm clipping at 1. Both FP32 arms process 1,599,968 examples, with no early stopping, checkpoint selection, zero-LR transitions, guard rescue or recovery extension. The fixed sample-weighted validation CE contrast is primary; test metrics are evaluated only after the final states exist. The same 8,192 training examples form the checkpoint monitor subset.\n\nThe pinned DBpedia revision is 9abd46cf7fc8b4c64290f26993c540b92aa145ac. The archived split arrays reproduce the fixed 476,000/84,000 development partition and 70,000 official test identities. The prospective sampler's explicit permutation/cursor implementation is separate from the historical loader. All seven first-transition fingerprints match; full initial model states match; first-step Adam moments match while parameter states differ. Both RNG streams remain matched at every retained checkpoint.\n\n## Table S18 Complete validation trajectory for the configured-peak pair\n\nThe ten evaluation times were fixed before execution. Accuracy differences below are peak-first minus clean; the primary CE difference has the same sign convention. No favorable horizon is selected.\n\n'''
    def curve_table(pop):
        text='| Step | Clean CE | Peak-first CE | CE difference | Clean accuracy (%) | Peak-first accuracy (%) |\n|--:|--:|--:|--:|--:|--:|\n'
        for row in curves:
            if row['population']!=pop:continue
            text+=f"| {row['step']} | {float(row['clean_logit_ce']):.9f} | {float(row['peak_first_logit_ce']):.9f} | {float(row['ce_peak_first_minus_clean']):+.6g} | {float(row['clean_accuracy_pct']):.4f} | {float(row['peak_first_accuracy_pct']):.4f} |\n"
        return text
    supp+=curve_table('validation')+'\n## Table S19 Complete fixed training-subset trajectory\n\nThe subset contains 8,192 fixed training examples; these are not full-training-set metrics.\n\n'+curve_table('train_monitor_subset')
    supp+='''\n![All online training CE blocks, fixed training-subset checkpoint loss and prespecified validation contrasts. Online losses come from changing model states and dropout; checkpoint subset losses use evaluation mode. The two statistics are distinct. Differences are peak-first minus clean.](../figures/figS9_peak_first_training_and_differences.png){width=98%}\n\nEvery 100-transition online-loss block and the final 99-transition block is included in peak_first_online_training.csv: 500 blocks per arm, 1,000 total. The complete 42 checkpoint/population/arm records are verified from raw logits and labels; peak_first_prediction_recheck.csv gives saved and independently recomputed CE and accuracy. Stable log-sum-exp is applied directly to uncapped logits and averaged over examples. Maximum absolute CE disagreement is 7.99e-15; all accuracies agree exactly. Example identities and labels agree across both arms and all checkpoints in each population.\n\n## Table S20 Per-arm budget and measured cost\n\n| Arm | Positive-LR steps | Zero-LR steps | Examples | Instrumented training (min) | Evaluation (min) |\n|:--|--:|--:|--:|--:|--:|\n'''
    for arm in ('clean','peak_first'):
        c=result['committed_trajectory_cost'][arm]
        supp+=f"| {arm.replace('_','-')} | 49,999 | 0 | 1,599,968 | {c['instrumented_training_seconds']/60:.3f} | {c['evaluation_seconds']/60:.3f} |\n"
    supp+='''\nShared recorded data-loading time is 83.685 seconds and checkpoint time 31.106 seconds. The closed training session totals 4,969.857 seconds (82.831 minutes); these component clocks do not account for every logging/control operation and must not be added to the total again. Training includes finite-state checks and the specified displacement instrumentation. GPU process peaks during the two arm intervals are 513,183,744 and 531,718,656 allocated bytes; both models remain resident. The maximum reported reserved memory is 601,882,624 bytes. These are not independent per-model memory footprints or an uninstrumented overhead comparison.\n\n## Retained state and execution checks for the new pair\n\nThe three returned bundles contain 49 record/source files, 42 prediction files and five paired checkpoint files, with identical cross-bundle file manifests. Every member size and SHA-256 digest was checked. The audit reconstructs all 49,999 sampled batches from the saved sampler contract and checks 99,998 consumed-LR values, all 42 evaluation events and all 1,000 online-loss blocks. Each arm's rates match the prescribed schedule; rates after transition one match each other exactly. Independent cross-platform cosine evaluation differs only in the final floating-point bits, so the schedule audit uses relative tolerance 1e-12 and zero absolute tolerance. No failure, resumed segment, truncated record or zero-LR transition is recorded.\n\nFull paired states are retained at 0, 1, 8,000 and 49,999, with a latest copy. Restricted tensor loading verifies finite parameters and Adam moments, all 55 active parameter-state counters, the prescribed groups/rates, paired RNG identity, sampler permutation/cursor/generator state and stored histories. The final retained state and latest state have identical scientific contents. The two first displacements recomputed from saved tensors match the trace. First-step optimizer moments are identical before later gradients diverge. These state checks are independent of the reported final accuracy.\n\nThe execution environment is Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.1.3, datasets 4.8.5, huggingface-hub 1.33.0 and pyarrow 23.0.1 on one Tesla T4. A short serialization/resume probe with the actual model, loader and device matched uninterrupted execution exactly. The new research run itself completed in one uninterrupted segment. This evidence concerns the new paired runner; it does not retroactively certify the historical execution host, rollback path or cross-device reproducibility.\n\nThe primary final validation CE difference is +4.6952637067e-6. Official test CE is 0.057490380 versus 0.057494741, with accuracy 98.5986% versus 98.6029%. There is no prespecified equivalence margin or between-seed interval for this single pair. Its early lower CE and temporarily lower accuracy do not establish a damage-and-recovery mechanism. The tiny fixed-final contrast limits extrapolation from the LR=1.0 stress result without invalidating the historical observations.\n'''
    supp=supp.replace('included in peak_first_online_training.csv:', 'included in the machine-readable online-training table:')
    supp=supp.replace('peak_first_prediction_recheck.csv gives', 'the raw-logit verification table gives')
    supp=supp.replace('Checkpoint weights are absent from the archive.', 'Historical checkpoint weights are absent from the October archive; new-pair states are reported separately below.')
    assert main.splitlines()[0]==title
    (M/'Research_Manuscript.md').write_text(main,encoding='utf-8');(M/'All_Seed_Results_and_Provenance.md').write_text(supp,encoding='utf-8')
    (M/'Experimental_Evidence_and_Revision_Rationale.md').write_text('# Revision rationale\n\nThe main manuscript and supplement incorporate the completed fixed-budget configured-peak pair. Its small observed final CE/accuracy contrast limits practical extrapolation from the historical LR=1.0 stress test. Both full curves, positive-LR budgets, training costs and single-seed limits are retained. Historical checkpoint selections, the AG News weighting uncertainty, WNUT misses and small pretrained effects are unchanged.\n\nThe new pair has independently checked predictions and retained full states. Historical missing candidate-epoch assets remain a distinct limitation; no historical checkpoint was reselected. The deferred full recovery extension was not performed.\n',encoding='utf-8')


if __name__=='__main__':build()

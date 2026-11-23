"""Apply the 13 November editorial revision without changing scientific inputs."""
from pathlib import Path
import re,json,shutil
import integrate_peak_pair
R=Path(__file__).resolve().parent; M=R/'manuscript'

def build():
    integrate_peak_pair.build()
    main=(M/'Research_Manuscript.md').read_text(encoding='utf-8')
    supp=(M/'All_Seed_Results_and_Provenance.md').read_text(encoding='utf-8')
    title=main.splitlines()[0]
    def section(a,b,text):
        nonlocal main
        i=main.index(a);j=main.index(b,i);main=main[:i]+text.rstrip()+'\n\n'+main[j:]
    section('## Abstract','## 2. Related work',r'''## Abstract

Large early updates can harm neural-network training, but that fact does not distinguish a single incorrect update from continuing scheduler delay or inadequate subsequent learning. We study these distinctions by tracing the learning rates actually consumed by Adam in text-classification experiments. In archived five-seed comparisons, a first rate of 1.0 followed by the correct stable schedule reduces selected test accuracy by 47.19-70.99 percentage points. Separate legacy-schedule contrasts show that repairing the first update can leave later scheduling delay intact. Across all 95 shared-step comparisons, selected endpoints can obscure substantial differences in learning progress: a DBpedia comparison has a 44.29-point validation gap at a shared step but a 5.72-point selected-test gap. Stable scheduling reaches zero at step 50,000, so later epochs provide no further parameter-learning budget. A new matched pair instead uses the configured peak rate of $10^{-4}$ at the first step, with 49,999 positive-rate updates per arm. Its final validation cross-entropy difference is only $4.70\times10^{-6}$. This small contrast limits extrapolation from the extreme fault; it establishes neither equivalence nor recovery. Historical selections and evaluation limitations are retained. The practical lesson is to assert the rate before mutation, record realized parameter displacement, and report shared-step performance alongside the available positive-rate budget. These checks answer different questions; displacement monitoring misses five of 40 tested mismatches.

**Keywords:** learning-rate warmup; Adam; scheduler execution; training diagnostics; validation-based selection; optimization budget.

## 1. Introduction

A training log can display the intended learning rate even when the optimizer has already used a different value. Installing a rate after `optimizer.step()` changes the next stored value; it cannot undo the preceding update. When performance is subsequently poor, the useful question is not only whether the first update was large. We must distinguish the first incorrect transition, any continuing schedule delay, and the learning opportunities available afterward.

We examine these distinctions in scratch-trained Adam classifiers on AG News and DBpedia. Stable versus single-bad-then-stable changes the first consumed rate while retaining the later stable schedule. Within a separate legacy schedule, rescue versus late execution compares first-step repair while delay remains; corrected execution versus rescue examines the remaining delay conditional on repair. These legacy contrasts are exploratory: stopping budgets differ and one factorial condition is missing. They therefore do not identify an additive causal decomposition.

The experiments add three observations to the established fact that oversized early updates can be harmful. First, a repaired first update need not repair subsequent scheduler timing. Second, a selected final classifier can conceal learning delay: in DBpedia medium-4, the rescue-minus-late validation gap at a shared step is 44.293 percentage points, whereas the selected-test gap is 5.716 points. These are different endpoints, so the smaller selected gap does not show that the earlier difference was unimportant. Third, once the stable schedule reaches zero, extra epochs no longer test parameter recovery even though Adam's state can continue to change.

Fault magnitude also matters to the interpretation. The historical first rate of 1.0 is 10,000 times the configured peak. The new single-pair test uses that peak, $10^{-4}$, at the first transition and the correct schedule thereafter. Its final difference is very small. Different protocols prevent using the two cohorts as a controlled dose-response, but together they require a narrower claim than treating every first-step mismatch as consequential damage.

The practical lesson is to separate prevention, diagnosis and outcome measurement. A pre-step rate assertion prevents this known scheduling mismatch. Realized displacement records how much the parameters actually moved; it does not certify task harm. Shared-step validation and positive-rate budgets reveal learning progress that selected endpoints and epoch counts can hide. The contribution is this measured separation in the tested execution paths, not new Adam mathematics, a general recovery mechanism or a superior debugging tool.
''')
    section('## 2. Related work','## 3. Method',r'''## 2. Related work

Warmup and early-update control are established topics. Gotmare et al. (2019) relate warmup to changes in later layers and test early layer freezing. Liu et al. (2020) analyze adaptive-rate variance; Ma and Yarats (2021) study untuned warmup. Kalra and Barkeshli (2024) examine conditioning and target-rate tolerance, while Abuduweili and Liu (2025) change second-moment initialization. Adam's bias correction (Kingma and Ba, 2015) and Adafactor's update clipping (Shazeer and Stern, 2018) already explain why adaptive update magnitudes deserve attention. Our first-step derivation applies these known equations to a recorded execution mismatch.

Kosson, Messmer and Jaggi (2024) analyze update magnitude, direction and representation change. Their Appendix A studies enlarged initial updates in ResNet-20/CIFAR-10, applies the intervention over the first five epochs and extends training to 800 epochs. Early-update harm, classifier experiments and persistence after longer training are therefore not distinctive findings by themselves. Our comparisons distinguish a single incorrect consumed rate from continuing one-step scheduler lag and distinguish selected endpoints from shared-step performance and positive-rate budgets. They do not reproduce Kosson et al.'s mechanism analysis or identify a general recovery law.

TheDeepChecker (Ben Braiek and Khomh, 2023) checks properties before, during and after fitting. Its parameter-dynamics checks already compare weight changes with parameter magnitude and flag unstable or stalled updates. DeepDiagnosis (Wardat et al., 2022) monitors training symptoms and maps them to likely fault locations and suggested fixes, including learning-rate changes. It also uses relative weight/gradient-magnitude heuristics. Thus, monitoring updates or diagnosing poor learning-rate settings is not novel here. We contribute a controlled analysis of particular consumed-rate sequences and their learning histories. Neither tool was run on our cases; no detection, repair or efficiency advantage over either is claimed.

Recent optimizer-stability and warmup theory (Bai et al., 2026; Alimisis, Islamov and Lucchi, 2026) further separates mechanisms from the observations available here. We do not measure curvature or identify representation damage. Prediction calibration metrics (Guo et al., 2017) describe outcomes rather than establish a cause. The empirical scope is a specified execution fault under recorded schedules, stopping rules and budgets.
''')
    key='The initial moments are $m_1=(1-\\beta_1)g$ and $v_1=(1-\\beta_2)g^2$.'
    assert key in main
    end=main.index('\n\n### 3.2',main.index(key))
    main=main[:end]+r'''

Intuitively, Adam first records the same gradient information in both arms; the rate then scales the movement produced from that information. Identical first moments do not mean identical model states. The larger movement changes where the next gradient is evaluated, so later gradients and moments may diverge without any direct first-step change to the moments.

The executable two-step example makes this distinction observable at the optimizer call. Stable execution consumes $S(1)=1.25\times10^{-8}$ and then $S(2)=2.5\times10^{-8}$; the extreme isolated fault consumes 1.0 and then the same $S(2)$. Its first-step moments agree with the control while weights differ. In the legacy repair path, rejecting 1.0 and replaying at $L(1)$ restores the first transition, but the next transition still consumes $L(1)$ rather than $L(2)$. Repairing the first displacement has therefore not removed the continuing lag. The example uses fixed gradients to check execution and state restoration; it does not predict later task gradients or prove a new theorem.
'''+main[end:]
    # Two explanatory figures duplicate the schedule equations and sequence table.
    main=re.sub(r'!\[[^\n]*\]\(\.\./figures/fig[123]_[^)]+\)\{[^}]+\}\n\n','',main)
    # Move long provenance out of the main argument; complete records already in SI.
    section('### 3.9 Execution evidence and reproducibility','### 3.10',r'''### 3.9 Execution evidence and reproducibility

The archived source, configurations, consumed-rate traces, recorded splits and initial fingerprints are checked against per-job outcomes; 180 contain trained test endpoints. Master completion flags match the pre-execution package and were not changed to simulate execution evidence. The supplement gives the source-to-result mappings, log comparisons, hash checks and two-step example. Original candidate-epoch weights, detailed validation outputs and complete execution-host records remain unavailable. The supplied analysis reproduces saved-record results and CPU fixtures; it does not claim an exact rerun of the original training host.

ChatGPT and Codex (OpenAI) assisted with analysis-code preparation and manuscript revision. Reported numerical results derive from retained experimental records; the analysis scripts and verification checks accompany the paper.
''')
    main=main.replace('Figures 4-5 show','Figures 1-2 show')
    main=main.replace('../figures/fig4_shared_step_stable_validation.png','../figures/fig1_computing_stable_progress.pdf')
    main=main.replace('![Legacy validation outcomes at all steps shared within each contrast and all five seeds. Dots show seeds; markers and bars show mean and sample SD. Each contrast retains its own full shared-step intersection.](../figures/fig5_shared_step_legacy_validation.png){width=97%}',
        '![First-step repair while legacy lag remains: DBpedia rescue versus late execution at every step shared by both arms and all five seeds. Points and bars show mean validation accuracy and sample SD. These shared-step outcomes measure learning progress; they are not validation-selected test endpoints or equal-compute causal estimates. Complete legacy contrasts are in the supplement.](../figures/fig2_computing_legacy_progress.pdf){width=97%}')
    main=main.replace('../figures/fig6_peak_first_fixed_budget.png','../figures/fig3_computing_peak_progress.pdf')
    main=main.replace('Supporting condition-level results appear in Supplementary Table S13.\n\nSupporting condition-level results appear in Supplementary Table S14.', 'Supplementary Tables S13-S14 report training performance and positive-rate, zero-rate and replay budgets separately for every condition.')
    # Keep all findings, but remove the procedural description already in SI.
    section('### 7.2 Replay contract and validation scope','## 8.',r'''### 7.2 Replay contract and validation scope

Archived rescue restores weights, Adam moments/counters, parameter groups and clipped gradients, then replays the same gradient. It does not repeat the stochastic forward pass. CPU fixtures check this contract; the supplement describes finite-state checks and their limits. The new configured-peak runner separately passed an actual-device/data resume check and has no rollback intervention. Neither check certifies original-host stochastic replay or the unperformed recovery extension.
''')
    needle='An LR assertion checks a specified execution invariant; displacement monitoring measures parameter motion; task metrics measure downstream utility.'
    main=main.replace(needle,needle+' When the intended schedule is known and the assertion checks every parameter group immediately before mutation, it is sufficient to prevent this particular consumed-rate mismatch. Displacement adds the realized magnitude and the difference between a rejected candidate and its retained replay. It can describe an unusual update even when the rate itself is correct, but these experiments do not establish detection coverage for other fault classes. A small endpoint difference and the WNUT misses show why an alarm is neither a diagnosis of downstream harm nor a guarantee that no mismatch occurred.')
    needle='The configured-peak first step produces a small fixed-final CE difference:'
    main=main.replace(needle,r'''The extreme and configured-peak findings constrain each other's interpretation. In historical DBpedia medium-4 runs, the stable and LR=1.0 conditions have selected-test accuracies of $98.62\pm0.02\%$ and $33.09\pm7.20\%$ (five-seed mean and sample SD; Table 2). In the new fixed-budget pair, the corresponding test accuracies are 98.5986% and 98.6029% (Table 6). The historical result shows a severe deficit under its procedure; the new result shows that a much smaller first-rate error need not produce a comparable final deficit in the tested configuration. The comparison varies seed count, stopping, sampler implementation and environment, so it does not isolate fault magnitude as the sole cause of the cross-study difference.

The configured-peak first step produces a small fixed-final CE difference:''')
    section('## 9. Discussion and limitations','## References',r'''## 9. Discussion and limitations

The experiments support a practical sequence of checks. Verify the rate immediately before the optimizer mutates state; inspect the resulting displacement; then compare validation at shared training steps and report how many positive-rate updates remain. A rate check establishes execution conformity, a displacement trace measures parameter movement, and task performance establishes the observed consequence. Substituting one for another obscures the problem: a corrected exit log can hide the first fault, a selected endpoint can hide learning delay, and extra epochs at zero rate can look like recovery training without moving the parameters.

The contrast between the severe LR=1.0 deficit and the tiny configured-peak endpoint difference restricts the practical claim. A large relative first-update displacement is not by itself evidence of a large final deficit. The new pair's early loss and accuracy contrasts also change differently, so a single crossing or favorable endpoint does not demonstrate damage followed by recovery. The recorded interventions and learning curves are observations; representation damage, curvature changes and a recovery mechanism remain untested explanations. The single new seed supplies neither an equivalence bound nor a general safe-rate threshold.

Historical selection remains part of the measured procedure. Unequal stopping and the missing legacy factorial cell prevent a complete interaction or equal-compute causal estimate. The shared-step comparisons are descriptive and preserve every common logged horizon. Five-seed intervals are unadjusted; small differences do not establish equivalence. AG News's partial validation batch may have changed historical checkpoint rankings, which cannot be recomputed without candidate-level outputs. The supplement preserves that uncertainty, the clipped-probability NLL distinction and the unavailable historical evidence.

Smaller pretrained effects cannot be attributed to pretraining alone because task, architecture, optimizer, schedule and fault magnitude differ. Depth and width also vary together. Displacement monitoring requires calibration and misses all five WNUT cases at its primary threshold; fault prevalence and guard overhead are unmeasured. A direct LR assertion is adequate for preventing the specified schedule mismatch, while displacement remains additional diagnostic evidence. We do not establish general guard superiority, permanent damage or recovery beyond the available positive-rate budget.

## 10. Conclusion

Executed learning rates, shared-step learning progress and positive-rate budgets give different information about a training failure. The recorded comparisons separate a single incorrect first update from continuing scheduler lag, show how selected endpoints can conceal substantial earlier differences, and identify when additional epochs cease to provide parameter-learning updates. The configured-peak pair's small endpoint contrast limits extrapolation from the extreme first-rate fault. The practical result is a reproducible way to interpret these particular execution failures while preserving uncertainty about mechanisms, historical selections and generalization.
''')
    # Complete shared-step legacy curves remain available without duplicating endpoints.
    supp+='''\n\n## Additional schedule and shared-step views\n\n![Stable and legacy schedules from the preserved configuration: widths 128/256, warmup 8,000 and stable zero boundary 50,000. Historical warmup 4,000 belongs to a separate cohort.](../figures/fig2_actual_configuration_schedule.png){width=98%}\n\n![Complete legacy validation comparisons at every common logged step within each contrast and all five seeds. Markers and bars show mean and sample SD. Stopping budgets differ; no interpolation or stopped-run carry-forward is used.](../figures/fig5_shared_step_legacy_validation.png){width=98%}\n'''
    main=main.replace('Consequently, $I(B)=Y(1;1;B)-Y(1;0;B)-Y(0;1;B)+Y(0;0;B)$ is unidentified.', 'Consequently, the interaction\n\n$$I(B)=Y(1;1;B)-Y(1;0;B)-Y(0;1;B)+Y(0;0;B)$$\n\nis unidentified.')
    main=re.sub(r'\nTopollai, K\.[^\n]+\n','\n',main)
    main+='\nBen Braiek, H., and Khomh, F. (2023). Testing Feedforward Neural Networks Training Programs. ACM Transactions on Software Engineering and Methodology 32(4), 1-61. https://doi.org/10.1145/3529318\n\nWardat, M., Cruz, B. D., Le, W., and Rajan, H. (2022). DeepDiagnosis: Automatically Diagnosing Faults and Recommending Actionable Fixes in Deep Learning Programs. Proceedings of the 44th International Conference on Software Engineering, 561-572. https://doi.org/10.1145/3510003.3510071\n'
    main=main.replace('**batch-mean logit CE**','**batch-mean CE** computed from logits')
    main=main.replace('## References', '''## Statements and declarations

### Research support

This research received no financial funding. It used resources provided by Innopolis University.

### Competing interests

The authors have no relevant financial or non-financial competing interests to disclose.

### Author contributions

Tivdzua Lubem Noah developed the implementation, analysed the results, and drafted and revised the manuscript. Bader Rasheed supervised the research, executed experiments, and reviewed and revised the manuscript. Dmitriy S. Vatolin contributed to the research.

### Data and code availability

The code, configurations, retained experimental records, tests, execution logs and editable manuscript sources are available in the [versioned research repository](https://github.com/tivdzualubem/warmup-update-budget-acml2026/tree/computing-submission-2026-11-18-r2) (version `computing-submission-2026-11-18-r2`). The repository preserves the earlier experiment and rebuttal code; `computing/` contains the materials for this article.

Online Resource 1 is the supplementary PDF containing the complete seed-level endpoint results, recorded training trajectories, budgets, provenance and evaluation limitations. Online Resource 2 is the reproducibility package containing the code and retained records used to regenerate the reported tables and figures. The larger prediction and retained-state archives for the configured-peak pair are stored separately and can be requested from the authors; their filenames and SHA-256 hashes are registered in the package. Historical candidate-epoch weights and matching validation predictions were not recovered and cannot be supplied. The public dataset sources are [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [DBpedia](https://huggingface.co/datasets/fancyzhx/dbpedia_14), [WNUT-17](https://huggingface.co/datasets/flaitenberger/wnut_17) and [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10). Exact revisions, split definitions and hashes are retained in the configurations and provenance records.

## References''')
    # The figure captions identify these views; avoid an isolated section heading.
    supp=supp.replace('## Additional schedule and shared-step views\n\n','')
    supplementary_identity = ('\n\nSupplementary information for **' + title.removeprefix('# ') + '**.\n\n'
        'Journal: Computing.\n\n'
        'Authors: Tivdzua Lubem Noah; Bader Rasheed; Dmitriy S. Vatolin.\n\n'
        'Corresponding author: Tivdzua Lubem Noah, Innopolis University, Innopolis, Russia; '
        'l.tivdzua@innopolis.university.\n')
    first, rest = supp.split('\n', 1)
    supplementary_identity += '\nResearch code, retained records and editable sources: [version computing-submission-2026-11-18-r2](https://github.com/tivdzualubem/warmup-update-budget-acml2026/tree/computing-submission-2026-11-18-r2).\n'
    supp = first + supplementary_identity + '\n' + rest
    assert main.splitlines()[0]==title
    (M/'Research_Manuscript.md').write_text(main,encoding='utf-8')
    (M/'All_Seed_Results_and_Provenance.md').write_text(supp,encoding='utf-8')
    abstract=main.split('## Abstract\n\n')[1].split('\n\n**Keywords:')[0]
    (R/'evidence/computing/editorial_checks.json').write_text(json.dumps({'abstract_words':len(abstract.split()),'title_unchanged':True,'main_figure_count':len(re.findall(r'!\[',main)),'new_training':False,'original_tables_retained':True},indent=2))
    print('Computing editorial revision applied; historical tables and title preserved.')

if __name__=='__main__':build()

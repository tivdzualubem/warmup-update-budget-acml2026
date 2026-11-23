"""Apply venue presentation only after the preserved scientific manuscript build."""
from pathlib import Path
import csv
from check_final_clarifications import run
R=Path(__file__).resolve().parent
def build():
    verified=run()
    p=R/'manuscript/Research_Manuscript.md'
    s=p.read_text(encoding='utf8')
    s=s.replace('## Abstract\n\nLarge early', '## Abstract\n\n**Purpose** Large early')
    s=s.replace(' We study these distinctions', '\n\n**Methods** We study these distinctions')
    s=s.replace(' In archived five-seed comparisons', '\n\n**Results** In archived five-seed comparisons')
    s=s.replace(' This small contrast limits extrapolation', '\n\n**Conclusion** This small contrast limits extrapolation')
    s=s.replace('### Research support', '### Funding')
    s=s.replace('## Statements and declarations','## Statements and Declarations')
    s=s.replace('`computing/` contains the materials for this article.', '`computing/` preserves the scientific code and records used in this article. The accompanying Online Resource 2 also includes the current journal-formatted manuscript sources.')
    s=s.replace('a 5.72-point selected-test gap', 'a 5.82-point selected-checkpoint validation gap')
    s=s.replace('whereas the selected-test gap is 5.716 points. These are different endpoints, so the smaller selected gap does not show that the earlier difference was unimportant.', 'whereas the validation-accuracy gap at the historically selected checkpoints is 5.823 points. The corresponding selected-test gap is 5.716 points. The same-metric validation comparison shows that selected endpoints can conceal earlier learning differences; it does not establish recovery, because checkpoint steps and stopping budgets differ.')
    anchor='### 3.5 Evaluation definitions and archived selection caveat'
    rates=r'''The pretrained first-step fault uses the configured peak, not the scratch rate of 1.0: $2\times10^{-5}$ for AG News, DBpedia and WNUT-17, and $5\times10^{-5}$ for CIFAR-10. The respective warmup lengths are 957, 4,463, 64 and 423 updates. Correct-order first consumed rates are the peak divided by these lengths: approximately $2.089864\times10^{-8}$, $4.481291\times10^{-9}$, $3.125\times10^{-7}$ and $1.182033\times10^{-7}$. These values agree across all five seeds in each task. The supplement reports the preserved manifests and first-transition traces.

'''
    assert anchor in s;s=s.replace(anchor,rates+anchor)
    anchor='### 5.2 Budgets and training performance'
    clar='For DBpedia medium-4, comparing validation accuracy with validation accuracy gives a rescue-minus-late gap of 44.292857 percentage points at step 44,625 and 5.822619 points at the historically selected checkpoints, averaged over the same five paired seeds. The supplement lists each selected epoch and paired value. This comparison retains the original validation-loss selections; it does not reselect by accuracy or recompute predictions. The large rescue-minus-late gap concerns first-step repair while both arms retain later lag. The separate corrected-minus-rescue contrast is small and uncertain, conditional on first-step repair and the recorded stopping rules.\n\n'
    assert anchor in s;s=s.replace(anchor,clar+anchor)
    anchor='An LR assertion checks a specified execution invariant;'
    wnut='WNUT-17 reaches its peak at update 64, within the safe calibration window of updates 1-100. The reference window therefore includes peak-rate safe updates, unlike the other three pretrained tasks. This can raise the reference displacement scale relative to the first-step fault and is a possible explanation for the five misses. The records do not isolate that explanation causally; the misses should not be attributed simply to architecture or pretraining.\n\n'
    assert anchor in s;s=s.replace(anchor,wnut+anchor)
    p.write_text(s,encoding='utf8')
    p=R/'manuscript/All_Seed_Results_and_Provenance.md'
    s=p.read_text(encoding='utf8').replace('Journal: Computing.', 'Journal: SN Computer Science.')
    s=s.replace('Research code, retained records and editable sources:', 'Preserved scientific code, retained records and earlier editable sources:')
    s+='\n\n## Same-metric selected-checkpoint validation comparison\n\nDBpedia medium-4 rescue versus late execution: both retain later scheduler lag. Accuracy below is validation accuracy at the historically selected minimum-validation-loss epoch, not a newly selected accuracy maximum. The shared-step mean gap is 44.292857 percentage points at step 44,625; the selected-checkpoint mean gap is 5.822619 points. Unequal selected steps prevent interpreting the endpoint comparison as matched-compute recovery. Exact records and source paths are in selected_validation_endpoint_pairs.csv.\n\n| Seed | Late selected epoch | Rescue selected epoch | Late validation (%) | Rescue validation (%) | Difference (pp) |\n|--:|--:|--:|--:|--:|--:|\n'
    for r in csv.DictReader((R/'tables/selected_validation_endpoint_pairs.csv').open()):
        s+=f"| {r['seed']} | {r['late_selected_epoch']} | {r['rescue_selected_epoch']} | {100*float(r['late_selected_validation_accuracy']):.6f} | {100*float(r['rescue_selected_validation_accuracy']):.6f} | {float(r['selected_gap_pp']):.6f} |\n"
    s+='\n## Pretrained first rates and warmup lengths\n\nThe following values are identical across all five seeds within each task. The mismatch first rate is the configured peak; the safe first rate is peak divided by warmup length. All 20 safe/fault pairs are checked directly against archived manifests and first-transition traces in pretrained_first_rates_and_warmup.csv.\n\n| Task | Total planned updates | Warmup updates | Fault first LR / peak | Safe first LR |\n|:--|--:|--:|--:|--:|\n'
    for r in csv.DictReader((R/'tables/pretrained_first_rates_and_warmup.csv').open()):
        if r['seed']=='42':s+=f"| {r['dataset']} | {r['total_steps']} | {r['warmup_steps']} | {float(r['peak_lr']):.6g} | {float(r['safe_first_consumed_lr']):.9g} |\n"
    s+='\nThe safe WNUT-17 trace reaches the peak at step 64 in every seed, inside the 1-100 calibration window. This may explain why its first-step fault does not exceed the primary displacement threshold, but no causal ablation tests that explanation. All five WNUT misses and the primary 35/40 count remain unchanged. Architecture and pretraining alone do not explain these records.\n'
    p.write_text(s,encoding='utf8')
    from link_supplementary_materials import build as add_links
    add_links()
if __name__=='__main__':build()

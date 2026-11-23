"""Rebuild all shared logged-step comparisons directly from archived histories.

No interpolation, stopped-run carry-forward, checkpoint reselection or training.
"""
from pathlib import Path
import csv, collections, statistics, math, json, zipfile, hashlib
from scipy.stats import t, binomtest

ROOT=Path(__file__).resolve().parent
SEEDS=[42,123,456,789,2026]
PAIRS=[('stable_minus_single_bad','stable_with_guard','single_bad_first_step_then_stable'),
       ('rescue_minus_late','buggy_rescue_guard','buggy_no_guard'),
       ('corrected_minus_rescue','fixed_order_with_guard','buggy_rescue_guard')]

def validation_layout(result):
    """Describe the archived batch weighting for this validation population."""
    n=int(result['dataset_manifest']['split']['validation_count']); batch=32
    full,remainder=divmod(n,batch); batches=full+bool(remainder)
    return dict(validation_examples=n,validation_batch_size=batch,
        validation_full_batches=full,validation_batches=batches,
        validation_final_batch_examples=remainder or batch,
        validation_has_partial_batch=bool(remainder),
        validation_batch_weighting_matches_example_weighting=not bool(remainder),
        final_batch_example_influence_ratio=n/(batches*remainder) if remainder else 1.,
        partial_batch_selection_effect_status='unknown_without_candidate_epoch_assets' if remainder else 'not_applicable_full_validation_batches')

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build(outdir=None):
    out=Path(outdir) if outdir else ROOT/'tables'
    out.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip'
    traces={};members={};failure_ledger=[];metric_labels=[];layouts={}
    with zipfile.ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            if not member.endswith('/result.json') or '/results/' not in member:
                continue
            r=json.loads(z.read(member))
            if not r.get('job_complete') or 'condition' not in r:
                continue
            family='scratch' if '/results/scratch/' in member else 'pretrained'
            model=r.get('model','deit-tiny' if r['dataset']=='cifar10' else 'distilbert')
            prefix=member.rsplit('/',1)[0]+'/'
            history=json.loads(z.read(prefix+'history.json'))
            k=(r['dataset'],model,r['condition'],int(r['seed']))
            if family=='scratch':
                steps=[int(h['global_step']) for h in history]
                assert len(steps)==len(set(steps)),('Duplicate history step',member)
                traces[k]={int(h['global_step']):h for h in history}
                members[k]=prefix+'history.json'
                layout=validation_layout(r)
                if r['dataset'] in layouts:assert layouts[r['dataset']]==layout
                layouts[r['dataset']]=layout
                metric_labels.append(dict(dataset=r['dataset'],model=model,seed=r['seed'],condition=r['condition'],
                    archived_loss_label='mean_of_batch_mean_logit_cross_entropy',
                    archived_nll_label='clipped_probability_NLL_floor_1e-12',
                    selection_rule='minimum_archived_validation_batch_mean_CE',
                    historical_selection_preserved=True,
                    checkpoint_weights_available=False,source=member,**layout))
            reason=r.get('terminal_reason','')
            diagnostic=r['condition'] in ('buggy_failfast_guard','update_guard_failfast','executed_lr_assertion')
            failure_ledger.append(dict(family=family,dataset=r['dataset'],model=model,seed=r['seed'],condition=r['condition'],
                terminal_reason=reason,diagnostic_condition=diagnostic,
                trained_test_endpoint=bool(r.get('test_metrics')),
                recorded_numerical_failure=('nonfinite' in reason.lower() or 'nan' in reason.lower()),
                selected_checkpoint_replaced=False,source=member))
    detail=[];summary=[];coverage=[]
    settings=sorted({k[:2] for k in traces})
    for ds,model in settings:
        for label,a,b in PAIRS:
            maps=[traces[(ds,model,c,s)] for s in SEEDS for c in (a,b)]
            common=sorted(set.intersection(*(set(m) for m in maps)))
            union=sorted(set.union(*(set(m) for m in maps)))
            coverage.append(dict(dataset=ds,model=model,contrast=label,arm_a=a,arm_b=b,n_seeds=5,
                shared_logged_steps=';'.join(map(str,common)),n_shared_steps=len(common),
                excluded_nonshared_steps=';'.join(map(str,sorted(set(union)-set(common)))),
                interpolation=False,carry_forward=False,horizon_selection='all_shared_logged_steps'))
            for step in common:
                block=[]
                for seed in SEEDS:
                    ah=traces[ds,model,a,seed][step];bh=traces[ds,model,b,seed][step]
                    aa=float(ah['validation']['accuracy']);ba=float(bh['validation']['accuracy'])
                    al=float(ah['validation']['loss']);bl=float(bh['validation']['loss'])
                    row=dict(dataset=ds,model=model,contrast=label,global_step=step,seed=seed,arm_a=a,arm_b=b,
                        arm_a_validation_accuracy=aa,arm_b_validation_accuracy=ba,accuracy_difference_pp=100*(aa-ba),
                        arm_a_archived_validation_loss=al,arm_b_archived_validation_loss=bl,
                        archived_validation_loss_difference=al-bl,
                        loss_definition='archived_mean_of_batch_mean_logit_cross_entropy',
                        validation_batch_weighting_matches_example_weighting=layouts[ds]['validation_batch_weighting_matches_example_weighting'],
                        arm_a_online_train_loss=ah.get('train_loss',''),arm_b_online_train_loss=bh.get('train_loss',''),
                        arm_a_online_train_accuracy=ah.get('train_accuracy',''),arm_b_online_train_accuracy=bh.get('train_accuracy',''),
                        analysis_status='exploratory_shared_step_validation',checkpoint_reselection=False,
                        arm_a_source=members[ds,model,a,seed],arm_b_source=members[ds,model,b,seed])
                    block.append(row);detail.append(row)
                diffs=[r['accuracy_difference_pp'] for r in block]
                mean=statistics.mean(diffs);sd=statistics.stdev(diffs);radius=t.ppf(.975,4)*sd/math.sqrt(5)
                pos=sum(v>0 for v in diffs);neg=sum(v<0 for v in diffs)
                summary.append(dict(dataset=ds,model=model,contrast=label,global_step=step,n=5,arm_a=a,arm_b=b,
                    arm_a_validation_accuracy_mean=statistics.mean(r['arm_a_validation_accuracy'] for r in block),
                    arm_a_validation_accuracy_sample_sd=statistics.stdev(r['arm_a_validation_accuracy'] for r in block),
                    arm_b_validation_accuracy_mean=statistics.mean(r['arm_b_validation_accuracy'] for r in block),
                    arm_b_validation_accuracy_sample_sd=statistics.stdev(r['arm_b_validation_accuracy'] for r in block),
                    mean_accuracy_difference_pp=mean,sample_sd_difference_pp=sd,
                    descriptive_ci95_lower_pp=mean-radius,descriptive_ci95_upper_pp=mean+radius,
                    exact_sign_p=binomtest(pos,pos+neg,.5).pvalue if pos+neg else 1.,
                    analysis_status='exploratory_unadjusted_all_shared_logged_steps',
                    stable_family_zero_lr_steps=max(0,step-49999) if label==PAIRS[0][0] else 0))
    assert len(detail)==475 and len(summary)==95
    assert sum(r['trained_test_endpoint'] for r in failure_ledger)==180
    write(out/'matched_step_per_seed.csv',detail)
    write(out/'matched_step_summary.csv',summary)
    write(out/'matched_step_coverage.csv',coverage)
    write(out/'archived_failure_and_endpoint_ledger.csv',failure_ledger)
    write(out/'archived_evaluator_provenance.csv',metric_labels)
    write(out/'validation_batch_layout.csv',[dict(dataset=ds,**layout) for ds,layout in sorted(layouts.items())])
    report=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        seed_set=SEEDS,contrast_settings=len(coverage),summary_rows=len(summary),paired_seed_rows=len(detail),
        scientific_records=len(failure_ledger),trained_test_endpoints=180,
        recorded_numerical_failures=sum(r['recorded_numerical_failure'] for r in failure_ledger),
        interpolation=False,carry_forward=False,reselected_checkpoints=False,
        loss_recomputed_from_predictions=False,model_training=False)
    (out/'matched_step_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return detail,summary,coverage

if __name__=='__main__':build()

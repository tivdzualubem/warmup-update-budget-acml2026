#!/usr/bin/env python3
"""Deterministic analysis of the October ACML return. No model training.
Usage: python reanalyse.py --archive ACML_RESULTS_FOR_STUDENT.zip --outdir ./tables
"""
from __future__ import annotations
import argparse, json, zipfile, csv, math, hashlib, collections, statistics
from pathlib import Path
import numpy as np
from scipy import stats

SEEDS=[42,123,456,789,2026]
SCRATCH=['stable_with_guard','fixed_order_with_guard','buggy_no_guard','buggy_failfast_guard','buggy_rescue_guard','single_bad_first_step_then_stable']
PRETRAINED=['safe_correct_order','late_schedule_no_guard','single_bad_then_correct','update_guard_failfast','update_guard_rescue','executed_lr_assertion']
PAIRS=[('stable_with_guard','single_bad_first_step_then_stable','isolated_first_step: stable - bad'),('fixed_order_with_guard','buggy_no_guard','scheduler_order: corrected - late'),('buggy_rescue_guard','fixed_order_with_guard','rescue_vs_corrected: rescue - corrected'),('buggy_rescue_guard','buggy_no_guard','exploratory_first_step_repair: rescue - late'),('fixed_order_with_guard','buggy_rescue_guard','exploratory_remaining_delay: corrected - rescue')]

def writetable(path,rows,fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    if fields is None: fields=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def safe_float(v):
    return float(v) if isinstance(v,(float,int)) and math.isfinite(v) else ''

def load(z,name):return json.loads(z.read(name))

def mean_sd_ci(values):
    a=np.asarray(values,dtype=float)
    avg=float(a.mean());sd=float(a.std(ddof=1)) if len(a)>1 else 0.0
    radius=float(stats.t.ppf(.975,len(a)-1)*sd/math.sqrt(len(a))) if len(a)>1 else 0.0
    return avg,sd,avg-radius,avg+radius

def run(archive,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip();assert bad is None,f'archive corrupted: {bad}'
        allnames=set(z.namelist());results=[]; histories=[]; traces={}; manifests={}; missing=[]; checkpoints=[]; anchors=collections.defaultdict(dict)
        for name in sorted(n for n in allnames if n.endswith('/result.json') and '/results/' in n and '/seed' in n):
            data=load(z,name)
            if not isinstance(data,dict) or not data.get('job_complete'):continue
            if 'dataset' not in data or 'condition' not in data:continue
            if data.get('pilot_mode'):raise AssertionError(f'pilot result in scientific archive: {name}')
            cond=data['condition'];seed=int(data['seed']);dataset=data['dataset'];model=data.get('model','distilbert' if dataset!='cifar10' else 'deit-tiny')
            family='scratch' if cond in SCRATCH else 'pretrained'
            key=(family,dataset,model,seed,cond)
            prefix=name.rsplit('/',1)[0]+'/';hist_name=prefix+'history.json';trace_name=prefix+'trace.json';manifest_name=prefix+'job_manifest.json'
            h=load(z,hist_name) if hist_name in allnames else []
            t=load(z,trace_name) if trace_name in allnames else []
            j=load(z,manifest_name) if manifest_name in allnames else {}
            best_i=None;actual_best=None;report_best=data.get('best_validation_loss') if family=='scratch' else data.get('best_validation_primary')
            if h:
                if family=='scratch':
                    best_i=min(range(len(h)),key=lambda i:h[i]['validation']['loss']);actual_best=h[best_i]['validation']['loss']
                else:
                    best_i=max(range(len(h)),key=lambda i:h[i]['validation_primary']);actual_best=h[best_i]['validation_primary']
            best_epoch=int(h[best_i]['epoch']) if best_i is not None else None
            checkpoint_valid=(report_best is None and actual_best is None) or (report_best is not None and actual_best is not None and math.isclose(report_best,actual_best,rel_tol=1e-8,abs_tol=1e-8))
            if family=='pretrained' and best_i is not None:checkpoint_valid=checkpoint_valid and int(data.get('best_epoch',-1))==best_epoch
            ts=data.get('test_metrics') or {}
            metric=('macro_f1' if family=='pretrained' and dataset in ('ag_news','dbpedia_14') else 'entity_f1' if dataset=='wnut_17' else 'accuracy')
            frac=data.get('train_fraction')
            if frac is not None and float(frac)!=1.:raise AssertionError('scientific training fraction should be 1')
            last=int(data.get('global_steps') or 0)
            is_stable_schedule=family=='scratch' and cond in ('stable_with_guard','single_bad_first_step_then_stable')
            row=dict(family=family,dataset=dataset,model=model,seed=seed,condition=cond,metric=metric,test_value=safe_float(ts.get(metric)),test_accuracy=safe_float(ts.get('accuracy')),test_macro_f1=safe_float(ts.get('macro_f1')),checkpoint_selection_metric=('validation_loss' if family=='scratch' else data.get('primary_metric')),train_accuracy=safe_float((data.get('train_metrics') or {}).get('accuracy')),train_loss=safe_float((data.get('train_metrics') or {}).get('loss')),test_entity_f1=safe_float(ts.get('entity_f1')),test_ece=safe_float(ts.get('ece')),test_nll=safe_float(ts.get('nll')),test_brier=safe_float(ts.get('brier')),threshold=safe_float(data.get('threshold')),terminal_reason=data.get('terminal_reason'),global_steps=last,best_validation_epoch=best_epoch or '',best_validation_objective=safe_float(actual_best),reported_best_validation_objective=safe_float(report_best),selection_metadata_consistent=checkpoint_valid,checkpoint_bytes_in_archive=False,positive_lr_steps=(min(last,49999) if is_stable_schedule else ''),post_zero_steps=(max(0,last-49999) if is_stable_schedule else ''),result_path=name)
            results.append(row)
            if not checkpoint_valid:missing.append('checkpoint selection mismatch: '+name)
            if data.get('anchor_evidence'):anchors[(family,dataset,model,seed)][cond]=data['anchor_evidence']
            if data.get('best_state_file') and prefix+data['best_state_file'] in allnames:row['checkpoint_bytes_in_archive']=True
            for r in h:
                step=int(r['global_step']);is_post_zero=is_stable_schedule and step>=50000
                histories.append(dict(family=family,dataset=dataset,model=model,seed=seed,condition=cond,epoch=r['epoch'],global_step=step,train_loss=r.get('train_loss'),train_accuracy=r.get('train_accuracy'),validation_macro_f1=r.get('validation',{}).get('macro_f1'),validation_primary=r.get('validation_primary'),validation_loss=r.get('validation',{}).get('loss'),validation_accuracy=r.get('validation',{}).get('accuracy'),validation_entity_f1=r.get('validation',{}).get('entity_f1'),stable_cosine_first_zero_step=50000 if is_stable_schedule else '',after_stable_zero=is_post_zero,post_zero_steps=max(0,step-49999) if is_stable_schedule else '',best_validation_epoch=best_epoch or '',selected_by_validation=(int(r['epoch'])==best_epoch)))
            if t:traces[key]=t
            if j:manifests[key]=j
        wr=writetable
        wr(out/'all_condition_results.csv',results)
        wr(out/'optimizer_step_histories.csv',histories)
        group=collections.defaultdict(dict)
        for row in results:group[(row['dataset'],row['model'],row['seed'])][row['condition']]=row
        paired=[];summary=[]
        for (ds,mod,seed),vals in sorted(group.items()):
            for control,treatment,label in PAIRS:
                if control not in vals or treatment not in vals:continue
                a,b=vals[control],vals[treatment]
                if a['test_accuracy']=='' or b['test_accuracy']=='':continue
                paired.append(dict(dataset=ds,model=mod,seed=seed,contrast=label,control=control,treatment=treatment,control_accuracy=float(a['test_accuracy']),treatment_accuracy=float(b['test_accuracy']),accuracy_difference_pp=100*(float(a['test_accuracy'])-float(b['test_accuracy'])),analysis_status=('exploratory' if label.startswith(('exploratory','rescue_vs_corrected')) else 'existing contrast'),control_source_path=a['result_path'],treatment_source_path=b['result_path'],control_optimizer_steps=a['global_steps'],treatment_optimizer_steps=b['global_steps']))
        wr(out/'paired_per_seed.csv',paired)
        bypair=collections.defaultdict(list)
        for p in paired:bypair[p['dataset'],p['model'],p['contrast']].append(p['accuracy_difference_pp'])
        for (ds,mod,label),dif in sorted(bypair.items()):
            m,s,lo,hi=mean_sd_ci(dif);pos=sum(x>0 for x in dif);neg=sum(x<0 for x in dif);neq=pos+neg
            signp=float(stats.binomtest(pos,neq,0.5).pvalue) if neq else 1.
            summary.append(dict(dataset=ds,model=mod,contrast=label,n=len(dif),mean_difference_pp=m,sd_pp=s,ci95_lower_pp=lo,ci95_upper_pp=hi,positive_seeds=pos,negative_seeds=neg,two_sided_exact_sign_p=signp,analysis_status=('exploratory' if label.startswith(('exploratory','rescue_vs_corrected')) else 'existing contrast')))
        wr(out/'paired_contrasts_corrected.csv',summary)
        # report meanВ±SD per experimental condition, not pooled across protocols
        agg=[];gr=collections.defaultdict(list)
        for r in results:
            if r['test_value']!='':gr[r['family'],r['dataset'],r['model'],r['condition'],r['metric']].append(float(r['test_value']))
        for key,values in sorted(gr.items()):
            m,s,lo,hi=mean_sd_ci(values)
            agg.append(dict(family=key[0],dataset=key[1],model=key[2],condition=key[3],metric=key[4],n=len(values),mean=m,sd=s,ci95_lo=lo,ci95_hi=hi))
        wr(out/'condition_mean_sd.csv',agg)
        budgets=[]
        for r in results:
            if r['family']!='scratch':continue
            stable=r['condition'] in ('stable_with_guard','single_bad_first_step_then_stable')
            step=r['global_steps'];ep=r['best_validation_epoch']
            stepper=int(step/max([x['epoch'] for x in histories if x['dataset']==r['dataset'] and x['model']==r['model'] and x['seed']==r['seed'] and x['condition']==r['condition']] or [1])) if step else 0
            budget=dict(dataset=r['dataset'],model=r['model'],seed=r['seed'],condition=r['condition'],schedule_family='stable_cosine' if stable else 'legacy_inverse_sqrt',termination_steps=step,best_validation_epoch=ep,steps_per_epoch=stepper,first_zero_lr_step=50000 if stable else '',positive_lr_steps_before_stop=min(step,49999) if stable else step,optimizer_steps_after_zero=max(0,step-49999) if stable else 0,best_validation_step=next((x['global_step'] for x in histories if x['dataset']==r['dataset'] and x['model']==r['model'] and x['seed']==r['seed'] and x['condition']==r['condition'] and x['selected_by_validation']), ''),terminal_reason=r['terminal_reason'])
            budgets.append(budget)
        wr(out/'scratch_optimization_budgets.csv',budgets)
        # Intrinsic LR behavior & guard outcome from primary first step records
        guard=[]
        for k,t in sorted(traces.items()):
            fam,ds,mod,seed,cond=k
            if not t:continue
            x=t[0]
            if cond in ('buggy_failfast_guard','buggy_rescue_guard','update_guard_failfast','update_guard_rescue'):
                guard.append(dict(family=fam,dataset=ds,model=mod,seed=seed,condition=cond,threshold=group.get((ds,mod,seed),{}).get(cond,{}).get('threshold',''),candidate_update_ratio=x.get('candidate_update_ratio'),committed_update_ratio=x.get('committed_update_ratio'),guard_triggered=x.get('guard_triggered'),candidate_lr_consumed=x.get('candidate_lr_consumed'),committed_lr_consumed=x.get('committed_lr_consumed'),target_lr=x.get('intended_lr')))
        wr(out/'guard_first_transition.csv',guard)
        # Recompute every sensitivity row directly from archived safe and faulty traces.
        from rebuild_sensitivity import regenerate
        regenerate(z, results, traces, out)
        # Twenty separately recorded pretrained first-transition severity diagnostics.
        severity=[]
        for n in sorted(allnames):
            if '/results/pretrained_severity/' not in n or not n.endswith('/result.json'):continue
            j=load(z,n)
            trials=[x for x in j.get('trials',[]) if x.get('label')=='natural_late_schedule_peak']
            if len(trials)!=1:raise RuntimeError('severity record lacks exactly one natural candidate: '+n)
            severity.append(dict(dataset=j['dataset'],seed=j['seed'],intended_lr=j['intended_step1_lr'],peak_lr=j['peak_lr'],lr_multiplier=j['natural_late_multiplier'],realized_ratio_multiplier=trials[0]['update_ratio_relative_to_1x'],source_path=n))
        if len(severity)!=20:missing.append(f'expected 20 severity records, found {len(severity)}')
        wr(out/'pretrained_severity_per_seed.csv',severity)
        sg=collections.defaultdict(list)
        for row in severity:sg[row['dataset']].append(row)
        sevsum=[dict(dataset=ds,n=len(vals),mean_lr_multiplier=float(np.mean([r['lr_multiplier'] for r in vals])),mean_realized_ratio_multiplier=float(np.mean([r['realized_ratio_multiplier'] for r in vals]))) for ds,vals in sorted(sg.items())]
        wr(out/'pretrained_severity_summary_recomputed.csv',sevsum)
        # archive inventory and fingerprint audit
        expected_scratch=2*2*5*6;expected_pretrained=4*5*6
        counts=collections.Counter(r['family'] for r in results)
        if counts['scratch'] != expected_scratch or counts['pretrained']!=expected_pretrained:missing.append(f'expected 120 scratch and 120 pretrained, actual {counts}')
        bad_anchor=[]
        for group_key,conds in anchors.items():
            if len(conds)<2:continue
            ref=next(iter(conds.values()))
            for cond,a in conds.items():
                if a!=ref:bad_anchor.append([group_key,cond])
        # No recomputation of predictions can be performed without checkpoint weights.
        checkpoint_n=sum(int(r['checkpoint_bytes_in_archive']) for r in results)
        audit={'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),'archive_member_count':len(allnames),'scientific_jobs_by_family':dict(counts),'distinct_anchor_groups':len(anchors),'anchor_hash_mismatch':bad_anchor,'selection_metadata_failures':missing,'model_checkpoint_files_in_archive':checkpoint_n,'cannot_independently_rerun_test_evaluation_without_checkpoint_bytes':True,'seed_set':SEEDS,'note':'best-validation metadata compared with per-epoch validation histories; actual test recalculation requires original model checkpoint bytes; historical notebook outcomes are supplemented by archived run sources, raw AG News traces and DBpedia trace-audit exports in evidence/historical_warmup'}
        (out/'audit_summary.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps(audit,indent=2))
        return results,histories,paired,summary,budgets

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',default='/mnt/data/ACML_RESULTS_FOR_STUDENT.zip');p.add_argument('--outdir',default='/mnt/data/acml_corrected_analysis/tables');a=p.parse_args();run(a.archive,a.outdir)

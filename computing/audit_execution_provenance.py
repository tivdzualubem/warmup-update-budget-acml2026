"""Audit returned execution records without importing or running training code.

Inputs are immutable source/result ZIPs and the original partial console log.
All comparisons are independently derived; false design flags are never edited.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import re
import zipfile

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence/execution_provenance'
SEEDS = (42, 123, 456, 789, 2026)
SCRATCH = ('stable_with_guard', 'single_bad_first_step_then_stable',
           'fixed_order_with_guard', 'buggy_no_guard', 'buggy_failfast_guard', 'buggy_rescue_guard')
PRETRAINED = ('safe_correct_order', 'single_bad_then_correct', 'late_schedule_no_guard',
              'update_guard_failfast', 'update_guard_rescue', 'executed_lr_assertion')
FOLDERS = {'ag_news':'AG_News','dbpedia_14':'DBPedia','wnut_17':'WNUT_17','cifar10':'CIFAR_10'}
ANCHORS = ('model_state','optimizer_non_lr','first_batch','rng_before_forward','logits','loss','clipped_gradient')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')


def write_csv(name, rows):
    with (ROOT/'tables'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stable(t):
    if t <= 8000:
        return 1e-4*t/8000
    progress = min(1.0,(t-8000)/(50000-8000))
    return 1e-4*.5*(1+math.cos(math.pi*progress))


def legacy(t, model):
    width = {'medium-4':128,'large-6':256}[model]
    return width**-.5*min(t**-.5,t*8000**-1.5)


def expected_rates(kind, r, manifest, t, triggered):
    c = r['condition']
    if kind=='scratch':
        s = (lambda n:stable(n)) if c in SCRATCH[:2] else (lambda n:legacy(n,r['model']))
        initial = 1e-4 if c=='stable_with_guard' else 1.
        safe = c in ('stable_with_guard','fixed_order_with_guard') or (c==SCRATCH[1] and t>1)
        entry = initial if t==1 else (1. if c==SCRATCH[1] and t==2 else s(t-1))
        candidate = s(t) if safe else entry
        committed = s(t) if c=='buggy_rescue_guard' and triggered else candidate
        exit_lr = 1. if c==SCRATCH[1] and t==1 else s(t)
    else:
        peak,w,total=manifest['peak_lr'],manifest['warmup_steps'],manifest['total_optimizer_steps']
        def s(n):
            return peak*n/w if n<=w else peak*max(0.,1.-(n-w)/(total-w))
        entry = peak if t==1 else s(t-1)
        safe = c=='safe_correct_order' or (c=='single_bad_then_correct' and t>1)
        candidate = s(t) if safe else entry
        committed = s(t) if c=='update_guard_rescue' and triggered else candidate
        exit_lr = s(t)
    return dict(entry_lr=entry,intended_lr=s(t),candidate_lr_consumed=candidate,
                committed_lr_consumed=committed,actual_exit_lr=exit_lr)


def run():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    failures=[]
    def check(ok, item, detail=''):
        if not ok:
            failures.append({'check':item,'detail':detail})
        return bool(ok)
    def close(a,b):
        return isinstance(a,(int,float)) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-18)
    archive=ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip'
    original=EVIDENCE/'original_source_launcher.zip'
    log_path=ROOT/'evidence/execution_logs/AG_News.log'
    z=zipfile.ZipFile(archive);o=zipfile.ZipFile(original)
    names=set(z.namelist()); original_names=set(o.namelist()); prefix='ACML_ALL_DATASETS_ONE_COMMAND/'
    check(z.testzip() is None,'returned archive CRC')
    check(o.testzip() is None,'source launcher CRC')
    original_sha=sha(original.read_bytes())
    check(original_sha=='a007bb397c9cd361c41a430a7f830eaadf9b9604c829e81a69a21695702d984b','source launcher identity')
    check(sha(archive.read_bytes())=='b05f9d4545b9377429ddaf3c9d5d778b5125d85fb9921f32768b7a478c6944a7','returned archive identity')
    metadata=[]
    for n in sorted(names):
        if '/results/' in n: continue
        original_name=prefix+n
        equal=original_name in original_names and z.read(n)==o.read(original_name)
        check(equal,'returned design file equals original launcher',n)
        d=json.loads(z.read(n))
        flags={k:v for k,v in d.items() if k in ('full_gpu_scientific_runs_completed','scientific_training_run_in_this_package')}
        metadata.append({'returned_member':n,'original_member':original_name,'sha256':sha(z.read(n)),
                         'byte_identical':equal,'historical_completion_fields':flags})
    source_map=[]
    source=ROOT/'source/warmup-update-budget-acml2026'
    for n in sorted(original_names):
        if '/publication_expansion/' not in n or not n.endswith(('.py','.json')): continue
        rel='publication_expansion/'+n.split('/publication_expansion/',1)[1]
        p=source/rel
        equal=p.is_file() and p.read_bytes()==o.read(n)
        check(equal,'local frozen code/config equals original launcher',n)
        source_map.append({'original_member':n,'local_path':p.relative_to(ROOT).as_posix(),
                           'sha256':sha(o.read(n)),'byte_identical':equal})
    hyper=json.loads(z.read('datasets/AG_News/publication_expansion/configs/hyperparameters_v1.json'))
    check(hyper['scratch_controlled']['stable_schedule']=={'base_lr':1e-4,'warmup_steps':8000,'total_steps':50000,'post_warmup':'cosine'},'scratch stable config')
    check(hyper['scratch_controlled']['legacy_schedule']['warmup_steps']==8000,'scratch legacy warmup')
    expected={(kind,ds,model,s,c) for kind,datasets,models,conditions in
              [('scratch',('ag_news','dbpedia_14'),('medium-4','large-6'),SCRATCH),
               ('pretrained',tuple(FOLDERS),('',),PRETRAINED)]
              for ds in datasets for model in models for s in SEEDS for c in conditions}
    records={}; per_run=[]; anchors=defaultdict(list); splits=defaultdict(list)
    trace_rows=trace_fields=0; history_rows=0; terminal=Counter(); severity=[]
    for n in sorted(names):
        if '/results/' not in n or not n.endswith('/result.json'): continue
        r=json.loads(z.read(n));kind=n.split('/results/',1)[1].split('/')[0]
        if kind=='pretrained_severity':
            severity.append({'member':n,'sha256':sha(z.read(n)),'dataset':r['dataset'],'seed':r['seed'],'trials':len(r['trials'])})
            continue
        key=(kind,r['dataset'],r.get('model',''),r['seed'],r['condition'])
        check(key not in records,'unique scientific identity',str(key))
        base=n.rsplit('/',1)[0]+'/'
        files={f:base+f+'.json' for f in ('job_manifest','status','history','trace','result')}
        check(all(f in names for f in files.values()),'all required scientific record members',n)
        m,status,hist,trace=[json.loads(z.read(files[f])) for f in ('job_manifest','status','history','trace')]
        records[key]={'r':r,'m':m,'h':hist,'trace':trace,'member':n}
        problems_before=len(failures)
        check(r['job_complete'] is True and status['status']=='complete','per-job completion and status',n)
        check(status['terminal_reason']==r['terminal_reason'],'terminal reason consistency',n)
        check(all(r[k]==m[k] for k in ('job_id','dataset','seed','condition')),'result-manifest identity',n)
        check(r['job_id']==status['job_id'],'result-status identity',n)
        check(r['scientific_results'] and not r['pilot_mode'] and m['scientific_results'] and not m['pilot_mode'],'full scientific, not pilot',n)
        check(m['split_seed']==42 and m['train_fraction']==1.0 and m['max_optimizer_steps'] is None,'split/full-data/budget job metadata',n)
        check(m['device']=='cuda','declared execution device',n)
        check(len({x['epoch'] for x in hist})==len(hist),'unique epochs',n)
        check(all(a['global_step']<b['global_step'] for a,b in zip(hist,hist[1:])),'increasing history steps',n)
        check(not hist or hist[-1]['global_step']==r['global_steps'],'final history step matches result',n)
        check(all(x['global_step']<=r['global_steps'] for x in hist),'history bounded by terminal step',n)
        check(all(a['step']<b['step'] for a,b in zip(trace,trace[1:])),'increasing sparse trace steps',n)
        check(not trace or trace[0]['step']==1,'trace starts at first executed step',n)
        check(all(x['step']<=r['global_steps'] for x in trace),'trace bounded by terminal step',n)
        if kind=='scratch':
            expected_config={'batch_size':32,'max_length':128,'num_epochs':30,'patience':5,'gradient_clip':1.0,'guard_window_steps':100,'trace_window_steps':300,'validation_ratio':.15}
            check(all(m.get(k)==v for k,v in expected_config.items()),'scratch frozen job configuration',n)
            d=r['dataset_manifest']; split=d['split'];splits[r['dataset']].append((n,d))
            check(split['split_seed']==42 and split['val_ratio']==.15 and split['train_fraction']==1.0,'scratch split parameters',n)
            check(m['hf_revision']==d['hf_revision_requested'],'scratch data revision consistency',n)
            check(split['train_count']+split['validation_count']==d['official_train_count'],'scratch split population',n)
            check(d['max_length']==128 and d['vocab_policy']['source']=='final_training_subset_only','scratch tokenization provenance',n)
            check(r['best_validation_loss'] is None if not hist else close(r['best_validation_loss'],min(x['validation']['loss'] for x in hist)), 'preserved scratch validation selection',n)
        else:
            check(m['warmup_steps']==math.ceil(m['total_optimizer_steps']*.1),'pretrained warmup ceil',n)
            check(m['epochs']==3 and m['weight_decay']==.01 and m['adamw_betas']==[.9,.999] and m['adamw_eps']==1e-8,'pretrained frozen optimizer config',n)
            primary={'ag_news':'macro_f1','dbpedia_14':'macro_f1','wnut_17':'entity_f1','cifar10':'accuracy'}[r['dataset']]
            check(m['primary_metric']==r['primary_metric']==primary,'pretrained primary/selection metric',n)
            if hist:
                best=max(hist,key=lambda x:x['validation_primary'])
                check(close(r['best_validation_primary'],best['validation_primary']) and r['best_epoch']==best['epoch'],'preserved pretrained validation selection',n)
                check(all(close(x['validation_primary'],x['validation'][primary]) for x in hist),'pretrained history primary consistency',n)
            if r['condition']=='executed_lr_assertion':
                a=r['assertion_record']; rate=m['peak_lr']/m['warmup_steps']
                check(r['global_steps']==0 and not trace and not hist and a['passed'] is False and a['step']==1 and close(a['intended_lr'],rate) and close(a['entry_lr'],m['peak_lr']),'pre-mutation assertion outcome',n)
        for row in trace:
            rates=expected_rates(kind,r,m,row['step'],row['guard_triggered'])
            for field,value in rates.items():
                check(close(row[field],value),'consumed LR trace field',f'{n}:{row["step"]}:{field}: observed {row[field]} expected {value}')
                trace_fields+=1
        anchors[key[:4]].append((n,r['anchor_evidence']))
        trace_rows+=len(trace);history_rows+=len(hist);terminal[(kind,r['terminal_reason'],bool(r.get('test_metrics')))]+=1
        runner='publication_expansion/src/'+('scratch_job_runner.py' if kind=='scratch' else 'pretrained_job_runner.py')
        per_run.append({'cohort':kind,'dataset':r['dataset'],'model':r.get('model',''),'seed':r['seed'],'condition':r['condition'],
                        'job_id':r['job_id'],'terminal_reason':r['terminal_reason'],'global_steps':r['global_steps'],
                        'trained_test_endpoint':bool(r.get('test_metrics')),'history_rows':len(hist),'trace_rows':len(trace),
                        'all_record_checks_passed':len(failures)==problems_before,
                        'result_member':n,'result_sha256':sha(z.read(n)),'job_manifest_member':files['job_manifest'],
                        'job_manifest_sha256':sha(z.read(files['job_manifest'])),'trace_member':files['trace'],'trace_sha256':sha(z.read(files['trace'])),
                        'history_member':files['history'],'history_sha256':sha(z.read(files['history'])),
                        'frozen_runner':'source/warmup-update-budget-acml2026/'+runner,'frozen_runner_sha256':sha((source/runner).read_bytes()),
                        'source_execution_attestation':'source-compatible saved records; executed source bytes not independently captured'})
    check(set(records)==expected,'complete scientific Cartesian matrix',str({'missing':sorted(expected-set(records)),'unexpected':sorted(set(records)-expected)}))
    check({(x['dataset'],x['seed']) for x in severity}=={(ds,s) for ds in FOLDERS for s in SEEDS} and len(severity)==20,'complete severity diagnostic grid')
    anchor_rows=[]
    for key,group in sorted(anchors.items()):
        check(len(group)==6,'six conditions per initialization group',str(key))
        matching={k:len({a[k] for _,a in group})==1 for k in ANCHORS}
        check(all(matching.values()),'matched initialization and first-transition fingerprints',str(key))
        anchor_rows.append({'cohort':key[0],'dataset':key[1],'model':key[2],'seed':key[3],'conditions':len(group),**matching})
    split_rows=[]
    for ds,items in sorted(splits.items()):
        distinct={json.dumps(d,sort_keys=True) for _,d in items}
        check(len(distinct)==1,'identical scratch split/tokenization manifests across all jobs',ds)
        d=items[0][1]
        split_rows.append({'dataset':ds,'jobs':len(items),**d['split'],'hf_revision':d['hf_revision_requested'],'vocab_size':d['vocab_size'],'same_manifest_across_jobs':len(distinct)==1})
    # Read and reconcile EVERY line of the separate partial console log.
    lines=log_path.read_text(encoding='utf-8').splitlines(); log_rows=[];completions=[];errors=[]
    log_jobs={r['r']['job_id']:r for r in records.values() if r['r']['dataset']=='ag_news' and '/scratch/' in r['member']}
    epoch_pattern=re.compile(r'\| epoch=(\d+) step=(\d+) train_loss=([\d.eE+-]+) val_loss=([\d.eE+-]+)')
    complete_pattern=re.compile(r'COMPLETE: (\S+) (\S+)')
    for line_number,line in enumerate(lines,1):
        match=epoch_pattern.search(line)
        if match:
            job_id=re.search(r'(ag_news__(?:medium-4|large-6)__seed\d+__\S+) \|',line).group(1)
            ep,step,train,val=match.groups();record=log_jobs.get(job_id)
            h=next((x for x in record['h'] if x['epoch']==int(ep)),None) if record else None
            ok=h is not None and h['global_step']==int(step) and f'{h["train_loss"]:.4f}'==train and f'{h["validation"]["loss"]:.4f}'==val
            check(ok,'rounded console epoch agrees with returned history',f'line {line_number} {job_id} epoch {ep}')
            log_rows.append({'log_line':line_number,'job_id':job_id,'epoch':int(ep),'global_step':int(step),'logged_train_loss':train,'logged_validation_loss':val,'returned_history_match':ok})
        match=complete_pattern.search(line)
        if match:
            job,reason=match.groups();record=log_jobs.get(job)
            check(record is not None and record['r']['terminal_reason']==reason,'console completion agrees with record',f'line {line_number}: {job}')
            completions.append({'line':line_number,'job_id':job,'terminal_reason':reason})
        if 'Traceback ' in line or re.search(r'(TypeError|RuntimeError):',line):
            errors.append({'line':line_number,'text':line})
    check(len(log_rows)==635 and len(completions)==40,'partial AG console coverage counts')
    check(len({(x['job_id'],x['epoch']) for x in log_rows})==len(log_rows),'unique console epoch rows')
    complete_jobs={x['job_id'] for x in completions}
    check(sum(len(log_jobs[j]['h']) for j in complete_jobs)==len(log_rows),'all returned epochs covered for 40 console-completed jobs')
    write_csv('execution_provenance_per_run.csv',per_run)
    write_csv('execution_provenance_anchor_groups.csv',anchor_rows)
    write_csv('execution_provenance_splits.csv',split_rows)
    write_csv('execution_provenance_AG_log_epochs.csv',log_rows)
    write_json(EVIDENCE/'design_manifest_comparison.json',metadata)
    write_json(EVIDENCE/'source_launcher_comparison.json',source_map)
    write_json(EVIDENCE/'AG_log_reconciliation.json',{'original_sha256':sha(log_path.read_bytes()),'lines_read':len(lines),'completed_jobs':completions,'epoch_rows':len(log_rows),'all_rounded_epoch_matches':all(x['returned_history_match'] for x in log_rows),'errors':errors,'log_is_complete_study_record':False,'other_logs_in_returned_results_zip':sorted(n for n in names if n.endswith('.log'))})
    summary={'status':'passed' if not failures else 'failed','scope':'saved-record and source provenance audit; no model training or prediction reevaluation',
             'results_archive_sha256':sha(archive.read_bytes()),'original_source_launcher_sha256':original_sha,
             'scientific_job_records':len(records),'trained_test_endpoints':sum(bool(x['r'].get('test_metrics')) for x in records.values()),
             'severity_diagnostic_records':len(severity),'metadata_files_byte_identical_to_original_launcher':sum(x['byte_identical'] for x in metadata),
             'historical_completion_manifests_unchanged':sum(bool(x['historical_completion_fields']) for x in metadata),
             'source_file_copy_matches':sum(x['byte_identical'] for x in source_map),'distinct_scientific_source_files':len({x['local_path'] for x in source_map}),
             'matched_initialization_groups':len(anchor_rows),'anchor_fields_per_group':len(ANCHORS),'history_rows_checked':history_rows,
             'trace_rows_checked':trace_rows,'trace_lr_fields_checked':trace_fields,'log_lines_read':len(lines),'log_epoch_rows_checked':len(log_rows),
             'log_completed_jobs':len(completions),'terminal_outcomes':[{'cohort':k[0],'reason':k[1],'has_test_endpoint':k[2],'count':v} for k,v in sorted(terminal.items())],
             'failures':failures,
             'interpretation':'The five completion flags are byte-identical pre-execution design metadata. Per-job result/status/history/trace records cover the full scientific matrix. This establishes internally consistent recorded completion, not bit-exact reproduction of original training.',
             'recovered_hotfix':'ACML_LAUNCHER_HOTFIX_RESUME.zip is preserved and audited separately in launcher_hotfix_audit.json; its text transformation addresses both observed launch errors.',
             'unresolved':['Final executed launcher bytes/application log not recovered. The supplied hotfix can reconstruct corrected wrappers but is not a captured execution-time copy.',
                           'No complete final master run_status/run_all_status.json or original console logs for remaining phases/datasets in returned results.',
                           'No original execution-host package/version lock or execution-time source-code digest; source_commit fields are provenance declarations.',
                           'Original candidate checkpoint weights, prediction-level validation evidence, and complete paired states at step 49999 not recovered.',
                           'Split and initialization fingerprints match saved records; underlying original index/token-to-ID/state bytes are not recovered by these hashes.']}
    write_json(EVIDENCE/'execution_provenance_summary.json',summary)
    z.close();o.close()
    print(json.dumps({k:v for k,v in summary.items() if k not in ('terminal_outcomes','unresolved')},indent=2))
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    run()

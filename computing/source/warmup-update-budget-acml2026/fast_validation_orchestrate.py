"""Bounded two-GPU validation of every frozen dataset/model/condition path.
Only genuine HF examples; subset inputs are externally overridden, deliberately NON-SCIENTIFIC.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path('validation_runtime_results')
LOGS = ROOT/'logs'
ARCHIVE = Path('/kaggle/working/ACML_REDUCED_VALIDATION_PROGRESS.zip')
FULL_SOURCE_COMMIT='bfcbc0bae87d6969fcd4d416cdc457de52c92048'

SCRATCH = ('stable_with_guard','fixed_order_with_guard','buggy_no_guard',
           'single_bad_first_step_then_stable','buggy_rescue_guard','buggy_failfast_guard')
PRE = ('safe_correct_order','late_schedule_no_guard','single_bad_then_correct',
       'update_guard_rescue','update_guard_failfast','executed_lr_assertion')

# Exactly 8 model/task combinations x 6 conditions = 48 tests, seed 42.
def jobs():
    out=[]
    for dataset in ('ag_news','dbpedia_14'):
        for model in ('medium-4','large-6'):
            out += [dict(family='scratch',dataset=dataset,model=model,condition=c)
                    for c in SCRATCH]
    for dataset in ('ag_news','dbpedia_14','wnut_17','cifar10'):
        model='deit-tiny' if dataset=='cifar10' else 'distilbert'
        out += [dict(family='pretrained',dataset=dataset,model=model,condition=c)
                for c in PRE]
    return out

# Non-transferable thresholds from prior heldout-seed42 FULL/25%-DB calibration.
# Used only to test runtime branch plumbing, not to claim coverage on tiny subsets.
SNAPSHOT={
 'pretrained':{'ag_news':1.6832995817e-4,'dbpedia_14':1.3743198480e-4,
               'wnut_17':1.5367398317e-3,'cifar10':5.0955677076e-4},
 'scratch':{'ag_news':2.5007260712e-4,'dbpedia_14':3.0661662075e-4},
}

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1048576),b''):
            h.update(part)
    return h.hexdigest()

def jid(job):return f'{job["family"]}__{job["dataset"]}__{job["model"]}__{job["condition"]}'

def result_path(job):
    p=ROOT/job['family']/job['dataset']
    if job['family']=='scratch':p=p/job['model']
    return p/'seed42'/job['condition']/'result.json'

def complete(job):
    p=result_path(job)
    if not p.exists():return False
    try:
        o=json.loads(p.read_text())
        return bool(o.get('job_complete') and o.get('pilot_mode') and o.get('validation_only_data_override'))
    except (ValueError,KeyError):return False

def load_thresholds():
    # Prefer the actual original calibration manifests when attached; fallback
    # permits new notebooks with no input attachment but embeds exact recorded
    # thresholds. No claim of fresh calibration on reduced input.
    import glob
    found={}
    for family,filename in [('pretrained','pilot_thresholds.json'),
                            ('scratch','scratch_thresholds.json')]:
        matches=glob.glob('/kaggle/input/**/'+filename,recursive=True)
        if matches:
            obj=json.loads(Path(matches[0]).read_text())
            fields={r['dataset']:float(r['threshold']) for r in obj['records']}
            for dataset,value in SNAPSHOT[family].items():
                if dataset not in fields or not abs(fields[dataset]-value)<=abs(value)*1e-7:
                    raise RuntimeError(f'calibration mismatch: {family}/{dataset}')
            found[family]={'path':matches[0],'sha256':hash_file(matches[0]),'values':fields}
        else:
            found[family]={'path':'embedded previous-session threshold snapshot',
                           'values':SNAPSHOT[family]}
    return found

def save_progress(progress, thresholds):
    ROOT.mkdir(exist_ok=True,parents=True)
    obj={'schema_version':1,'source_commit':FULL_SOURCE_COMMIT,
         'scientific_results':False,
         'purpose':'complete reduced end-to-end functional verification only',
         'total_jobs':len(jobs()),'completed_jobs':len(progress),
         'jobs':progress,'threshold_sources':thresholds,
         'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    p=ROOT/'progress.json'
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)

def compact_archive():
    # Progress can be downloaded at any time; never include model/optimizer weights.
    if not ROOT.exists():return
    tmp=ARCHIVE.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file() or p.suffix in ('.pt','.bin','.safetensors'):
                continue
            z.write(p,p.relative_to(ROOT.parent))
        manifest=Path('validation_runtime_data/subset_manifest.json')
        if manifest.exists():z.write(manifest,manifest.as_posix())
    os.replace(tmp,ARCHIVE)

def audit(all_jobs,strict=True):
    summary={'schema_version':1,'scientific_results':False,
             'source_commit':FULL_SOURCE_COMMIT,'jobs':[],'checks':[],'errors':[]}
    by_group={}
    for job in all_jobs:
        if not complete(job):
            summary['errors'].append('missing/incomplete: '+jid(job))
            continue
        r=json.loads(result_path(job).read_text())
        tpath=result_path(job).with_name('trace.json')
        trace=json.loads(tpath.read_text()) if tpath.exists() else []
        row={'family':job['family'],'dataset':job['dataset'],'model':job['model'],
             'condition':job['condition'],'terminal':r['terminal_reason'],
             'steps':r['global_steps'],'test_metrics':r.get('test_metrics'),
             'first':trace[0] if trace else None}
        summary['jobs'].append(row)
        key=(job['family'],job['dataset'],job['model'])
        by_group.setdefault(key,{})[job['condition']]=r
        if job['condition']=='executed_lr_assertion' and r['terminal_reason']!='executed_lr_assertion_failed':
            summary['errors'].append('assertion did not fail: '+jid(job))
        if job['condition']=='buggy_failfast_guard' and r['terminal_reason']!='guard_failfast':
            summary['errors'].append('scratch failfast did not stop: '+jid(job))
        if job['condition']=='update_guard_failfast' and r['terminal_reason'] not in ('guard_failfast','completed_3_epochs'):
            summary['errors'].append('unexpected pretrained failfast terminal: '+jid(job))
        if job['condition'] not in ('executed_lr_assertion','buggy_failfast_guard','update_guard_failfast') and r['terminal_reason'] not in ('max_epochs','completed_3_epochs','early_stopping','nonfinite_loss'):
            summary['errors'].append('unexpected training terminal: '+jid(job))
    for group, conditions in by_group.items():
        if len(conditions)!=6:continue
        anchors=[r.get('anchor_evidence') for r in conditions.values()]
        if not all(anchors) or any(a!=anchors[0] for a in anchors[1:]):
            summary['errors'].append('first-batch/model/gradient anchor mismatch: '+str(group))
        else:summary['checks'].append('matched anchors: '+str(group))
    report=ROOT/'validation_audit.json'
    report.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    csv=ROOT/'validation_results.csv'
    import csv as _csv
    with csv.open('w',newline='') as f:
        w=_csv.DictWriter(f,fieldnames=['family','dataset','model','condition','terminal','steps','test_metrics','first'])
        w.writeheader()
        for row in summary['jobs']:
            w.writerow({**row,'test_metrics':json.dumps(row['test_metrics']),
                        'first':json.dumps(row['first'])})
    print(f'[AUDIT] {len(summary["jobs"])}/{len(all_jobs)} jobs, '
          f'{len(summary["checks"])} matched-anchor groups, errors={len(summary["errors"])}',flush=True)
    for e in summary['errors']: print('[AUDIT ERROR]',e,flush=True)
    if strict and summary['errors']:
        raise RuntimeError('reduced validation incomplete; review errors. Not ready for professor.')
    return summary

def run(args):
    work=jobs()
    print(f'[PLAN] {len(work)} REAL-DATA REDUCED JOBS | 2 Kaggle GPUs | seed 42 | 3 epochs',flush=True)
    if args.dry_run:
        for i,j in enumerate(work,1):print(f'  {i:02d}/{len(work)} {jid(j)}')
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError('This Kaggle notebook requires GPU T4 x2 enabled')
    if not Path('validation_runtime_data/READY').exists():
        raise RuntimeError('Run the data-prep cell first')
    thresholds=load_thresholds()
    for fam,x in thresholds.items():print(f'[THRESHOLD] {fam} source={x["path"]}',flush=True)
    LOGS.mkdir(parents=True,exist_ok=True)
    pending=queue.Queue()
    for j in work:
        if not complete(j):pending.put(j)
        else:print('[SKIP complete]',jid(j),flush=True)
    progress={jid(j):{'status':'complete','result_file':str(result_path(j))} for j in work if complete(j)}
    save_progress(progress,thresholds)
    compact_archive()
    print(f'[RUN] remaining={pending.qsize()} | per-job time cap={args.job_minutes}min '
          f'| total cap={args.total_minutes}min',flush=True)
    lock=threading.Lock()
    stop=threading.Event()
    failures=[]
    started=time.monotonic()

    def gpu_lane(gpu):
        while not stop.is_set():
            try:job=pending.get_nowait()
            except queue.Empty:return
            name=jid(job)
            threshold=None
            if job['condition'] in ('buggy_rescue_guard','buggy_failfast_guard',
                                    'update_guard_rescue','update_guard_failfast'):
                threshold=thresholds[job['family']]['values'][job['dataset']]
            cmd=[sys.executable,'-u','fast_validation_worker.py',
                 '--family',job['family'],'--dataset',job['dataset'],
                 '--model',job['model'],'--condition',job['condition']]
            if threshold is not None:cmd += ['--threshold',repr(float(threshold))]
            env=os.environ.copy()
            env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env['PYTHONUNBUFFERED']='1'
            path=LOGS/(name+'.log')
            print(f'\n[GPU{gpu}] START {name}',flush=True)
            t0=time.monotonic()
            try:
                with path.open('a',buffering=1) as logfile:
                    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                          text=True,bufsize=1,env=env)
                    def relay():
                        for line in iter(proc.stdout.readline,''):
                            line=line.rstrip('\n')
                            print(f'[GPU{gpu} {name}] {line}',flush=True)
                            logfile.write(line+'\n')
                    reader=threading.Thread(target=relay,daemon=True)
                    reader.start()
                    next_heartbeat=time.monotonic()+45
                    timed_out=False
                    while proc.poll() is None:
                        time.sleep(2)
                        now=time.monotonic()
                        if now>=next_heartbeat:
                            print(f'[GPU{gpu}] ACTIVE {name} | elapsed={int((now-t0)/60)}min '
                                  f'| completed={len(progress)}/{len(work)}',flush=True)
                            next_heartbeat=now+45
                        if now-t0>args.job_minutes*60 or now-started>args.total_minutes*60:
                            timed_out=True
                            print(f'[TIME LIMIT] {name}; terminating to prevent another multi-hour run',flush=True)
                            proc.terminate()
                            try:proc.wait(timeout=20)
                            except subprocess.TimeoutExpired:proc.kill()
                            break
                    ret=proc.wait()
                    reader.join(timeout=15)
                elapsed=round((time.monotonic()-t0)/60,2)
                with lock:
                    if ret==0 and complete(job) and not timed_out:
                        progress[name]={'status':'complete','minutes':elapsed,'result_file':str(result_path(job))}
                        print(f'[GPU{gpu}] SAVED {name} ({elapsed}min)',flush=True)
                        # Outputs are already complete; remove large non-essential weights.
                        for p in result_path(job).parent.glob('*.pt'):
                            p.unlink(missing_ok=True)
                    else:
                        progress[name]={'status':'FAILED_OR_TIMED_OUT','minutes':elapsed,'exit_status':ret}
                        failures.append(name)
                        stop.set()
                        print(f'[GPU{gpu}] STOPPED/FAILED {name}; no false success claimed',flush=True)
                    save_progress(progress,thresholds)
                    compact_archive()
                    print(f'[CHECKPOINT] {ARCHIVE} refreshed (download anytime)',flush=True)
            except Exception as exc:
                with lock:
                    failures.append(name)
                    progress[name]={'status':'ERROR','detail':str(exc)}
                    save_progress(progress,thresholds)
                    compact_archive()
                    stop.set()
                print(f'[ERROR] GPU{gpu} {name}: {exc}',flush=True)
            finally:pending.task_done()
    threads=[threading.Thread(target=gpu_lane,args=(i,),daemon=False) for i in (0,1)]
    for t in threads:t.start()
    for t in threads:t.join()
    audit(work,strict=False)
    compact_archive()
    if failures:
        raise RuntimeError('Failed/timed-out jobs: '+', '.join(failures)+'; validation_progress archive preserved.')
    audit(work,strict=True)
    compact_archive()
    print('[PASS] FULL REDUCED 48-JOB INTEGRATION VALIDATION COMPLETED; NOT SCIENTIFIC RESULTS.',flush=True)
    print('[NEXT] Run the professor bundle cell to package exact frozen source and handoff.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--job-minutes',type=int,default=15)
    p.add_argument('--total-minutes',type=int,default=150)
    run(p.parse_args())

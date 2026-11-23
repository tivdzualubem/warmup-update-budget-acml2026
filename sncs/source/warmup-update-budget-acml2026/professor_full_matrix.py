#!/usr/bin/env python3
"""Publication FULL jobs, NOT the reduced notebook. Run on professor hardware.
Uses only the frozen repository's own matrix builders and job runners.
"""
import argparse, concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path

# Execute from repository root containing publication_expansion.
from publication_expansion.src import scratch_matrix as scratch
from publication_expansion.src import pretrained_matrix as pretrained

SCRATCH_ROOT='publication_expansion/results/scratch'
PRETRAINED_ROOT='publication_expansion/results/pretrained'
PHASES=('scratch-safe','scratch-nonguarded','scratch-thresholds','scratch-guarded',
        'pretrained-safe','pretrained-nonguarded','pretrained-thresholds',
        'pretrained-rescue','pretrained-diagnostic','pretrained-severity','pretrained-anchor-audit')

def main():
    p=argparse.ArgumentParser(description='Reproducible, restartable full ACML experiment matrix')
    p.add_argument('--phase',required=True,choices=PHASES)
    p.add_argument('--gpus',nargs='+',type=int,default=[0])
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--execute',action='store_true',help='required: otherwise print all proposed jobs')
    args=p.parse_args()
    print('PHASE:',args.phase,'GPUs:',args.gpus,'execute:',args.execute,flush=True)
    if args.phase in ('scratch-thresholds','pretrained-thresholds'):
        cmd=[sys.executable,'-u','publication_expansion/scripts/build_'+('scratch' if args.phase.startswith('scratch') else 'pretrained')+'_thresholds.py']
        print('COMMAND:', ' '.join(cmd),flush=True)
        if args.execute:subprocess.run(cmd,check=True)
        return
    if args.phase=='pretrained-anchor-audit':
        cmd=[sys.executable,'-u','publication_expansion/scripts/audit_pretrained_anchors.py']
        print('COMMAND:',' '.join(cmd),flush=True)
        if args.execute:subprocess.run(cmd,check=True)
        return
    if args.phase=='pretrained-severity':
        jobs=[dict(dataset=d,seed=s,condition='severity') for d in pretrained.DATASETS for s in pretrained.SEEDS]
        command=lambda j:[sys.executable,'-u','publication_expansion/scripts/run_pretrained_severity.py',
          '--dataset',j['dataset'],'--seed',str(j['seed']),'--output-root',
          'publication_expansion/results/pretrained_severity','--device','cuda','--num-workers',str(args.num_workers)]
    elif args.phase.startswith('scratch'):
        group=args.phase.removeprefix('scratch-')
        tf=Path(SCRATCH_ROOT)/'loso_thresholds.json'
        if group=='guarded' and not tf.exists():
            raise SystemExit('Missing scratch LOSO manifest. Execute scratch-safe and scratch-thresholds FIRST.')
        manifest=json.loads(tf.read_text()) if group=='guarded' else None
        jobs=scratch.build_jobs(group,threshold_manifest=manifest)
        command=lambda j: scratch.command_for_job(j,output_root=SCRATCH_ROOT,device='cuda',num_workers=args.num_workers)
    elif args.phase.startswith('pretrained'):
        group=args.phase.removeprefix('pretrained-')
        tf=Path(PRETRAINED_ROOT)/'loso_thresholds.json'
        if group in ('rescue','diagnostic') and not tf.exists():
            raise SystemExit('Missing pretrained LOSO manifest. Execute pretrained-safe and pretrained-thresholds FIRST.')
        manifest=json.loads(tf.read_text()) if group in ('rescue','diagnostic') else None
        jobs=pretrained.build_jobs(group,threshold_manifest=manifest)
        command=lambda j: pretrained.command_for_job(j,output_root=PRETRAINED_ROOT,device='cuda',num_workers=args.num_workers)
    print('TOTAL FULL-SCALE JOBS:',len(jobs),flush=True)
    if not args.execute:
        for j in jobs:print(' '.join(command(j)))
        return
    if len(set(args.gpus))!=len(args.gpus):raise RuntimeError('duplicate GPU ID')
    # Each lane serially consumes jobs, dynamically balancing wall time.
    import queue
    q=queue.Queue()
    for j in jobs:q.put(j)
    failed=[]
    def lane(gpu):
        while True:
            try:j=q.get_nowait()
            except queue.Empty:return
            tag=f'{j["dataset"]}/{j.get("model","distilbert")}/seed{j["seed"]}/{j["condition"]}'
            try:
                env=os.environ.copy()
                env['CUDA_VISIBLE_DEVICES']=str(gpu)
                env['PYTHONUNBUFFERED']='1'
                c=command(j)
                print(f'[GPU{gpu}] START {tag}',flush=True)
                log=Path('publication_expansion/results/full_run_logs')
                log.mkdir(parents=True,exist_ok=True)
                f=log/('GPU'+str(gpu)+'__'+tag.replace('/','__')+'.log')
                with f.open('a',buffering=1) as handle:
                    # A reader thread allows a real heartbeat before the first epoch ends.
                    proc=subprocess.Popen(c,env=env,text=True,bufsize=1,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    import threading
                    def relay():
                        for line in proc.stdout:
                            print(f'[GPU{gpu}] {tag}: {line.rstrip()}',flush=True)
                            handle.write(line)
                    reader=threading.Thread(target=relay,daemon=True)
                    reader.start()
                    while proc.poll() is None:
                        try:proc.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            print(f'[GPU{gpu}] ACTIVE {tag} (waiting for next epoch)',flush=True)
                    rc=proc.wait();reader.join(timeout=15)
                    if rc:raise RuntimeError(f'exit {rc}; see {f}')
                print(f'[GPU{gpu}] COMPLETE {tag}',flush=True)
            except Exception as e:
                failed.append((tag,str(e)))
                print(f'[GPU{gpu}] FAILED {tag}: {e}',flush=True)
            finally:q.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        futures=[ex.submit(lane,gpu) for gpu in args.gpus]
        for f in futures:f.result()
    if failed:
        raise SystemExit(f'{len(failed)} failed; rerun same phase after repair. '+str(failed[:4]))
    print(f'[FULL PHASE PASS] {args.phase}: {len(jobs)} jobs.',flush=True)

if __name__=='__main__':main()

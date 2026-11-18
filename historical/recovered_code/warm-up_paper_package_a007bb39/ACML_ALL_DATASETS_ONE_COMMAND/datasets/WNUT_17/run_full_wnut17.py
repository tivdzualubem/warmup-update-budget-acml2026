#!/usr/bin/env python3
"""One-command FULL scientific experiment driver for WNUT-17.

Uses the original frozen publication_expansion implementations unchanged.
All 5 seeds, full training data, full epoch counts, held-out calibration.
This script is a new execution wrapper; it is not an alteration of model logic.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from publication_expansion.src import scratch_matrix as scratch
from publication_expansion.src import pretrained_matrix as pre

DATASET = 'wnut_17'
INCLUDES_SCRATCH = False
SEEDS = (42, 123, 456, 789, 2026)
SCRATCH_ROOT = 'publication_expansion/results/scratch'
PRE_ROOT = 'publication_expansion/results/pretrained'
SEVERITY_ROOT = 'publication_expansion/results/pretrained_severity'
LOG_ROOT = Path('publication_expansion/results/full_run_logs')

SCRATCH_PHASES = (
    'scratch-safe', 'scratch-thresholds', 'scratch-nonguarded',
    'scratch-guarded', 'scratch-anchor-audit',
)
PRETRAINED_PHASES = (
    'pretrained-safe', 'pretrained-thresholds',
    'pretrained-nonguarded', 'pretrained-rescue',
    'pretrained-diagnostic', 'pretrained-severity', 'pretrained-anchor-audit',
)
ORDER = (
    list(SCRATCH_PHASES) + list(PRETRAINED_PHASES)
    if INCLUDES_SCRATCH else list(PRETRAINED_PHASES)
)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def run_tests_and_environment():
    import torch
    print(f'[PREFLIGHT] Python={sys.version.split()[0]} Torch={torch.__version__}', flush=True)
    print(f'[PREFLIGHT] GPU count={torch.cuda.device_count()}', flush=True)
    if not torch.cuda.is_available():
        raise SystemExit('GPU unavailable. Install GPU-enabled PyTorch and check CUDA first.')
    try:
        import transformers, datasets, seqeval
        print(f'[PREFLIGHT] transformers={transformers.__version__}, datasets={datasets.__version__}', flush=True)
    except ImportError as ex:
        raise SystemExit(f'Missing dependency: {ex}. See README_RUN_FIRST.md.')
    cmd = [sys.executable, '-m', 'unittest', 'discover', '-s',
           'publication_expansion/tests', '-t', '.', '-q']
    print('[PREFLIGHT] Original frozen unit test suite...', flush=True)
    subprocess.run(cmd, check=True)
    print('[PREFLIGHT] UNIT TESTS PASSED', flush=True)


def threshold_phase(phase, dry_run):
    if phase == 'scratch-thresholds':
        from publication_expansion.src.scratch_matrix import compute_loso_thresholds
        root = Path(SCRATCH_ROOT)
        path = root / 'loso_thresholds.json'
        print(f'[CALIBRATION] 5-seed LOSO, dataset={DATASET}, both scratch models -> {path}', flush=True)
        if not dry_run:
            manifest = compute_loso_thresholds(root, datasets=(DATASET,), models=scratch.MODELS,
                                               seeds=scratch.SEEDS)
            write_json_atomic(path, manifest)
            assert len(manifest['records']) == len(scratch.MODELS) * len(scratch.SEEDS)
    else:
        from publication_expansion.src.pretrained_matrix import compute_loso_thresholds
        root = Path(PRE_ROOT)
        path = root / 'loso_thresholds.json'
        print(f'[CALIBRATION] 5-seed LOSO, dataset={DATASET} -> {path}', flush=True)
        if not dry_run:
            manifest = compute_loso_thresholds(root, datasets=(DATASET,), seeds=pre.SEEDS)
            write_json_atomic(path, manifest)
            assert len(manifest['records']) == len(pre.SEEDS)


def audit_phase(phase, dry_run):
    if dry_run:
        print('[AUDIT] Will check all seeds for exact initial anchors; FAIL on mismatch.', flush=True)
        return
    if phase == 'pretrained-anchor-audit':
        report = pre.audit_matched_anchors(Path(PRE_ROOT), datasets=(DATASET,), seeds=SEEDS)
        assert report['all_matched'] and len(report['checks']) == len(SEEDS), report
        write_json_atomic(Path(PRE_ROOT) / f'{DATASET}_matched_anchor_audit.json', report)
        print(f'[AUDIT] Pretrained {DATASET}: {len(report["checks"])} exact matched-anchor groups PASS', flush=True)
    else:
        checked = []
        for model in scratch.MODELS:
            for seed in SEEDS:
                anchors = {}
                for condition in (*scratch.SAFE_CONDITIONS, *scratch.NONGUARDED_CONDITIONS, *scratch.GUARDED_CONDITIONS):
                    p = Path(SCRATCH_ROOT) / DATASET / model / f'seed{seed}' / condition / 'result.json'
                    result = json.loads(p.read_text())
                    assert result['job_complete'] and result['scientific_results'] is True, p
                    anchors[condition] = result['anchor_evidence']
                assert all(a == next(iter(anchors.values())) for a in anchors.values()), (
                    'Mismatched scratch first-transition anchors', DATASET, model, seed)
                checked.append({'dataset':DATASET,'model':model,'seed':seed,'matched':True})
        write_json_atomic(Path(SCRATCH_ROOT) / f'{DATASET}_matched_anchor_audit.json',
                          {'schema_version':1,'scientific_results':True,'checks':checked,'all_matched':True})
        print(f'[AUDIT] Scratch {DATASET}: {len(checked)} exact matched-anchor groups PASS', flush=True)


def build_jobs(phase, dry_run):
    if phase == 'pretrained-severity':
        return [({'dataset':DATASET,'seed':seed,'condition':'severity'},
                 [sys.executable,'-u','publication_expansion/scripts/run_pretrained_severity.py',
                  '--dataset',DATASET,'--seed',str(seed),'--output-root',SEVERITY_ROOT,
                  '--device','cuda']) for seed in SEEDS]

    if phase.startswith('scratch-'):
        group = phase[len('scratch-'):]
        path = Path(SCRATCH_ROOT) / 'loso_thresholds.json'
        if group == 'guarded' and not path.exists() and not dry_run:
            raise RuntimeError(f'Missing scratch calibration: {path}. Run scratch-safe then scratch-thresholds.')
        if group == 'guarded' and dry_run and not path.exists():
            # Display the exact workload without using fabricated thresholds.
            conditions = scratch.GUARDED_CONDITIONS
            return [({'dataset':DATASET,'model':model,'seed':seed,'condition':condition,'threshold':None},
                     ['(threshold computed from 4 other seed safe traces before execution)'])
                    for model in scratch.MODELS for seed in SEEDS for condition in conditions]
        manifest = json.loads(path.read_text()) if group == 'guarded' else None
        jobs = [j for j in scratch.build_jobs(group, threshold_manifest=manifest)
                if j['dataset'] == DATASET]
        return [(j,scratch.command_for_job(j,output_root=SCRATCH_ROOT,device='cuda',num_workers=2))
                for j in jobs]

    group = phase[len('pretrained-'):]
    path = Path(PRE_ROOT) / 'loso_thresholds.json'
    if group in ('rescue','diagnostic') and not path.exists() and not dry_run:
        raise RuntimeError(f'Missing pretrained calibration: {path}. Run pretrained-safe then pretrained-thresholds.')
    if group in ('rescue','diagnostic') and dry_run and not path.exists():
        conditions = (pre.RESCUE_CONDITION,) if group == 'rescue' else pre.DIAGNOSTIC_CONDITIONS
        return [({'dataset':DATASET,'seed':seed,'condition':condition,'threshold':None},
                 ['(threshold computed from 4 other seed safe traces before execution)'])
                for seed in SEEDS for condition in conditions]
    manifest = json.loads(path.read_text()) if group in ('rescue','diagnostic') else None
    jobs = [j for j in pre.build_jobs(group, threshold_manifest=manifest)
            if j['dataset'] == DATASET]
    return [(j,pre.command_for_job(j,output_root=PRE_ROOT,device='cuda',num_workers=2))
            for j in jobs]


def execute_jobs(phase, args):
    jobs = build_jobs(phase, args.dry_run)
    print(f'[PHASE] {phase}: {len(jobs)} jobs, GPUs {args.gpus}, data={DATASET}', flush=True)
    if args.dry_run:
        for job, command in jobs:
            print('  ', job['dataset'], job.get('model',''), 'seed', job['seed'],
                  job['condition'], ':', ' '.join(command), flush=True)
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    errors = []
    stop = threading.Event()
    mutex = threading.Lock()

    def worker(gpu):
        while not stop.is_set():
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            tag = '__'.join([phase,job['dataset'],job.get('model',''),f'seed{job["seed"]}',job['condition']])
            command = None
            try:
                # Build command with explicit requested worker count. Original protocol unchanged.
                if phase == 'pretrained-severity':
                    command = [sys.executable,'-u','publication_expansion/scripts/run_pretrained_severity.py',
                               '--dataset',DATASET,'--seed',str(job['seed']),
                               '--output-root',SEVERITY_ROOT,'--device','cuda',
                               '--num-workers',str(args.num_workers)]
                elif phase.startswith('scratch'):
                    command = scratch.command_for_job(job,output_root=SCRATCH_ROOT,device='cuda',num_workers=args.num_workers)
                else:
                    command = pre.command_for_job(job,output_root=PRE_ROOT,device='cuda',num_workers=args.num_workers)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                env['PYTHONUNBUFFERED'] = '1'
                logfile = LOG_ROOT / f'{tag}.log'
                print(f'[GPU{gpu}] START {tag}',flush=True)
                with logfile.open('a', encoding='utf-8', buffering=1) as log:
                    proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
                    def relay():
                        for line in proc.stdout:
                            msg = f'[GPU{gpu}] {tag} | {line.rstrip()}'
                            print(msg,flush=True)
                            log.write(msg + '\n')
                    thread = threading.Thread(target=relay,daemon=True)
                    thread.start()
                    while proc.poll() is None:
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            print(f'[GPU{gpu}] ACTIVE {tag} — awaiting epoch output', flush=True)
                    rc = proc.wait()
                    thread.join(timeout=20)
                    if rc != 0:
                        raise RuntimeError(f'job exit={rc}; log={logfile}')
                print(f'[GPU{gpu}] DONE {tag}',flush=True)
            except BaseException as exc:
                with mutex:
                    errors.append(f'{tag}: {exc}')
                stop.set()
                print(f'[GPU{gpu}] ERROR {tag}: {exc}',flush=True)
            finally:
                job_queue.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker,gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    if errors:
        raise RuntimeError(f'Phase {phase} stopped; retry this phase after fixing: {errors}')
    print(f'[PASS] {phase}: {len(jobs)} jobs finished. Restartable results saved.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=f'FULL, five-seed {DATASET} publication experiment driver')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU IDs e.g. --gpus 0 1')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--phase', default='all', choices=['all'] + ORDER)
    parser.add_argument('--dry-run', action='store_true', help='print phase plan without training')
    parser.add_argument('--skip-tests', action='store_true', help='skip repeated preflight unit tests when resuming')
    args=parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        raise SystemExit('Each GPU ID must be unique and nonnegative')
    if args.num_workers < 0: raise SystemExit('--num-workers must be nonnegative')
    phases = ORDER if args.phase == 'all' else [args.phase]
    print(f'[STUDY] dataset={DATASET} seed list={SEEDS} mode={"DRY RUN" if args.dry_run else "FULL PUBLICATION"}',flush=True)
    if not args.dry_run and not args.skip_tests:
        run_tests_and_environment()
    for index,phase in enumerate(phases,1):
        print(f'\n{"="*72}\n[{index}/{len(phases)}] {phase.upper()}\n{"="*72}',flush=True)
        if phase.endswith('thresholds'):
            threshold_phase(phase,args.dry_run)
        elif phase.endswith('anchor-audit'):
            audit_phase(phase,args.dry_run)
        else:
            execute_jobs(phase,args)
    print(f'[FULL DATASET {DATASET}: {"PLAN LISTED" if args.dry_run else "ALL PHASES COMPLETED"}]',flush=True)

if __name__=='__main__': main()

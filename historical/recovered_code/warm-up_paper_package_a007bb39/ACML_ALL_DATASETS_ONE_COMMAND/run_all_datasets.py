#!/usr/bin/env python3
"""Run the frozen full ACML publication expansion over all 4 datasets.

This orchestrator never changes scientific runner code or frozen configs.
Each dataset lives in an isolated source/results folder to prevent threshold
manifest collisions. Completed jobs are resumed/skipped by original runners.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
DATASETS = [
    ('AG_News', 'ag_news', 'run_full_agnews.py', True),
    ('DBPedia', 'dbpedia_14', 'run_full_dbpedia.py', True),
    ('WNUT_17', 'wnut_17', 'run_full_wnut17.py', False),
    ('CIFAR_10', 'cifar10', 'run_full_cifar10.py', False),
]
SEEDS = (42, 123, 456, 789, 2026)
SCRATCH_CONDITIONS = (
    'stable_with_guard', 'fixed_order_with_guard', 'buggy_no_guard',
    'single_bad_first_step_then_stable', 'buggy_rescue_guard',
    'buggy_failfast_guard',
)
PRETRAINED_CONDITIONS = (
    'safe_correct_order', 'late_schedule_no_guard',
    'single_bad_then_correct', 'update_guard_rescue',
    'update_guard_failfast', 'executed_lr_assertion',
)
EXPECTED_VERSIONS = {
    'transformers': '5.17.0',
    'datasets': '5.0.0',
    'scikit-learn': '1.9.1',
    'seqeval': '1.2.2',
}


def info(msg):
    print(msg, flush=True)


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def verify_files():
    info('[VERIFY] Checking all four source packages against SHA256SUMS.txt')
    root_hashes = {}
    for label, dataset, entry, scratch in DATASETS:
        pkg = ROOT / 'datasets' / label
        sums = pkg / 'SHA256SUMS.txt'
        if not sums.is_file():
            raise RuntimeError(f'Missing source checksum manifest: {sums}')
        entries = 0
        for ln in sums.read_text(encoding='utf-8').splitlines():
            if not ln.strip():
                continue
            parts = ln.split('  ', 1)
            if len(parts) != 2:
                raise RuntimeError(f'Malformed SHA256SUMS entry: {ln!r}')
            expected, rel = parts
            path = pkg / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen source missing/changed: {path}')
            if rel.startswith(('publication_expansion/', 'src/')):
                previous = root_hashes.setdefault(rel, expected)
                if previous != expected:
                    raise RuntimeError(f'Dataset source diverges across packages: {rel}')
            entries += 1
        manifest = json.loads((pkg/'full_run_manifest.json').read_text())
        if manifest['dataset'] != dataset or set(manifest['experiment_seeds']) != set(SEEDS):
            raise RuntimeError(f'Unexpected dataset/seeds in {pkg}/full_run_manifest.json')
        if manifest.get('pilot_results_eligible_for_publication') is not False:
            raise RuntimeError(f'Unexpected pilot/scientific status in {pkg}')
        if manifest['entry_command'] != f'python -u {entry}':
            raise RuntimeError(f'Unexpected run command for {label}')
        info(f'[VERIFY] {label}: {entries} files + manifest OK')
    info(f'[VERIFY] All four datasets share identical {len(root_hashes)} frozen source files')


def preflight(gpus, *, min_disk_gb, min_gpu_gb, check_hub=True):
    info('[PREFLIGHT] Checking versions, free storage, GPUs, unit tests and pinned sources')
    import torch
    versions = {'python': sys.version.split()[0], 'torch': torch.__version__}
    for library, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f'Missing dependency {library}=={expected}; see README_RUN_FIRST.md')
        versions[library] = actual
        if actual != expected:
            raise RuntimeError(f'Version mismatch: {library}=={actual}, expected {expected}. Use isolated environment.')
    for library in ('numpy', 'scipy', 'Pillow', 'torchvision', 'huggingface-hub'):
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f'Missing dependency: {library}')
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Full scientific matrix requires GPU; use --dry-run without GPU.')
    n_gpu = torch.cuda.device_count()
    selected = list(range(min(n_gpu, 2))) if not gpus else gpus
    if not selected or any(g >= n_gpu or g < 0 for g in selected) or len(set(selected)) != len(selected):
        raise RuntimeError(f'Requested GPUs {selected} unavailable; only {n_gpu} visible')
    gpu_info = []
    for i in selected:
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free = free_bytes/(1024**3)
        name = torch.cuda.get_device_name(i)
        info(f'[PREFLIGHT] GPU{i} {name}, free={free:.1f} GiB')
        if free < min_gpu_gb:
            raise RuntimeError(f'GPU{i} has only {free:.1f} GiB free; requires {min_gpu_gb:.1f} GiB minimum for reliable launch')
        gpu_info.append({'index':i,'name':name,'free_gib':round(free,2),'total_gib':round(total_bytes/(1024**3),2)})
    free = shutil.disk_usage(ROOT).free/(1024**3)
    info(f'[PREFLIGHT] Disk free={free:.1f} GiB')
    if free < min_disk_gb:
        raise RuntimeError(f'Only {free:.1f} GiB disk; need at least {min_disk_gb:.1f} GiB before starting multi-day run')
    pkg = ROOT / 'datasets' / 'AG_News'
    cmd = [sys.executable, '-m', 'unittest', 'discover', '-s',
           'publication_expansion/tests', '-t', '.', '-q']
    subprocess.run(cmd, cwd=pkg, check=True)
    info('[PREFLIGHT] Frozen scientific unit test suite PASSED')
    if check_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        sources = json.loads((pkg/'publication_expansion/configs/pretrained_sources_v1.json').read_text())
        names = set()
        for kind, source_type in [('models','model'),('datasets','dataset')]:
            for item in sources[kind].values():
                name, rev = item['repository'], item['revision']
                if (source_type,name,rev) in names: continue
                names.add((source_type,name,rev))
                try:
                    if source_type=='model':
                        api.model_info(repo_id=name, revision=rev)
                    else:
                        api.dataset_info(repo_id=name, revision=rev)
                except Exception as exc:
                    raise RuntimeError(f'Pinned Hugging Face {source_type} unavailable: {name}@{rev}: {exc}') from exc
                info(f'[PREFLIGHT] HF {source_type} {name}@{rev[:10]} accessible')
    write_json_atomic(ROOT/'run_status'/'preflight.json', {
        'verified_source_commit': 'bfcbc0bae87d6969fcd4d416cdc457de52c92048',
        'versions': versions, 'gpu_info':gpu_info, 'disk_free_gib':free,
        'hub_revisions_checked':check_hub, 'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    })
    info('[PREFLIGHT] PASS. Full-data execution is permitted (not guaranteed failure-free).')
    return selected


def validate_dataset(dataset_tuple):
    label, dataset, entry, with_scratch = dataset_tuple
    pkg = ROOT/'datasets'/label
    scrroot = pkg/'publication_expansion/results/scratch'/dataset
    pre_root = pkg/'publication_expansion/results/pretrained'/dataset
    sev_root = pkg/'publication_expansion/results/pretrained_severity'/dataset
    problems=[]
    checked=0
    for seed in SEEDS:
        if with_scratch:
            for model in ('medium-4','large-6'):
                for condition in SCRATCH_CONDITIONS:
                    file = scrroot/model/f'seed{seed}'/condition/'result.json'
                    try:
                        result=json.loads(file.read_text())
                        assert result.get('job_complete') is True
                        assert result.get('scientific_results') is True
                        assert result.get('seed') == seed and result.get('condition') == condition
                        # Frozen scratch runner uses 'max_epochs' or 'early_stopping',
                        # not 'completed_30_epochs' (the pretrained naming convention).
                        reason = result.get('terminal_reason')
                        valid = {'max_epochs', 'early_stopping'}
                        if condition == 'buggy_failfast_guard':
                            valid.add('guard_failfast')
                        assert reason in valid, reason
                        assert result.get('configured_epochs') == 30
                        assert result.get('train_fraction') == 1.0
                        assert result.get('max_optimizer_steps') is None
                        assert result.get('pilot_mode') is False
                    except Exception as exc:
                        problems.append(f'{file}: {exc}')
                    checked+=1
        for condition in PRETRAINED_CONDITIONS:
            file=pre_root/f'seed{seed}'/condition/'result.json'
            try:
                result=json.loads(file.read_text())
                assert result.get('job_complete') is True
                assert result.get('scientific_results') is True
                assert result.get('seed') == seed and result.get('condition') == condition
                assert result.get('configured_epochs') == 3
                assert result.get('train_fraction') == 1.0
                assert result.get('max_optimizer_steps') is None
                assert result.get('pilot_mode') is False
                if condition in ('safe_correct_order','late_schedule_no_guard','single_bad_then_correct','update_guard_rescue'):
                    assert result.get('terminal_reason') == 'completed_3_epochs', result.get('terminal_reason')
                elif condition == 'executed_lr_assertion':
                    assert result.get('terminal_reason') == 'executed_lr_assertion_failed', result.get('terminal_reason')
            except Exception as exc:
                problems.append(f'{file}: {exc}')
            checked+=1
        file=sev_root/f'seed{seed}'/'result.json'
        try:
            severity=json.loads(file.read_text())
            assert severity.get('dataset')==dataset and severity.get('seed')==seed
            assert len(severity.get('trials',[]))==8
        except Exception as exc:
            problems.append(f'{file}: {exc}')
        checked+=1
    scratch_threshold = pkg/'publication_expansion/results/scratch/loso_thresholds.json'
    pretrained_threshold = pkg/'publication_expansion/results/pretrained/loso_thresholds.json'
    try:
        pt=json.loads(pretrained_threshold.read_text())
        assert len(pt['records']) == 5
        assert all(r['dataset']==dataset for r in pt['records'])
    except Exception as exc:
        problems.append(f'{pretrained_threshold}: {exc}')
    if with_scratch:
        try:
            st=json.loads(scratch_threshold.read_text())
            assert len(st['records']) == 10
            assert all(r['dataset']==dataset for r in st['records'])
        except Exception as exc:
            problems.append(f'{scratch_threshold}: {exc}')
    if problems:
        raise RuntimeError(f'{label}: result integrity check FAILED ({len(problems)} problems), examples: {problems[:5]}')
    info(f'[RESULT-AUDIT] {label}: {checked} complete jobs including severity + thresholds OK')
    return checked


def stream_dataset(label, dataset, entry, selected, workers):
    pkg=ROOT/'datasets'/label
    logdir=ROOT/'run_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    log_file=logdir/f'{label}.log'
    cmd=[sys.executable,'-u',entry,'--gpus',*[str(g) for g in selected],
         '--num-workers',str(workers),'--skip-tests']
    env=os.environ.copy()
    env['PYTHONUNBUFFERED']='1'
    env['PYTHONHASHSEED']=env.get('PYTHONHASHSEED','0')
    env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    # The scientific algorithm is unmodified; each child runs in the same environment.
    info(f'[DATASET] {label} starting: {" ".join(cmd)}')
    p=None
    try:
        with log_file.open('a',encoding='utf-8',buffering=1) as out:
            if os.name=='posix':
                p=subprocess.Popen(cmd,cwd=pkg,env=env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            else:
                p=subprocess.Popen(cmd,cwd=pkg,env=env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in p.stdout:
                s=f'[{label}] {line.rstrip()}'
                info(s)
                out.write(s+'\n')
            rc=p.wait()
            if rc:
                raise RuntimeError(f'{label} exited code {rc}; see {log_file}. Relaunch the SAME command to resume finished work.')
    except BaseException:
        if p is not None and p.poll() is None:
            if os.name=='posix':
                os.killpg(p.pid,signal.SIGTERM)
            else: p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill()
        raise
    return log_file


def main(argv=None):
    parser=argparse.ArgumentParser(description='One command: AG News + DBPedia + WNUT17 + CIFAR10, full 5-seed ACML matrix')
    parser.add_argument('--gpus', nargs='+', type=int, default=None, help='GPU IDs; default auto up to two')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true', help='verify package and dry-run all four dataset entry scripts WITHOUT TRAINING')
    parser.add_argument('--verify-only', action='store_true', help='verify frozen source checksums only; no GPU needed')
    parser.add_argument('--preflight-only', action='store_true', help='verify environment and upstream pinned revisions; no training')
    parser.add_argument('--skip-hub-check', action='store_true', help='only use if all pinned datasets and models are already cached')
    parser.add_argument('--min-free-disk-gb',type=float,default=15.0)
    parser.add_argument('--min-free-gpu-gb',type=float,default=10.0)
    parser.add_argument('--only',nargs='+',choices=[x[0] for x in DATASETS],default=None,help='Select subset of datasets (useful for manual recovery)')
    args=parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if args.gpus and (any(x<0 for x in args.gpus) or len(set(args.gpus))!=len(args.gpus)):
        parser.error('GPU IDs must be unique and nonnegative')
    verify_files()
    if args.verify_only:
        info('[VERIFY] PASS: source-only verification, no experiments were run')
        return 0
    chosen=[x for x in DATASETS if not args.only or x[0] in args.only]
    if args.dry_run:
        for label,dataset,entry,scr in chosen:
            info(f'[DRY RUN] {label}: frozen dataset entry script, all phases:')
            subprocess.run([sys.executable,'-u',entry,'--dry-run'],cwd=ROOT/'datasets'/label,check=True)
        info(f'[DRY RUN] PASS: all {len(chosen)} source drivers emitted valid full experiment plans; no GPU used.')
        return 0
    selected=preflight(args.gpus,min_disk_gb=args.min_free_disk_gb,
                       min_gpu_gb=args.min_free_gpu_gb,check_hub=not args.skip_hub_check)
    if args.preflight_only:
        info('[PREFLIGHT] PASS: environment ready; NO full training run started.')
        return 0
    state_file=ROOT/'run_status'/'run_all_status.json'
    if state_file.exists():
        state=json.loads(state_file.read_text())
    else:
        state={'source_commit':'bfcbc0bae87d6969fcd4d416cdc457de52c92048','datasets':{}}
    try:
        for index,item in enumerate(chosen,1):
            label,dataset,entry,scr=item
            info(f'\n[{index}/{len(chosen)}] FULL DATASET {label} (5 seeds, original full epochs)')
            state['datasets'][label]={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            write_json_atomic(state_file,state)
            stream_dataset(label,dataset,entry,selected,args.num_workers)
            checked=validate_dataset(item)
            state['datasets'][label]={'status':'complete','verified_job_count':checked,
                                      'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            write_json_atomic(state_file,state)
        info('[FULL STUDY] ALL REQUESTED DATASETS COMPLETE AND VERIFIED')
        return 0
    except BaseException as exc:
        state['last_error']=repr(exc)
        write_json_atomic(state_file,state)
        info(f'[STOPPED SAFELY] {exc}')
        info('[RESUME] Fix the cause, then run: python -u run_all_datasets.py (completed original jobs are skipped).')
        raise


if __name__=='__main__':
    raise SystemExit(main())

"""Prospective configured-peak pair. Default/describe never launches training.

preflight: real-machine/data smoke, resume and timing checks, discarded afterward.
run: fixed-budget research pair; requires a matching passed preflight report.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent


def load_config(path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    fixed = dict(protocol_version='configured_peak_pair_v1_proposed', dataset='dbpedia_14', model='medium-4', seed=42, split_seed=42,
                 validation_ratio=.15, batch_size=32, max_length=128, peak_lr=1e-4,
                 warmup_steps=8000, schedule_horizon=50000, gradient_clip=1., precision='fp32',
                 primary_endpoint='validation_logit_ce_peak_first_minus_clean_at_fixed_final_step',
                 test_policy='evaluate_only_after_both_arms_reach_fixed_final_step')
    for key, value in fixed.items():
        if c.get(key) != value:
            raise ValueError(f'{key} differs from this scoped protocol; make a separately reviewed protocol version.')
    for key in ('budget_steps', 'latest_checkpoint_every', 'online_loss_every', 'train_monitor_size', 'preflight_timing_steps'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if not 8000 <= c['budget_steps'] < c['schedule_horizon']:
        raise ValueError('Budget must finish warmup and stay strictly before the zero-LR boundary.')
    if c['train_monitor_size'] > 476000:
        raise ValueError('Training monitor exceeds the fixed population.')
    for key in ('evaluation_steps', 'retained_checkpoint_steps'):
        values = c[key]
        if values != sorted(set(values)) or any(type(x) is not int or not 0 <= x <= c['budget_steps'] for x in values):
            raise ValueError('Checkpoints must be unique ordered integer steps inside the budget.')
        if not {0, 1, 8000, c['budget_steps']}.issubset(values):
            raise ValueError('Required initialization/first-step/warmup/final checkpoints missing.')
    if c['dataset_revision'] != '9abd46cf7fc8b4c64290f26993c540b92aa145ac':
        raise ValueError('Pinned dataset revision differs.')
    if c['expected_validation_indices_sha256'] != '75cc3a5c716ae024bd95de7e7d579546bef5a63038d940c7ff5d5a66aa08f3fa':
        raise ValueError('Expected archived split fingerprint differs.')
    return c


def source_hashes():
    paths = list(ROOT.glob('*.py'))
    for folder in ('frozen_source', 'reference_utils', 'tests'):
        paths.extend((ROOT/folder).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def check_frozen_source():
    expected = json.loads((ROOT/'FROZEN_SOURCE_HASHES.json').read_text())
    for name, digest in expected.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source changed: {name}')


def environment(device):
    versions = {}
    for name in ('torch', 'numpy', 'datasets', 'huggingface-hub', 'pyarrow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'packages': versions, 'device': str(device), 'torch_cuda_build': torch.version.cuda,
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'tf32': False,
              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    result['torch_threads'] = {'intraop': torch.get_num_threads(), 'interop': torch.get_num_interop_threads()}
    result['installed_distributions'] = sorted({(d.metadata['Name'], d.version)
                                               for d in importlib.metadata.distributions() if d.metadata['Name']})
    result['installed_distributions'] = [list(item) for item in result['installed_distributions']]
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        result['cuda_device'] = {'name': props.name, 'total_memory': props.total_memory,
                                'capability': list(torch.cuda.get_device_capability(device)),
                                'cudnn_version': torch.backends.cudnn.version()}
    return result


def data_content_hash(dataset):
    h = hashlib.sha256()
    for text, label in zip(dataset.texts, dataset.labels):
        encoded = text.encode('utf-8')
        h.update(len(encoded).to_bytes(8, 'little'))
        h.update(encoded)
        h.update(int(label).to_bytes(8, 'little', signed=True))
    return h.hexdigest()


def prepare(c, device, out, expected_identity=None):
    from publication_expansion.src.scratch_dataset_adapter import load_scratch_hf_dataset
    from publication_expansion.src.scratch_data_contract import make_split_indices
    from publication_expansion.src.scratch_model_adapter import build_scratch_model
    check_frozen_source()
    prepared = load_scratch_hf_dataset(c['dataset'], max_length=c['max_length'], split_seed=c['split_seed'],
        val_ratio=c['validation_ratio'], train_fraction=1., revision=c['dataset_revision'])
    m = prepared.manifest
    if (len(prepared.train_dataset), len(prepared.validation_dataset), len(prepared.test_dataset)) != (476000, 84000, 70000):
        raise ValueError('Unexpected official populations; do not silently substitute data.')
    if m['split']['validation_indices_sha256'] != c['expected_validation_indices_sha256']:
        raise ValueError('Reconstructed validation indices do not match the archived digest.')
    split = make_split_indices(560000, split_seed=c['split_seed'], val_ratio=c['validation_ratio'])
    data = {'manifest': m, 'vocabulary_sha256': core.json_hash(prepared.vocab),
            'ordered_train_content_sha256': data_content_hash(prepared.train_dataset),
            'ordered_validation_content_sha256': data_content_hash(prepared.validation_dataset),
            'ordered_test_content_sha256': data_content_hash(prepared.test_dataset)}
    identity = {'protocol': core.json_hash(c), 'code': core.json_hash(source_hashes()),
                'data': core.json_hash(data), 'environment': core.json_hash(environment(device))}
    if expected_identity is not None and identity != expected_identity:
        raise ValueError('Preflight code/config/data/environment differs; run a new preflight.')
    metadata = {'protocol.json': c, 'data_manifest.json': data,
                'environment.json': environment(device), 'source_hashes.json': source_hashes(),
                'vocabulary.json': prepared.vocab}
    arrays = dict(train=split.train, validation=split.validation,
                  test=np.arange(70000, dtype=np.int64), train_monitor=split.train[:c['train_monitor_size']])
    # Check ALL existing metadata before writing anything during resume.
    for name, value in metadata.items():
        if (out/name).exists() and json.loads((out/name).read_text(encoding='utf-8')) != value:
            raise ValueError(f'Existing run metadata differs: {name}')
    if (out/'split_indices.npz').exists():
        with np.load(out/'split_indices.npz', allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError('Existing saved split arrays differ.')
    out.mkdir(parents=True, exist_ok=True)
    for name, value in metadata.items():
        if not (out/name).exists(): core.atomic_json(out/name, value)
    if not (out/'split_indices.npz').exists():
        np.savez_compressed(out/'split_indices.npz', **arrays)
    factory = lambda: build_scratch_model(model_name=c['model'], vocab_size=len(prepared.vocab),
        num_classes=prepared.num_classes, max_seq_len=c['max_length'])
    return prepared, split, factory, identity


def new_engine(factory, prepared, c, device):
    return core.PairEngine(factory, prepared.train_dataset, c, device)


def preflight(c, device, out):
    if out.exists():
        raise FileExistsError('Use a new preflight folder; preserve earlier checks.')
    prepared, split, factory, identity = prepare(c, device, out)
    engine = new_engine(factory, prepared, c, device)
    for _ in range(3):
        engine.step()
    state = engine.state_dict(identity)
    checkpoint = out/'resume_probe.pt'
    core.save_verified_checkpoint(checkpoint, state)
    for _ in range(2):
        engine.step()
    uninterrupted = engine.scientific_state_digest()
    # Restore through serialization on the SAME actual device and real dataset.
    restored = new_engine(factory, prepared, c, device)
    restored.load_state_dict(core.load_verified_checkpoint(checkpoint), identity)
    for _ in range(2):
        restored.step()
    if restored.scientific_state_digest() != uninterrupted:
        raise RuntimeError('Actual-device/data resume probe differs from uninterrupted execution.')
    del engine, restored, state
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    timing = new_engine(factory, prepared, c, device)
    for _ in range(c['preflight_timing_steps']):
        timing.step()
    evaluation_seconds = {}
    for arm in core.ARMS:
        core.synchronize(device)
        start = time.perf_counter()
        core.evaluate(timing.models[arm], prepared.validation_dataset, device, c['batch_size'])
        core.synchronize(device)
        evaluation_seconds[arm] = time.perf_counter()-start
    core.synchronize(device)
    start = time.perf_counter()
    core.save_verified_checkpoint(out/'timing_state.pt', timing.state_dict(identity))
    core.synchronize(device)
    checkpoint_seconds = time.perf_counter()-start
    estimate = {arm: {'instrumented_training_seconds': timing.cost[arm]['instrumented_training_seconds']/c['preflight_timing_steps']*c['budget_steps'],
                     'validation_seconds': evaluation_seconds[arm]*len(c['evaluation_steps'])} for arm in core.ARMS}
    report = {'status': 'passed', 'scope': 'Actual selected device and real-data loader smoke/resume/timing check; discarded diagnostic trajectories, not scientific results.',
              'identity': identity, 'resume_probe_exact': True, 'first_transition_anchors_match': timing.anchors['clean']==timing.anchors['peak_first'],
              'measured_steps_per_arm': c['preflight_timing_steps'], 'cost': timing.cost,
              'full_validation_seconds_per_arm': evaluation_seconds, 'paired_checkpoint_seconds': checkpoint_seconds,
              'paired_checkpoint_bytes': (out/'timing_state.pt').stat().st_size,
              'rough_cost_projection': estimate,
              'projection_limits': 'Short-prefix instrumented throughput is approximate. Add data, train-monitor/test evaluation, prediction serialization and checkpoint costs; no wall-clock guarantee.',
              'research_pair_run': False}
    core.atomic_json(out/'preflight.json', report)
    print(json.dumps(report, indent=2))


def evaluate_pair(engine, prepared, split, out, include_test=False, emit=None):
    from torch.utils.data import Subset
    if include_test and engine.step_count != engine.config['budget_steps']:
        raise RuntimeError('Test evaluation is allowed only after both arms reach fixed final budget.')
    c = engine.config
    populations = [('validation', prepared.validation_dataset, split.validation),
                   ('train_monitor_subset', Subset(prepared.train_dataset, range(c['train_monitor_size'])), split.train[:c['train_monitor_size']])]
    if include_test:
        populations = [('test', prepared.test_dataset, np.arange(len(prepared.test_dataset)))]
    for population, dataset, indices in populations:
        for arm in core.ARMS:
            prior = [r for r in engine.history if (r['arm'], r['population'], r['step']) == (arm, population, engine.step_count)]
            if prior:
                saved = out/'predictions'/prior[-1]['prediction_file']
                if not saved.exists() or hashlib.sha256(saved.read_bytes()).hexdigest() != prior[-1]['prediction_sha256']:
                    raise ValueError('Saved evaluation predictions missing or changed.')
                continue
            core.synchronize(engine.device)
            core.begin_memory_interval(engine.device)
            start = time.perf_counter()
            path = out/'predictions'/f'{population}_{arm}_step{engine.step_count:05d}.npz'
            metrics = core.evaluate(engine.models[arm], dataset, engine.device, c['batch_size'], path, indices)
            core.synchronize(engine.device)
            elapsed = time.perf_counter()-start
            core.record_memory_interval(engine.cost[arm], engine.device)
            engine.cost[arm]['evaluation_seconds'] += elapsed
            engine.history.append({'arm': arm, 'population': population, 'step': engine.step_count, **metrics})
            if emit is not None:
                emit({'event': 'evaluation', 'arm': arm, 'population': population, 'step': engine.step_count,
                      'seconds': elapsed, 'prediction_bytes': path.stat().st_size})


def save_checkpoint(engine, identity, out, emit, retain=False):
    core.synchronize(engine.device)
    start = time.perf_counter()
    payload = engine.state_dict(identity)
    core.save_verified_checkpoint(out/'checkpoints/latest.pt', payload)
    if retain:
        core.save_verified_checkpoint(out/'checkpoints'/f'pair_step{engine.step_count:05d}.pt', payload)
    core.synchronize(engine.device)
    elapsed = time.perf_counter()-start
    engine.shared_cost['checkpoint_seconds'] += elapsed
    emit({'event': 'checkpoint', 'step': engine.step_count, 'seconds': elapsed, 'retained': retain})


def run_pair(c, device, out, preflight_path, resume=False):
    if not resume and out.exists():
        raise FileExistsError('Use a new run folder; never overwrite an earlier pair.')
    if resume and (not (out/'checkpoints/latest.pt').exists() or (out/'result.json').exists()):
        raise ValueError('Resume requires an unfinished run with its own latest full pair state.')
    gate = json.loads(Path(preflight_path).read_text())
    if gate.get('status') != 'passed' or not gate.get('resume_probe_exact'):
        raise ValueError('A passed actual-machine preflight is required.')
    # Resume identity files are verified before replacement.
    old_protocol = json.loads((out/'protocol.json').read_text()) if resume else None
    if resume and old_protocol != c:
        raise ValueError('Cannot change the fixed protocol while resuming.')
    prepared, split, factory, identity = prepare(c, device, out, expected_identity=gate['identity'])
    if gate['identity'] != identity:
        raise ValueError('Preflight code/config/data/environment differs; run a new preflight.')
    engine = new_engine(factory, prepared, c, device)
    if resume:
        engine.load_state_dict(core.load_verified_checkpoint(out/'checkpoints/latest.pt'), identity)
    segment = out/'logs'/('segment_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.jsonl')
    segment.parent.mkdir(exist_ok=True)
    start_session = time.perf_counter()
    with segment.open('x', encoding='utf-8') as f:
        def emit(value):
            f.write(json.dumps(value, allow_nan=False)+'\n')
            f.flush()
        emit({'event': 'start', 'resume': resume, 'start_step': engine.step_count, 'identity': identity})
        try:
            if not resume:
                evaluate_pair(engine, prepared, split, out, emit=emit)
                save_checkpoint(engine, identity, out, emit, retain=True)
            while engine.step_count < c['budget_steps']:
                emit({'event': 'paired_step', **engine.step()})
                if engine.step_count in c['evaluation_steps']:
                    evaluate_pair(engine, prepared, split, out, emit=emit)
                    core.atomic_json(out/'evaluation_history.json', engine.history)
                if engine.step_count % c['latest_checkpoint_every'] == 0 or engine.step_count in c['retained_checkpoint_steps']:
                    save_checkpoint(engine, identity, out, emit, retain=engine.step_count in c['retained_checkpoint_steps'])
            evaluate_pair(engine, prepared, split, out, include_test=True, emit=emit)
            save_checkpoint(engine, identity, out, emit, retain=True)
            core.atomic_json(out/'evaluation_history.json', engine.history)
            core.atomic_json(out/'online_training_history.json', engine.online_history)
            final = {(row['arm'], row['population']): row for row in engine.history if row['step']==c['budget_steps']}
            emit({'event': 'session_end', 'status': 'completed', 'active_seconds': time.perf_counter()-start_session})
            result = {'status': 'completed', 'identity': identity, 'protocol': c,
                'fixed_positive_lr_steps_per_arm': engine.step_count, 'zero_lr_steps_per_arm': 0,
                'validation_ce_peak_first_minus_clean': final[('peak_first','validation')]['logit_ce']-final[('clean','validation')]['logit_ce'],
                'validation_accuracy_clean_minus_peak_first': final[('clean','validation')]['accuracy']-final[('peak_first','validation')]['accuracy'],
                'final_metrics': {arm: {pop: final[(arm,pop)] for pop in ('validation','test','train_monitor_subset')} for arm in core.ARMS},
                'committed_trajectory_cost': engine.cost, 'shared_cost': engine.shared_cost,
                'recorded_attempt_cost': aggregate_logs(out/'logs'),
                'retained_file_bytes': retained_bytes(out),
                'interpretation': 'One prespecified seed/configuration. No equivalence, population-confidence or permanent-damage claim. Final-state evaluation only; no checkpoint selection.'}
            core.atomic_json(out/'result.json', result)
        except Exception as error:
            emit({'event': 'failure', 'step': engine.step_count, 'error_type': type(error).__name__, 'message': str(error),
                  'active_seconds': time.perf_counter()-start_session, 'partial_uncommitted_work_possible': True})
            raise


def aggregate_logs(folder):
    report = {'completed_pair_step_events': 0, 'training_seconds': {arm: 0. for arm in ('clean','peak_first')},
              'evaluation_seconds': {arm: 0. for arm in ('clean','peak_first')},
              'checkpoint_seconds': 0., 'closed_session_active_seconds': 0., 'failure_events': 0,
              'segments_without_end': 0, 'truncated_final_log_lines': 0}
    for file in sorted(Path(folder).glob('segment_*.jsonl')):
        closed = False
        lines = file.read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines)-1: raise
                report['truncated_final_log_lines'] += 1
                continue
            if row['event']=='paired_step':
                report['completed_pair_step_events'] += 1
                for arm in report['training_seconds']:
                    report['training_seconds'][arm] += row['arms'][arm]['instrumented_training_seconds']
            if row['event']=='checkpoint': report['checkpoint_seconds'] += row['seconds']
            if row['event']=='evaluation': report['evaluation_seconds'][row['arm']] += row['seconds']
            if row['event'] in ('session_end','failure'):
                closed = True
                report['closed_session_active_seconds'] += row['active_seconds']
            if row['event']=='failure': report['failure_events'] += 1
        report['segments_without_end'] += int(not closed)
    report['scope'] = 'Recorded attempts include repeated work after resume. Open/killed segments or failed partial steps can leave unmeasured cost; do not call these totals complete when such gaps exist.'
    return report


def retained_bytes(out):
    return {'predictions_per_arm': {arm: sum(p.stat().st_size for p in (out/'predictions').glob(f'*_{arm}_step*.npz'))
                                   for arm in ('clean', 'peak_first')},
            'paired_checkpoints_shared': sum(p.stat().st_size for p in (out/'checkpoints').glob('*.pt')),
            'note': 'Full checkpoints contain both arms and shared sampler/metadata; archive bytes are shared, not charged twice.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', choices=('describe','preflight','run'), default='describe')
    p.add_argument('--config', type=Path, default=ROOT/'protocol.json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    c = load_config(a.config)
    if a.action=='describe':
        print(json.dumps({'status':'proposed_not_run','protocol':c,'positive_transitions_total':2*c['budget_steps'],
                          'next':'Run the required real-data/device preflight on the professor training machine, review measured resources, then execute the fixed pair. No repeat scope approval is required.'},indent=2))
        return
    if a.output is None:
        p.error('--output is required for preflight/run.')
    if a.action=='run' and a.preflight is None:
        p.error('--preflight report is required for run.')
    global core, torch, np
    import pair_core as core
    import torch
    import numpy as np
    core.configure_determinism()
    device = torch.device(a.device)
    if device.type not in ('cpu','cuda'):
        raise ValueError('This version supports one CPU or CUDA device only.')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device unavailable; no silent CPU fallback.')
    if a.action=='preflight':
        preflight(c, device, a.output)
    else:
        run_pair(c, device, a.output, a.preflight, a.resume)


if __name__=='__main__':
    main()

"""Paired, fixed-budget training mechanics. No data downloads or CLI execution."""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'frozen_source'))
from publication_expansion.src.anchor import object_sha256, model_fingerprint, optimizer_non_lr_fingerprint, gradient_fingerprint
from publication_expansion.src.scheduler_semantics import stable_lr
from publication_expansion.src.scratch_model_adapter import build_scratch_adam
from reference_utils.resumable_state import capture_rng, restore_rng, preserve_rng
from reference_utils.evaluation import CrossEntropyAccumulator, EVALUATOR_VERSION

ARMS = ('clean', 'peak_first')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def begin_memory_interval(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def record_memory_interval(cost, device):
    if torch.device(device).type == 'cuda':
        for key, value in (('peak_process_cuda_allocated_bytes', torch.cuda.max_memory_allocated(device)),
                           ('peak_process_cuda_reserved_bytes', torch.cuda.max_memory_reserved(device))):
            cost[key] = max(cost.get(key) or 0, value)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def save_verified_checkpoint(path, value):
    path = Path(path)
    atomic_save(path, value)
    atomic_json(path.with_suffix('.sha256.json'), {'file': path.name,
        'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


def load_verified_checkpoint(path):
    path = Path(path)
    record = json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))
    if record['file'] != path.name or record['bytes'] != path.stat().st_size or record['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Checkpoint checksum mismatch; preserve files and investigate.')
    return torch.load(path, map_location='cpu', weights_only=False)


def finite_tensors(values):
    values = [v for v in values if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex())]
    # Scalar optimizer counters may be CPU tensors even when parameters are CUDA.
    groups = {}
    for value in values:
        groups.setdefault(str(value.device), []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(group).all().item()) for group in groups.values())


def finite_model_optimizer(model, optimizer):
    values = list(model.state_dict().values())
    for state in optimizer.state.values():
        values.extend(state.values())
    return finite_tensors(values)


class CursorSampler:
    """Explicit permutation/cursor; shared between the two arms, saved on resume.

    This is a new paired sampling implementation, not a claim to reconstruct the
    historical DataLoader's implicit generator-consumption sequence.
    """
    def __init__(self, count, batch_size, seed):
        if count < 1 or batch_size < 1:
            raise ValueError('Nonempty dataset and positive batch size required.')
        self.count, self.batch_size = count, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(count, generator=self.generator)
        self.cursor, self.epoch = 0, 0

    def next(self):
        if self.cursor == self.count:
            self.epoch += 1
            self.cursor = 0
            self.permutation = torch.randperm(self.count, generator=self.generator)
        end = min(self.count, self.cursor+self.batch_size)
        indices = self.permutation[self.cursor:end].clone()
        self.cursor = end
        return indices

    def state_dict(self):
        return dict(count=self.count, batch_size=self.batch_size, permutation=self.permutation.clone(),
                    cursor=self.cursor, epoch=self.epoch, generator_state=self.generator.get_state().clone())

    def load_state_dict(self, value):
        if value['count'] != self.count or value['batch_size'] != self.batch_size:
            raise ValueError('Sampler population or batch size mismatch.')
        permutation = value['permutation']
        if permutation.dtype != torch.int64 or len(permutation) != self.count or not torch.equal(torch.sort(permutation).values, torch.arange(self.count)):
            raise ValueError('Invalid saved permutation.')
        if not 0 <= value['cursor'] <= self.count:
            raise ValueError('Invalid sampler cursor.')
        self.permutation = permutation.clone()
        self.cursor, self.epoch = value['cursor'], value['epoch']
        self.generator.set_state(value['generator_state'])


class PairEngine:
    def __init__(self, model_factory, dataset, config, device='cpu'):
        self.config = copy.deepcopy(config)
        self.device = torch.device(device)
        self.dataset = dataset
        seed_all(config['seed'])
        base = model_factory().to(self.device)
        self.models = {arm: copy.deepcopy(base) for arm in ARMS}
        del base
        self.optimizers = {arm: build_scratch_adam(self.models[arm], initial_lr=config['peak_lr']) for arm in ARMS}
        initial_rng = capture_rng()
        self.rng = {arm: copy.deepcopy(initial_rng) for arm in ARMS}
        self.sampler = CursorSampler(len(dataset), config['batch_size'], config['seed'])
        self.step_count = 0
        self.anchors = {}
        self.history = []
        self.online_history = []
        self.windows = {arm: {'sum_loss': 0., 'examples': 0, 'first_step': 1} for arm in ARMS}
        self.cost = {arm: {'instrumented_training_seconds': 0., 'evaluation_seconds': 0., 'examples': 0,
                          'positive_lr_steps': 0, 'zero_lr_steps': 0,
                          'peak_process_cuda_allocated_bytes': None, 'peak_process_cuda_reserved_bytes': None,
                          'memory_scope': 'Process CUDA peak during this arm; BOTH arms remain resident. CPU peak RAM not measured.'} for arm in ARMS}
        self.shared_cost = {'data_loading_seconds': 0., 'checkpoint_seconds': 0.}
        self.last_observed_lr = {}
        self.failed = False
        for arm in ARMS:
            self.optimizers[arm].register_step_pre_hook(self._hook(arm))
        assert model_fingerprint(self.models['clean']) == model_fingerprint(self.models['peak_first'])

    def _hook(self, arm):
        def observe(optimizer, args, kwargs):
            rates = [float(group['lr']) for group in optimizer.param_groups]
            expected = self.expected_rate(arm, self.step_count+1)
            if rates != [expected]*len(rates):
                raise RuntimeError('Consumed LR violates the prespecified arm schedule.')
            self.last_observed_lr[arm] = rates
        return observe

    def expected_rate(self, arm, step):
        if arm == 'peak_first' and step == 1:
            return self.config['peak_lr']
        return stable_lr(step, base_lr=self.config['peak_lr'], warmup_steps=self.config['warmup_steps'], total_steps=self.config['schedule_horizon'])

    def step(self):
        if self.failed:
            raise RuntimeError('A failed in-memory pair must not continue.')
        if self.step_count >= self.config['budget_steps']:
            raise RuntimeError('Fixed budget reached; no automatic extension.')
        try:
            return self._step()
        except Exception:
            self.failed = True
            raise

    def _step(self):
        step = self.step_count+1
        start = time.perf_counter()
        indices = self.sampler.next()
        batch = default_collate([self.dataset[int(i)] for i in indices])
        batch_hash = object_sha256(batch)
        self.shared_cost['data_loading_seconds'] += time.perf_counter()-start
        record = {'step': step, 'epoch': self.sampler.epoch, 'sampler_cursor': self.sampler.cursor,
                  'batch_indices_sha256': object_sha256(indices), 'batch_sha256': batch_hash, 'arms': {}}
        first_anchors = {}
        track_ratio = step <= 100 or step in self.config['evaluation_steps']
        for arm in ARMS:
            model, optimizer = self.models[arm], self.optimizers[arm]
            restore_rng(self.rng[arm])
            model.train()
            synchronize(self.device)
            begin_memory_interval(self.device)
            begin = time.perf_counter()
            rng_before = object_sha256(capture_rng()) if step == 1 else None
            initial_model = model_fingerprint(model) if step == 1 else None
            initial_optimizer = optimizer_non_lr_fingerprint(optimizer) if step == 1 else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['input_ids'].to(self.device))
            loss = F.cross_entropy(logits, batch['labels'].to(self.device))
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Nonfinite loss in {arm} at step {step}.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'], error_if_nonfinite=True)
            if not finite_tensors([p.grad for p in model.parameters() if p.grad is not None]):
                raise FloatingPointError('Nonfinite gradient after clipping.')
            if step == 1:
                first_anchors[arm] = dict(model_state=initial_model, optimizer_non_lr=initial_optimizer,
                    first_batch=batch_hash, rng_before_forward=rng_before, logits=object_sha256(logits),
                    loss=object_sha256(loss), clipped_gradient=gradient_fingerprint(model))
                if arm == 'peak_first' and first_anchors[arm] != first_anchors['clean']:
                    raise RuntimeError('First-transition anchor mismatch; pair invalid.')
            rate = self.expected_rate(arm, step)
            if not rate > 0 or not math.isfinite(rate):
                raise RuntimeError('No zero/nonfinite LR step is permitted in this fixed budget.')
            before = [p.detach().clone() for p in model.parameters()] if track_ratio else None
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.step()
            if not finite_model_optimizer(model, optimizer):
                raise FloatingPointError('Nonfinite model/optimizer state after update.')
            ratio = None
            if before is not None:
                displacement = sum((p.detach().double()-v.double()).square().sum() for p, v in zip(model.parameters(), before)).sqrt()
                norm = sum(v.double().square().sum() for v in before).sqrt()
                ratio = float((displacement/(norm+1e-12)).item())
            self.rng[arm] = capture_rng()
            synchronize(self.device)
            elapsed = time.perf_counter()-begin
            record_memory_interval(self.cost[arm], self.device)
            count = len(batch['labels'])
            self.cost[arm]['instrumented_training_seconds'] += elapsed
            self.cost[arm]['examples'] += count
            self.cost[arm]['positive_lr_steps'] += 1
            window = self.windows[arm]
            window['sum_loss'] += float(loss.item())*count
            window['examples'] += count
            record['arms'][arm] = {'lr_consumed': self.last_observed_lr[arm], 'online_batch_ce': float(loss.item()),
                'examples': count, 'candidate_and_committed_ratio': ratio, 'instrumented_training_seconds': elapsed}
        self.step_count = step
        if step == 1:
            self.anchors = first_anchors
        if step % self.config['online_loss_every'] == 0 or step == self.config['budget_steps']:
            for arm in ARMS:
                window = self.windows[arm]
                self.online_history.append({'arm': arm, 'first_step': window['first_step'], 'last_step': step,
                    'num_examples': window['examples'], 'example_weighted_online_ce': window['sum_loss']/window['examples']})
                self.windows[arm] = {'sum_loss': 0., 'examples': 0, 'first_step': step+1}
        return record

    def state_dict(self, identity):
        if self.failed:
            raise RuntimeError('Cannot record a partial/failed pair as resumable.')
        return {'format': 'peak_pair_state_v1', 'identity': identity, 'config': self.config,
            'device': str(self.device), 'step': self.step_count,
            'models': {arm: copy.deepcopy(self.models[arm].state_dict()) for arm in ARMS},
            'optimizers': {arm: copy.deepcopy(self.optimizers[arm].state_dict()) for arm in ARMS},
            'rng': copy.deepcopy(self.rng), 'sampler': self.sampler.state_dict(),
            'anchors': copy.deepcopy(self.anchors), 'history': copy.deepcopy(self.history),
            'online_history': copy.deepcopy(self.online_history), 'windows': copy.deepcopy(self.windows),
            'cost': copy.deepcopy(self.cost), 'shared_cost': copy.deepcopy(self.shared_cost)}

    def load_state_dict(self, state, identity):
        if state['format'] != 'peak_pair_state_v1' or state['identity'] != identity or state['config'] != self.config:
            raise ValueError('Checkpoint protocol/code/data/environment identity mismatch.')
        if state['device'] != str(self.device):
            raise ValueError('Resume device must match the checked environment.')
        if not 0 <= state['step'] <= self.config['budget_steps']:
            raise ValueError('Checkpoint step lies outside the fixed budget.')
        for arm in ARMS:
            self.models[arm].load_state_dict(state['models'][arm])
            self.optimizers[arm].load_state_dict(state['optimizers'][arm])
            if not finite_model_optimizer(self.models[arm], self.optimizers[arm]):
                raise FloatingPointError('Checkpoint contains nonfinite state.')
            for value in self.optimizers[arm].state.values():
                if 'step' in value and float(value['step'].item()) != state['step']:
                    raise ValueError('Adam counter differs from completed transition count.')
        self.sampler.load_state_dict(state['sampler'])
        self.step_count = state['step']
        for name in ('rng', 'anchors', 'history', 'online_history', 'windows', 'cost', 'shared_cost'):
            setattr(self, name, copy.deepcopy(state[name]))
        self.failed = False

    def scientific_state_digest(self):
        state = self.state_dict({})
        for field in ('cost', 'shared_cost'):
            state.pop(field)
        return object_sha256(state)


def evaluate(model, dataset, device, batch_size, prediction_path=None, example_indices=None):
    """Uncapped example-weighted logit CE; retain raw logits for re-evaluation."""
    previous = model.training
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    acc = CrossEntropyAccumulator()
    all_logits, all_labels = [], []
    correct = 0
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            for batch in loader:
                labels = batch['labels'].to(device)
                logits = model(batch['input_ids'].to(device))
                acc.update(logits, labels)
                correct += int((logits.argmax(1) == labels).sum().item())
                if prediction_path is not None:
                    all_logits.append(logits.detach().cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
    finally:
        model.train(previous)
    result = dict(logit_ce=acc.mean, accuracy=correct/acc.num_examples, num_examples=acc.num_examples,
                  evaluator_version=EVALUATOR_VERSION)
    if prediction_path is not None:
        arrays = {'logits': np.concatenate(all_logits), 'labels': np.concatenate(all_labels),
                  'example_indices': np.asarray(example_indices, dtype=np.int64)}
        if len(arrays['example_indices']) != acc.num_examples:
            raise ValueError('Prediction identities do not match evaluation population.')
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if prediction_path.exists():
            with np.load(prediction_path, allow_pickle=False) as old:
                if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                    raise RuntimeError('Repeated evaluation differs from retained predictions; do not overwrite.')
        else:
            temp = prediction_path.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f, **arrays)
            os.replace(temp, prediction_path)
        result['prediction_file'] = prediction_path.name
        result['prediction_sha256'] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    return result

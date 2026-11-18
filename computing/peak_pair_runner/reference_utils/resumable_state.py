"""Complete boundary-state primitives; actual-device recovery gate remains required."""
from __future__ import annotations
import copy, random, platform
from contextlib import contextmanager
import numpy as np
import torch
from .transition_executor import snapshot_step_state, restore_step_state, snapshot_is_finite

STATE_VERSION='boundary_state_v2'
REQUIRED_IDENTITIES={'dataset_split','preprocessing','architecture','config','code'}

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state().clone(),
            'torch_cuda':[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda'])!=torch.cuda.device_count():
            raise ValueError('CUDA RNG/device topology does not match checkpoint.')
        torch.cuda.set_rng_state_all(state['torch_cuda'])

@contextmanager
def preserve_rng():
    state=capture_rng()
    try:yield
    finally:restore_rng(state)

def capture_boundary_state(*,model,optimizer,global_step,scheduler_state,sampler_state,
                           fingerprints,precision_state):
    if global_step<0:
        raise ValueError('Global step must be nonnegative.')
    if not REQUIRED_IDENTITIES.issubset(fingerprints) or not all(fingerprints[k] for k in REQUIRED_IDENTITIES):
        raise ValueError('Missing split/preprocessing/architecture/config/code fingerprint.')
    if not {'permutation','cursor','epoch','generator_state'}.issubset(sampler_state):
        raise ValueError('Sampler permutation, cursor, epoch and generator state are required.')
    permutation=sampler_state['permutation'];cursor=sampler_state['cursor']
    if cursor<0 or cursor>len(permutation):
        raise ValueError('Invalid sampler cursor.')
    snapshot=snapshot_step_state(model,optimizer)
    if not snapshot_is_finite(snapshot):
        raise FloatingPointError('Cannot save a nonfinite boundary state.')
    return copy.deepcopy(dict(state_version=STATE_VERSION,step_snapshot=snapshot,
        global_step=int(global_step),scheduler_state=scheduler_state,sampler_state=sampler_state,
        fingerprints=fingerprints,precision_state=precision_state,rng=capture_rng(),
        training_mode=model.training,environment={'torch':torch.__version__,'python':platform.python_version(),
            'device_types':sorted({str(p.device) for p in model.parameters()})}))

def restore_boundary_state(*,model,optimizer,state,expected_fingerprints):
    if state.get('state_version')!=STATE_VERSION:
        raise ValueError('Incompatible checkpoint state version.')
    if state['fingerprints']!=expected_fingerprints:
        raise ValueError('Checkpoint identities do not match the requested run.')
    devices=sorted({str(p.device) for p in model.parameters()})
    if devices!=state['environment']['device_types']:
        raise ValueError('Resume on the original device topology must be verified first.')
    restore_step_state(model,optimizer,state['step_snapshot'])
    model.train(state['training_mode']);restore_rng(state['rng'])
    return copy.deepcopy({k:state[k] for k in ['global_step','scheduler_state','sampler_state','precision_state']})

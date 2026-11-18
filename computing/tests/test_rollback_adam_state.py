"""Executable CPU test of snapshot/restore and transactional replay.
Run from this distribution: python -m unittest discover -s tests -v
This uses the archived frozen transition executor, without modifying it.
"""
import copy, unittest, sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parents[1]
SOURCE=HERE/'source'/'warmup-update-budget-acml2026'
if not SOURCE.exists():
    raise RuntimeError('The frozen source tree is required for optimizer-state tests')
sys.path.insert(0,str(SOURCE))
from publication_expansion.src.transition_executor import (snapshot_step_state,restore_step_state,transactional_step)


def recursively_equal(a,b,loc='root'):
    if torch.is_tensor(a):
        if not torch.equal(a,b): raise AssertionError(f'tensor mismatch: {loc}')
    elif isinstance(a,dict):
        assert set(a)==set(b),loc
        for k in a: recursively_equal(a[k],b[k],f'{loc}.{k}')
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),loc
        for i,(x,y) in enumerate(zip(a,b)):recursively_equal(x,y,f'{loc}[{i}]')
    else:
        if a!=b: raise AssertionError(f'non-tensor mismatch: {loc}: {a} != {b}')

class TestAdamRollback(unittest.TestCase):
    def build(self):
        torch.manual_seed(530)
        model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Tanh(),torch.nn.Linear(8,2))
        opt=torch.optim.Adam(model.parameters(),lr=0.05,betas=(0.9,0.98),eps=1e-9)
        x=torch.tensor([[1.,-1.,2.,.5],[-.3,.2,.9,-.5],[.1,.5,.3,-1.]])
        y=torch.tensor([0,1,0])
        for i in range(3):
            opt.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(model(x),y).backward();opt.step()
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x),y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        return model,opt

    def test_state_restore_after_unsafe_candidate(self):
        model,opt=self.build()
        pre=snapshot_step_state(model,opt)
        self.assertGreater(len(opt.state),0)
        self.assertTrue(all('exp_avg' in v and 'exp_avg_sq' in v and 'step' in v for v in opt.state.values()))
        opt.step()
        with self.assertRaises(AssertionError):recursively_equal(opt.state_dict(),pre.optimizer_state)
        restore_step_state(model,opt,pre)
        recursively_equal(model.state_dict(),pre.model_state)
        recursively_equal(opt.state_dict(),pre.optimizer_state)
        for name,p in model.named_parameters():
            recursively_equal(p.grad,pre.gradients[name],f'gradient_{name}')

    def test_replay_matches_clean_safe_step_including_moments_and_counters(self):
        model,opt=self.build();pre=snapshot_step_state(model,opt)
        clean_model=copy.deepcopy(model)
        clean_opt=torch.optim.Adam(clean_model.parameters(),lr=.05,betas=(.9,.98),eps=1e-9)
        clean_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        for (name,a),(name2,b) in zip(model.named_parameters(),clean_model.named_parameters()):
            self.assertEqual(name,name2)
            b.grad=a.grad.detach().clone() if a.grad is not None else None
        safe=transactional_step(model=clean_model,optimizer=clean_opt,intended_lr=1e-4,threshold=None,mode='safe')
        rescue=transactional_step(model=model,optimizer=opt,intended_lr=1e-4,threshold=1e-8,mode='rescue')
        self.assertTrue(rescue.guard_triggered)
        self.assertTrue(rescue.rolled_back)
        self.assertTrue(rescue.replayed)
        self.assertFalse(rescue.stopped)
        self.assertEqual(rescue.committed_lr_by_group,[1e-4])
        self.assertGreater(rescue.candidate_update_ratio,rescue.committed_update_ratio)
        recursively_equal(model.state_dict(),clean_model.state_dict())
        recursively_equal(opt.state_dict(),clean_opt.state_dict())
        for name,st in opt.state.items():
            self.assertIn('exp_avg',st)
            self.assertIn('exp_avg_sq',st)
            self.assertIn('step',st)
        for name,p in model.named_parameters():
            recursively_equal(p.grad,pre.gradients[name],f'restored gradient {name}')

    def test_known_scheduler_zero_at_50000(self):
        from publication_expansion.src.scheduler_semantics import stable_lr
        self.assertGreater(stable_lr(49999),0)
        self.assertEqual(stable_lr(50000),0)
        self.assertEqual(stable_lr(70000),0)

if __name__=='__main__':unittest.main()

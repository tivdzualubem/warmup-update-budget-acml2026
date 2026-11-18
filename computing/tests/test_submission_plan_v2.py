"""Regression counterexamples and finite-state contracts requested in the PDF."""
from pathlib import Path
import sys, unittest, copy, ast, json, random, io
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from future_runs.evaluation import CrossEntropyAccumulator
from future_runs.transition_executor import (transactional_step, snapshot_step_state,
    restore_step_state, state_is_finite, snapshot_is_finite)
from future_runs.resumable_state import capture_boundary_state,restore_boundary_state,preserve_rng,capture_rng

def equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

def fixture(lr=1.):
    model=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad():model.weight.fill_(1.)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.98),eps=1e-9)
    model.weight.grad=torch.ones_like(model.weight)
    return model,optimizer

class EvaluatorV2(unittest.TestCase):
    def test_full_v2_evaluator_uses_uncapped_weighted_ce_and_restores_mode(self):
        from future_runs.scratch_evaluator import evaluate
        class Identity(torch.nn.Module):
            def forward(self,x):return x
        model=Identity();model.train()
        logits=torch.tensor([[0.,0.]]*32+[[0.,10.]],dtype=torch.float64);labels=torch.zeros(33,dtype=torch.long)
        loader=[{'input_ids':logits[:32],'labels':labels[:32]},{'input_ids':logits[32:],'labels':labels[32:]}]
        result=evaluate(model,loader,'cpu',2)
        self.assertTrue(model.training);self.assertEqual(result['num_examples'],33)
        self.assertAlmostEqual(result['loss'],float(F.cross_entropy(logits,labels)),places=13)
        self.assertEqual(result['loss'],result['nll']);self.assertAlmostEqual(result['accuracy'],32/33)
        extreme=evaluate(model,[{'input_ids':torch.tensor([[1000.,-1000.]]),'labels':torch.tensor([1])}],'cpu',2)
        self.assertEqual(extreme['nll'],2000.)

    def test_extreme_logits_have_uncapped_cross_entropy(self):
        logits=torch.tensor([[1000.,-1000.]],dtype=torch.float64)
        labels=torch.tensor([1]);a=CrossEntropyAccumulator();a.update(logits,labels)
        self.assertEqual(a.mean,2000.)
        clipped=-np.log(np.clip(torch.softmax(logits,1).numpy()[0,1],1e-12,1.))
        self.assertAlmostEqual(clipped,27.631021115928547)

    def test_partial_batch_is_weighted_by_examples_and_partition_invariant(self):
        # 32 near-zero losses plus one CE=10: a deliberately uneven batch.
        logits=torch.tensor([[20.,0.]]*32+[[0.,10.]],dtype=torch.float64)
        labels=torch.zeros(33,dtype=torch.long)
        a=CrossEntropyAccumulator();a.update(logits[:32],labels[:32]);a.update(logits[32:],labels[32:])
        b=CrossEntropyAccumulator();b.update(logits,labels)
        self.assertAlmostEqual(a.mean,b.mean,places=14)
        self.assertEqual(a.num_examples,33)
        old=float((F.cross_entropy(logits[:32],labels[:32])+F.cross_entropy(logits[32:],labels[32:]))/2)
        self.assertGreater(old,4.9);self.assertLess(a.mean,.31)

    def test_invalid_logits_and_empty_evaluation_fail_explicitly(self):
        a=CrossEntropyAccumulator()
        with self.assertRaises(ValueError):_ = a.mean
        with self.assertRaises(FloatingPointError):a.update(torch.tensor([[float('nan'),0.]]),torch.tensor([0]))

    def test_archived_evaluator_counterexamples_are_preserved(self):
        p=ROOT/'source/warmup-update-budget-acml2026/publication_expansion/src/scratch_job_runner.py'
        tree=ast.parse(p.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        ns={'torch':torch,'F':F,'np':np}
        helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='macro_f1']
        exec(compile(ast.Module(body=helpers+[node],type_ignores=[]),str(p),'exec'),ns)
        class Identity(torch.nn.Module):
            def forward(self,x):return x
        loader=[{'input_ids':torch.tensor([[1000.,-1000.]]),'labels':torch.tensor([1])}]
        result=ns['evaluate'](Identity(),loader,'cpu',2)
        self.assertEqual(result['loss'],2000.)
        self.assertAlmostEqual(result['nll'],27.631021,places=5)
        logits=torch.tensor([[0.,0.]]*32+[[0.,10.]],dtype=torch.float64);labels=torch.zeros(33,dtype=torch.long)
        loader=[{'input_ids':logits[:32],'labels':labels[:32]},{'input_ids':logits[32:],'labels':labels[32:]}]
        old=ns['evaluate'](Identity(),loader,'cpu',2)
        correct=CrossEntropyAccumulator();correct.update(logits,labels)
        self.assertAlmostEqual(old['loss'],5.346596,places=5)
        self.assertAlmostEqual(correct.mean,.975174,places=5)

class ExecutorV2(unittest.TestCase):
    def test_replay_ceiling_at_steps_one_one_hundred_and_one_hundred_one(self):
        for step in (1,100,101):
            model,opt=fixture();before=snapshot_step_state(model,opt)
            r=transactional_step(model=model,optimizer=opt,intended_lr=.1,threshold=.001,mode='rescue',step=step)
            if step<=100:
                self.assertTrue(r.stopped);self.assertTrue(r.rolled_back);self.assertTrue(r.replayed)
                self.assertEqual(r.failure_reason,'replay_exceeds_threshold')
                self.assertEqual(r.optimizer_step_calls,2);self.assertEqual(r.committed_transitions,0)
                self.assertIsNone(r.committed_lr_by_group);self.assertIsNone(r.committed_update_ratio)
                self.assertAlmostEqual(r.replay_update_ratio,.1,places=7)
                equal(model.state_dict(),before.model_state);equal(opt.state_dict(),before.optimizer_state)
            else:
                self.assertFalse(r.stopped);self.assertFalse(r.replayed)
                self.assertEqual(r.optimizer_step_calls,1);self.assertEqual(r.committed_transitions,1)

    def test_successful_replay_matches_clean_adam_with_warmed_moments(self):
        model,opt=fixture(lr=.01)
        for _ in range(3):opt.step()
        opt.param_groups[0]['lr']=1.
        before=snapshot_step_state(model,opt)
        clean=copy.deepcopy(model);clean_opt=torch.optim.Adam(clean.parameters(),lr=1.,betas=(.9,.98),eps=1e-9)
        clean_opt.load_state_dict(copy.deepcopy(opt.state_dict()));clean.weight.grad=model.weight.grad.clone()
        c=transactional_step(model=clean,optimizer=clean_opt,intended_lr=1e-4,threshold=None,mode='safe')
        r=transactional_step(model=model,optimizer=opt,intended_lr=1e-4,threshold=.001,mode='rescue')
        self.assertFalse(r.stopped);self.assertTrue(r.replayed);self.assertLessEqual(r.committed_update_ratio,.001)
        equal(model.state_dict(),clean.state_dict());equal(opt.state_dict(),clean_opt.state_dict())
        equal(model.weight.grad,before.gradients['weight'])

    def test_nonfinite_gradients_stop_before_update_at_all_boundaries(self):
        for step in (1,100,101):
            for value in (float('nan'),float('inf')):
                model,opt=fixture();good=snapshot_step_state(model,opt)
                model.weight.grad.fill_(value)
                r=transactional_step(model=model,optimizer=opt,intended_lr=.1,threshold=.001,mode='rescue',step=step,verified_snapshot=good)
                self.assertTrue(r.stopped);self.assertEqual(r.failure_reason,'nonfinite_gradient')
                self.assertEqual(r.optimizer_step_calls,0);self.assertTrue(state_is_finite(model,opt))
                self.assertEqual(r.candidate_lr_by_group,[])
                equal(model.state_dict(),good.model_state);equal(opt.state_dict(),good.optimizer_state)
                equal(model.weight.grad,good.gradients['weight'])

    def test_nonfinite_candidate_moments_restore_and_do_not_replay(self):
        for step in (1,100,101):
            model,opt=fixture();before=snapshot_step_state(model,opt);original=opt.step
            def corrupt():
                original();opt.state[model.weight]['exp_avg'].fill_(float('nan'))
            opt.step=corrupt
            r=transactional_step(model=model,optimizer=opt,intended_lr=1e-4,threshold=.001,mode='rescue',step=step)
            self.assertTrue(r.stopped);self.assertFalse(r.replayed)
            self.assertEqual(r.failure_reason,'nonfinite_candidate_state');self.assertEqual(r.optimizer_step_calls,1)
            equal(model.state_dict(),before.model_state);equal(opt.state_dict(),before.optimizer_state)

    def test_nonfinite_replay_is_rolled_back_once(self):
        for step in (1,100):
            model,opt=fixture();before=snapshot_step_state(model,opt);original=opt.step;calls=[]
            def corrupt_second_call():
                original();calls.append(1)
                if len(calls)==2:
                    with torch.no_grad():model.weight.fill_(float('inf'))
            opt.step=corrupt_second_call
            r=transactional_step(model=model,optimizer=opt,intended_lr=1e-4,threshold=.001,mode='rescue',step=step)
            self.assertTrue(r.stopped);self.assertEqual(r.failure_reason,'nonfinite_replay_state')
            self.assertEqual(len(calls),2);equal(model.state_dict(),before.model_state);equal(opt.state_dict(),before.optimizer_state)

    def test_invalid_snapshot_is_never_restored(self):
        model,opt=fixture();bad=snapshot_step_state(model,opt);bad.model_state['weight'].fill_(float('nan'))
        self.assertFalse(snapshot_is_finite(bad))
        with self.assertRaises(ValueError):restore_step_state(model,opt,bad)
        self.assertTrue(state_is_finite(model,opt))

    def test_nonfinite_entry_uses_verified_state_or_explicit_failure(self):
        model,opt=fixture();good=snapshot_step_state(model,opt)
        with torch.no_grad():model.weight.fill_(float('nan'))
        r=transactional_step(model=model,optimizer=opt,intended_lr=.1,threshold=None,mode='safe')
        self.assertEqual(r.failure_reason,'nonfinite_entry_state_no_verified_snapshot');self.assertEqual(r.optimizer_step_calls,0)
        r=transactional_step(model=model,optimizer=opt,intended_lr=.1,threshold=None,mode='safe',verified_snapshot=good)
        self.assertEqual(r.failure_reason,'nonfinite_entry_state');self.assertTrue(state_is_finite(model,opt))

    def test_lr_assertion_rejects_before_mutation(self):
        model,opt=fixture();before=snapshot_step_state(model,opt)
        r=transactional_step(model=model,optimizer=opt,intended_lr=1e-4,threshold=None,mode='lr_assertion')
        self.assertEqual(r.failure_reason,'lr_mismatch_before_update');self.assertEqual(r.optimizer_step_calls,0)
        self.assertEqual(r.candidate_lr_by_group,[])
        equal(model.state_dict(),before.model_state);equal(opt.state_dict(),before.optimizer_state)

class ResumeCPU(unittest.TestCase):
    def test_stochastic_forward_and_mid_permutation_resume_match_cpu(self):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Dropout(.3),torch.nn.Tanh(),torch.nn.Linear(7,2))
        opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.98),eps=1e-9)
        x=torch.arange(33,dtype=torch.float32).reshape(11,3)/33;labels=torch.arange(11)%2
        gen=torch.Generator().manual_seed(42)
        sampler={'permutation':torch.randperm(11,generator=gen),'cursor':0,'epoch':0,'generator_state':gen.get_state()}
        def advance(model,opt,state):
            if state['cursor']==11:
                gen=torch.Generator();gen.set_state(state['generator_state'])
                state['permutation']=torch.randperm(11,generator=gen);state['cursor']=0;state['epoch']+=1
                state['generator_state']=gen.get_state()
            end=min(11,state['cursor']+3);idx=state['permutation'][state['cursor']:end];state['cursor']=end
            opt.zero_grad(set_to_none=True);F.cross_entropy(model(x[idx]),labels[idx]).backward()
            # All three RNG domains are consumed in this fixture.
            random.random();np.random.random();opt.step()
        for _ in range(2):advance(model,opt,sampler)
        ids={k:'fixture_'+k for k in ['dataset_split','preprocessing','architecture','config','code']}
        checkpoint=capture_boundary_state(model=model,optimizer=opt,global_step=2,scheduler_state={'lr':1e-4},sampler_state=sampler,fingerprints=ids,precision_state={'amp':False})
        # Exercise actual serialization, including Adam state and RNG tensors.
        buf=io.BytesIO();torch.save(checkpoint,buf);buf.seek(0);loaded=torch.load(buf,weights_only=False)
        for _ in range(7):advance(model,opt,sampler)
        final_model=copy.deepcopy(model.state_dict());final_opt=copy.deepcopy(opt.state_dict());final_rng=capture_rng();final_sampler=copy.deepcopy(sampler)
        next_draws=(random.random(),np.random.random(5),torch.rand(5))
        restored=torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Dropout(.3),torch.nn.Tanh(),torch.nn.Linear(7,2))
        restored_opt=torch.optim.Adam(restored.parameters(),lr=1e-4,betas=(.9,.98),eps=1e-9)
        state=restore_boundary_state(model=restored,optimizer=restored_opt,state=loaded,expected_fingerprints=ids)
        for _ in range(7):advance(restored,restored_opt,state['sampler_state'])
        equal(restored.state_dict(),final_model);equal(restored_opt.state_dict(),final_opt);equal(state['sampler_state'],final_sampler)
        self.assertEqual(random.getstate(),final_rng['python'])
        resumed_numpy=np.random.get_state()
        self.assertEqual(resumed_numpy[0],final_rng['numpy'][0])
        self.assertTrue(np.array_equal(resumed_numpy[1],final_rng['numpy'][1]))
        self.assertEqual(resumed_numpy[2:],final_rng['numpy'][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),final_rng['torch_cpu']))
        self.assertEqual(random.random(),next_draws[0])
        self.assertTrue(np.array_equal(np.random.random(5),next_draws[1]))
        self.assertTrue(torch.equal(torch.rand(5),next_draws[2]))

    def test_evaluation_context_restores_training_rng_on_success_and_failure(self):
        for fail in (False,True):
            before=capture_rng()
            try:
                with preserve_rng():
                    random.random();np.random.random();torch.rand(5)
                    if fail:raise RuntimeError('fixture evaluation failure')
            except RuntimeError:pass
            after=capture_rng()
            self.assertEqual(before['python'],after['python'])
            self.assertEqual(before['numpy'][0],after['numpy'][0])
            self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
            self.assertEqual(before['numpy'][2:],after['numpy'][2:])
            self.assertTrue(torch.equal(before['torch_cpu'],after['torch_cpu']))

    def test_incomplete_sampler_or_identity_is_rejected(self):
        model,opt=fixture()
        with self.assertRaises(ValueError):
            capture_boundary_state(model=model,optimizer=opt,global_step=1,scheduler_state={},sampler_state={},fingerprints={},precision_state={})

if __name__=='__main__':unittest.main()

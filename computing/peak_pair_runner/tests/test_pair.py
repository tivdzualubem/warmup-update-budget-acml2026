"""CPU synthetic checks only: no downloaded dataset or research training."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pair_core as core
import peak_pair as cli

cli.core, cli.torch, cli.np = core, torch, np
core.configure_determinism()


class Data:
    def __init__(self, n=11): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return {'input_ids': torch.tensor([1+i%10, 2+i%9, 3]), 'labels': torch.tensor(i%3)}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.dropout = nn.Dropout(.3)
        self.head = nn.Linear(8, 3)
    def forward(self, x): return self.head(self.dropout(self.embedding(x).mean(1)))


def config():
    return dict(seed=42, peak_lr=1e-4, batch_size=4, warmup_steps=3,
                schedule_horizon=9, budget_steps=7, gradient_clip=1.,
                evaluation_steps=[0,1,3,5,7], retained_checkpoint_steps=[0,1,3,7],
                latest_checkpoint_every=3, online_loss_every=3, train_monitor_size=5,
                preflight_timing_steps=2)


class PairTests(unittest.TestCase):
    def test_first_anchors_and_actual_lr_then_identical_schedule(self):
        engine = core.PairEngine(Model, Data(), config())
        one, two = engine.step(), engine.step()
        self.assertEqual(len(engine.anchors['clean']), 7)
        self.assertEqual(engine.anchors['clean'], engine.anchors['peak_first'])
        self.assertEqual(one['arms']['clean']['lr_consumed'], [1e-4/3])
        self.assertEqual(one['arms']['peak_first']['lr_consumed'], [1e-4])
        self.assertEqual(two['arms']['clean']['lr_consumed'], two['arms']['peak_first']['lr_consumed'])
        self.assertNotEqual(core.model_fingerprint(engine.models['clean']), core.model_fingerprint(engine.models['peak_first']))
        self.assertGreater(one['arms']['peak_first']['candidate_and_committed_ratio'], one['arms']['clean']['candidate_and_committed_ratio'])

    def test_arm_rng_streams_match_despite_interleaving(self):
        engine = core.PairEngine(Model, Data(), config())
        for _ in range(7):
            engine.step()
            self.assertEqual(core.object_sha256(engine.rng['clean']), core.object_sha256(engine.rng['peak_first']))

    def test_serialized_resume_across_permutation_boundary_is_exact(self):
        engine = core.PairEngine(Model, Data(), config())
        for _ in range(2): engine.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            core.atomic_save(path, engine.state_dict({'test': 'identity'}))
            for _ in range(5): engine.step()
            restored = core.PairEngine(Model, Data(), config())
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), {'test': 'identity'})
            for _ in range(5): restored.step()
            self.assertEqual(engine.scientific_state_digest(), restored.scientific_state_digest())

    def test_real_medium4_architecture_synthetic_resume(self):
        from publication_expansion.src.scratch_model_adapter import build_scratch_model, get_scratch_model_config
        cfg = get_scratch_model_config('medium-4')
        self.assertEqual((cfg['d_model'], cfg['n_layers'], cfg['n_heads'], cfg['d_ff']), (128,4,8,512))
        factory = lambda: build_scratch_model(model_name='medium-4', vocab_size=16, num_classes=3, max_seq_len=3)
        engine = core.PairEngine(factory, Data(5), config())
        engine.step()
        state = engine.state_dict({})
        engine.step()
        restored = core.PairEngine(factory, Data(5), config())
        restored.load_state_dict(state, {})
        restored.step()
        self.assertEqual(engine.scientific_state_digest(), restored.scientific_state_digest())

    def test_fixed_budget_and_weighted_online_blocks(self):
        engine = core.PairEngine(Model, Data(), config())
        records = [engine.step() for _ in range(7)]
        for arm in core.ARMS:
            rows = [r for r in engine.online_history if r['arm']==arm]
            for row in rows:
                selected = records[row['first_step']-1:row['last_step']]
                expected = sum(r['arms'][arm]['online_batch_ce']*r['arms'][arm]['examples'] for r in selected)/sum(r['arms'][arm]['examples'] for r in selected)
                self.assertEqual(row['example_weighted_online_ce'], expected)
            self.assertEqual(engine.cost[arm]['positive_lr_steps'], 7)
            self.assertEqual(engine.cost[arm]['zero_lr_steps'], 0)
            self.assertEqual(engine.cost[arm]['examples'], 26)
        with self.assertRaisesRegex(RuntimeError, 'budget'): engine.step()

    def test_zero_lr_is_rejected(self):
        c = config(); c['budget_steps']=10
        engine = core.PairEngine(Model, Data(), c)
        for _ in range(8): engine.step()
        with self.assertRaisesRegex(RuntimeError, 'zero'): engine.step()

    def test_partial_failed_pair_cannot_continue_or_checkpoint(self):
        engine = core.PairEngine(Model, Data(), config())
        with torch.no_grad(): next(engine.models['peak_first'].parameters()).fill_(float('nan'))
        with self.assertRaises(FloatingPointError): engine.step()
        self.assertEqual(engine.step_count, 0)
        with self.assertRaises(RuntimeError): engine.step()
        with self.assertRaises(RuntimeError): engine.state_dict({})

    def test_resume_rejects_identity_and_adam_counter_change(self):
        engine = core.PairEngine(Model, Data(), config()); engine.step()
        state = engine.state_dict({'code': 'a'})
        other = core.PairEngine(Model, Data(), config())
        with self.assertRaises(ValueError): other.load_state_dict(state, {'code': 'b'})
        state['optimizers']['clean']['state'][0]['step'].fill_(2)
        with self.assertRaisesRegex(ValueError, 'counter'): other.load_state_dict(state, {'code': 'a'})

    def test_sampler_rejects_invalid_permutation(self):
        sampler = core.CursorSampler(11,4,42); state = sampler.state_dict()
        state['permutation'][0] = state['permutation'][1]
        with self.assertRaises(ValueError): sampler.load_state_dict(state)

    def test_checkpoint_corruption_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'
            core.save_verified_checkpoint(path,{'tensor':torch.ones(2)})
            self.assertTrue(torch.equal(core.load_verified_checkpoint(path)['tensor'],torch.ones(2)))
            with path.open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'checksum'):core.load_verified_checkpoint(path)

    def test_actual_adam_pre_hook_rejects_wrong_rate(self):
        engine = core.PairEngine(Model, Data(), config())
        optimizer = engine.optimizers['clean']
        optimizer.param_groups[0]['lr'] = 1.
        with self.assertRaisesRegex(RuntimeError, 'Consumed LR'): optimizer.step()


class EvaluationTests(unittest.TestCase):
    def test_uneven_batches_uncapped_ce_and_saved_predictions(self):
        logits = torch.tensor([[0.,0.], [0.,0.], [0.,0.], [0.,0.], [100.,-100.]])
        labels = torch.tensor([0,0,0,0,1])
        dataset = [{'input_ids': logits[i], 'labels': labels[i]} for i in range(5)]
        model = nn.Identity().train()
        before = core.object_sha256(core.capture_rng())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'predictions.npz'
            result = core.evaluate(model,dataset,'cpu',4,path,np.arange(5))
            expected = F.cross_entropy(logits.double(), labels).item()
            self.assertAlmostEqual(result['logit_ce'], expected, places=12)
            self.assertGreater(result['logit_ce'], 27.7)
            self.assertEqual(result['accuracy'], .8)
            digest = result['prediction_sha256']
            self.assertEqual(core.evaluate(model,dataset,'cpu',4,path,np.arange(5))['prediction_sha256'], digest)
            dataset[-1]['input_ids'] = torch.tensor([99.,-99.])
            with self.assertRaisesRegex(RuntimeError, 'differs'): core.evaluate(model,dataset,'cpu',4,path,np.arange(5))
        self.assertEqual(before, core.object_sha256(core.capture_rng()))
        self.assertTrue(model.training)

    def test_evaluation_does_not_change_subsequent_dropout_trajectory(self):
        engine = core.PairEngine(Model, Data(), config()); engine.step()
        state = engine.state_dict({})
        for arm in core.ARMS: core.evaluate(engine.models[arm], Data(5),'cpu',4)
        engine.step()
        restored = core.PairEngine(Model, Data(), config()); restored.load_state_dict(state,{})
        restored.step()
        self.assertEqual(engine.scientific_state_digest(), restored.scientific_state_digest())

    def test_nonfinite_eval_rejected_and_mode_rng_restored(self):
        model=nn.Identity().train()
        before=core.object_sha256(core.capture_rng())
        with self.assertRaises((FloatingPointError, ValueError)):
            core.evaluate(model,[{'input_ids':torch.tensor([float('nan'),0.]),'labels':torch.tensor(0)}],'cpu',4)
        self.assertTrue(model.training)
        self.assertEqual(before,core.object_sha256(core.capture_rng()))

    def test_test_population_gated_until_both_arms_finish(self):
        engine=core.PairEngine(Model,Data(),config())
        with self.assertRaisesRegex(RuntimeError,'fixed final'):
            cli.evaluate_pair(engine,None,None,Path('.'),include_test=True)


class InterfaceTests(unittest.TestCase):
    def test_default_describe_is_stdlib_only(self):
        # -S excludes site packages: torch/datasets cannot be imported here.
        p=subprocess.run([sys.executable,'-S','-B',str(cli.ROOT/'peak_pair.py')],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout)['positive_transitions_total'],99998)

    def test_cli_config_rejects_zero_lr_budget_and_wrong_metric(self):
        c=cli.load_config(cli.ROOT/'protocol.json')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.json'
            for key,value in [('budget_steps',50000),('primary_endpoint','accuracy'),('dataset','ag_news'),('train_monitor_size',476001)]:
                bad=copy.deepcopy(c);bad[key]=value;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):cli.load_config(path)

    def test_frozen_original_source_hashes(self): cli.check_frozen_source()

    def test_prepare_identity_mismatch_does_not_write_run_files(self):
        prepared=SimpleNamespace(train_dataset=Data(476000),validation_dataset=Data(84000),test_dataset=Data(70000),
                                 vocab={'<PAD>':0,'<UNK>':1},num_classes=14,
                                 manifest={'split':{'validation_indices_sha256': '75cc3a5c716ae024bd95de7e7d579546bef5a63038d940c7ff5d5a66aa08f3fa'}})
        c=cli.load_config(cli.ROOT/'protocol.json')
        with tempfile.TemporaryDirectory() as tmp, patch('publication_expansion.src.scratch_dataset_adapter.load_scratch_hf_dataset',return_value=prepared), patch.object(cli,'data_content_hash',side_effect=lambda d:str(len(d))):
            out=Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError,'Preflight'):
                cli.prepare(c,torch.device('cpu'),out,expected_identity={'wrong':True})
            self.assertFalse(out.exists())
            cli.prepare(c,torch.device('cpu'),out)
            (out/'environment.json').write_text('{"original":"preserve"}')
            before={p.name:p.read_bytes() for p in out.iterdir()}
            with self.assertRaisesRegex(ValueError,'Existing run metadata'):
                cli.prepare(c,torch.device('cpu'),out)
            self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})

    def test_truncated_log_tail_accounted_but_interior_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'segment_1.jsonl';path.write_text('{"event":"start"}\n{"event":')
            report=cli.aggregate_logs(tmp)
            self.assertEqual(report['truncated_final_log_lines'],1)
            self.assertEqual(report['segments_without_end'],1)
            path.write_text('{broken\n{"event":"start"}\n')
            with self.assertRaises(json.JSONDecodeError): cli.aggregate_logs(tmp)

    def test_full_synthetic_orchestration_interruption_and_resume(self):
        prepared=SimpleNamespace(train_dataset=Data(),validation_dataset=Data(5),test_dataset=Data(5))
        split=SimpleNamespace(train=np.arange(11),validation=np.arange(5))
        identity={'fixture':'synthetic-only'}
        def prepare(c,device,out,expected_identity=None):
            if expected_identity is not None and expected_identity!=identity: raise ValueError('identity mismatch')
            out.mkdir(parents=True,exist_ok=True)
            core.atomic_json(out/'protocol.json',c)
            return prepared,split,Model,identity
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli,'prepare',side_effect=prepare):
            root=Path(tmp);gate=root/'preflight.json'
            core.atomic_json(gate,{'status':'passed','resume_probe_exact':True,'identity':identity})
            cli.run_pair(config(),torch.device('cpu'),root/'full',gate)
            original=core.PairEngine.step
            def interrupt(engine):
                if engine.step_count==5: raise RuntimeError('synthetic interruption')
                return original(engine)
            with patch.object(core.PairEngine,'step',interrupt):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    cli.run_pair(config(),torch.device('cpu'),root/'resumed',gate)
            cli.run_pair(config(),torch.device('cpu'),root/'resumed',gate,resume=True)
            full=torch.load(root/'full/checkpoints/latest.pt',weights_only=False)
            resumed=torch.load(root/'resumed/checkpoints/latest.pt',weights_only=False)
            for s in (full,resumed):
                s.pop('cost');s.pop('shared_cost')
            self.assertEqual(core.object_sha256(full),core.object_sha256(resumed))
            result=json.loads((root/'resumed/result.json').read_text())
            self.assertEqual(result['fixed_positive_lr_steps_per_arm'],7)
            self.assertEqual(result['recorded_attempt_cost']['failure_events'],1)
            self.assertEqual(result['recorded_attempt_cost']['completed_pair_step_events'],9)
            self.assertEqual(len(resumed['history']),22)
            self.assertGreater(result['retained_file_bytes']['paired_checkpoints_shared'],0)
            with self.assertRaises(ValueError):cli.run_pair(config(),torch.device('cpu'),root/'resumed',gate,resume=True)


if __name__=='__main__': unittest.main(verbosity=2)

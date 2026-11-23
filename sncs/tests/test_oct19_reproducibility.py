"""Tests for the specific reproducibility defects identified on 19 October."""
from pathlib import Path
import unittest,sys,copy,json,csv,zipfile,ast,math,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'source/warmup-update-budget-acml2026'
sys.path[:0]=[str(ROOT),str(SOURCE)]
from analysis.reference_scheduler_semantics import scratch_transition_plan,pretrained_transition_plan,resolve_transition
from publication_expansion.src.transition_executor import snapshot_step_state,restore_step_state,transactional_step
from publication_expansion.src.anchor import capture_rng_state,restore_rng_state,object_sha256
from test_rollback_adam_state import recursively_equal
import test_rollback_adam_state as rollback_checks

def read(name):
 with (ROOT/'tables'/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

class TestOctober19(unittest.TestCase):
 def test_threshold_exports_match_original_results(self):
  rows=read('guard_first_transition.csv');self.assertEqual(len(rows),80)
  with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
   index={}
   for n in z.namelist():
    if n.endswith('/result.json') and '/results/' in n:
     r=json.loads(z.read(n))
     if 'condition' in r:index[r['dataset'],r.get('model','deit-tiny' if r['dataset']=='cifar10' else 'distilbert'),int(r['seed']),r['condition']]=r
   for row in rows:
    self.assertNotEqual(row['threshold'],'')
    r=index[row['dataset'],row['model'],int(row['seed']),row['condition']]
    self.assertEqual(float(row['threshold']),r['threshold'])

 def test_sensitivity_rebuild_is_deterministic_and_source_calibrated(self):
  import tempfile
  from reanalyse import run
  expected={name:(ROOT/'tables'/name).read_bytes() for name in ['guard_first_transition.csv','retrospective_kappa_sensitivity.csv','retrospective_kappa_summary.csv']}
  with tempfile.TemporaryDirectory() as directory:
   run(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip',Path(directory))
   for name,data in expected.items():self.assertEqual((Path(directory)/name).read_bytes(),data)
  rows=read('retrospective_kappa_sensitivity.csv');self.assertEqual(len(rows),400)
  main=[r for r in rows if float(r['kappa'])==10]
  self.assertEqual(sum(r['detected']=='True' for r in main),35)
  self.assertTrue(all(int(r['seed']) not in map(int,r['calibration_seeds'].split(';')) for r in rows))
  self.assertTrue(all(float(r['threshold'])==float(r['kappa'])*float(r['safe_calibration_max']) for r in rows))

 def test_reference_matches_actual_runner_specifications_and_guard_boundary(self):
  # Extract pure specification functions from the original runner AST, avoiding dataset imports.
  for family,file,function in [('scratch','scratch_job_runner.py','step_semantics'),('pretrained','pretrained_job_runner.py','transition_spec')]:
   module=ast.parse((SOURCE/'publication_expansion/src'/file).read_text())
   definition=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==function)
   env=dict(GUARD_WINDOW=100,WARMUP_STEPS=8000,BASE_LR=1e-4,TOTAL_STEPS=50000,LEGACY_INITIAL_LR=1.,Optional=__import__('typing').Optional)
   from analysis.reference_scheduler_semantics import legacy_transformer_lr,stable_lr
   env.update(legacy_transformer_lr=legacy_transformer_lr,stable_lr=stable_lr)
   exec(compile(ast.Module(body=[definition],type_ignores=[]),file,'exec'),env)
   conditions=['buggy_failfast_guard','buggy_rescue_guard','single_bad_first_step_then_stable'] if family=='scratch' else ['update_guard_failfast','update_guard_rescue','single_bad_then_correct']
   for cond in conditions:
    for t in [1,2,100,101,300]:
     entry=1. if family=='scratch' else 2e-5
     if family=='scratch':
      plan=scratch_transition_plan(condition=cond,t=t,entry_lr=entry,d_model=128)
      spec=env[function](condition=cond,step=t,d_model=128,threshold=.0001)
      target=spec['executor_intended_lr']
     else:
      plan=pretrained_transition_plan(condition=cond,t=t,entry_lr=entry,peak_lr=2e-5,total_steps=1000)
      spec=env[function](condition=cond,step=t,intended_lr=plan.intended_lr,threshold=.0001)
      target=plan.intended_lr
     self.assertEqual(plan.guard_mode in ('failfast','rescue'),spec['mode'] in ('failfast','rescue'))
     for triggered in (False,True):
      outcome=resolve_transition(plan,guard_triggered=triggered)
      self.assertEqual(outcome.actual_exit_lr,target)
      self.assertEqual(outcome.replay,spec['mode']=='rescue' and triggered)
      self.assertEqual(outcome.stop,spec['mode']=='failfast' and triggered)

 def test_reference_matches_all_saved_transition_records(self):
  count=0
  with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
   for row in read('all_condition_results.csv'):
    pre=row['result_path'].rsplit('/',1)[0]+'/'
    trace=json.loads(z.read(pre+'trace.json'));j=json.loads(z.read(pre+'job_manifest.json'))
    for t in trace:
     if row['family']=='scratch':plan=scratch_transition_plan(condition=row['condition'],t=int(t['step']),entry_lr=t['entry_lr'],d_model=128 if row['model']=='medium-4' else 256)
     else:plan=pretrained_transition_plan(condition=row['condition'],t=int(t['step']),entry_lr=t['entry_lr'],peak_lr=j['peak_lr'],total_steps=j['total_optimizer_steps'])
     r=resolve_transition(plan,guard_triggered=t['guard_triggered'])
     self.assertAlmostEqual(r.committed_lr_consumed,t['committed_lr_consumed'],places=15)
     self.assertAlmostEqual(r.actual_exit_lr,t['actual_exit_lr'],places=15)
     count+=1
  self.assertGreater(count,50000)

 def test_fixed_gradient_adam_rescue_preserves_rng_and_matches_clean_step(self):
  random.seed(420);np.random.seed(420);torch.manual_seed(420)
  model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.25),torch.nn.Tanh(),torch.nn.Linear(8,2))
  opt=torch.optim.Adam(model.parameters(),lr=.05,betas=(.9,.98),eps=1e-9)
  x=torch.tensor([[1.,-1.,2.,.5],[-.3,.2,.9,-.5],[.1,.5,.3,-1.]])
  y=torch.tensor([0,1,0])
  for _ in range(3):
   opt.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(model(x),y).backward();opt.step()
  opt.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(model(x),y).backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  # Gradients were computed once before the candidate; no second stochastic forward is replayed.
  clean=copy.deepcopy(model);clean_opt=torch.optim.Adam(clean.parameters(),lr=.05,betas=(.9,.98),eps=1e-9)
  clean_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
  for a,b in zip(model.parameters(),clean.parameters()):b.grad=None if a.grad is None else a.grad.clone()
  before=capture_rng_state();state=snapshot_step_state(model,opt)
  unsafe=transactional_step(model=model,optimizer=opt,intended_lr=1e-4,threshold=1e-8,mode='rescue')
  self.assertTrue(unsafe.replayed);self.assertEqual(object_sha256(before),object_sha256(capture_rng_state()))
  transactional_step(model=clean,optimizer=clean_opt,intended_lr=1e-4,threshold=None,mode='safe')
  self.assertEqual(object_sha256(before),object_sha256(capture_rng_state()))
  recursively_equal(model.state_dict(),clean.state_dict());recursively_equal(opt.state_dict(),clean_opt.state_dict())
  for name,p in model.named_parameters():recursively_equal(p.grad,state.gradients[name])

 def test_explicit_random_state_restore_for_python_numpy_and_torch(self):
  initial=capture_rng_state()
  expected=(random.random(),np.random.rand(3),torch.rand(3))
  random.random();np.random.rand(10);torch.rand(10)
  restore_rng_state(initial)
  actual=(random.random(),np.random.rand(3),torch.rand(3))
  self.assertEqual(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1]);self.assertTrue(torch.equal(actual[2],expected[2]))

 def test_exploratory_contrasts_have_all_five_pairs_and_no_equivalence_claim(self):
  rows=read('paired_per_seed.csv');new=[r for r in rows if r['contrast'].startswith('exploratory')]
  self.assertEqual(len(new),40)
  for r in new:
   self.assertEqual(r['analysis_status'],'exploratory')
   self.assertAlmostEqual(float(r['accuracy_difference_pp']),100*(float(r['control_accuracy'])-float(r['treatment_accuracy'])))
   self.assertTrue(r['control_source_path'].endswith('/result.json'))
  from scipy.stats import binomtest
  self.assertEqual(binomtest(5,5,.5).pvalue,.0625)

 def test_all_display_table_sources_exist_and_prior_pairs_are_preserved(self):
  manifest=json.loads((ROOT/'tables/manuscript_table_sources.json').read_text())
  with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
   names=set(z.namelist())
   for r in manifest['records']:
    for path in r['source_files']:self.assertTrue(path in names or (ROOT/path).exists(),path)
  with (ROOT/'evidence/previous_revision/paired_per_seed.csv').open(newline='',encoding='utf-8') as f:old=list(csv.DictReader(f))
  current=read('paired_per_seed.csv');self.assertEqual(len(old),60)
  for r in old:
   match=next(n for n in current if all(n[k]==r[k] for k in ['dataset','model','seed','contrast']))
   for field,value in r.items():self.assertEqual(match[field],value)

if __name__=='__main__':unittest.main()

"""Scientific record/provenance checks, independent of model-training dependencies."""
from pathlib import Path
import unittest,csv,json,zipfile,hashlib,collections,math,re,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/warmup-update-budget-acml2026'))
from publication_expansion.src.scheduler_semantics import stable_lr,legacy_transformer_lr,pretrained_linear_lr

def rows(name):
 with (ROOT/'tables'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

class RevisionEvidence(unittest.TestCase):
 def test_01_saved_primary_metrics_and_validation_selection(self):
  index={r['result_path']:r for r in rows('all_condition_results.csv')};trained=0
  with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
   for path,row in index.items():
    result=json.loads(z.read(path));test=result.get('test_metrics') or {}
    metric='accuracy' if row['family']=='scratch' or row['dataset']=='cifar10' else 'entity_f1' if row['dataset']=='wnut_17' else 'macro_f1'
    self.assertEqual(row['metric'],metric)
    if test:self.assertAlmostEqual(float(row['test_value']),test[metric],places=14)
    if row['family']=='pretrained' and test:
     trained+=1;history=json.loads(z.read(path.rsplit('/',1)[0]+'/history.json'))
     best=max(history,key=lambda h:h['validation'][metric])
     self.assertEqual(result['primary_metric'],metric)
     self.assertEqual(result['best_epoch'],best['epoch'])
     self.assertAlmostEqual(result['best_validation_primary'],best['validation'][metric],places=14)
  self.assertEqual(len(index),240);self.assertEqual(trained,80)
  selection=rows('pretrained_selection_and_primary_metric.csv');self.assertEqual(len(selection),80)
  for row in selection:
   self.assertEqual(row['selection_metric'],row['test_primary_metric'])
   self.assertAlmostEqual(float(row['validation_primary_at_selected_epoch']),float(row['same_validation_metric']),places=14)
  for row in rows('condition_mean_sd.csv'):
   if row['family']=='pretrained' and row['dataset'] in ('ag_news','dbpedia_14'):
    self.assertEqual(row['metric'],'macro_f1')
    a=[float(r['test_macro_f1']) for r in index.values() if r['family']==row['family'] and r['dataset']==row['dataset'] and r['condition']==row['condition'] and r['test_macro_f1']]
    self.assertAlmostEqual(float(row['mean']),sum(a)/len(a),places=14)

 def test_02_condition_specific_positive_and_zero_budgets(self):
  data=rows('condition_training_and_budget_per_run.csv');self.assertEqual(len(data),240)
  groups=collections.Counter((r['family'],r['dataset'],r['model'],r['condition']) for r in data)
  self.assertEqual(len(groups),48);self.assertEqual(set(groups.values()),{5})
  self.assertEqual(len(rows('condition_training_and_budget_summary.csv')),48)
  with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
   for r in data:
    n=int(r['committed_optimizer_steps']);pos=int(r['positive_lr_committed_steps']);zero=int(r['zero_lr_committed_steps'])
    self.assertEqual(pos+zero,n)
    self.assertEqual(n+int(r['rejected_candidate_replay_calls']),int(r['optimizer_step_calls_including_replays']))
    if r['family']=='scratch':
     expected=min(n,49999) if r['condition'] in ('stable_with_guard','single_bad_first_step_then_stable') else n
    else:
     expected=n-int(n>0 and r['condition'] in ('safe_correct_order','single_bad_then_correct'))
    self.assertEqual(pos,expected)
    prefix=r['result_path'].rsplit('/',1)[0]+'/'
    d=json.loads(z.read(r['result_path']));self.assertEqual(n,d.get('global_steps') or 0)
    train=d.get('train_metrics') or {}
    for field,source in [('selected_checkpoint_train_loss','loss'),('selected_checkpoint_train_accuracy','accuracy')]:
     if source in train:self.assertAlmostEqual(float(r[field]),train[source],places=14)
    trace=json.loads(z.read(prefix+'trace.json'))
    expected_replays=sum(bool(t.get('guard_triggered')) for t in trace) if r['condition'] in ('buggy_rescue_guard','update_guard_rescue') else 0
    self.assertEqual(int(r['rejected_candidate_replay_calls']),expected_replays)

 def test_03_original_warmup_evidence(self):
  base=ROOT/'evidence/historical_warmup';v=json.loads((base/'warmup_verification.json').read_text())
  self.assertEqual(v['historical_warmup_steps'],4000);self.assertEqual(v['trainer_default_steps'],8000)
  for source in v['sources']:
   self.assertEqual(hashlib.sha256((ROOT/source['local_evidence_path']).read_bytes()).hexdigest(),source['sha256'])
  for dataset in ('ag_news','dbpedia_14'):
   runner=(base/dataset/'run_initial_lr_sweep.py').read_text()
   self.assertRegex(runner,r'"warmup_steps"\s*:\s*4000')
   self.assertIn('warmup_steps=settings["warmup_steps"]',runner)
  ag=[t for t in v['trace_checks'] if t['dataset']=='ag_news' and 'rows_checked' in t];db=[t for t in v['trace_checks'] if t['dataset']=='dbpedia_14']
  self.assertEqual(len(ag),30);self.assertEqual(len(db),30)
  self.assertEqual(sum(t['rows_checked'] for t in ag),1268824)
  self.assertTrue(all(t['peak_step']==4000 and t['max_absolute_lr_error']<1e-15 for t in ag))
  self.assertTrue(all(abs(t['recovered_warmup_from_first_installed_lr']-4000)<1e-8 for t in db))
  for f in (base/'ag_news/trace_excerpts').glob('*.json'):
   width=128 if f.name.startswith('medium-4') else 256
   for row in json.loads(f.read_text()):self.assertAlmostEqual(row['lr_after_step'],legacy_transformer_lr(row['step'],width,4000),places=15)

 def test_04_configuration_derived_schedule_plot(self):
  cfg=json.loads((ROOT/'source/warmup-update-budget-acml2026/publication_expansion/configs/condition_semantics_v1.json').read_text())
  widths=cfg['scratch_schedules']['legacy_transformer']['d_model'];self.assertEqual(widths,{'medium-4':128,'large-6':256})
  curves=rows('schedule_curves_from_configuration.csv');steps={int(r['step']) for r in curves}
  self.assertTrue({1,2,7999,8000,8001,49999,50000,50001}.issubset(steps))
  for r in curves:
   t=int(r['step']);self.assertAlmostEqual(float(r['stable_lr']),stable_lr(t),places=15)
   for model,width in widths.items():self.assertAlmostEqual(float(r['legacy_'+model+'_lr']),legacy_transformer_lr(t,width),places=15)

 def test_05_guard_misses_retained(self):
  first=[r for r in rows('guard_first_transition.csv') if r['condition'] in ('buggy_failfast_guard','update_guard_failfast')]
  self.assertEqual(len(first),40);self.assertEqual(sum(r['guard_triggered']=='True' for r in first),35)
  wnut=[r for r in first if r['dataset']=='wnut_17'];self.assertEqual(len(wnut),5)
  self.assertTrue(all(r['guard_triggered']=='False' for r in wnut))

 def test_06_original_data_training_code_and_contrasts_preserved(self):
  expected=json.loads((ROOT/'tests/expected_preserved_hashes.json').read_text())
  self.assertGreater(len(expected['files']),50)
  for name,sha in expected['files'].items():
   self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),sha,name)

 def test_07_documents_consistent(self):
  from pypdf import PdfReader
  md=(ROOT/'manuscript/Research_Manuscript.md').read_text(encoding='utf-8')
  self.assertNotIn('{{',md);self.assertNotIn('Primary task measure is accuracy except',md)
  self.assertIn('selected by maximum validation macro-F1',md)
  self.assertIn('AG News and DBpedia use macro-F1',md)
  self.assertEqual(md.count('| Macro-F1 (%) |'),2)
  for target in re.findall(r'!\[.*?\]\((.*?)\)',md):self.assertTrue((ROOT/'manuscript'/target).exists(),target)
  for name in ('Research_Manuscript','All_Seed_Results_and_Provenance'):
   reader=PdfReader(ROOT/'manuscript'/(name+'.pdf'));text=' '.join(p.extract_text() for p in reader.pages)
   self.assertIn('macro-F1',text);self.assertNotIn('\ufffd',text)
   self.assertNotIn('Primary task measure is accuracy except',text)
   self.assertGreater(len(text),3000)

if __name__=='__main__':unittest.main(verbosity=2)

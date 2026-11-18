"""Independent shared-step, mathematical and requirement-coverage checks."""
from pathlib import Path
import csv,json,zipfile,collections,unittest,sys,tempfile,math,re,statistics
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def rows(name):return list(csv.DictReader((ROOT/'tables'/name).open(encoding='utf-8-sig')))

class SubmissionAnalysis(unittest.TestCase):
    def test_every_pair_matches_raw_history_and_has_no_checkpoint_reselection(self):
        detail=rows('matched_step_per_seed.csv');self.assertEqual(len(detail),475)
        with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
            cache={}
            for r in detail:
                values=[]
                for arm in ['a','b']:
                    path=r[f'arm_{arm}_source']
                    if path not in cache:cache[path]={int(h['global_step']):h for h in json.loads(z.read(path))}
                    h=cache[path][int(r['global_step'])]
                    val=h['validation']['accuracy'];values.append(val)
                    self.assertEqual(float(r[f'arm_{arm}_validation_accuracy']),val)
                    self.assertEqual(float(r[f'arm_{arm}_archived_validation_loss']),h['validation']['loss'])
                self.assertAlmostEqual(float(r['accuracy_difference_pp']),100*(values[0]-values[1]),places=12)
                self.assertEqual(r['checkpoint_reselection'],'False')

    def test_all_and_only_common_steps_across_ten_trajectories_are_exported(self):
        from analyse_matched_steps import PAIRS,SEEDS
        coverage=rows('matched_step_coverage.csv');self.assertEqual(len(coverage),12)
        index=collections.defaultdict(dict)
        for r in rows('all_condition_results.csv'):
            if r['family']=='scratch':index[(r['dataset'],r['model'],r['condition'])][int(r['seed'])]=r['result_path']
        with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
            for r in coverage:
                sets=[]
                for c in (r['arm_a'],r['arm_b']):
                    for seed in SEEDS:
                        p=index[r['dataset'],r['model'],c][seed].rsplit('/',1)[0]+'/history.json'
                        sets.append({int(h['global_step']) for h in json.loads(z.read(p))})
                self.assertEqual(set(map(int,r['shared_logged_steps'].split(';'))),set.intersection(*sets))
                self.assertEqual(r['interpolation'],'False');self.assertEqual(r['carry_forward'],'False')

    def test_shared_step_regeneration_is_deterministic(self):
        from analyse_matched_steps import build
        names=['matched_step_per_seed.csv','matched_step_summary.csv','matched_step_coverage.csv','archived_failure_and_endpoint_ledger.csv','archived_evaluator_provenance.csv','validation_batch_layout.csv','matched_step_audit.json']
        with tempfile.TemporaryDirectory() as directory:
            build(directory)
            for name in names:self.assertEqual((Path(directory)/name).read_bytes(),(ROOT/'tables'/name).read_bytes(),name)
        summary=rows('matched_step_summary.csv');self.assertEqual(len(summary),95)
        for r in summary:
            self.assertEqual(int(r['n']),5)
            radius=2.7764451051977987*float(r['sample_sd_difference_pp'])/math.sqrt(5)
            self.assertAlmostEqual(float(r['descriptive_ci95_upper_pp'])-float(r['mean_accuracy_difference_pp']),radius,places=11)

    def test_first_adam_lr_changes_weights_not_initial_moments_and_obeys_bounds(self):
        gradient=torch.tensor([1.,-.2,0.,1e-10],dtype=torch.float64)
        states=[];weights=[]
        for lr in (1.,1.25e-8):
            p=torch.nn.Parameter(torch.ones(4,dtype=torch.float64));p.grad=gradient.clone()
            opt=torch.optim.Adam([p],lr=lr,betas=(.9,.98),eps=1e-9,weight_decay=0)
            before=p.detach().clone();opt.step();states.append(opt.state[p]);weights.append(p.detach().clone())
            expected=-lr*gradient/(gradient.abs()+1e-9)
            torch.testing.assert_close(p.detach()-before,expected,rtol=1e-7,atol=1e-15)
            self.assertLessEqual(float(torch.max(torch.abs(p.detach()-before))),lr+1e-15)
            self.assertLessEqual(float(torch.linalg.vector_norm(p.detach()-before)),lr*math.sqrt(4)+1e-15)
        self.assertTrue(torch.equal(states[0]['exp_avg'],states[1]['exp_avg']))
        self.assertTrue(torch.equal(states[0]['exp_avg_sq'],states[1]['exp_avg_sq']))
        self.assertFalse(torch.equal(weights[0],weights[1]))

    def test_figure_reading_order_title_and_scientific_scope(self):
        main=(ROOT/'manuscript/Research_Manuscript.md').read_text(encoding='utf-8')
        prior=(ROOT/'evidence/previous_revision/Research_Manuscript_2026-10-19.md').read_text(encoding='utf-8')
        self.assertEqual(main.splitlines()[0],prior.splitlines()[0])
        files=re.findall(r'!\[.*?\]\((.*?)\)',main)
        self.assertEqual([int(re.search(r'fig(\d+)_',p)[1]) for p in files],[1,2,3])
        for p in files:self.assertTrue((ROOT/'manuscript'/p).exists())
        for phrase in ['clipped-probability NLL','batch-mean CE','95 shared-step','180 contain trained test endpoints','bad/no-lag legacy condition is missing','actual API and version','not a paired superiority test']:
            self.assertIn(phrase,main)

    def test_requirements_and_endpoint_failure_ledger_keep_conditional_work_explicit(self):
        ledger=rows('submission_plan_requirements.csv')
        self.assertGreaterEqual(len(ledger),65);self.assertEqual(len(ledger),len({r['id'] for r in ledger}))
        self.assertEqual({int(r['page']) for r in ledger},{1,2,3,4,5})
        self.assertTrue(any(r['status']=='conditional_protocol_choices' for r in ledger))
        results=rows('archived_failure_and_endpoint_ledger.csv')
        self.assertEqual(len(results),240)
        self.assertEqual(sum(r['trained_test_endpoint']=='True' for r in results),180)
        self.assertTrue(all(r['selected_checkpoint_replaced']=='False' for r in results))

if __name__=='__main__':unittest.main()

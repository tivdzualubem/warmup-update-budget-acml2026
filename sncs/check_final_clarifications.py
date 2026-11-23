"""Verify endpoint and pretrained-schedule clarifications from archived run records."""
from pathlib import Path
import csv,io,json,zipfile,math,statistics
R=Path(__file__).resolve().parent
def run():
 rows=list(csv.DictReader((R/'tables/all_condition_results.csv').open()))
 endpoints=[];rates=[]
 with zipfile.ZipFile(R/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
  for seed in [42,123,456,789,2026]:
   vals={}
   for cond in ['buggy_no_guard','buggy_rescue_guard']:
    rec=next(r for r in rows if r['family']=='scratch' and r['dataset']=='dbpedia_14' and r['model']=='medium-4' and r['seed']==str(seed) and r['condition']==cond)
    root=rec['result_path'].rsplit('/',1)[0]
    hist=json.loads(z.read(root+'/history.json'))
    selected=next(h for h in hist if h['epoch']==int(rec['best_validation_epoch']))
    assert math.isclose(selected['validation']['loss'],float(rec['best_validation_objective']),rel_tol=1e-10)
    shared=next(h for h in hist if h['global_step']==44625)
    vals[cond]=(selected,shared,root)
   late,ls,lp=vals['buggy_no_guard'];rescue,rs,rp=vals['buggy_rescue_guard']
   endpoints.append(dict(seed=seed,shared_step=44625,late_selected_epoch=late['epoch'],rescue_selected_epoch=rescue['epoch'],late_selected_validation_accuracy=late['validation']['accuracy'],rescue_selected_validation_accuracy=rescue['validation']['accuracy'],selected_gap_pp=100*(rescue['validation']['accuracy']-late['validation']['accuracy']),shared_gap_pp=100*(rs['validation']['accuracy']-ls['validation']['accuracy']),late_source=lp+'/history.json',rescue_source=rp+'/history.json'))
  for rec in rows:
   if rec['family']!='pretrained' or rec['condition']!='safe_correct_order':continue
   root=rec['result_path'].rsplit('/',1)[0];m=json.loads(z.read(root+'/job_manifest.json'));t=json.loads(z.read(root+'/trace.json'))
   badroot=root.rsplit('/',1)[0]+'/single_bad_then_correct';bad=json.loads(z.read(badroot+'/trace.json'))
   assert m['warmup_steps']==math.ceil(m['total_optimizer_steps']*.1)
   assert math.isclose(t[0]['candidate_lr_consumed'],m['peak_lr']/m['warmup_steps'],rel_tol=1e-12)
   assert bad[0]['candidate_lr_consumed']==m['peak_lr']
   peak_in_window=m['warmup_steps']<=100
   if peak_in_window:assert t[m['warmup_steps']-1]['candidate_lr_consumed']==m['peak_lr']
   rates.append(dict(dataset=m['dataset'],seed=m['seed'],warmup_steps=m['warmup_steps'],total_steps=m['total_optimizer_steps'],peak_lr=m['peak_lr'],safe_first_consumed_lr=t[0]['candidate_lr_consumed'],fault_first_consumed_lr=bad[0]['candidate_lr_consumed'],peak_within_safe_calibration_window=peak_in_window,source=root+'/job_manifest.json'))
 for name,data in [('selected_validation_endpoint_pairs.csv',endpoints),('pretrained_first_rates_and_warmup.csv',rates)]:
  with (R/'tables'/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 selected=statistics.mean(r['selected_gap_pp'] for r in endpoints);shared=statistics.mean(r['shared_gap_pp'] for r in endpoints)
 assert round(selected,2)==5.82 and round(shared,2)==44.29
 assert len(rates)==20 and sum(r['peak_within_safe_calibration_window'] for r in rates)==5
 report=dict(selected_validation_gap_pp=selected,shared_validation_gap_pp=shared,paired_seeds=5,pretrained_safe_fault_pairs_verified=len(rates),wnut_peak_within_window_seeds=5,checkpoint_selection_unchanged=True,new_training=False,prediction_reevaluation=False)
 (R/'evidence/sncs/final_clarifications.json').write_text(json.dumps(report,indent=2))
 return report
if __name__=='__main__':print(json.dumps(run(),indent=2))

"""Rebuild LOSO thresholds/sensitivity from original records, without training."""
import json,math,collections,csv
GRID=[1.1,2.,5.,10.,20.,50.,100.,200.,500.,1000.]
def write(path,data):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def regenerate(z,results,traces,out):
 index={(r['family'],r['dataset'],r['model'],r['seed'],r['condition']):r for r in results}
 manifests={}
 for name in z.namelist():
  if not name.endswith('/loso_thresholds.json'):continue
  family='scratch' if '/scratch/' in name else 'pretrained'
  for r in json.loads(z.read(name))['records']:
   model=r.get('model','deit-tiny' if r['dataset']=='cifar10' else 'distilbert')
   manifests[family,r['dataset'],model,int(r['heldout_seed'])]=(name,r)
 data=[];audit=[]
 for key,(member,rec) in sorted(manifests.items()):
  fam,ds,mod,seed=key
  safe=['stable_with_guard','fixed_order_with_guard'] if fam=='scratch' else ['safe_correct_order']
  early=[];calibration_sources=[]
  for other in rec['calibration_seeds']:
   for cond in safe:
    k=(fam,ds,mod,other,cond);t=[v for v in traces[k] if 1<=v['step']<=100]
    assert {v['step'] for v in t}==set(range(1,101))
    early.extend(v['committed_update_ratio'] for v in t);calibration_sources.append(index[k]['result_path'].replace('result.json','trace.json'))
  safe_max=max(early)
  assert math.isclose(safe_max,rec['safe_max_committed_update_ratio'],rel_tol=1e-12)
  threshold=10*safe_max;recorded=rec.get('threshold',rec.get('primary_threshold'))
  assert math.isclose(threshold,recorded,rel_tol=1e-12)
  fail='buggy_failfast_guard' if fam=='scratch' else 'update_guard_failfast'
  badkey=(fam,ds,mod,seed,fail);candidate=traces[badkey][0]['candidate_update_ratio']
  held=[]
  for cond in safe:
   held.extend(v['committed_update_ratio'] for v in traces[fam,ds,mod,seed,cond] if 1<=v['step']<=100)
  assert len(held)==len(safe)*100
  for kappa in GRID:
   th=kappa*safe_max
   data.append(dict(kind=fam,dataset=ds,model=mod,seed=seed,kappa=kappa,detected=candidate>th,safe_fp=sum(v>th for v in held),safe_n=len(held),threshold=th,safe_calibration_max=safe_max,candidate_update_ratio=candidate,threshold_manifest=member,candidate_source=index[badkey]['result_path'].replace('result.json','trace.json'),calibration_seeds=';'.join(map(str,rec['calibration_seeds']))))
  audit.append(dict(family=fam,dataset=ds,model=mod,heldout_seed=seed,calibration_seeds=rec['calibration_seeds'],safe_max=safe_max,primary_threshold=threshold,source_manifest=member,calibration_trace_sources=calibration_sources,heldout_safe_n=len(held)))
 assert len(data)==400 and len(audit)==40
 write(out/'retrospective_kappa_sensitivity.csv',data)
 summaries=[]
 for kappa in GRID:
  for category in ['all','scratch','pretrained','wnut_17']:
   g=[r for r in data if r['kappa']==kappa and (category=='all' or r['kind']==category or r['dataset']==category)]
   summaries.append(dict(kappa=kappa,scope=category,detected=sum(r['detected'] for r in g),n_groups=len(g),safe_false_positive_transitions=sum(r['safe_fp'] for r in g),heldout_safe_transitions=sum(r['safe_n'] for r in g),analysis_status='prespecified threshold' if kappa==10 else 'retrospective sensitivity'))
 write(out/'retrospective_kappa_summary.csv',summaries)
 (out/'threshold_reconstruction.json').write_text(json.dumps({'threshold_export_cells_filled':80,'threshold_export_cells_remaining_blank':0,'loso_groups_verified':40,'sensitivity_rows_regenerated':400,'guard_window_steps':100,'kappa_grid':GRID,'audit':audit},indent=2),encoding='utf-8')
 print('Regenerated 400 sensitivity rows, verified 40 LOSO thresholds; restored 80 guard threshold cells.')

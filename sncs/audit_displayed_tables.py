"""Independently compare every displayed numeric table cell with source records."""
from pathlib import Path
import json,csv,zipfile,statistics,math,re
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parent
SEEDS=[42,123,456,789,2026]
SETTINGS=[(d,m) for d in ['ag_news','dbpedia_14'] for m in ['medium-4','large-6']]
def rows(name):
 with (ROOT/'tables'/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def table(text,marker):
 start=text.index(marker);block=text[start:];start=block.index('\n|');lines=block[start:].splitlines()[1:]
 out=[]
 for line in lines:
  if not line.startswith('|'):break
  cols=[v.strip() for v in line.strip('|').split('|')]
  if all(re.fullmatch(r':?-+:?',v) for v in cols):continue
  out.append(cols)
 return out[1:]
def value(v,scale=1,digits=0):
 return 'n.a.' if not v else f'{statistics.mean(v)*scale:,.{digits}f}'
def metric(d):return 'accuracy' if d=='cifar10' else 'entity_f1' if d=='wnut_17' else 'macro_f1'
def run():
 count=0;data={};traces={};hist={}
 with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
  for r in rows('all_condition_results.csv'):
   key=(r['family'],r['dataset'],r['model'],int(r['seed']),r['condition']);name=r['result_path'];pre=name.rsplit('/',1)[0]+'/'
   data[key]=json.loads(z.read(name));traces[key]=json.loads(z.read(pre+'trace.json'));hist[key]=json.loads(z.read(pre+'history.json'))
 def check(actual,expected):
  nonlocal count
  assert actual==expected,(actual,expected);count+=1
 def records(family,ds,mod,cond):return [data[family,ds,mod,seed,cond] for seed in SEEDS]
 def acc(ds,mod,cond):return [d['test_metrics']['accuracy'] for d in records('scratch',ds,mod,cond)]
 def mean_sd(v):return f'{100*statistics.mean(v):.2f} ± {100*statistics.stdev(v):.2f}'
 main=(ROOT/'manuscript/Research_Manuscript.md').read_text(encoding='utf-8')
 supplement=(ROOT/'manuscript/All_Seed_Results_and_Provenance.md').read_text(encoding='utf-8')
 corpus=main+'\n\n'+supplement
 conditions=['stable_with_guard','single_bad_first_step_then_stable','fixed_order_with_guard','buggy_no_guard','buggy_rescue_guard']
 for row,(ds,mod) in zip(table(main,'**Table 2.'),SETTINGS):
  for cell,cond in zip(row[1:],conditions):check(cell,mean_sd(acc(ds,mod,cond)))
 pairs=[('stable_with_guard','single_bad_first_step_then_stable'),('buggy_rescue_guard','buggy_no_guard'),('fixed_order_with_guard','buggy_rescue_guard')]
 for row,(ds,mod) in zip(table(main,'**Table 3.'),SETTINGS):
  for cell,(a,b) in zip(row[1:],pairs):
   diff=[100*(x-y) for x,y in zip(acc(ds,mod,a),acc(ds,mod,b))];mean=statistics.mean(diff);radius=student_t.ppf(.975,4)*statistics.stdev(diff)/math.sqrt(5)
   check(cell,f'{mean:+.2f} [{mean-radius:+.2f}, {mean+radius:+.2f}]')
 order=['stable_with_guard','single_bad_first_step_then_stable','fixed_order_with_guard','buggy_no_guard','buggy_rescue_guard','buggy_failfast_guard']
 for i,row in enumerate(table(corpus,'## Table S13 ')):
  ds,mod=SETTINGS[i//6];cond=order[i%6];r=records('scratch',ds,mod,cond)
  for cell,group,key,scale,digits in [(row[2],'train_metrics','loss',1,3),(row[3],'train_metrics','accuracy',100,2),(row[4],'test_metrics','accuracy',100,2)]:
   check(cell,value([d[group][key] for d in r if d.get(group) and key in d[group]],scale,digits))
 for i,row in enumerate(table(corpus,'## Table S14 ')):
  ds,mod=SETTINGS[i//6];cond=order[i%6];r=records('scratch',ds,mod,cond);n=[d['global_steps'] for d in r]
  pos=[min(k,49999) if cond in order[:2] else k for k in n];zero=[a-b for a,b in zip(n,pos)]
  replays=[sum(bool(t['guard_triggered']) for t in traces['scratch',ds,mod,seed,cond]) if cond=='buggy_rescue_guard' else 0 for seed in SEEDS]
  for cell,v,digits in [(row[2],pos,0),(row[3],zero,0),(row[4],replays,1)]:check(cell,value(v,digits=digits))
 historical=rows('historical_initial_lr_per_seed.csv')
 for row,(ds,mod) in zip(table(corpus,'## Table S16 '),SETTINGS):
  for cell,rate in zip(row[1:],[1.,.3,.1,.03,.01]):
   v=[float(r['test_acc']) for r in historical if r['dataset']==ds and r['model_name']==mod and float(r['initial_lr'])==rate]
   check(cell,mean_sd(v))
 tasks=['ag_news','dbpedia_14','wnut_17','cifar10'];preorder=['safe_correct_order','late_schedule_no_guard','single_bad_then_correct','update_guard_rescue']
 for row,ds in zip(table(main,'**Table 5.'),tasks):
  mod='deit-tiny' if ds=='cifar10' else 'distilbert'
  for cell,cond in zip(row[1:5],preorder):check(cell,mean_sd([d['test_metrics'][metric(ds)] for d in records('pretrained',ds,mod,cond)]))
 prebudget=['safe_correct_order','late_schedule_no_guard','single_bad_then_correct','update_guard_rescue','update_guard_failfast','executed_lr_assertion']
 for i,row in enumerate(table(corpus,'## Table S15 ')):
  ds=tasks[i//6];cond=prebudget[i%6];mod='deit-tiny' if ds=='cifar10' else 'distilbert';r=records('pretrained',ds,mod,cond)
  h=[hist['pretrained',ds,mod,seed,cond] for seed in SEEDS];check(row[2],value([v[-1]['train_loss'] for v in h if v],digits=3))
  n=[d['global_steps'] for d in r];zero=[int(k>0 and cond in ('safe_correct_order','single_bad_then_correct')) for k in n]
  check(row[3],value([a-b for a,b in zip(n,zero)]));check(row[4],value(zero))
 sensitivity=rows('retrospective_kappa_sensitivity.csv')
 for row in table(corpus,'## Table S17 '):
  k=float(row[0]);r=[d for d in sensitivity if float(d['kappa'])==k];w=[d for d in r if d['dataset']=='wnut_17']
  check(row[1],f"{sum(v['detected']=='True' for v in r)}/40");check(row[2],f"{sum(v['detected']=='True' for v in w)}/5")
  check(row[3],f"{sum(int(v['safe_fp']) for v in r)}/{sum(int(v['safe_n']) for v in r)}")
 for number,(ds,mod) in enumerate(SETTINGS,1):
  for row,seed in zip(table(supplement,f'## Table S{number} '),SEEDS):
   for cell,cond in zip(row[1:],['stable_with_guard','single_bad_first_step_then_stable','buggy_no_guard','buggy_rescue_guard','fixed_order_with_guard']):check(cell,f"{100*data['scratch',ds,mod,seed,cond]['test_metrics']['accuracy']:.4f}")
 for number,ds in enumerate(tasks,5):
  mod='deit-tiny' if ds=='cifar10' else 'distilbert'
  for row,seed in zip(table(supplement,f'## Table S{number} '),SEEDS):
   for cell,cond in zip(row[1:],['safe_correct_order','single_bad_then_correct','late_schedule_no_guard','update_guard_rescue']):check(cell,f"{100*data['pretrained',ds,mod,seed,cond]['test_metrics'][metric(ds)]:.4f}")
 for number,(ds,mod) in enumerate(SETTINGS,9):
  for row,seed in zip(table(supplement,f'## Table S{number} '),SEEDS):
   for cell,(a,b) in zip(row[1:],pairs):check(cell,f"{100*(data['scratch',ds,mod,seed,a]['test_metrics']['accuracy']-data['scratch',ds,mod,seed,b]['test_metrics']['accuracy']):+.5f}")
 # Independently reconstruct the six additional shared-step illustration cells
 # directly from raw archive histories and test outcomes, not the new CSV.
 for row,mod in zip(table(main,'**Table 4.'),['medium-4','large-6']):
  step=44625;check(row[1],f'{step:,}')
  differences=[]
  for seed in SEEDS:
   a=next(h for h in hist['scratch','dbpedia_14',mod,seed,'buggy_rescue_guard'] if int(h['global_step'])==step)
   b=next(h for h in hist['scratch','dbpedia_14',mod,seed,'buggy_no_guard'] if int(h['global_step'])==step)
   differences.append(100*(a['validation']['accuracy']-b['validation']['accuracy']))
  check(row[2],f'{statistics.mean(differences):.3f}')
  endpoint=[100*(x-y) for x,y in zip(acc('dbpedia_14',mod,'buggy_rescue_guard'),acc('dbpedia_14',mod,'buggy_no_guard'))]
  check(row[3],f'{statistics.mean(endpoint):.3f}')
 assert count==560,count
 return {'displayed_numeric_cells_checked':count,'discrepancies':0,'basis':'original October result/history/trace JSON; historical per-seed notebook extraction; regenerated sensitivity rows; raw shared-step history reconstruction'}
if __name__=='__main__':print(json.dumps(run(),indent=2))

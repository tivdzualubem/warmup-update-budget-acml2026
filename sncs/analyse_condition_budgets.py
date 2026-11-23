"""Extract training endpoints and condition-specific positive-LR budgets. No training."""
from pathlib import Path
import csv,json,zipfile,math,collections,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent

def write(name,rows):
 with (ROOT/'tables'/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def extract():
 rows=[];selection=[];secondary=[]
 cfg=json.loads((ROOT/'source/warmup-update-budget-acml2026/publication_expansion/configs/condition_semantics_v1.json').read_text())
 widths=cfg['scratch_schedules']['legacy_transformer']['d_model']
 with zipfile.ZipFile(ROOT/'data/ACML_RESULTS_FOR_STUDENT.zip') as z:
  names=set(z.namelist())
  for name in sorted(names):
   if not name.endswith('/result.json') or '/results/' not in name:continue
   d=json.loads(z.read(name))
   if not d.get('job_complete') or 'condition' not in d:continue
   pre=name.rsplit('/',1)[0]+'/'
   h=json.loads(z.read(pre+'history.json'));trace=json.loads(z.read(pre+'trace.json'));j=json.loads(z.read(pre+'job_manifest.json'))
   cond=d['condition'];ds=d['dataset'];fam='scratch' if '/results/scratch/' in name else 'pretrained';mod=d.get('model','deit-tiny' if ds=='cifar10' else 'distilbert');n=int(d.get('global_steps') or 0)
   t=np.arange(1,n+1,dtype=float)
   if fam=='scratch':
    if cond in ('stable_with_guard','single_bad_first_step_then_stable'):
     lr=np.where(t<=8000,1e-4*t/8000,np.where(t>=50000,0.,1e-4*.5*(1+np.cos(np.pi*(t-8000)/42000))))
     if n and cond=='single_bad_first_step_then_stable':lr[0]=1.
    else:
     index=t if cond=='fixed_order_with_guard' else np.maximum(1,t-1)
     lr=widths[mod]**-.5*np.minimum(index**-.5,index*8000**-1.5)
     if n and cond!='fixed_order_with_guard':lr[0]=1.
   else:
    T=j['total_optimizer_steps'];W=j['warmup_steps'];peak=j['peak_lr']
    index=t if cond in ('safe_correct_order','single_bad_then_correct') else np.maximum(1,t-1)
    lr=np.where(index<=W,peak*index/W,peak*np.maximum(0,(T-index)/(T-W)))
    if n and cond!='safe_correct_order':lr[0]=peak
   for tr in trace:
    k=int(tr['step'])-1
    if k<n and tr.get('committed_lr_consumed') is not None:lr[k]=tr['committed_lr_consumed']
   positive=int(np.count_nonzero(lr>0));zero=int(np.count_nonzero(lr==0));assert positive+zero==n
   replay=sum(1 for tr in trace if tr.get('guard_triggered') and cond in ('buggy_rescue_guard','update_guard_rescue'))
   train=d.get('train_metrics') or {};test=d.get('test_metrics') or {}
   if h:
    best=min(h,key=lambda x:x['validation']['loss']) if fam=='scratch' else max(h,key=lambda x:x['validation_primary'])
    last=h[-1]
   else:best={};last={}
   metric='accuracy' if fam=='scratch' or ds=='cifar10' else 'entity_f1' if ds=='wnut_17' else 'macro_f1'
   rows.append(dict(family=fam,dataset=ds,model=mod,seed=d['seed'],condition=cond,committed_optimizer_steps=n,positive_lr_committed_steps=positive,zero_lr_committed_steps=zero,rejected_candidate_replay_calls=replay,optimizer_step_calls_including_replays=n+replay,cumulative_committed_lr=float(lr.sum()),best_validation_epoch=best.get('epoch',''),best_validation_step=best.get('global_step',''),last_logged_epoch=last.get('epoch',''),last_online_train_loss=last.get('train_loss',''),last_online_train_accuracy=last.get('train_accuracy',''),selected_checkpoint_train_loss=train.get('loss',''),selected_checkpoint_train_accuracy=train.get('accuracy',''),selected_checkpoint_test_primary=test.get(metric,''),primary_metric=metric,terminal_reason=d['terminal_reason'],result_path=name))
   if fam=='pretrained' and h and test:
    selection.append(dict(dataset=ds,seed=d['seed'],condition=cond,selection_metric=d['primary_metric'],best_epoch=d['best_epoch'],validation_primary_at_selected_epoch=best['validation_primary'],same_validation_metric=best['validation'][d['primary_metric']],test_primary_metric=metric,test_primary_value=test.get(metric,''),test_accuracy_secondary=test.get('accuracy','')))
   if fam=='pretrained' and test:secondary.append(dict(dataset=ds,model=mod,seed=d['seed'],condition=cond,accuracy=test.get('accuracy',''),macro_f1=test.get('macro_f1',''),entity_f1=test.get('entity_f1','')))
 assert len(rows)==240
 write('condition_training_and_budget_per_run.csv',rows)
 write('pretrained_selection_and_primary_metric.csv',selection)
 write('pretrained_secondary_metrics_per_seed.csv',secondary)
 groups=collections.defaultdict(list)
 for r in rows:groups[r['family'],r['dataset'],r['model'],r['condition']].append(r)
 summaries=[]
 fields=['committed_optimizer_steps','positive_lr_committed_steps','zero_lr_committed_steps','rejected_candidate_replay_calls','optimizer_step_calls_including_replays','cumulative_committed_lr','last_online_train_loss','selected_checkpoint_train_loss','selected_checkpoint_train_accuracy','selected_checkpoint_test_primary']
 for key,g in sorted(groups.items()):
  row=dict(zip(['family','dataset','model','condition'],key));row['n_jobs']=len(g);row['primary_metric']=g[0]['primary_metric']
  for field in fields:
   values=[r[field] for r in g if r[field]!=''];row[field+'_n']=len(values);row[field+'_mean']=statistics.mean(values) if values else '';row[field+'_sd']=statistics.stdev(values) if len(values)>1 else ''
  summaries.append(row)
 write('condition_training_and_budget_summary.csv',summaries)
 print('Condition budgets:',len(rows),'runs;',len(summaries),'separate condition summaries; primary metrics:',len(selection))
 return rows,summaries

if __name__=='__main__':extract()

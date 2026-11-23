#!/usr/bin/env python3
"""Recover displayed historical 3-seed LR-sweep tables from executed 2026 notebooks.
The preserved notebooks contain tables, not original per-seed JSON/checkpoint bytes.
"""
import argparse,io,json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import t

def extract(ag,db,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 evidence=Path(__file__).resolve().parent/'evidence/historical_warmup/warmup_verification.json'
 if not evidence.exists(): raise RuntimeError('Historical warmup provenance is required; do not infer a value from the notebook trainer default')
 verified=json.loads(evidence.read_text())
 warmup=verified['historical_warmup_steps']
 assert verified['AG_News_raw_traces_checked']==30 and verified['DBpedia_first_step_trace_export_rows_checked']==30
 assert verified['sources'] and warmup==4000
 for item in verified['sources']:
  p=Path(__file__).resolve().parent/item['local_evidence_path']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
 rows=[];summaries=[];reports=[]
 for dataset,path in [('ag_news',Path(ag)),('dbpedia_14',Path(db))]:
  nb=json.loads(path.read_text());data=[x for x in nb['cells'][45]['outputs'] if 'text/html' in x.get('data',{})]
  assert len(data)==2, f'expected two displayed tables in {path}'
  summary=pd.read_html(io.StringIO(''.join(data[0]['data']['text/html'])))[0]
  per=pd.read_html(io.StringIO(''.join(data[1]['data']['text/html'])))[0]
  assert len(summary)==10 and len(per)==30
  for df in [summary,per]:
   df.drop(columns=['Unnamed: 0'],errors='ignore',inplace=True)
   df.insert(0,'dataset',dataset)
   df['historical_protocol']=f'legacy_buggy_warmup_{warmup}_steps_3seeds_source_and_trace_verified'
   df['provenance']='executed_notebook_display_table'
  for r in per.to_dict('records'):rows.append(r)
  for r in summary.to_dict('records'):summaries.append(r)
  reports.append({'dataset':dataset,'notebook':path.name,'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows_summary':len(summary),'rows_per_seed':len(per),'warmup_steps_verified':warmup,'warmup_evidence':'evidence/historical_warmup/warmup_verification.json','note':'Outcomes remain extracted from saved notebook display output; original checkpoint tensors are absent.'})
 full=pd.DataFrame(rows);summ=pd.DataFrame(summaries)
 assert set(full['seed'])=={42,123,456};assert set(np.round(full.initial_lr,2))=={1.0,0.3,0.1,0.03,0.01}
 # internal displayed summary vs per-seed mean/std checks (display rounds to 6 decimals)
 for k,g in full.groupby(['dataset','model_name','initial_lr']):
  sg=summ[(summ.dataset==k[0])&(summ.model_name==k[1])&np.isclose(summ.initial_lr,k[2])].iloc[0]
  assert abs(g.test_acc.mean()-sg.test_acc_mean)<2e-6,k
  assert abs(g.test_acc.std(ddof=0)-sg.test_acc_std)<2e-6,k
 # Preserve original notebook summary, and provide properly labeled sample-SD estimates.
 adjusted=[]
 for (dataset,model,lr),g in full.groupby(['dataset','model_name','initial_lr']):
  from scipy.stats import t
  n=len(g);m=float(g.test_acc.mean());sd=float(g.test_acc.std(ddof=1));rad=float(t.ppf(.975,n-1)*sd/(n**.5))
  adjusted.append(dict(dataset=dataset,model_name=model,initial_lr=lr,n=n,mean_test_acc=m,sample_sd=sd,ci95_t_low=m-rad,ci95_t_high=m+rad,mean_ece=float(g.test_ece.mean()),mean_nll=float(g.test_nll.mean()),protocol=f'historical_{warmup}_warmup_3_seeds_verified',provenance='saved_notebook_output'))
 pd.DataFrame(adjusted).to_csv(out/'historical_initial_lr_recomputed_statistics.csv',index=False)
 full.to_csv(out/'historical_initial_lr_per_seed.csv',index=False)
 summ.to_csv(out/'historical_initial_lr_summary.csv',index=False)
 (out/'historical_provenance.json').write_text(json.dumps(reports,indent=2)+'\n')
 print('RECOVERED',len(full),'historical seed runs and',len(summ),'summary combinations from notebook outputs')
 print(summ[['dataset','model_name','initial_lr','test_acc_mean','test_acc_std','ece_mean','nll_mean']].round(5).to_string(index=False))
 return full,summ
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--ag',default='/mnt/data/Transformer_Artifacts(AGNEWS).ipynb');a.add_argument('--db',default='/mnt/data/Transformer_Artifacts(DBPEDIA).ipynb');a.add_argument('--out',default='/mnt/data/acml_corrected_analysis/tables');v=a.parse_args();extract(v.ag,v.db,v.out)

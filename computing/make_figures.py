#!/usr/bin/env python3
"""Rebuild main research figures from preserved October and historical records."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;TABLES=ROOT/'tables';FIGS=ROOT/'figures';FIGS.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':210,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8})
labels={'stable_with_guard':'Stable cosine','single_bad_first_step_then_stable':'One bad step + stable','fixed_order_with_guard':'Correct legacy','buggy_no_guard':'Late legacy','buggy_rescue_guard':'Rollback rescue'}
colors={'stable_with_guard':'#245a83','single_bad_first_step_then_stable':'#b65a40','fixed_order_with_guard':'#319277','buggy_no_guard':'#ad7e27','buggy_rescue_guard':'#7a65a4'}
D=[('ag_news','medium-4'),('ag_news','large-6'),('dbpedia_14','medium-4'),('dbpedia_14','large-6')]
C=lambda ds: 'AG News' if ds=='ag_news' else 'DBpedia'

# Paired actual values, precisely the correct stable control.
paired=pd.read_csv(TABLES/'paired_per_seed.csv')
fig,axes=plt.subplots(2,2,figsize=(11,7.3),sharey=True)
for ax,(ds,mod) in zip(axes.flat,D):
 s=paired[(paired.dataset==ds)&(paired.model==mod)&(paired.contrast.str.startswith('isolated_first_step'))].sort_values('seed')
 for _,r in s.iterrows():ax.plot([0,1],[100*r.control_accuracy,100*r.treatment_accuracy],color='#9aa6b0',lw=1.3,alpha=.75)
 ax.scatter(np.repeat(0,len(s)),100*s.control_accuracy,color=colors['stable_with_guard'],zorder=4,label='Stable control')
 ax.scatter(np.repeat(1,len(s)),100*s.treatment_accuracy,color=colors['single_bad_first_step_then_stable'],zorder=4,label='Isolated bad step')
 ax.set_xticks([0,1],['Stable','Single bad step']);ax.set_xlim(-.28,1.28);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.22)
 ax.set_title(f'{C(ds)} · {mod}\nmean loss = {s.accuracy_difference_pp.mean():.1f} percentage points')
 for i,y in enumerate([100*s.control_accuracy.mean(),100*s.treatment_accuracy.mean()]):ax.annotate(f'{y:.1f}%',(i,y),xytext=(0,8),textcoords='offset points',ha='center',fontweight='bold',fontsize=9)
for ax in axes[:,0]:ax.set_ylabel('Test accuracy (%)')
fig.suptitle('Isolated first-step intervention: correct within-schedule comparison',weight='semibold');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(FIGS/'fig1_isolated_first_step_paired.png');plt.close(fig)

# order-only comparison on same legacy schedule
fig,axes=plt.subplots(2,2,figsize=(11,7.2),sharey=True)
for ax,(ds,mod) in zip(axes.flat,D):
 s=paired[(paired.dataset==ds)&(paired.model==mod)&(paired.contrast.str.startswith('scheduler_order'))].sort_values('seed')
 for _,r in s.iterrows():ax.plot([0,1],[100*r.control_accuracy,100*r.treatment_accuracy],color='#9aa6b0',lw=1.3,alpha=.75)
 ax.scatter([0]*len(s),100*s.control_accuracy,color=colors['fixed_order_with_guard'],label='Correct order');ax.scatter([1]*len(s),100*s.treatment_accuracy,color=colors['buggy_no_guard'],label='Late order')
 ax.set_xticks([0,1],['Correct order','Late assignment']);ax.set_xlim(-.32,1.32);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.22)
 ax.set_title(f'{C(ds)} · {mod}\nmean gap = {s.accuracy_difference_pp.mean():.1f} pp')
for ax in axes[:,0]:ax.set_ylabel('Test accuracy (%)')
fig.suptitle('Schedule-order intervention: common target formula, different consumed sequence',weight='semibold');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(FIGS/'fig2_legacy_schedule_order_paired.png');plt.close(fig)

# Archived validation histories on optimizer-step axis, show cap.
history=pd.read_csv(TABLES/'optimizer_step_histories.csv')
fig,axes=plt.subplots(2,2,figsize=(12.5,8.4))
for ax,(ds,mod) in zip(axes.flat,D):
 for cond in labels:
  h=history[(history.dataset==ds)&(history.model==mod)&(history.condition==cond)]
  if h.empty:continue
  g=h.groupby('global_step').validation_accuracy.agg(['mean','std']).sort_index()
  ax.plot(g.index,g['mean']*100,label=labels[cond],color=colors[cond],lw=1.6)
  if g['std'].notna().any():ax.fill_between(g.index,(g['mean']-g['std'].fillna(0))*100,(g['mean']+g['std'].fillna(0))*100,alpha=.10,color=colors[cond])
 ax.axvline(50000,color='#333333',ls='--',lw=1.0)
 ax.set_title(f'{C(ds)} · {mod}');ax.grid(alpha=.2);ax.set_xlabel('Executed optimizer transition index');ax.set_ylabel('Validation accuracy (%)')
 ax.set_ylim(bottom=0,top=105)
fig.suptitle('Validation trajectories in optimizer steps · dashed = stable cosine LR reaches zero',weight='semibold')
h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01));fig.tight_layout(rect=(0,.045,1,.96));fig.savefig(FIGS/'fig3_step_indexed_training_histories.png',bbox_inches='tight');plt.close(fig)

# Schedules loaded from the frozen model configuration and pure source functions.
import json,sys
SOURCE=ROOT/'source/warmup-update-budget-acml2026'
sys.path.insert(0,str(SOURCE))
from publication_expansion.src.scheduler_semantics import stable_lr,legacy_transformer_lr
cfg=json.loads((SOURCE/'publication_expansion/configs/condition_semantics_v1.json').read_text())
widths=cfg['scratch_schedules']['legacy_transformer']['d_model']
assert widths=={'medium-4':128,'large-6':256}
step=np.unique(np.r_[np.arange(1,125001,125),[1,2,7999,8000,8001,49999,50000,50001,125000]])
S=np.array([stable_lr(int(t)) for t in step])
fig,ax=plt.subplots(figsize=(10.5,4.6))
ax.plot(step,S*1e4,color=colors['stable_with_guard'],label='Stable cosine',lw=2)
curve={'step':step,'stable_lr':S}
for mod,color in [('medium-4','#319277'),('large-6','#9b6b32')]:
 values=np.array([legacy_transformer_lr(int(t),widths[mod]) for t in step])
 curve[f'legacy_{mod}_lr']=values
 ax.plot(step,values*1e4,color=color,label=f'Legacy {mod}, d_model={widths[mod]}',lw=1.8)
ax.axvline(8000,c='gray',ls=':',label='Warmup boundary 8,000')
ax.axvline(50000,c='black',ls='--',label='Stable zero LR at 50,000')
ax.axvspan(50000,125000,color='gray',alpha=.07)
ax.set_xlim(0,125000);ax.set_ylim(bottom=0);ax.set_xlabel('Optimizer transition');ax.set_ylabel('Learning rate × 10⁴')
ax.set_title('Schedules from the frozen configuration');ax.grid(alpha=.2);ax.legend(ncol=2,loc='upper right',fontsize=8)
fig.tight_layout();fig.savefig(FIGS/'fig4_scheduler_and_zero_boundary.png');plt.close(fig)
pd.DataFrame(curve).to_csv(TABLES/'schedule_curves_from_configuration.csv',index=False)

# Historical 4000-step sweep, not mixed with October.
hist=pd.read_csv(TABLES/'historical_initial_lr_recomputed_statistics.csv')
fig,axes=plt.subplots(2,2,figsize=(11,7.1),sharex=True)
for ax,(ds,mod) in zip(axes.flat,D):
 q=hist[(hist.dataset==ds)&(hist.model_name==mod)].sort_values('initial_lr')
 ax.errorbar(q.initial_lr,100*q.mean_test_acc,yerr=100*q.sample_sd,fmt='o-',lw=1.6,capsize=4,color='#285c82',label='Historical 3 seeds')
 ax.set_xscale('log');ax.set_xlim(.007,1.4);ax.grid(alpha=.2);ax.set_title(f'{C(ds)} · {mod}');ax.set_xticks([.01,.03,.1,.3,1],labels=['0.01','0.03','0.1','0.3','1'])
for ax in axes[:,0]:ax.set_ylabel('Test accuracy (%), mean ± sample SD')
for ax in axes[1,:]:ax.set_xlabel('Initial Adam LR consumed by faulty first update')
fig.suptitle('Historical initial-LR training sweep (three seeds; 4,000-step warmup)',weight='semibold');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(FIGS/'fig5_historical_initial_lr_sweep.png');plt.close(fig)

# Each scratch condition is shown separately, including diagnostic fail-fast.
bud=pd.read_csv(TABLES/'condition_training_and_budget_per_run.csv')
order=['stable_with_guard','single_bad_first_step_then_stable','fixed_order_with_guard','buggy_no_guard','buggy_rescue_guard','buggy_failfast_guard']
short=['Stable','Bad then\nstable','Correct\nlegacy','Late\nlegacy','Rescue','Fail-fast']
fig,axes=plt.subplots(2,2,figsize=(12,8.2))
for ax,(ds,mod) in zip(axes.flat,D):
 q=bud[(bud.family=='scratch')&(bud.dataset==ds)&(bud.model==mod)]
 for i,cond in enumerate(order):
  v=q[q.condition==cond];a=v.positive_lr_committed_steps.mean();b=v.zero_lr_committed_steps.mean()
  ax.bar(i,a,color=colors.get(cond,'#8c8c8c'),width=.64,label='Positive LR' if i==0 else None)
  ax.bar(i,b,bottom=a,color='#c9cdd2',width=.64,label='Zero LR' if i==0 else None)
 ax.set_xticks(range(len(order)),short,fontsize=8);ax.set_title(f'{C(ds)} · {mod}');ax.grid(axis='y',alpha=.18)
 ax.set_ylabel('Mean committed optimizer transitions')
fig.suptitle('Positive-LR budget and zero-LR computation by condition',weight='semibold')
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(FIGS/'fig6_ineffective_postzero_steps.png');plt.close(fig)

# Logged online training loss is separate from validation/test evaluation.
fig,axes=plt.subplots(2,2,figsize=(12,8.2))
for ax,(ds,mod) in zip(axes.flat,D):
 for cond in labels:
  h=history[(history.dataset==ds)&(history.model==mod)&(history.condition==cond)]
  for j,(seed,v) in enumerate(h.groupby('seed')):
   ax.plot(v.global_step,v.train_loss,color=colors[cond],alpha=.65,lw=1,label=labels[cond] if j==0 else None)
 ax.axvline(50000,color='black',ls='--',lw=.8);ax.set_yscale('log');ax.grid(alpha=.18)
 ax.set_title(f'{C(ds)} · {mod}');ax.set_xlabel('Optimizer transition');ax.set_ylabel('Online epoch training CE loss')
fig.suptitle('Training loss by condition and seed',weight='semibold')
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(FIGS/'fig8_condition_training_loss.png');plt.close(fig)

# aggregate detection vs kappa derived from existing threshold sweep csv
try:
 df=pd.read_csv(TABLES/'retrospective_kappa_sensitivity.csv')
 q=df.groupby('kappa').detected.agg(['sum','count']).reset_index().sort_values('kappa')
 fig,ax=plt.subplots(figsize=(8.3,4.2));ax.plot(q.kappa,q['sum'],marker='o',color='#285c82',lw=2);ax.axvline(10,color='black',ls='--',label='Prespecified κ = 10');ax.set_xscale('log');ax.set_ylim(0,42);ax.set_xlabel('Threshold multiplier κ');ax.set_ylabel('First-step induced mismatches detected / 40');ax.set_title('Observed threshold sensitivity (retrospective operating points)');ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIGS/'fig7_guard_threshold_sensitivity.png');plt.close(fig)
except Exception as e:print('Guard sensitivity plot unavailable:',e)
print('Created',len(list(FIGS.glob('*.png'))),'figures')

# Exploratory repair and remaining-delay decomposition, all five archived seeds.
runs=pd.read_csv(TABLES/'all_condition_results.csv')
fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharey=True)
for ax,(ds,mod) in zip(axes.flat,D):
 q=runs[(runs.family=='scratch')&(runs.dataset==ds)&(runs.model==mod)]
 for seed in [42,123,456,789,2026]:
  values=[100*q[(q.seed==seed)&(q.condition==cond)].iloc[0].test_accuracy for cond in ['buggy_no_guard','buggy_rescue_guard','fixed_order_with_guard']]
  ax.plot(range(3),values,marker='o',alpha=.75,lw=1,label=str(seed))
 ax.set_xticks(range(3),['Late legacy','First-step rescue','Correct legacy']);ax.set_ylim(0,105);ax.grid(alpha=.2);ax.set_title(f'{C(ds)} / {mod}')
for ax in axes[:,0]:ax.set_ylabel('Test accuracy (%)')
fig.suptitle('Exploratory repair and remaining delay; unequal realized budgets',fontweight='bold')
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=5,title='Seed',frameon=False)
fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(FIGS/'fig9_exploratory_repair_delay.png');plt.close(fig)

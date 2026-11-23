"""Three complementary main figures; all underlying rows remain in the supplement."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;F=R/'figures'
plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
s=pd.read_csv(R/'tables/matched_step_summary.csv');p=pd.read_csv(R/'tables/matched_step_per_seed.csv')
def panel(ax,ds,model,contrast,names,zero=False):
    b=s[(s.dataset==ds)&(s.model==model)&(s.contrast==contrast)].sort_values('global_step')
    seeds=p[(p.dataset==ds)&(p.model==model)&(p.contrast==contrast)]
    for arm,color,marker,name in [('a','#245b8c','o',names[0]),('b','#bd5827','s',names[1])]:
        ax.scatter(seeds.global_step/1000,100*seeds[f'arm_{arm}_validation_accuracy'],s=6,color=color,alpha=.2)
        ax.errorbar(b.global_step/1000,100*b[f'arm_{arm}_validation_accuracy_mean'],yerr=100*b[f'arm_{arm}_validation_accuracy_sample_sd'],fmt=marker,ms=3,capsize=2,lw=.7,color=color,label=name)
    if zero:ax.axvline(50,color='.4',ls='--',lw=.7)
    ax.set_title(('AG News' if ds=='ag_news' else 'DBpedia')+' '+model)
    ax.set_ylim(0,103);ax.set_ylabel('Validation accuracy (%)');ax.set_xlabel('Transitions (thousands)');ax.grid(alpha=.15)
    ax.legend(loc='lower right',frameon=False)
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});fig.savefig(F/(name+'.png'),dpi=220);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(4.9,5.5),layout='constrained')
for ax,(ds,model) in zip(axes.flat,[(d,m) for d in ['ag_news','dbpedia_14'] for m in ['medium-4','large-6']]):panel(ax,ds,model,'stable_minus_single_bad',('Stable','First LR 1.0'),True)
save(fig,'fig1_computing_stable_progress')
fig,axes=plt.subplots(2,1,figsize=(4.9,5.4),layout='constrained')
for ax,model in zip(axes,['medium-4','large-6']):panel(ax,'dbpedia_14',model,'rescue_minus_late',('First-step repair','Late execution'))
save(fig,'fig2_computing_legacy_progress')
c=pd.read_csv(R/'tables/peak_first_curves.csv');c=c[c.population=='validation']
fig,axes=plt.subplots(2,1,figsize=(4.9,5.3),layout='constrained')
for arm,color,style,mark in [('clean','#245b8c','-','o'),('peak_first','#bd5827','--','x')]:
    for ax,key in zip(axes,['logit_ce','accuracy_pct']):
        ax.plot(c.step,c[f'{arm}_{key}'],style,marker=mark,ms=4,color=color,label='Clean first step' if arm=='clean' else 'Peak rate at first step')
for ax in axes:
    ax.set_xscale('symlog',linthresh=100);ax.set_xticks([0,100,1000,8000,49999],['0','100','1,000','8,000','49,999']);ax.set_xlabel('Completed positive-rate transitions');ax.grid(alpha=.15)
axes[0].set_ylabel('Validation cross-entropy');axes[0].set_yscale('log');axes[0].legend(frameon=False)
axes[1].set_ylabel('Validation accuracy (%)');axes[1].set_ylim(0,103)
save(fig,'fig3_computing_peak_progress')
print('Three vector main figures regenerated from unchanged tables.')

"""Publication figures from the fixed-budget pair's complete saved tables."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
curves=pd.read_csv(ROOT/'tables/peak_first_curves.csv')
online=pd.read_csv(ROOT/'tables/peak_first_online_training.csv')
colors={'clean':'#245b8c','peak_first':'#bd5827'}
labels={'clean':'Clean first step','peak_first':'Peak LR at first step'}

def axis(ax,ylabel):
    ax.set_xscale('symlog',linthresh=100)
    ax.set_xticks([0,100,1000,8000,49999],['0','100','1,000','8,000','49,999'])
    ax.set_xlabel('Completed positive-LR transitions');ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)

v=curves[curves.population=='validation']
fig,axes=plt.subplots(1,2,figsize=(10.3,3.7),layout='constrained')
for arm,style,marker in [('clean','-','o'),('peak_first','--','x')]:
    axes[0].plot(v.step,v[f'{arm}_logit_ce'],style,marker=marker,color=colors[arm],label=labels[arm],markersize=4)
    axes[1].plot(v.step,v[f'{arm}_accuracy_pct'],style,marker=marker,color=colors[arm],label=labels[arm],markersize=4)
axis(axes[0],'Validation logit CE');axes[0].set_yscale('log');axes[0].set_title('(a) Absolute loss')
axis(axes[1],'Validation accuracy (%)');axes[1].set_ylim(0,103);axes[1].set_title('(b) Absolute accuracy')
axes[0].legend(frameon=False,fontsize=9)
fig.savefig(ROOT/'figures/fig6_peak_first_fixed_budget.png');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10.3,7.0),layout='constrained')
t=curves[curves.population=='train_monitor_subset']
for arm,style in [('clean','-'),('peak_first','--')]:
    part=online[online.arm==arm]
    axes[0,0].plot(part.last_step,part.example_weighted_online_ce,style,color=colors[arm],alpha=.8,lw=.8,label=labels[arm])
    axes[0,1].plot(t.step,t[f'{arm}_logit_ce'],style,marker='o' if arm=='clean' else 'x',color=colors[arm],label=labels[arm],markersize=4)
for ax,label in [(axes[0,0],'Online training CE (100-step blocks)'),(axes[0,1],'Fixed training-subset logit CE')]:
    axis(ax,label);ax.set_yscale('log')
axes[0,0].legend(frameon=False,fontsize=9)
axes[0,0].set_title('(a) All 500 blocks per arm');axes[0,1].set_title('(b) Same 8,192 examples at every checkpoint')
axes[1,0].plot(v.step,v.ce_peak_first_minus_clean,'o-',color='#534585',markersize=4)
axis(axes[1,0],'Validation CE: peak-first minus clean');axes[1,0].axhline(0,color='black',lw=.6);axes[1,0].set_title('(c) All prespecified loss contrasts')
axes[1,1].plot(v.step,v.accuracy_peak_first_minus_clean_pp,'o-',color='#534585',markersize=4)
axis(axes[1,1],'Validation accuracy: peak-first minus clean (pp)');axes[1,1].axhline(0,color='black',lw=.6);axes[1,1].set_title('(d) All prespecified accuracy contrasts')
fig.savefig(ROOT/'figures/figS9_peak_first_training_and_differences.png');plt.close(fig)

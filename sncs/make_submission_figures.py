"""Scientific figures from preserved records and every shared logged step."""
from pathlib import Path
import csv,collections,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parent
F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
SETTINGS=[('ag_news','medium-4'),('ag_news','large-6'),('dbpedia_14','medium-4'),('dbpedia_14','large-6')]
CONTRASTS=['stable_minus_single_bad','rescue_minus_late','corrected_minus_rescue']
TITLES=['Stable / single bad','Rescue / late legacy','Corrected / rescue legacy']
def rows(name):return list(csv.DictReader((ROOT/'tables'/name).open(encoding='utf-8-sig')))
def label(ds,model):return ('AG News' if ds=='ag_news' else 'DBpedia')+' '+model
def save(fig,name):
    fig.savefig(F/name,bbox_inches='tight',facecolor='white');plt.close(fig)

def build():
    fig,ax=plt.subplots(figsize=(10,3.8));ax.set_xlim(0,10);ax.set_ylim(-.5,5);ax.axis('off')
    conditions=[('Stable','S(1)','S(t)'),('Single bad','1.0','S(t)'),('Corrected legacy','L(1)','L(t)'),('Late legacy','1.0','L(t-1)'),('Recorded rescue','reject 1.0; commit L(1)','L(t-1)')]
    for i,(name,first,later) in enumerate(conditions):
        y=4.1-i*.85
        ax.text(.05,y,name,va='center',fontsize=11)
        for x,w,text in [(2.3,3.6,first),(6.6,2.9,later)]:
            ax.add_patch(FancyBboxPatch((x,y-.27),w,.54,boxstyle='round,pad=0.035',facecolor='#edf2f7',edgecolor='#52708b'))
            ax.text(x+w/2,y,text,ha='center',va='center',fontsize=10)
        ax.annotate('',xy=(6.5,y),xytext=(6.0,y),arrowprops={'arrowstyle':'->','color':'#52708b'})
    ax.text(4.1,4.7,'Committed first transition',ha='center',weight='bold')
    ax.text(8.05,4.7,'Transitions t >= 2',ha='center',weight='bold')
    save(fig,'fig1_execution_conditions.png')
    # Same selected endpoints and contrasts as the archived result analysis.
    pairs=rows('paired_per_seed.csv');prefixes=['isolated','exploratory_first_step_repair','exploratory_remaining_delay']
    fig,axes=plt.subplots(1,3,figsize=(10.4,4),gridspec_kw={'wspace':.45})
    colors=['#205b87','#b65c26','#36764b','#735190']
    for col,(prefix,title) in enumerate(zip(prefixes,['Stable - single bad','Rescue - late','Corrected - rescue'])):
        ax=axes[col]
        for i,(ds,model) in enumerate(SETTINGS):
            r=[r for r in pairs if r['dataset']==ds and r['model']==model and r['contrast'].startswith(prefix)]
            vals=[float(r['accuracy_difference_pp']) for r in sorted(r,key=lambda r:int(r['seed']))]
            ax.scatter(vals,np.arange(5)*.055+i-.11,s=24,color=colors[i],alpha=.8)
            ax.plot(np.mean(vals),i,'D',color='black',markersize=5)
        ax.axvline(0,color='.55',lw=.8);ax.set_title(title,fontsize=11)
        ax.set_yticks(range(4),[label(*x) for x in SETTINGS] if col==0 else [])
        ax.set_xlabel('Selected test accuracy difference (pp)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    save(fig,'fig2_selected_paired_endpoints.png')
    data=rows('matched_step_per_seed.csv');summary=rows('matched_step_summary.csv')
    # Split the two schedule families into legible panels. Every pair uses all
    # steps common to its own ten trajectories; no missing outcomes are filled.
    for columns,name in [([0],'fig3_shared_step_stable_validation.png'),([1,2],'fig4_shared_step_legacy_validation.png')]:
        fig,axes=plt.subplots(2,len(columns)*2,figsize=(10.5,6.4),squeeze=False,sharey=True)
        for i,(ds,model) in enumerate(SETTINGS):
            for j,ci in enumerate(columns):
                ax=axes[i//2,(i%2)*len(columns)+j]
                block=[r for r in summary if r['dataset']==ds and r['model']==model and r['contrast']==CONTRASTS[ci]]
                seeds=[r for r in data if r['dataset']==ds and r['model']==model and r['contrast']==CONTRASTS[ci]]
                x=np.array([int(r['global_step']) for r in block])/1000
                for arm,color,marker in [('a','#225b86','o'),('b','#bb5a25','s')]:
                    ax.scatter([int(r['global_step'])/1000 for r in seeds],[100*float(r[f'arm_{arm}_validation_accuracy']) for r in seeds],s=7,color=color,alpha=.25)
                    means=[100*float(r[f'arm_{arm}_validation_accuracy_mean']) for r in block]
                    sd=[100*float(r[f'arm_{arm}_validation_accuracy_sample_sd']) for r in block]
                    armnames=[('Stable','Single bad'),('Rescue','Late'),('Corrected','Rescue')][ci]
                    ax.errorbar(x,means,yerr=sd,fmt=marker,ms=4,capsize=2,color=color,label=armnames[0 if arm=='a' else 1],lw=.7)
                if ci==0:ax.axvline(50,color='.4',ls='--',lw=.8)
                ax.set_title(label(ds,model)+'\n'+TITLES[ci],fontsize=10)
                ax.set_ylim(0,103);ax.grid(alpha=.15);ax.tick_params(labelsize=9.5)
                if i//2==1:ax.set_xlabel('Transitions (thousands)',fontsize=10)
                if (i%2)*len(columns)+j==0:ax.set_ylabel('Validation accuracy (%)',fontsize=10.5)
                ax.legend(fontsize=9.5,loc='lower right',frameon=False)
        fig.tight_layout();save(fig,name)
    # Supplementary paired uncertainty at every common step, no chosen horizon.
    fig,axes=plt.subplots(4,3,figsize=(10.5,10.8),squeeze=False)
    for i,(ds,model) in enumerate(SETTINGS):
        for j,c in enumerate(CONTRASTS):
            ax=axes[i,j];block=[r for r in summary if r['dataset']==ds and r['model']==model and r['contrast']==c]
            x=np.array([int(r['global_step']) for r in block])/1000
            mean=np.array([float(r['mean_accuracy_difference_pp']) for r in block])
            lo=np.array([float(r['descriptive_ci95_lower_pp']) for r in block]);hi=np.array([float(r['descriptive_ci95_upper_pp']) for r in block])
            ax.errorbar(x,mean,yerr=[mean-lo,hi-mean],fmt='o',ms=3,capsize=2,color=colors[i],lw=.7)
            ax.axhline(0,color='.5',lw=.7)
            if j==0:ax.set_ylabel(label(ds,model)+'\nAccuracy difference (pp)',fontsize=10)
            if i==0:ax.set_title(TITLES[j],fontsize=10)
            if i==3:ax.set_xlabel('Transitions (thousands)',fontsize=10)
            ax.tick_params(labelsize=9.5);ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'figS1_all_shared_step_paired_intervals.png')
    print('Generated five complementary figures from preserved records.')

if __name__=='__main__':build()

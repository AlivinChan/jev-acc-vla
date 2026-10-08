"""Durable report figures from saved diagnostics and completed developer pilots."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'figures';out.mkdir(exist_ok=True)
style_source=Path('C:/Users/admin/.codex/skills/deepscientist-figure-polish/assets/deepscientist-academic.mplstyle')
style=(out/'academic.mplstyle')
if style_source.exists():
    text=style_source.read_text(encoding='utf-8')
    # Quote hex colors: unquoted '#' is otherwise interpreted as a style-file comment.
    text=re.sub(r'(:\s*)(#[0-9A-Fa-f]{6})',r'\1"\2"',text)
    style.write_text(text,encoding='utf-8',newline='\n')
plt.style.use(style)
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'savefig.pad_inches':.12})
diagnosis=json.loads((ROOT/'analysis/diagnosis_summary.json').read_text())
reprs=json.loads((ROOT/'analysis/representations_v1_summary.json').read_text())
cadence=json.loads((ROOT/'analysis/cadence_v1_summary.json').read_text())
catalog=[]
def export(fig,name,claim,sources):
    for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    catalog.append(dict(id=name,surface_class='internal_review',claim=claim,
        source_data={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        generating_script='scripts/build_figures.py',exports=[f'figures/{name}.{ext}' for ext in ['png','svg']],
        review_status='rendered_pending_visual_inspection'))

names=['original3','numbers3','fullhead3','binary_full','binary_short','binary_compact']
labels=['Original: 3 choices','Rounded numbers: 3 choices','Larger head budget: 3 choices',
        'Long binary question','Short binary question','Compact state + short binary']
fig,ax=plt.subplots(figsize=(7.8,3.35))
for i,name in enumerate(names):
    row=diagnosis[name];stats=row['max_probability'];color='#AF817C' if name=='binary_full' else '#6A9390'
    ax.plot([stats['min'],stats['max']],[i,i],color=color,lw=2)
    ax.scatter([stats['median']],[i],s=32,color=color,zorder=3)
    ax.text(.902,i,f"{row['cross_07']:3d} / 270",va='center',ha='left',fontsize=9,color='#4B5563')
ax.axvline(.7,color='#958C80',ls='--',lw=1.1)
ax.text(.705,-.51,'0.7 cutoff',ha='left',va='bottom',fontsize=9,color='#756C62')
ax.text(.902,-.51,'Above cutoff',ha='left',va='bottom',fontsize=9,color='#4B5563')
ax.set(yticks=range(6),yticklabels=labels,xlim=(.4,1.01),ylim=(5.65,-.65),
       xticks=[.4,.5,.6,.7,.8,.9],xlabel='Maximum returned probability (median and min–max range)')
ax.grid(axis='y',visible=False);ax.spines['left'].set_visible(False)
fig.suptitle('A larger score did not establish a better decision',x=.01,ha='left',y=1.02,fontsize=12,fontweight='bold')
fig.text(.01,-.10,'270 paired saved states. The long binary question chose CONTINUE in every case.',fontsize=9,color='#6B7280')
export(fig,'diagnostic_probability_ranges','Score inflation in the long binary variant coexists with complete continue collapse.',
       [ROOT/'analysis/diagnosis_summary.json'])

colors={'Smol70':'#969A9E','Short / check5':'#B6A286','Compact / check5':'#B8847C','Compact / check0.3H':'#628F8A'}
markers={'Smol70':'o','Short / check5':'^','Compact / check5':'s','Compact / check0.3H':'D'}
fig,axes=plt.subplots(1,2,figsize=(7.8,3.8),sharey=True)
for ax,h in zip(axes,[50,200]):
    ref=reprs['references'][f'smol70_h{h}']
    rows={'Smol70':ref,'Short / check5':reprs['groups'][f'short5_h{h}'],
          'Compact / check5':reprs['groups'][f'compact5_h{h}'],
          'Compact / check0.3H':cadence['groups'][f'compact_r30_h{h}']}
    for label,row in rows.items():
        n=row['n'];p=row['successes']/n;z=1.95996398454
        center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        x=row['total_service_seconds']/ref['total_service_seconds']
        ax.errorbar(x,100*p,yerr=[[100*(p-center+half)],[100*(center+half-p)]],fmt=markers[label],
                    color=colors[label],ms=6 if label!='Compact / check0.3H' else 7,
                    elinewidth=1,alpha=.95,capsize=2.4,label=label)
    ax.set_xscale('log');ax.set_xticks([.5,1,2,4,8]);ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set(xlim=(.45,9),ylim=(0,100),title=f'Native action length H = {h}',
           xlabel='Total service cost / Smol70 (log scale)')
    ax.axvline(1,color='#B9B3AA',ls='--',lw=.9);ax.grid(axis='x',visible=False)
axes[0].set_ylabel('Successful episodes (%)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.52,-.12),columnspacing=1.7)
fig.suptitle('Sparser checks reduced cost, with a success-rate tradeoff',x=.01,ha='left',y=1.02,fontsize=12,fontweight='bold')
fig.text(.01,-.18,'Developer state0: 10 paired tasks per point. Bars: marginal 95% Wilson intervals.\nCost = synchronized VLA service + full gate wall time; these are not real-robot speedups.',fontsize=8.5,color='#6B7280')
fig.tight_layout(w_pad=1.6)
export(fig,'developer_cost_success_frontier','Frequent compact gating has higher cost; sparse compact gating approaches Smol70 cost on this small developer pilot.',
       [ROOT/'analysis/representations_v1_summary.json',ROOT/'analysis/cadence_v1_summary.json'])
(out/'figure_catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
print(json.dumps(dict(figures=len(catalog),status='rendered; inspect PNG previews before acceptance')))

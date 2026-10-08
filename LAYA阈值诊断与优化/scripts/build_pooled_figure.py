"""Paired developer results after confirming the selected policies on all 30 starts."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'figures'
plt.style.use(out/'academic.mplstyle')
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':10,'savefig.pad_inches':.15})
source=ROOT/'analysis/pooled_development.json';data=json.loads(source.read_text())['groups']
policies=[('compact_r30','Check every 0.3H','#628F8A'),('compact_minr30_i5','Min. age 0.3H, check5','#B8847C')]
fig,axes=plt.subplots(1,2,figsize=(8.0,3.7),sharey=True)
labels=[]
for hi,h in enumerate([50,100,200]):
    for pi,(policy,label,color) in enumerate(policies):
        y=hi*2+pi;row=data[f'{policy}_h{h}'];comp=row['comparisons']['smol70']
        labels.append(f'H={h} / '+('periodic' if pi==0 else 'minimum age'))
        values=[(comp['total_service_cost_ratio'],comp['cost_ratio_95_task_bootstrap']),
                (comp['success_delta']*100,[v*100 for v in comp['success_delta_95_task_bootstrap']])]
        for ax,(point,bounds) in zip(axes,values):
            ax.errorbar(point,y,xerr=[[max(0,point-bounds[0])],[max(0,bounds[1]-point)]],fmt='o',ms=5,
                        color=color,elinewidth=1.2,capsize=2.5,label=label if hi==0 else None)
        axes[0].annotate(f'{comp["total_service_cost_ratio"]:.3f}',(values[0][0],y),xytext=(0,-13),
                         textcoords='offset points',ha='center',fontsize=8,color=color)
        axes[1].annotate(f'{row["successes"]}/30; W/L {comp["wins"]}/{comp["losses"]}',(values[1][0],y),
                         xytext=(0,-13),textcoords='offset points',ha='center',fontsize=8,color=color)
for ax in axes:
    ax.set(ylim=(5.6,-.7),yticks=range(6),yticklabels=labels)
    ax.grid(axis='y',visible=False)
    for y in [1.5,3.5]:ax.axhline(y,color='#E5E7EB',lw=.7)
    ax.spines['left'].set_visible(False)
axes[0].axvline(1,color='#A8A29A',ls='--',lw=1)
axes[0].set(xlim=(.59,1.50),xlabel='Total service cost / paired Smol70',title='Below 1 means lower service cost')
axes[1].axvline(0,color='#A8A29A',ls='--',lw=1)
axes[1].set(xlim=(-28,28),xlabel='Paired success-rate difference (pp)',title='Positive values mean more successes')
fig.suptitle('H100/200 did not retain a net efficiency gain',x=.01,ha='left',y=1.025,fontsize=12,fontweight='bold')
fig.tight_layout(w_pad=1.8)
fig.text(.01,-.07,'30 developer starts per row; 10 tasks × 3 initial states. Bars: task-cluster bootstrap 95% intervals.\n'
         'Descriptive intervals can collapse when task-level differences cancel. Includes candidate selection; not an independent test.',fontsize=8.5,color='#6B7280')
name='pooled_development_paired'
for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight')
plt.close(fig)
catalog_path=out/'figure_catalog.json';catalog=json.loads(catalog_path.read_text())
catalog=[r for r in catalog if r['id']!=name]
catalog.append(dict(id=name,surface_class='internal_review',claim='Selected sparse policies show lower service cost only at H50 on pooled developer starts.',
    source_data={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()},
    generating_script='scripts/build_pooled_figure.py',exports=[f'figures/{name}.{ext}' for ext in ['png','svg']],
    review_status='rendered_pending_visual_inspection'))
catalog_path.write_text(json.dumps(catalog,indent=2))
print(json.dumps(dict(figure=name,status='rendered; visual inspection required')))

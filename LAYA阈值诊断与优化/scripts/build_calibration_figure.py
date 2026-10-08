"""Plot the frozen state2 diagnostic without implying episode-level accuracy."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];out=ROOT/'figures'
source=ROOT/'analysis/calibrators_v1_state2_evaluation.json'
data=json.loads(source.read_text(encoding='utf-8'))
assert not data['fit_performed'] and not data['proposed_deployment_models']
plt.style.use(out/'academic.mplstyle')
variants=['binary_short','binary_compact','binary_window','binary_window3']
labels=['Short','Compact','Window (full)','Window (3 samples)']
kinds=[('raw','Original binary argmax','#628F8A',''),
       ('linear_calibrated','Frozen fitted readout + calibration','#B8847C','///')]
fig,axes=plt.subplots(1,3,figsize=(11.6,4.0),sharex=True)
x=np.arange(4);width=.34
for ki,(kind,label,color,hatch) in enumerate(kinds):
    rows=[data['results'][f'utility__{v}__{kind}']['metrics'] for v in variants]
    assert all(r['n']==54 and r['replan_labels']==18 and r['continue_labels']==36 for r in rows)
    values=[[100*r['coverage_07'] for r in rows],
            [100*r['replan_recall'] for r in rows],
            [100*r['continue_recall'] for r in rows]]
    counts=[[r['above_07'] for r in rows],
            [round(18*r['replan_recall']) for r in rows],
            [round(36*r['continue_recall']) for r in rows]]
    for ai,ax in enumerate(axes):
        xx=x+(ki-.5)*width
        ax.bar(xx,values[ai],width,label=label,color=color,hatch=hatch,edgecolor='white',linewidth=.5)
        for xi,yi,count in zip(xx,values[ai],counts[ai]):
            ax.text(xi,yi+2.4,f'{count}/{[54,18,36][ai]}',ha='center',va='bottom',fontsize=7.6)
for ax,title in zip(axes,['Maximum probability reaches 0.7','Recall when replan is required','Recall when continuing is preferred']):
    ax.set(xticks=x,xticklabels=labels,ylim=(0,116),yticks=[0,25,50,75,100],ylabel='Percent',title=title)
    ax.tick_params(axis='x',labelrotation=27)
    for tick in ax.get_xticklabels():tick.set_horizontalalignment('right')
    ax.grid(axis='x',visible=False)
fig.suptitle('Larger probabilities did not restore replan detection',x=.01,ha='left',y=1.035,fontsize=13,fontweight='bold')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.006,1.005),ncol=2)
fig.tight_layout(rect=(0,0,1,.91),w_pad=1.5)
fig.text(.01,-.095,'Frozen before state2; 54 resolved opportunities out of 178 (30.3%), from 6 tasks. Correlated opportunities across horizons.\n'
         '124 both-failure opportunities are unlabelled. Counts describe this subset; they are not robot episode success rates.',fontsize=9,color='#6B7280')
name='calibration_state2_diagnostic'
for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
catalog_path=out/'figure_catalog.json';catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
catalog=[r for r in catalog if r['id']!=name]
catalog.append(dict(id=name,surface_class='internal_review',
    claim='Frozen readouts yield high maximum probability while missing every replan label in the resolved state2 subset.',
    source_data={source.relative_to(ROOT).as_posix():hashlib.sha256(source.read_bytes()).hexdigest()},
    generating_script='scripts/build_calibration_figure.py',exports=[f'figures/{name}.{ext}' for ext in ['png','svg']],
    review_status='rendered_pending_visual_inspection'))
catalog_path.write_text(json.dumps(catalog,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(dict(figure=name,status='rendered; visual inspection required')))

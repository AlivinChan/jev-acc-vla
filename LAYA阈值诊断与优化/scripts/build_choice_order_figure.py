"""Probability sensitivity to option order, with independently checked semantic mapping."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'raw/choice_order_audit_v1/data/decisions.jsonl'
summary_path = ROOT/'analysis/choice_order_audit_v1_summary.json'
assert json.loads((ROOT/'checks/choice_order_audit_v1_audit.json').read_text(encoding='utf-8'))['errors'] == 0
pairs = defaultdict(dict)
for line in source.read_text(encoding='utf-8').splitlines():
    r = json.loads(line)
    if r['kind'] != 'synthetic_explicit_semantic_control':
        pairs[(r['variant'], r['id'])][tuple(r['order'])] = r
plt.style.use(ROOT/'figures/academic.mplstyle')
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9})
fig, axes = plt.subplots(2, 2, figsize=(7.7, 6.1), sharex=True, sharey=True)
labels = [('binary_full', 'Long binary question'), ('binary_short', 'Short binary question'),
          ('binary_compact', 'Compact state + short question'), ('binary_window3', 'Three-sample window (state0)')]
points = []
for ax, (variant, label) in zip(axes.flat, labels):
    selected = [p for (v, _), p in pairs.items() if v == variant]
    x = np.array([p[(0,1)]['canonical_unrounded_probabilities']['replan'] for p in selected])
    y = np.array([p[(1,0)]['canonical_unrounded_probabilities']['replan'] for p in selected])
    changed = np.array([p[(0,1)]['choice'] != p[(1,0)]['choice'] for p in selected])
    ax.fill_between([.12, .5], .5, .72, color='#F4E6E4', alpha=.45, lw=0)
    ax.fill_between([.5, .72], .12, .5, color='#F4E6E4', alpha=.45, lw=0)
    ax.plot([.12, .72], [.12, .72], color='#CBC7C1', lw=.8)
    ax.axvline(.5, color='#ABA59C', ls='--', lw=.8)
    ax.axhline(.5, color='#ABA59C', ls='--', lw=.8)
    for mask, color, text in [(~changed, '#628F8A', 'Same choice'), (changed, '#B8847C', 'Changed choice')]:
        ax.scatter(x[mask], y[mask], s=12, color=color, alpha=.65, edgecolors='none', label=text)
    ax.set(xlim=(.12,.72), ylim=(.12,.72), xticks=[.2,.3,.4,.5,.6,.7], yticks=[.2,.3,.4,.5,.6,.7])
    ax.set_title(f'{label}\n{int(changed.sum())}/{len(selected)} choices changed', loc='left', pad=8)
    ax.set_aspect('equal')
    points.append(dict(variant=variant, n=len(selected), changed=int(changed.sum()), x=x.tolist(), y=y.tolist()))
for ax in axes[1]:
    ax.set_xlabel('p(replan), original option order')
for ax in axes[:,0]:
    ax.set_ylabel('p(replan), reversed option order')
axes[0,0].legend(loc='lower right', fontsize=8, frameon=False)
fig.suptitle('Option position changes the decision on the same state', x=.01, ha='left', y=1.025,
             fontsize=12, fontweight='bold')
fig.tight_layout(w_pad=1.8, h_pad=2)
fig.text(.01,-.055,'Only the two option slots were exchanged; semantic label mapping passed all checks.\n'
         'Shared developer inputs are repeated observations, not independent robot trials. Changed choices do not imply improved accuracy.',
         fontsize=8.3,color='#6B7280')
name='choice_order_sensitivity'
for ext in ['png','svg']:
    fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.14)
plt.close(fig)
values=ROOT/'checks/choice_order_plotted_values.json'
values.write_text(json.dumps(points,indent=2)+'\n',encoding='utf-8',newline='\n')
catalog_path=ROOT/'figures/figure_catalog.json'
catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
catalog=[r for r in catalog if r['id']!=name]
catalog.append(dict(id=name,surface_class='internal_review',
    claim='Reordering two choices changes many same-state decisions even when semantic probability mapping is correct.',
    source_data={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,summary_path,values]},
    generating_script='scripts/build_choice_order_figure.py',exports=[f'figures/{name}.{ext}' for ext in ['png','svg']],
    review_status='rendered_pending_visual_inspection'))
catalog_path.write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(figure=name,review='required')))

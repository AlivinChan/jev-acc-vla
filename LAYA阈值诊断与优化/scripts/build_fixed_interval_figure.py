"""Expose the horizon/period confound with complete, explicitly reused cells."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1];out=ROOT/'figures'


def main():
    source=ROOT/'analysis/fixed_interval_matrix.json';data=json.loads(source.read_text(encoding='utf-8'))
    assert data['unique_horizon_interval_initial_cells']==240
    plt.style.use(out/'academic.mplstyle')
    horizons=[50,100,200];intervals=[30,45,60]
    success=np.full((3,3),np.nan);calls=np.full((3,3),np.nan)
    for yi,i in enumerate(intervals):
        for xi,h in enumerate(horizons):
            row=data['groups'].get(f'H{h}_I{i}')
            if row is None:assert (h,i)==(50,60);continue
            assert row['n']==30
            success[yi,xi]=row['successes'];calls[yi,xi]=row['vla_calls']
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8))
    for ax,values,title,cmap in zip(axes,[success,calls],['Successful episodes / 30','Total VLA calls / 30 episodes'],['Blues','Oranges']):
        palette=ListedColormap(plt.colormaps[cmap](np.linspace(.03,.63,256)))
        palette.set_bad('#F1F0EE')
        maximum=30 if cmap=='Blues' else float(np.nanmax(calls))*1.18
        ax.imshow(values,cmap=palette,vmin=0,vmax=maximum,aspect='auto')
        for yi,i in enumerate(intervals):
            for xi,h in enumerate(horizons):
                v=values[yi,xi]
                label='Not run\nI > H' if np.isnan(v) else (f'{int(v)}/30' if cmap=='Blues' else f'{int(v)}')
                ax.text(xi,yi,label,ha='center',va='center',fontsize=12,color='#24323E')
                if (h,i) in [(100,30),(200,60)]:
                    ax.add_patch(Rectangle((xi-.47,yi-.45),.94,.9,fill=False,edgecolor='#4B5563',lw=1.3,ls='--'))
        ax.set(xticks=range(3),xticklabels=[f'H={h}' for h in horizons],
               yticks=range(3),yticklabels=[f'I={i} steps' for i in intervals],
               xlabel='Native output length',title=title)
        ax.grid(False);ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Separate output length from request frequency',x=.01,ha='left',y=1.04,fontsize=13,fontweight='bold')
    fig.tight_layout(w_pad=2)
    fig.text(.01,-.07,'Each cell: the same 10 tasks × 3 developer starts. Dashed borders: original Smol70 cells reused once.\n'
             'Frozen VLA, logical d=1, no LAYA. Counts are descriptive; this grid includes development data.',fontsize=9,color='#6B7280')
    name='fixed_interval_attribution'
    for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    catalog_path=out/'figure_catalog.json';catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    catalog=[r for r in catalog if r['id']!=name]
    catalog.append(dict(id=name,surface_class='internal_review',
        claim='Same-interval cells separate native horizon and request cadence on matched developer starts.',
        source_data={source.relative_to(ROOT).as_posix():hashlib.sha256(source.read_bytes()).hexdigest()},
        generating_script='scripts/build_fixed_interval_figure.py',exports=[f'figures/{name}.{ext}' for ext in ['png','svg']],
        review_status='rendered_pending_visual_inspection'))
    catalog_path.write_text(json.dumps(catalog,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(figure=name,status='rendered; visual inspection required')))


if __name__=='__main__':main()

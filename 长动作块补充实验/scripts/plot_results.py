"""Render verified aggregate results; CPU Matplotlib only."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['smol70', 'laya', 'naive_k5', 'vlash_style_k5']
LABELS = {'smol70':'SmolVLA 70% queue', 'laya':'LAYA (tau=0.7) + frozen SmolVLA',
          'naive_k5':'Naive K=5', 'vlash_style_k5':'VLASH-style K=5 (frozen transfer)'}
COLORS = {'smol70':'#91999E','laya':'#647D70','naive_k5':'#B7A99A','vlash_style_k5':'#B88C8C'}
MARKERS = {'smol70':'o','laya':'X','naive_k5':'s','vlash_style_k5':'D'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.metrics.read_text(encoding='utf-8'))
    assert data['completion_status']=='complete', 'Final figures require complete audited data'
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#D8D1C7','axes.labelcolor':'#4B5563',
                         'xtick.color':'#6B7280','ytick.color':'#6B7280',
                         'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'legend.frameon':False,'legend.fontsize':8.7,'svg.fonttype':'none',
                         'savefig.facecolor':'white','savefig.dpi':220})
    lookup = {(g['horizon'],g['method']):g for g in data['groups']}
    horizons = [50,100,200]
    x = np.arange(3)
    exports = []
    def export(fig, stem):
        for ext in ['png','svg']:
            path = args.out/(stem+'.'+ext)
            fig.savefig(path, bbox_inches='tight', pad_inches=.1)
            exports.append(str(path))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.3,4.25))
    for i, method in enumerate(METHODS):
        rows = [lookup[h,method] for h in horizons]
        means = np.array([r['success_rate']*100 for r in rows])
        low = means-np.array([r['wilson_low']*100 for r in rows])
        high = np.array([r['wilson_high']*100 for r in rows])-means
        ax.errorbar(x+(i-1.5)*.085, means, yerr=np.array([low,high]),
                    color=COLORS[method], marker=MARKERS[method], linestyle='none',
                    markersize=7 if method=='laya' else 5, capsize=3,
                    elinewidth=1.2, label=LABELS[method])
    ax.set(xticks=x, xticklabels=horizons, ylim=(-3,103),
           xlabel='Native predicted action count (H)', ylabel='Task success (%)',
           title='Frozen SmolVLA: output length and closed-loop quality')
    ax.grid(axis='y',color='#E7E5E4',linewidth=.65)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2)
    fig.text(.5, -.09, '30 paired starts / 10 tasks per point; Wilson 95% (binomial approximation).\nTask correlation is not corrected; exposed development states.',
             ha='center', va='top', fontsize=8, color='#6B7280')
    fig.subplots_adjust(bottom=.26)
    export(fig,'success_by_horizon')
    fig, axes = plt.subplots(1,2,figsize=(8.4,3.6))
    for method in METHODS:
        rows = [lookup[h,method] for h in horizons]
        for ax, values in zip(axes, [[r['calls_per100steps'] for r in rows],
                                   [100*r['vla_plus_gate_seconds']/r['steps'] for r in rows]]):
            ax.plot(x,values,color=COLORS[method],marker=MARKERS[method],
                    linewidth=2.2 if method=='laya' else 1.5,
                    markersize=6 if method=='laya' else 4.5,label=LABELS[method])
    for ax in axes:
        ax.set(xticks=x,xticklabels=horizons,xlabel='Native action count (H)',ylim=(0,None))
        ax.grid(axis='y',color='#E7E5E4',linewidth=.65)
        ax.set_axisbelow(True)
    axes[0].set(ylabel='VLA calls / 100 control steps',title='Call frequency')
    axes[1].set(ylabel='Seconds / 100 control steps',title='VLA + gate service wall cost')
    axes[0].text(.03,.81,'Smol70 and LAYA call counts coincide.',transform=axes[0].transAxes,
                 fontsize=8,color='#6B7280')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.025))
    fig.text(.5, -.11, 'Same frozen SmolVLA; 30 starts / 10 tasks per setting. Includes host and IPC overhead.\nUnpaced serial simulation; not GPU-active time or real asynchronous robot latency.',
             ha='center', va='top', fontsize=8, color='#6B7280')
    fig.subplots_adjust(bottom=.25,wspace=.35)
    export(fig,'cost_by_horizon')
    catalog = dict(source_metrics=str(args.metrics.resolve()),
                   generating_script=str(Path(__file__).resolve()),
                   surface_class='appendix', exports=exports,
                   interpretation='Same frozen SmolVLA only; reference pi0.5 is excluded. Success intervals are descriptive Wilson intervals. Cost includes host and gate overhead, not GPU-active time or real asynchronous robot latency.',
                   render_review_status='pending_visual_inspection')
    (args.out/'figure_catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    print(json.dumps(catalog))


if __name__=='__main__':
    main()

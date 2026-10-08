"""Visualize the registered row, state and noise contrasts without policy selection."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', default='noise_alignment_validation_d4_v1')
    args = parser.parse_args()
    assert args.cohort.isascii() and args.cohort.replace('_', '').isalnum()
    source = ROOT/'analysis'/f'{args.cohort}_attribution.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    out = ROOT/'figures'
    plt.style.use(out/'academic.mplstyle')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10})
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.4), sharey=True)
    headings = [
        ('execution_row', 'A. Execute row 0 instead of dropping the delayed prefix'),
        ('current_proprioception', 'B. Use current robot state with the same image-time noise'),
        ('noise_clock', 'C. Switch to prediction-time noise with the same input rule'),
    ]
    colors = ['#628F8A', '#B8847C', '#9D8EAE']
    points = []
    success_values, cost_values = [0.], [1.]
    for row, (name, title) in enumerate(headings):
        effect = data['effects'][name]
        for y, (h, color) in enumerate(zip([50, 100, 200], colors)):
            item = effect['by_horizon'][str(h)]
            pair = item['paired']
            intervals = pair['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
            success = 100*pair['success_delta']
            success_ci = [100*x for x in intervals['success_delta_95']]
            cost, cost_ci = pair['total_service_cost_ratio'], intervals['cost_ratio_95']
            success_values.extend([success, *success_ci])
            cost_values.extend([cost, *cost_ci])
            for col, value, ci in [(0, success, success_ci), (1, cost, cost_ci)]:
                axes[row, col].plot(ci, [y, y], color=color, lw=1.4)
                axes[row, col].plot(ci, [y, y], linestyle='none', marker='|', color=color, markersize=7)
                axes[row, col].scatter([value], [y], color=color, s=28, zorder=3)
                text = f'{success:+.0f} pp; W/L {pair["wins"]}/{pair["losses"]}' if col == 0 else f'{cost:.3f}'
                axes[row, col].annotate(text, (value, y), xytext=(0, -14), textcoords='offset points',
                                       ha='center', fontsize=8.5, color=color)
            points.append(dict(contrast=name, horizon=h, n=pair['n'], success_delta=pair['success_delta'],
                               success_delta_95=intervals['success_delta_95'], cost_ratio=cost, cost_ratio_95=cost_ci))
        axes[row, 0].set_title(title, loc='left', pad=15)
    success_extent = max(5, np.ceil(max(abs(v) for v in success_values)/5)*5+5)
    cost_width = max(cost_values)-min(cost_values)
    cost_pad = max(.05, cost_width*.12)
    for row in range(3):
        axes[row, 0].axvline(0, color='#A8A29A', ls='--', lw=1)
        axes[row, 0].set_xlim(-success_extent, success_extent)
        axes[row, 1].axvline(1, color='#A8A29A', ls='--', lw=1)
        axes[row, 1].set_xlim(max(0, min(cost_values)-cost_pad), max(cost_values)+cost_pad)
    for ax in axes.flat:
        ax.set(yticks=range(3), yticklabels=['H=50', 'H=100', 'H=200'], ylim=(2.6, -.6))
        ax.grid(axis='y', visible=False)
        ax.spines['left'].set_visible(False)
    axes[-1, 0].set_xlabel('Paired success-rate difference (pp)')
    axes[-1, 1].set_xlabel('Total model-service cost ratio')
    fig.suptitle('Same frozen SmolVLA: separate three K5 mechanisms', x=.01, ha='left', y=1.015,
                 fontsize=12, fontweight='bold')
    fig.tight_layout(w_pad=2.3, h_pad=2.5)
    fig.text(.01, -.045, f'{pair["n"]} paired starts per H; 10 tasks; logical delay d=4. Hierarchical bootstrap 95% intervals.\n'
             'Simulation pauses during inference. These controls do not include official VLASH offset training.',
             fontsize=8.5, color='#6B7280')
    name = args.cohort+'_attribution'
    for ext in ['png', 'svg']:
        fig.savefig(out/f'{name}.{ext}', dpi=240, bbox_inches='tight', pad_inches=.14)
    plt.close(fig)
    receipt = ROOT/'checks'/f'{name}_plotted_values.json'
    receipt.write_text(json.dumps(dict(source_sha256=sha(source), points=points), indent=2)+'\n', encoding='utf-8', newline='\n')
    catalog_path = out/'figure_catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    catalog = [r for r in catalog if r['id'] != name]
    catalog.append(dict(id=name, surface_class='internal_review',
                        claim='Full registered one-mechanism paired comparisons, with uncertainty and separate command-level controls.',
                        source_data={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [source, receipt]},
                        generating_script='scripts/build_noise_attribution_figure.py',
                        exports=[f'figures/{name}.{ext}' for ext in ['png', 'svg']],
                        review_status='rendered_pending_visual_inspection'))
    catalog_path.write_text(json.dumps(catalog, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(figure=name, status='rendered; visual inspection required')))


if __name__ == '__main__':
    main()

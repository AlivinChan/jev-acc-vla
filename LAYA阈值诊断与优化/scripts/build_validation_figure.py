"""Paired validation effects from a complete audited cohort; never refit a policy."""
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
    parser.add_argument('--cohort', default='validation_d1_v2')
    args = parser.parse_args()
    assert args.cohort.isascii() and args.cohort.replace('_', '').isalnum()
    source = ROOT/'analysis'/f'{args.cohort}_summary.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    definition_path = ROOT/'protocol'/args.cohort/'definition.json'
    definition = json.loads(definition_path.read_text(encoding='utf-8'))
    assert sha(definition_path) == data['definition_sha256']
    assert data['total_episodes'] == definition['total_episodes']
    assert not definition.get('implementation_pilot', False), 'Do not present a qualification pilot as validation'
    comparisons = [('smol70', 'Compared with Smol70'), ('fixed45', 'Compared with fixed 45-step period')]
    if args.cohort.startswith('budget_'):
        comparisons = [('smol70', 'Compared with Smol70'),
                       ('budget_period_h{h}', 'Compared with a cost-selected period')]
    elif args.cohort.startswith('random_skip_'):
        comparisons = [('random_skip_s11131', 'Compared with random skip (seed 11131)'),
                       ('random_skip_s22261', 'Compared with random skip (seed 22261)')]
    out = ROOT/'figures'
    plt.style.use(out/'academic.mplstyle')
    plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 9.5})
    fig, axes = plt.subplots(2, len(comparisons), figsize=(4.05*len(comparisons), 5.5), sharey=True)
    colors = ['#628F8A', '#B8847C', '#9D8EAE']
    collected = []
    success_limits, cost_limits = [], []
    for col, (reference_pattern, label) in enumerate(comparisons):
        for y, (h, color) in enumerate(zip(definition['horizons'], colors)):
            row = data['groups'][f'{data["primary_candidate"]}_h{h}']
            reference = reference_pattern.format(h=h)
            comp = row['comparisons'][reference]
            ci = comp['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
            n = row['n']
            success = 100*comp['success_delta']
            success_ci = [100*x for x in ci['success_delta_95']]
            cost = comp['total_service_cost_ratio']
            cost_ci = ci['cost_ratio_95']
            assert np.isfinite([success, cost, *success_ci, *cost_ci]).all()
            for ax, point, interval in [(axes[0, col], success, success_ci), (axes[1, col], cost, cost_ci)]:
                ax.plot(interval, [y, y], color=color, lw=1.5)
                ax.plot(interval, [y, y], linestyle='none', marker='|', color=color, markersize=7)
                ax.scatter([point], [y], color=color, s=30, zorder=3)
            axes[0, col].annotate(f'{success:+.0f} pp; W/L {comp["wins"]}/{comp["losses"]}',
                                 (success, y), xytext=(0, -15), textcoords='offset points',
                                 ha='center', fontsize=8.4, color=color)
            axes[1, col].annotate(f'{cost:.3f}', (cost, y), xytext=(0, -15),
                                 textcoords='offset points', ha='center', fontsize=8.4, color=color)
            success_limits.extend([success, *success_ci])
            cost_limits.extend([cost, *cost_ci])
            collected.append(dict(horizon=h, comparator=reference, n=n,
                                  success_delta=comp['success_delta'], success_delta_ci=ci['success_delta_95'],
                                  service_cost_ratio=cost, service_cost_ratio_ci=cost_ci))
        axes[0, col].set_title(label, pad=12)
        axes[0, col].axvline(0, color='#A8A29A', ls='--', lw=1)
        axes[0, col].set_xlabel('Paired success-rate difference (pp)')
        axes[1, col].axvline(1, color='#A8A29A', ls='--', lw=1)
        axes[1, col].set_xlabel('Total model-service cost ratio')
    extent = max(5, np.ceil(max(abs(x) for x in success_limits)/5)*5+5)
    cost_range = max(cost_limits+[1])-min(cost_limits+[1])
    padding = max(.08, cost_range*.12)
    for col in range(len(comparisons)):
        axes[0, col].set_xlim(-extent, extent)
        axes[1, col].set_xlim(max(0, min(cost_limits+[1])-padding), max(cost_limits+[1])+padding)
    for ax in axes.flat:
        ax.set(yticks=range(3), yticklabels=[f'H={h}' for h in definition['horizons']], ylim=(2.65, -.6))
        ax.grid(axis='y', visible=False)
        ax.spines['left'].set_visible(False)
    fig.suptitle('Frozen LAYA binary argmax: paired validation', x=.01, ha='left', y=1.025,
                 fontsize=12, fontweight='bold')
    fig.tight_layout(w_pad=2.5, h_pad=1.8)
    fig.text(.01, -.055, f'{n} paired starts per point; 10 tasks. Bars: hierarchical task-and-initial bootstrap 95% intervals.\n'
             'Descriptive uncertainty; no equivalence claim. Cost includes VLA + gate service; simulation pauses during inference.',
             fontsize=8.4, color='#6B7280')
    name = args.cohort+'_paired'
    for ext in ['png', 'svg']:
        fig.savefig(out/f'{name}.{ext}', dpi=240, bbox_inches='tight', pad_inches=.14)
    plt.close(fig)
    receipt_path = ROOT/'checks'/f'{name}_plotted_values.json'
    receipt_path.write_text(json.dumps(dict(source_sha256=sha(source), points=collected), indent=2)+'\n',
                            encoding='utf-8', newline='\n')
    catalog_path = out/'figure_catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    catalog = [r for r in catalog if r['id'] != name]
    catalog.append(dict(id=name, surface_class='internal_review',
                        claim='Complete frozen-cohort paired success and full-service cost comparisons with descriptive uncertainty.',
                        source_data={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [source, definition_path, receipt_path]},
                        generating_script='scripts/build_validation_figure.py',
                        exports=[f'figures/{name}.{ext}' for ext in ['png', 'svg']],
                        review_status='rendered_pending_visual_inspection'))
    catalog_path.write_text(json.dumps(catalog, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(figure=name, status='rendered; visual inspection required')))


if __name__ == '__main__':
    main()

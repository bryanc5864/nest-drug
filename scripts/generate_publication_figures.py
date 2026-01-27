#!/usr/bin/env python3
"""
Generate publication-quality figures for NEST-DRUG paper.

Figures:
1. L1 Ablation Bar Chart: Correct L1 vs Generic L1 across targets (V2 + V3)
2. Context Attribution Heatmap: Per-atom importance for different targets
3. Model Comparison Radar Chart: V1 vs V2 vs V3 across benchmarks

Usage:
    python scripts/generate_publication_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Style settings for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = Path('results/experiments')
OUTPUT_DIR = Path('results/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 1: L1 Ablation Bar Chart
# ============================================================
def figure_l1_ablation():
    """Bar chart comparing Correct L1 vs Generic L1 for V2 and V3."""
    # V2 ablation data
    v2_data = load_json(RESULTS_DIR / 'v2_ablation' / 'v3_ablation_correct_results.json')

    # V3 ablation data - use hardcoded values from RESULTS.md (JSON was overwritten)
    # These were captured from original terminal output before V1 overwrite
    v3_per_target = {
        'egfr':  {'with_correct_l1': 0.961, 'with_generic_l1': 0.826},
        'drd2':  {'with_correct_l1': 0.987, 'with_generic_l1': 0.905},
        'adrb2': {'with_correct_l1': 0.786, 'with_generic_l1': 0.715},
        'bace1': {'with_correct_l1': 0.656, 'with_generic_l1': 0.776},
        'esr1':  {'with_correct_l1': 0.899, 'with_generic_l1': 0.775},
        'hdac2': {'with_correct_l1': 0.929, 'with_generic_l1': 0.830},
        'jak2':  {'with_correct_l1': 0.908, 'with_generic_l1': 0.855},
        'pparg': {'with_correct_l1': 0.842, 'with_generic_l1': 0.761},
        'cyp3a4':{'with_correct_l1': 0.689, 'with_generic_l1': 0.638},
        'fxa':   {'with_correct_l1': 0.846, 'with_generic_l1': 0.826},
    }

    targets = list(v3_per_target.keys())
    target_labels = [t.upper() for t in targets]

    # V3 data
    v3_correct = [v3_per_target[t]['with_correct_l1'] for t in targets]
    v3_generic = [v3_per_target[t]['with_generic_l1'] for t in targets]

    # V2 data
    v2_correct = [v2_data['per_target'][t]['with_correct_l1'] for t in targets]
    v2_generic = [v2_data['per_target'][t]['with_generic_l1'] for t in targets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    x = np.arange(len(targets))
    width = 0.35

    # V3 panel
    ax = axes[0]
    bars1 = ax.bar(x - width/2, v3_correct, width, label='Correct L1', color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, v3_generic, width, label='Generic L1 (id=0)', color='#BBDEFB', edgecolor='white')
    ax.set_xlabel('Target')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('V3-FineTuned: L1 Ablation (+6% mean)')
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0.3, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random')

    # Add delta annotations
    for i, (c, g) in enumerate(zip(v3_correct, v3_generic)):
        delta = c - g
        color = '#4CAF50' if delta > 0 else '#F44336'
        ax.annotate(f'{delta:+.3f}', xy=(i, max(c, g) + 0.01),
                   ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

    # V2 panel
    ax = axes[1]
    bars1 = ax.bar(x - width/2, v2_correct, width, label='Correct L1', color='#FF9800', edgecolor='white')
    bars2 = ax.bar(x + width/2, v2_generic, width, label='Generic L1 (id=0)', color='#FFE0B2', edgecolor='white')
    ax.set_xlabel('Target')
    ax.set_title('V2-Expanded: L1 Ablation (+29% mean)')
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    for i, (c, g) in enumerate(zip(v2_correct, v2_generic)):
        delta = c - g
        color = '#4CAF50' if delta > 0 else '#F44336'
        ax.annotate(f'{delta:+.3f}', xy=(i, max(c, g) + 0.01),
                   ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig1_l1_ablation.png'
    plt.savefig(out)
    plt.savefig(OUTPUT_DIR / 'fig1_l1_ablation.pdf')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Figure 2: Context Attribution Heatmap
# ============================================================
def figure_context_attribution():
    """Heatmap of per-atom importance across targets for a single molecule."""
    data = load_json(RESULTS_DIR / 'context_attribution_fixed' / 'context_attribution_results.json')

    for mol_result in data['results']:
        mol_name = mol_result['name']
        attributions = mol_result['attributions']
        targets = list(attributions.keys())
        n_atoms = len(attributions[targets[0]]['importance'])

        # Build matrix: targets x atoms
        matrix = np.array([attributions[t]['importance'] for t in targets])

        # Normalize each row to [0, 1]
        for i in range(len(targets)):
            row = matrix[i]
            if row.max() > row.min():
                matrix[i] = (row - row.min()) / (row.max() - row.min())

        fig, ax = plt.subplots(figsize=(max(8, n_atoms * 0.35), 3))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets)
        ax.set_xlabel('Atom Index')
        ax.set_title(f'{mol_name}: Context-Conditional Attribution')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Normalized Importance')

        plt.tight_layout()
        out = OUTPUT_DIR / f'fig2_attribution_{mol_name.lower()}.png'
        plt.savefig(out)
        plt.savefig(OUTPUT_DIR / f'fig2_attribution_{mol_name.lower()}.pdf')
        plt.close()
        print(f"  Saved: {out}")

    # Also make a combined divergence summary
    fig, ax = plt.subplots(figsize=(6, 4))
    mol_names = [r['name'] for r in data['results']]
    mean_kls = []
    mean_cos = []
    for r in data['results']:
        kls = [d['kl_divergence'] for d in r['divergences']]
        cos = [d['cosine_similarity'] for d in r['divergences']]
        mean_kls.append(np.mean(kls))
        mean_cos.append(np.mean(cos))

    x = np.arange(len(mol_names))
    width = 0.35
    ax.bar(x - width/2, mean_kls, width, label='KL Divergence', color='#E91E63')
    ax.bar(x + width/2, [1 - c for c in mean_cos], width, label='1 - Cosine Sim', color='#9C27B0')
    ax.set_xticks(x)
    ax.set_xticklabels(mol_names)
    ax.set_ylabel('Divergence')
    ax.set_title('Attribution Divergence Across Targets (V3)')
    ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / 'fig2_attribution_divergence.png'
    plt.savefig(out)
    plt.savefig(OUTPUT_DIR / 'fig2_attribution_divergence.pdf')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Figure 3: Radar Chart - Model Comparison
# ============================================================
def figure_radar_comparison():
    """Radar chart comparing V1, V2, V3 across key metrics."""
    # Metrics from RESULTS.md (verified by audit)
    categories = [
        'DUD-E Mean\n(generic L1)',
        'DUD-E Mean\n(correct L1)',
        'Temporal\nROC-AUC',
        'TDC hERG',
        'TDC BBB',
        'L1 Delta',
    ]

    # Values (normalized to 0-1 scale where meaningful)
    v1_vals = [0.803, 0.805, 0.912, 0.727, 0.605, 0.0]
    v2_vals = [0.557, 0.850, 0.644, 0.450, 0.371, 0.293]
    v3_vals = [0.791, 0.850, 0.843, 0.628, 0.570, 0.060]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    v1_vals += v1_vals[:1]
    v2_vals += v2_vals[:1]
    v3_vals += v3_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, v1_vals, 'o-', linewidth=2, label='V1-Original', color='#4CAF50')
    ax.fill(angles, v1_vals, alpha=0.1, color='#4CAF50')

    ax.plot(angles, v2_vals, 's-', linewidth=2, label='V2-Expanded', color='#FF9800')
    ax.fill(angles, v2_vals, alpha=0.1, color='#FF9800')

    ax.plot(angles, v3_vals, '^-', linewidth=2, label='V3-FineTuned', color='#2196F3')
    ax.fill(angles, v3_vals, alpha=0.1, color='#2196F3')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    ax.set_title('NEST-DRUG Model Comparison', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig3_radar_comparison.png'
    plt.savefig(out)
    plt.savefig(OUTPUT_DIR / 'fig3_radar_comparison.pdf')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Figure 4: DUD-E Per-Target Comparison (V1 vs V3)
# ============================================================
def figure_dude_comparison():
    """Grouped bar chart of DUD-E per-target AUC for V1 vs V3."""
    targets = ['egfr', 'drd2', 'adrb2', 'bace1', 'esr1', 'hdac2', 'jak2', 'pparg', 'cyp3a4', 'fxa']
    target_labels = [t.upper() for t in targets]

    # From RESULTS.md (verified by audit)
    v1_aucs = [0.943, 0.960, 0.745, 0.672, 0.864, 0.866, 0.865, 0.787, 0.497, 0.833]
    v3_aucs = [0.899, 0.934, 0.763, 0.842, 0.817, 0.901, 0.862, 0.748, 0.782, 0.846]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(targets))
    width = 0.35

    bars1 = ax.bar(x - width/2, v1_aucs, width, label='V1-Original', color='#4CAF50', edgecolor='white')
    bars2 = ax.bar(x + width/2, v3_aucs, width, label='V3-FineTuned', color='#2196F3', edgecolor='white')

    ax.set_xlabel('Target')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('DUD-E Virtual Screening: V1 vs V3')
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    # Add mean lines
    ax.axhline(y=np.mean(v1_aucs), color='#4CAF50', linestyle=':', alpha=0.5)
    ax.axhline(y=np.mean(v3_aucs), color='#2196F3', linestyle=':', alpha=0.5)
    ax.text(len(targets)-0.5, np.mean(v1_aucs)+0.01, f'V1 mean={np.mean(v1_aucs):.3f}',
            color='#4CAF50', fontsize=9, ha='right')
    ax.text(len(targets)-0.5, np.mean(v3_aucs)+0.01, f'V3 mean={np.mean(v3_aucs):.3f}',
            color='#2196F3', fontsize=9, ha='right')

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig4_dude_comparison.png'
    plt.savefig(out)
    plt.savefig(OUTPUT_DIR / 'fig4_dude_comparison.pdf')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Figure 5: V2 "Rehabilitation" - L1 context is the key
# ============================================================
def figure_v2_rehabilitation():
    """Shows that V2 appears broken without L1 but achieves 0.850 with correct L1."""
    v2_data = load_json(RESULTS_DIR / 'v2_ablation' / 'v3_ablation_correct_results.json')

    targets = list(v2_data['per_target'].keys())
    target_labels = [t.upper() for t in targets]

    correct = [v2_data['per_target'][t]['with_correct_l1'] for t in targets]
    generic = [v2_data['per_target'][t]['with_generic_l1'] for t in targets]

    # Sort by delta for visual impact
    deltas = [c - g for c, g in zip(correct, generic)]
    sorted_idx = np.argsort(deltas)[::-1]

    targets_sorted = [target_labels[i] for i in sorted_idx]
    correct_sorted = [correct[i] for i in sorted_idx]
    generic_sorted = [generic[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(targets))
    height = 0.35

    ax.barh(y + height/2, correct_sorted, height, label='With Correct L1', color='#2196F3', edgecolor='white')
    ax.barh(y - height/2, generic_sorted, height, label='With Generic L1 (id=0)', color='#FFCDD2', edgecolor='white')

    ax.set_yticks(y)
    ax.set_yticklabels(targets_sorted)
    ax.set_xlabel('ROC-AUC')
    ax.set_title('V2-Expanded: "Broken" Model Rehabilitated by Correct L1 Context')
    ax.legend(loc='lower right')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, label='Random')
    ax.set_xlim(0.2, 1.05)

    # Add delta annotations
    for i, (c, g) in enumerate(zip(correct_sorted, generic_sorted)):
        delta = c - g
        ax.annotate(f'+{delta:.2f}' if delta > 0 else f'{delta:.2f}',
                   xy=(max(c, g) + 0.01, i),
                   ha='left', va='center', fontsize=9,
                   color='#4CAF50' if delta > 0 else '#F44336',
                   fontweight='bold')

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig5_v2_rehabilitation.png'
    plt.savefig(out)
    plt.savefig(OUTPUT_DIR / 'fig5_v2_rehabilitation.pdf')
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("Generating publication figures...")
    print()

    print("Figure 1: L1 Ablation Bar Chart")
    figure_l1_ablation()

    print("\nFigure 2: Context Attribution Heatmap")
    figure_context_attribution()

    print("\nFigure 3: Radar Chart - Model Comparison")
    figure_radar_comparison()

    print("\nFigure 4: DUD-E Per-Target Comparison")
    figure_dude_comparison()

    print("\nFigure 5: V2 Rehabilitation")
    figure_v2_rehabilitation()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

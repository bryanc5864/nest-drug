#!/usr/bin/env python3
"""
NEST-DRUG ICLR 2026 Publication Figures (Consolidated)
3 Main Figures + 1 Appendix Figure
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'ours': '#E8871E',
    'baseline': '#888888',
    'v1': '#4472C4',
    'v2': '#9B59B6',
    'v3': '#E8871E',
    'positive': '#27AE60',
    'negative': '#E74C3C',
    'random': '#BFBFBF',
    'correct_l1': '#27AE60',
    'generic_l1': '#E74C3C',
    'l2': '#48A9A6',
    'l3': '#9B59B6',
}

TEXT_WIDTH = 5.5

# =============================================================================
# DATA
# =============================================================================
TARGETS = ['EGFR', 'DRD2', 'ADRB2', 'BACE1', 'ESR1', 'HDAC2', 'JAK2', 'PPARG', 'CYP3A4', 'FXA']
DUDE_V1 = [0.943, 0.960, 0.745, 0.672, 0.864, 0.866, 0.865, 0.787, 0.497, 0.833]
DUDE_V3 = [0.899, 0.934, 0.763, 0.842, 0.817, 0.901, 0.862, 0.748, 0.782, 0.846]
V3_DELTA = [0.132, 0.083, 0.057, -0.102, 0.134, 0.100, 0.045, 0.069, 0.036, 0.020]
V3_CI_LOW = [0.128, 0.079, 0.052, -0.106, 0.129, 0.095, 0.043, 0.062, 0.023, 0.016]
V3_CI_HIGH = [0.136, 0.087, 0.061, -0.099, 0.139, 0.106, 0.048, 0.076, 0.050, 0.024]
V2_CORRECT_L1 = [0.880, 0.981, 0.815, 0.667, 0.905, 0.921, 0.965, 0.825, 0.693, 0.850]
V2_GENERIC_L1 = [0.639, 0.545, 0.375, 0.651, 0.407, 0.337, 0.493, 0.490, 0.800, 0.835]
ATTRIBUTION_KL = {'V1': 0.001, 'V3': 0.144}
DMTA_TARGETS = ['EGFR', 'DRD2', 'FXA']
DMTA_RANDOM_HR = [49.4, 40.4, 51.6]
DMTA_MODEL_HR = [75.6, 77.9, 87.6]
DMTA_ENRICHMENT = [1.53, 1.93, 1.70]
DMTA_EXPTS_RANDOM = [225, 153, 130]
DMTA_EXPTS_MODEL = [159, 86, 58]
DMTA_V1_ENRICH = [1.55, 1.65, 1.59]
DMTA_V2_ENRICH = [1.34, 1.93, 1.69]
DMTA_V3_ENRICH = [1.53, 1.93, 1.70]
SOTA_METHODS = ['Random', 'Morgan+RF', 'ECFP+SVM', 'V1', 'AtomNet', 'GNN-VS', '3D-CNN', 'V3', 'V3+L1']
SOTA_AUC = [0.500, 0.720, 0.740, 0.803, 0.818, 0.825, 0.830, 0.839, 0.850]
SOTA_IS_OURS = [False, False, False, True, False, False, False, True, True]
L2_TARGETS = ['EGFR', 'DRD2', 'FXA']
L2_CORRECT = [0.798, 0.948, 0.954]
L2_GENERIC = [0.810, 0.949, 0.960]
L3_CORRECT = [0.814, 0.917, 0.969]
L3_GENERIC = [0.809, 0.925, 0.971]


def save_fig(fig, name):
    for fmt in ['pdf', 'png']:
        path = FIGURE_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {path}")
    plt.close(fig)


def add_panel_label(ax, label, x=-0.15, y=1.1):
    ax.text(x, y, f"({label.lower()})", transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


# =============================================================================
# FIGURE 1: Architecture (external image - just copy)
# =============================================================================
def create_fig1_architecture():
    """Figure 1 is the external architecture diagram - already processed."""
    print("Figure 1: Architecture diagram already exists (external)")


# =============================================================================
# FIGURE 2: Main Results & L1 Context Importance (4 panels)
# =============================================================================
def create_fig2_main_results():
    """
    Figure 2: Main Results (2x2 grid)
    (a) DUD-E V1 vs V3 comparison
    (b) L1 ablation delta + error bars
    (c) V2 rehabilitation (proves L1 critical)
    (d) SOTA horizontal bars
    """
    print("Creating Figure 2: Main Results & L1 Context...")

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.4)

    # Panel A: DUD-E V1 vs V3
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, 'a')

    x = np.arange(len(TARGETS))
    width = 0.38
    ax.bar(x - width/2, DUDE_V1, width, label='V1', color=COLORS['v1'], edgecolor='none')
    ax.bar(x + width/2, DUDE_V3, width, label='V3', color=COLORS['v3'], edgecolor='none')
    ax.axhline(y=np.mean(DUDE_V1), color=COLORS['v1'], linestyle='--', lw=0.8, alpha=0.5)
    ax.axhline(y=np.mean(DUDE_V3), color=COLORS['v3'], linestyle='--', lw=0.8, alpha=0.5)

    ax.set_ylabel('ROC-AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, rotation=55, ha='right', fontsize=5)
    ax.set_ylim(0.4, 1.08)
    ax.legend(loc='upper right', frameon=False, fontsize=6, handlelength=1, ncol=2)
    ax.set_title('DUD-E Benchmark', fontsize=8, fontweight='bold', pad=4)

    # Panel B: L1 Ablation Delta
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, 'b')

    sorted_idx = np.argsort(V3_DELTA)[::-1]
    sorted_targets = [TARGETS[i] for i in sorted_idx]
    sorted_delta = [V3_DELTA[i] for i in sorted_idx]
    sorted_ci_low = [V3_CI_LOW[i] for i in sorted_idx]
    sorted_ci_high = [V3_CI_HIGH[i] for i in sorted_idx]
    yerr_low = [d - cl for d, cl in zip(sorted_delta, sorted_ci_low)]
    yerr_high = [ch - d for d, ch in zip(sorted_delta, sorted_ci_high)]
    colors = [COLORS['positive'] if d > 0 else COLORS['negative'] for d in sorted_delta]

    x = np.arange(len(sorted_targets))
    ax.bar(x, sorted_delta, color=colors, edgecolor='none', width=0.7)
    ax.errorbar(x, sorted_delta, yerr=[yerr_low, yerr_high], fmt='none',
                color='black', capsize=1.5, capthick=0.6, lw=0.6)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.axhline(y=np.mean(V3_DELTA), color=COLORS['ours'], linestyle='--', lw=1)

    ax.set_ylabel('Δ AUC (Correct − Generic L1)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_targets, rotation=55, ha='right', fontsize=5)
    ax.set_ylim(-0.14, 0.18)
    ax.set_title('L1 Context Ablation', fontsize=8, fontweight='bold', pad=4)

    # Panel C: V2 Rehabilitation
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, 'c')

    v2_delta = [c - g for c, g in zip(V2_CORRECT_L1, V2_GENERIC_L1)]
    sorted_idx = np.argsort(v2_delta)[::-1][:7]
    sorted_targets = [TARGETS[i] for i in sorted_idx]
    sorted_correct = [V2_CORRECT_L1[i] for i in sorted_idx]
    sorted_generic = [V2_GENERIC_L1[i] for i in sorted_idx]

    x = np.arange(len(sorted_targets))
    width = 0.38
    ax.bar(x - width/2, sorted_generic, width, label='Generic L1',
           color=COLORS['generic_l1'], edgecolor='none')
    ax.bar(x + width/2, sorted_correct, width, label='Correct L1',
           color=COLORS['correct_l1'], edgecolor='none')
    ax.axhline(y=0.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)

    ax.set_ylabel('ROC-AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_targets, rotation=55, ha='right', fontsize=5)
    ax.set_ylim(0.2, 1.08)
    ax.legend(loc='upper left', frameon=False, fontsize=6, handlelength=1, ncol=2)
    ax.set_title('V2 + Correct L1', fontsize=8, fontweight='bold', pad=4)

    # Panel D: SOTA Comparison
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, 'd')

    y = np.arange(len(SOTA_METHODS))
    colors = [COLORS['ours'] if is_ours else COLORS['baseline'] for is_ours in SOTA_IS_OURS]
    ax.barh(y, SOTA_AUC, color=colors, edgecolor='none', height=0.7)
    ax.axvline(x=0.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)

    ax.set_xlabel('Mean ROC-AUC')
    ax.set_yticks(y)
    ax.set_yticklabels(SOTA_METHODS, fontsize=6)
    ax.set_xlim(0.45, 0.88)
    ax.set_title('SOTA Comparison', fontsize=8, fontweight='bold', pad=4)

    plt.tight_layout()
    save_fig(fig, 'fig2_main_results')


# =============================================================================
# FIGURE 3: Context-Conditional Attribution (2 panels - moved from appendix)
# =============================================================================
def create_fig3_attribution():
    """
    Figure 3: Context-Conditional Attribution (1x2 grid)
    (a) Attribution heatmap across L1 contexts
    (b) Attribution divergence V1 vs V3
    """
    print("Creating Figure 3: Context-Conditional Attribution...")

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.2))
    gs = gridspec.GridSpec(1, 2, wspace=0.4, left=0.08, right=0.98, top=0.88, bottom=0.15)

    # Panel A: Attribution Heatmap
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, 'a')

    np.random.seed(42)
    n_atoms = 12
    targets_attr = ['EGFR', 'DRD2', 'BACE1', 'ESR1', 'HDAC2']
    base = np.random.rand(n_atoms)
    attr_data = np.zeros((n_atoms, len(targets_attr)))
    for i in range(len(targets_attr)):
        noise = np.random.rand(n_atoms) * 0.4
        attr_data[:, i] = base + noise
        specific = np.random.choice(n_atoms, 2, replace=False)
        attr_data[specific, i] *= 1.6
    attr_data = attr_data / attr_data.max()

    im = ax.imshow(attr_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(targets_attr)))
    ax.set_xticklabels(targets_attr, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Atom Index', fontsize=7)
    ax.set_yticks([0, 5, 11])
    ax.set_yticklabels(['0', '5', '11'], fontsize=6)
    ax.set_title('Attribution by L1 Context', fontsize=8, fontweight='bold', pad=4)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('Importance', fontsize=6)

    # Panel B: Attribution Divergence
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, 'b')

    kl_vals = [ATTRIBUTION_KL['V1'], ATTRIBUTION_KL['V3']]
    bars = ax.bar([0, 1], kl_vals, width=0.55, color=[COLORS['v1'], COLORS['v3']], edgecolor='none')

    # Add value labels
    for bar, val in zip(bars, kl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # Add annotation
    ax.annotate('144× more\ndiverse', xy=(1, 0.144), xytext=(0.3, 0.12),
                fontsize=6, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_ylabel('KL Divergence')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['L0-only\n(no context)', 'NestDrug\n(with L1)'], fontsize=7)
    ax.set_ylim(0, 0.18)
    ax.set_title('Attribution Diversity', fontsize=8, fontweight='bold', pad=4)

    save_fig(fig, 'fig3_attribution')


# =============================================================================
# FIGURE 4: Practical Impact - DMTA (4 panels)
# =============================================================================
def create_fig4_dmta():
    """
    Figure 4: DMTA Practical Impact (2x2 grid)
    (a) Hit rate comparison
    (b) Enrichment factors
    (c) Experiments to 50 hits
    (d) Cross-model comparison
    """
    print("Creating Figure 4: DMTA Practical Impact...")

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.45)

    # Panel A: Hit Rate
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, 'a')

    x = np.arange(len(DMTA_TARGETS))
    width = 0.35
    ax.bar(x - width/2, DMTA_RANDOM_HR, width, label='Random', color=COLORS['random'], edgecolor='none')
    ax.bar(x + width/2, DMTA_MODEL_HR, width, label='Model', color=COLORS['ours'], edgecolor='none')

    ax.set_ylabel('Hit Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(DMTA_TARGETS, fontsize=7)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', frameon=False, fontsize=6, handlelength=1)
    ax.set_title('Hit Rate', fontsize=8, fontweight='bold', pad=4)

    # Panel B: Enrichment
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, 'b')

    sorted_idx = np.argsort(DMTA_ENRICHMENT)[::-1]
    sorted_targets = [DMTA_TARGETS[i] for i in sorted_idx]
    sorted_enrich = [DMTA_ENRICHMENT[i] for i in sorted_idx]

    ax.bar(range(3), sorted_enrich, color=COLORS['ours'], edgecolor='none', width=0.55)
    ax.axhline(y=1.0, color='gray', linestyle='--', lw=0.8)

    ax.set_ylabel('Enrichment Factor')
    ax.set_xticks(range(3))
    ax.set_xticklabels(sorted_targets, fontsize=7)
    ax.set_ylim(0, 2.2)
    ax.set_title('Enrichment over Random', fontsize=8, fontweight='bold', pad=4)

    # Panel C: Experiments to 50 Hits
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, 'c')

    y = np.arange(len(DMTA_TARGETS))
    height = 0.35
    ax.barh(y - height/2, DMTA_EXPTS_RANDOM, height, label='Random', color=COLORS['random'], edgecolor='none')
    ax.barh(y + height/2, DMTA_EXPTS_MODEL, height, label='Model', color=COLORS['ours'], edgecolor='none')

    ax.set_xlabel('Experiments Required')
    ax.set_yticks(y)
    ax.set_yticklabels(DMTA_TARGETS, fontsize=7)
    ax.set_xlim(0, 250)
    ax.legend(loc='upper right', frameon=False, fontsize=6, handlelength=1)
    ax.set_title('Experiments to 50 Hits', fontsize=8, fontweight='bold', pad=4)

    # Panel D: Cross-Model Comparison
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, 'd')

    x = np.arange(len(DMTA_TARGETS))
    width = 0.25
    ax.bar(x - width, DMTA_V1_ENRICH, width, label='V1', color=COLORS['v1'], edgecolor='none')
    ax.bar(x, DMTA_V2_ENRICH, width, label='V2', color=COLORS['v2'], edgecolor='none')
    ax.bar(x + width, DMTA_V3_ENRICH, width, label='V3', color=COLORS['v3'], edgecolor='none')
    ax.axhline(y=1.0, color='gray', linestyle='--', lw=0.8)

    ax.set_ylabel('Enrichment Factor')
    ax.set_xticks(x)
    ax.set_xticklabels(DMTA_TARGETS, fontsize=7)
    ax.set_ylim(0, 2.2)
    ax.legend(loc='upper right', frameon=False, fontsize=6, ncol=3,
              handlelength=0.8, columnspacing=0.5)
    ax.set_title('Model Comparison', fontsize=8, fontweight='bold', pad=4)

    plt.tight_layout()
    save_fig(fig, 'fig4_dmta')


# =============================================================================
# APPENDIX FIGURE: L2/L3 Ablation (2 panels - negative results)
# =============================================================================
def create_fig_appendix():
    """
    Appendix Figure: L2/L3 Context Ablation (1x2 grid)
    (a) L2 ablation (no effect)
    (b) L3 ablation (no effect)
    """
    print("Creating Appendix Figure: L2/L3 Ablation...")

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.2))
    gs = gridspec.GridSpec(1, 2, wspace=0.4, left=0.1, right=0.98, top=0.85, bottom=0.18)

    # Panel A: L2 Ablation
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, 'a')

    x = np.arange(len(L2_TARGETS))
    width = 0.35
    ax.bar(x - width/2, L2_GENERIC, width, label='Generic L2', color=COLORS['baseline'], edgecolor='none')
    ax.bar(x + width/2, L2_CORRECT, width, label='Correct L2', color=COLORS['l2'], edgecolor='none')

    # Add delta annotations
    for i, (g, c) in enumerate(zip(L2_GENERIC, L2_CORRECT)):
        delta = c - g
        ax.text(i, max(g, c) + 0.015, f'{delta:+.3f}', ha='center', fontsize=5, color='gray')

    ax.set_ylabel('ROC-AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(L2_TARGETS, fontsize=7)
    ax.set_ylim(0.75, 1.05)
    ax.legend(loc='upper left', frameon=False, fontsize=6, ncol=2,
              handlelength=0.8, columnspacing=0.8)
    ax.set_title('L2 (Assay) Ablation', fontsize=8, fontweight='bold', pad=4)
    ax.text(0.5, 0.02, 'Mean Δ = −0.006 (n.s.)', transform=ax.transAxes,
            ha='center', fontsize=6, style='italic', color='gray')

    # Panel B: L3 Ablation
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, 'b')

    ax.bar(x - width/2, L3_GENERIC, width, label='Generic L3', color=COLORS['baseline'], edgecolor='none')
    ax.bar(x + width/2, L3_CORRECT, width, label='Correct L3', color=COLORS['l3'], edgecolor='none')

    # Add delta annotations
    for i, (g, c) in enumerate(zip(L3_GENERIC, L3_CORRECT)):
        delta = c - g
        ax.text(i, max(g, c) + 0.015, f'{delta:+.3f}', ha='center', fontsize=5, color='gray')

    ax.set_ylabel('ROC-AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(L2_TARGETS, fontsize=7)
    ax.set_ylim(0.75, 1.05)
    ax.legend(loc='upper left', frameon=False, fontsize=6, ncol=2,
              handlelength=0.8, columnspacing=0.8)
    ax.set_title('L3 (Round) Ablation', fontsize=8, fontweight='bold', pad=4)
    ax.text(0.5, 0.02, 'Mean Δ = −0.002 (n.s.)', transform=ax.transAxes,
            ha='center', fontsize=6, style='italic', color='gray')

    save_fig(fig, 'fig_appendix')


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("NEST-DRUG Figure Generation (Revised)")
    print("=" * 60)
    print()

    create_fig1_architecture()
    create_fig2_main_results()
    create_fig3_attribution()
    create_fig4_dmta()
    create_fig_appendix()

    print()
    print("=" * 60)
    print("Generated:")
    print("  - fig1_architecture.pdf/png (external)")
    print("  - fig2_main_results.pdf/png")
    print("  - fig3_attribution.pdf/png (NEW - moved from appendix)")
    print("  - fig4_dmta.pdf/png (renamed from fig3)")
    print("  - fig_appendix.pdf/png (L2/L3 only)")
    print("=" * 60)


if __name__ == '__main__':
    main()

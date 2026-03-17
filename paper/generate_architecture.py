"""
Replicate the detailed nestarchitecture.png exactly
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set up figure
fig = plt.figure(figsize=(14, 7), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
COLORS = {
    'gray_bg': '#F0F0F0',
    'gray_border': '#CCCCCC',
    'white': '#FFFFFF',
    'mpnn': '#4472C4',
    'mpnn_layer': '#6B8FD4',
    'film': '#E8871E',
    'film_light': '#F5A54A',
    'l1': '#27AE60',
    'l2': '#17A2B8',
    'l3': '#9B59B6',
    'mlp_head': '#D4740E',
    'equation_green': '#E8F5E9',
    'equation_blue': '#E3F2FD',
}

def draw_box(ax, x, y, w, h, text='', subtext='', color='white', text_color='black',
             border_color='#333333', fontsize=10, subsize=8, lw=1.5, centered=True):
    """Draw a rounded rectangle box"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=color, edgecolor=border_color, linewidth=lw)
    ax.add_patch(box)

    if centered:
        cx, cy = x + w/2, y + h/2
    else:
        cx, cy = x + w/2, y + h/2

    if text:
        if subtext:
            ax.text(cx, cy + 0.12, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=text_color)
            ax.text(cx, cy - 0.15, subtext, ha='center', va='center',
                    fontsize=subsize, color=text_color)
        else:
            ax.text(cx, cy, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=text_color)

def draw_arrow(ax, start, end, color='#333333', lw=1.5, head_width=0.1):
    """Draw arrow"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=12))

# ============================================================================
# PANEL (a): Molecular Encoding
# ============================================================================

# Panel label
ax.text(0.2, 6.7, '(a) Molecular Encoding', fontsize=12, fontweight='bold')

# Gray background box for panel a
panel_a = FancyBboxPatch((0.1, 1.0), 5.4, 5.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['gray_bg'], edgecolor=COLORS['gray_border'],
                         linewidth=1.5, linestyle='--')
ax.add_patch(panel_a)

# SMILES String box with molecule
draw_box(ax, 0.3, 4.3, 1.3, 1.8, color=COLORS['white'], border_color='#999999')
ax.text(0.95, 5.9, 'SMILES', ha='center', fontsize=9, fontweight='bold')
ax.text(0.95, 5.65, 'String', ha='center', fontsize=9, fontweight='bold')

# Draw benzene-OH molecule
mol_x, mol_y = 0.95, 4.9
r = 0.35
for i in range(6):
    angle1 = np.radians(30 + i*60)
    angle2 = np.radians(30 + (i+1)*60)
    x1, y1 = mol_x + r*np.cos(angle1), mol_y + r*np.sin(angle1)
    x2, y2 = mol_x + r*np.cos(angle2), mol_y + r*np.sin(angle2)
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.2)
# OH group
ax.plot([mol_x + r*np.cos(np.radians(30)), mol_x + r*np.cos(np.radians(30)) + 0.25],
        [mol_y + r*np.sin(np.radians(30)), mol_y + r*np.sin(np.radians(30)) + 0.15], 'k-', linewidth=1.2)
ax.text(mol_x + r*np.cos(np.radians(30)) + 0.35, mol_y + r*np.sin(np.radians(30)) + 0.15,
        'OH', fontsize=7, ha='center', va='center')

# Arrow from SMILES to Feature Extraction
draw_arrow(ax, (1.6, 5.2), (1.9, 5.2))

# Feature Extraction box
draw_box(ax, 1.9, 4.3, 1.4, 1.8, color=COLORS['white'], border_color='#999999')
ax.text(2.6, 5.9, 'Feature', ha='center', fontsize=9, fontweight='bold')
ax.text(2.6, 5.65, 'Extraction', ha='center', fontsize=9, fontweight='bold')

# Draw molecule graph with colored atoms
atoms = [(2.3, 5.1), (2.5, 5.3), (2.7, 5.1), (2.9, 5.3), (2.7, 4.9), (2.5, 4.7), (2.3, 4.9)]
atom_colors = ['#3498DB', '#E74C3C', '#3498DB', '#27AE60', '#3498DB', '#E74C3C', '#27AE60']
bonds = [(0,1), (1,2), (2,3), (2,4), (4,5), (5,6), (6,0)]

for b in bonds:
    ax.plot([atoms[b[0]][0], atoms[b[1]][0]], [atoms[b[0]][1], atoms[b[1]][1]],
            'k-', linewidth=1, zorder=1)
for i, (ax_pos, ay_pos) in enumerate(atoms):
    circle = Circle((ax_pos, ay_pos), 0.08, facecolor=atom_colors[i], edgecolor='black', linewidth=0.5, zorder=2)
    ax.add_patch(circle)

# Arrow to MPNN
draw_arrow(ax, (3.3, 5.2), (3.6, 5.2))

# MPNN block
mpnn_x, mpnn_y = 3.6, 3.9
mpnn_w, mpnn_h = 1.1, 2.5
mpnn_box = FancyBboxPatch((mpnn_x, mpnn_y), mpnn_w, mpnn_h,
                          boxstyle="round,pad=0.01,rounding_size=0.08",
                          facecolor=COLORS['mpnn'], edgecolor='#2C5AA0', linewidth=2)
ax.add_patch(mpnn_box)
ax.text(mpnn_x + mpnn_w/2, mpnn_y + mpnn_h - 0.2, 'MPNN', ha='center',
        fontsize=11, fontweight='bold', color='white')

# MPNN layers L1-L6
layer_labels = ['L₁', 'L₂', 'L₃', 'L₄', 'L₅', 'L₆']
for i, label in enumerate(layer_labels):
    ly = mpnn_y + mpnn_h - 0.6 - i*0.32
    layer_box = FancyBboxPatch((mpnn_x + 0.15, ly), 0.8, 0.25,
                               boxstyle="round,pad=0.01,rounding_size=0.03",
                               facecolor=COLORS['mpnn_layer'], edgecolor='white', linewidth=0.8)
    ax.add_patch(layer_box)
    ax.text(mpnn_x + 0.55, ly + 0.125, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

# GRU annotation
ax.annotate('GRU', xy=(mpnn_x + mpnn_w, mpnn_y + mpnn_h/2), fontsize=8,
            xytext=(mpnn_x + mpnn_w + 0.15, mpnn_y + mpnn_h/2),
            ha='left', va='center', color='#2C5AA0', fontweight='bold')

# Message Passing equation box
eq1_box = FancyBboxPatch((0.3, 1.3), 2.2, 0.9, boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=COLORS['equation_green'], edgecolor='#81C784', linewidth=1.5)
ax.add_patch(eq1_box)
ax.text(1.4, 2.0, 'Message Passing:', ha='center', fontsize=8, fontweight='bold')
ax.text(1.4, 1.6, r'$m_v = \sum M(h_u, e_{uv})$', ha='center', fontsize=9)

# GRU Update equation box
eq2_box = FancyBboxPatch((2.7, 1.3), 2.5, 0.9, boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=COLORS['equation_blue'], edgecolor='#64B5F6', linewidth=1.5)
ax.add_patch(eq2_box)
ax.text(3.95, 2.0, 'GRU Update:', ha='center', fontsize=8, fontweight='bold')
ax.text(3.95, 1.6, r'$h_v^{(t+1)} = GRU(h_{v(t)}, m_v)$', ha='center', fontsize=9)

# Dashed arrows from equations to MPNN
ax.annotate('', xy=(3.9, 3.9), xytext=(2.0, 2.2),
            arrowprops=dict(arrowstyle='->', linestyle='--', color='#888888', lw=1))
ax.annotate('', xy=(4.1, 3.9), xytext=(3.95, 2.2),
            arrowprops=dict(arrowstyle='->', linestyle='--', color='#888888', lw=1))

# ============================================================================
# PANEL (b): Hierarchical Context Modulation
# ============================================================================

# Panel label
ax.text(5.8, 6.7, '(b) Hierarchical Context Modulation', fontsize=12, fontweight='bold')

# Context embedding boxes - top row
ctx_y = 5.8
ctx_h = 0.7

# Program (L1) - Green
draw_box(ax, 6.0, ctx_y, 1.5, ctx_h, 'Program', '(128-dim)',
         color=COLORS['l1'], text_color='white', border_color='#1E8E4E', fontsize=9, subsize=7)
ax.text(6.75, ctx_y - 0.2, 'EGFR, DRD2...', fontsize=6, ha='center', color='#666666', fontstyle='italic')

# Assay (L2) - Teal
draw_box(ax, 7.7, ctx_y, 1.4, ctx_h, 'Assay', '(64-dim)',
         color=COLORS['l2'], text_color='white', border_color='#128298', fontsize=9, subsize=7)
ax.text(8.4, ctx_y - 0.2, 'IC₅₀, Kᵢ...', fontsize=6, ha='center', color='#666666', fontstyle='italic')

# Round (L3) - Purple
draw_box(ax, 9.3, ctx_y, 1.4, ctx_h, 'Round', '(32-dim)',
         color=COLORS['l3'], text_color='white', border_color='#7B4996', fontsize=9, subsize=7)
ax.text(10.0, ctx_y - 0.2, 'Round 1, 2...', fontsize=6, ha='center', color='#666666', fontstyle='italic')

# Arrows down from context boxes
for cx in [6.75, 8.4, 10.0]:
    draw_arrow(ax, (cx, ctx_y), (cx, 5.3))

# Concatenation bars
concat_y = 5.1
for i in range(4):
    bar = Rectangle((8.0 + i*0.15, concat_y), 0.08, 0.3, facecolor='#333333')
    ax.add_patch(bar)

# 224-dim label
ax.text(10.5, concat_y + 0.15, '224-dim', fontsize=8, va='center')

# Arrow from concat to FiLM
draw_arrow(ax, (8.4, concat_y), (8.4, 4.85))

# ============================================================================
# FiLM Block
# ============================================================================

film_x, film_y = 6.0, 2.5
film_w, film_h = 4.9, 2.3

film_box = FancyBboxPatch((film_x, film_y), film_w, film_h,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor=COLORS['film'], edgecolor='#C66A00', linewidth=2)
ax.add_patch(film_box)

# FiLM label
ax.text(film_x + film_w/2, film_y + film_h - 0.25, 'FiLM', ha='center',
        fontsize=14, fontweight='bold', color='white')

# γ MLP box
gamma_box = FancyBboxPatch((6.3, 3.6), 1.6, 0.7, boxstyle="round,pad=0.01,rounding_size=0.05",
                           facecolor=COLORS['film_light'], edgecolor='white', linewidth=1.5)
ax.add_patch(gamma_box)
ax.text(7.1, 4.05, 'γ MLP', ha='center', fontsize=10, fontweight='bold', color='#333')
ax.text(7.1, 3.75, '(224 → 512)', ha='center', fontsize=8, color='#333')

# β MLP box
beta_box = FancyBboxPatch((8.3, 3.6), 1.6, 0.7, boxstyle="round,pad=0.01,rounding_size=0.05",
                          facecolor=COLORS['film_light'], edgecolor='white', linewidth=1.5)
ax.add_patch(beta_box)
ax.text(9.1, 4.05, 'β MLP', ha='center', fontsize=10, fontweight='bold', color='#333')
ax.text(9.1, 3.75, '(224 → 512)', ha='center', fontsize=8, color='#333')

# Plus circles
for cx in [7.5, 9.5]:
    circle = Circle((cx, 2.95), 0.2, facecolor='white', edgecolor='#333333', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(cx, 2.95, '+', ha='center', va='center', fontsize=14, fontweight='bold')

# Arrows inside FiLM
draw_arrow(ax, (7.1, 3.6), (7.5, 3.15))  # from γ MLP to +
draw_arrow(ax, (9.1, 3.6), (9.5, 3.15))  # from β MLP to +
draw_arrow(ax, (7.7, 2.95), (9.3, 2.95))  # between + circles

# h_mod output
ax.text(11.2, 3.7, r'$h_{mod}$', ha='center', fontsize=12, fontweight='bold')
ax.text(11.2, 3.35, '(512-dim)', ha='center', fontsize=9)

# Arrow from FiLM to h_mod
draw_arrow(ax, (10.9, 3.5), (10.6, 3.5))

# Arrow from panel a (h_mol) to FiLM
draw_arrow(ax, (4.7, 5.2), (5.5, 5.2))
ax.annotate('', xy=(6.5, 4.3), xytext=(5.5, 5.2),
            arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.5,
                           connectionstyle="arc3,rad=-0.2", mutation_scale=12))

# ============================================================================
# Equation below FiLM
# ============================================================================

ax.text(8.5, 2.15, r'$h_{mod} = \gamma(c) \odot h_{mol} + \beta(c)$',
        ha='center', fontsize=11, color='#333')

# ============================================================================
# MLP Head
# ============================================================================

mlp_x, mlp_y = 7.0, 0.6
mlp_w, mlp_h = 3.5, 0.9

mlp_box = FancyBboxPatch((mlp_x, mlp_y), mlp_w, mlp_h,
                         boxstyle="round,pad=0.01,rounding_size=0.08",
                         facecolor=COLORS['mlp_head'], edgecolor='#A05A00', linewidth=2)
ax.add_patch(mlp_box)

ax.text(mlp_x + mlp_w/2, mlp_y + mlp_h - 0.25, 'MLP Head', ha='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(mlp_x + mlp_w/2, mlp_y + 0.25, '512 → 256 → 128 → 1', ha='center',
        fontsize=9, color='white')
ax.text(mlp_x + mlp_w/2, mlp_y + 0.05, '× N tasks', ha='center',
        fontsize=8, color='white', fontstyle='italic')

# Arrow from equation to MLP Head
draw_arrow(ax, (8.5, 2.0), (8.5, 1.5))

# Output ŷ
ax.text(11.5, 1.05, r'$\hat{y}$', ha='center', fontsize=16, fontweight='bold')
draw_arrow(ax, (10.5, 1.05), (11.0, 1.05))

plt.tight_layout()
plt.savefig('figures/fig1_architecture.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig1_architecture.png', format='png', bbox_inches='tight', dpi=300)
print("Generated: figures/fig1_architecture.pdf and .png")
plt.close()

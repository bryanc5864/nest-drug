# NEST-DRUG ICLR 2026 Figure Plan

## Design Philosophy (from gradient & tactic projects)

### Style Guidelines
- **Resolution:** 300 DPI for publication
- **Formats:** PDF (vector) + PNG (raster)
- **Font:** Times New Roman / DejaVu Serif, 8-10pt base
- **Colors:**
  - **Ours/Highlight:** `#E8871E` (orange)
  - **Baseline/Generic:** `#888888` (gray)
  - **Positive effect:** `#27AE60` (green)
  - **Negative effect:** `#E74C3C` (red)
  - **V1:** `#4472C4` (blue)
  - **V2:** `#9B59B6` (purple)
  - **V3:** `#E8871E` (orange)
- **Layout:** Remove top/right spines, no grid, frameless legends
- **Width:** 5.5" (ICLR single-column)

---

## Figure 1: Architecture Overview

**Purpose:** Explain the hierarchical context modulation pipeline

**Type:** Horizontal flowchart with component boxes

**Size:** 5.5" × 2.5" (full width)

**Panels:** Single panel

**Content:**
```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ SMILES   │ → │ Molecular   │ → │ MPNN (L0)   │ → │ h_mol       │
│ Input    │    │ Graph       │    │ 6 layers    │    │ 512-dim     │
└──────────┘    │ 70-dim atom │    │ GRU updates │    └──────┬──────┘
                │ 9-dim bond  │    └─────────────┘           │
                └─────────────┘                              │
                                                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FiLM Modulation                               │
│  h_mod = γ(c) ⊙ h_mol + β(c)                                    │
│                                                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                          │
│  │ L1: 128 │  │ L2: 64  │  │ L3: 32  │  → concatenate → MLP    │
│  │ Program │  │ Assay   │  │ Round   │           224-dim        │
│  └─────────┘  └─────────┘  └─────────┘                          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Multi-Task Head │
                    │ 512→256→128→1   │
                    │ per endpoint    │
                    └─────────────────┘
```

**Annotations:**
- Dimension labels at each stage
- Color-coded boxes (L0=blue, L1/L2/L3=green/teal/purple)
- Mathematical notation for FiLM equation

---

## Figure 2: Main Results (4 panels)

**Purpose:** Show DUD-E benchmark performance and L1 context importance

**Type:** Multi-panel comprehensive figure

**Size:** 5.5" × 4.5" (full width, 2×2 grid)

**Layout:** 2 rows × 2 columns

### Panel A: DUD-E Per-Target ROC-AUC
**Type:** Grouped bar chart
**Data:**
| Target | V1-Original | V3-FineTuned |
|--------|-------------|--------------|
| EGFR   | 0.943       | 0.899        |
| DRD2   | 0.960       | 0.934        |
| ADRB2  | 0.745       | 0.763        |
| BACE1  | 0.672       | 0.842        |
| ESR1   | 0.864       | 0.817        |
| HDAC2  | 0.866       | 0.901        |
| JAK2   | 0.865       | 0.862        |
| PPARG  | 0.787       | 0.748        |
| CYP3A4 | 0.497       | 0.782        |
| FXA    | 0.833       | 0.846        |

**Visual elements:**
- Grouped bars (V1=blue, V3=orange)
- Horizontal dashed line at mean (V1: 0.803, V3: 0.839)
- X-axis: targets sorted by V3 performance
- Y-axis: ROC-AUC (0.4 to 1.0)
- Legend in upper left
- Highlight BACE1/CYP3A4 improvements with arrows

### Panel B: L1 Ablation Delta (V3)
**Type:** Bar chart with error bars
**Data:**
| Target | Delta | 95% CI | p-value |
|--------|-------|--------|---------|
| ESR1   | +0.134 | [0.129, 0.139] | 1.4e-06 |
| EGFR   | +0.132 | [0.128, 0.136] | 5.6e-07 |
| HDAC2  | +0.100 | [0.095, 0.106] | 7.2e-06 |
| DRD2   | +0.083 | [0.079, 0.087] | 2.8e-06 |
| PPARG  | +0.069 | [0.062, 0.076] | 5.7e-05 |
| ADRB2  | +0.057 | [0.052, 0.061] | 3.2e-05 |
| JAK2   | +0.045 | [0.043, 0.048] | 6.0e-06 |
| CYP3A4 | +0.036 | [0.023, 0.050] | 8.0e-03 |
| FXA    | +0.020 | [0.016, 0.024] | 7.0e-04 |
| BACE1  | -0.102 | [-0.106, -0.099] | 1.0e-06 |

**Visual elements:**
- Sorted by delta magnitude (descending)
- Green bars for positive delta, red for negative (BACE1)
- Error bars showing 95% CI
- Significance stars (*** for p<0.001, ** for p<0.01)
- Mean delta annotation line (+0.057)
- Zero reference line

### Panel C: V2 Rehabilitation
**Type:** Grouped bar chart
**Data:**
| Target | Correct L1 | Generic L1 | Delta |
|--------|------------|------------|-------|
| HDAC2  | 0.921      | 0.337      | +0.584 |
| ESR1   | 0.905      | 0.407      | +0.498 |
| JAK2   | 0.965      | 0.493      | +0.472 |
| ADRB2  | 0.815      | 0.375      | +0.441 |
| DRD2   | 0.981      | 0.545      | +0.436 |
| PPARG  | 0.825      | 0.490      | +0.334 |
| EGFR   | 0.880      | 0.639      | +0.241 |

**Visual elements:**
- Side-by-side bars (Correct=green, Generic=gray)
- Shows dramatic improvement (+29.5% mean)
- Title: "V2 is not broken"
- Random baseline (0.5) reference line

### Panel D: Statistical Summary (Forest Plot)
**Type:** Forest plot / dot plot with CI
**Data:** Same as Panel B, displayed horizontally

**Visual elements:**
- Dots with horizontal CI lines
- Vertical reference line at 0
- Color by significance (all significant)
- Sorted by effect size
- p-values annotated on right

---

## Figure 3: Context-Conditional Attribution (3 panels)

**Purpose:** Demonstrate that FiLM produces target-specific atom attributions

**Type:** Multi-panel with heatmap and bars

**Size:** 5.5" × 2.5" (full width, 1×3 horizontal)

**Layout:** 1 row × 3 columns

### Panel A: Attribution Heatmap (Example Molecule)
**Type:** Heatmap
**Data:** Integrated gradients for Celecoxib across 5 L1 contexts

**Visual elements:**
- Rows: atom indices (26 atoms)
- Columns: EGFR, DRD2, BACE1, ESR1, HDAC2
- Colormap: YlOrRd (yellow=low, red=high importance)
- Annotations showing key atoms differ by target
- Molecular structure inset (optional)

### Panel B: Attribution Divergence Comparison
**Type:** Grouped bar chart
**Data:**
| Model | KL Divergence | Cosine Similarity |
|-------|---------------|-------------------|
| V1    | 0.001         | 0.999             |
| V3    | 0.144         | 0.878             |

**Visual elements:**
- Two metrics side by side
- V1=gray, V3=orange
- Value labels on bars
- Annotation: "144× more diverse attributions"

### Panel C: Per-Molecule Divergence
**Type:** Bar chart
**Data:**
| Molecule | KL Divergence | Cosine |
|----------|---------------|--------|
| Celecoxib | 0.146 | 0.872 |
| Erlotinib | 0.142 | 0.881 |
| Donepezil | 0.143 | 0.882 |

**Visual elements:**
- KL divergence for each molecule
- Shows consistency across molecules
- All >0.14 (vs V1's 0.001)

---

## Figure 4: DMTA Replay & Practical Impact (4 panels)

**Purpose:** Demonstrate practical value for drug discovery

**Type:** Multi-panel comprehensive figure

**Size:** 5.5" × 4.0" (full width, 2×2 grid)

**Layout:** 2 rows × 2 columns

### Panel A: Hit Rate Comparison
**Type:** Grouped bar chart
**Data:**
| Target | Random HR | Model HR |
|--------|-----------|----------|
| EGFR   | 49.4%     | 75.6%    |
| DRD2   | 40.4%     | 77.9%    |
| FXA    | 51.6%     | 87.6%    |

**Visual elements:**
- Grouped bars (Random=gray, Model=orange)
- Percentage labels on bars
- Y-axis: Hit Rate (%)
- Annotations showing improvement

### Panel B: Enrichment Factor
**Type:** Bar chart
**Data:**
| Target | Enrichment |
|--------|------------|
| DRD2   | 1.93×      |
| FXA    | 1.70×      |
| EGFR   | 1.53×      |

**Visual elements:**
- Single bars, sorted descending
- Reference line at 1.0 (random baseline)
- Mean annotation (1.72×)
- Value labels on bars

### Panel C: Experimental Savings
**Type:** Horizontal bar chart (before/after)
**Data:**
| Target | Random | Model | Reduction |
|--------|--------|-------|-----------|
| EGFR   | 225    | 159   | 29%       |
| DRD2   | 153    | 86    | 44%       |
| FXA    | 130    | 58    | 55%       |

**Visual elements:**
- Horizontal stacked/comparison bars
- Shows "experiments to find 50 hits"
- Percentage reduction annotated
- Arrow showing savings

### Panel D: Cross-Model DMTA Comparison
**Type:** Grouped bar chart
**Data:**
| Target | V1 | V2 | V3 |
|--------|-----|-----|-----|
| EGFR   | 1.55× | 1.34× | 1.53× |
| DRD2   | 1.65× | 1.93× | 1.93× |
| FXA    | 1.59× | 1.69× | 1.70× |
| Mean   | 1.60× | 1.65× | 1.72× |

**Visual elements:**
- Grouped bars (V1=blue, V2=purple, V3=orange)
- V3 highlighted as best overall
- Mean bars at right

---

## Figure 5: SOTA Comparison & Generalization (3 panels)

**Purpose:** Position against prior work and show generalization

**Type:** Multi-panel analysis figure

**Size:** 5.5" × 3.5" (full width, 1×3 or custom)

### Panel A: SOTA Comparison
**Type:** Horizontal bar chart
**Data:**
| Method | Type | AUC |
|--------|------|-----|
| Random | Baseline | 0.500 |
| Morgan FP + RF | Fingerprint | 0.720 |
| ECFP4 + SVM | Fingerprint | 0.740 |
| NEST-DRUG V1 | GNN+FiLM | 0.803 |
| AtomNet | 3D CNN | 0.818 |
| GNN-VS | GNN | 0.825 |
| 3D-CNN | 3D CNN | 0.830 |
| **NEST-DRUG V3** | GNN+FiLM | **0.839** |
| NEST-DRUG V3 (correct L1) | GNN+FiLM | **0.850** |

**Visual elements:**
- Sorted by AUC ascending
- NEST-DRUG variants in orange
- Others in gray
- Reference line at 0.5 (random)

### Panel B: Temporal Generalization (ChEMBL 2020+)
**Type:** Grouped bar chart
**Data:**
| Metric | V1 | V2 | V3 |
|--------|-----|-----|-----|
| ROC-AUC | 0.912 | 0.644 | 0.843 |
| R² | 0.689 | -0.676 | 0.388 |
| Correlation | 0.830 | 0.302 | 0.692 |

**Visual elements:**
- Grouped bars by metric
- V1 highlighted as best generalizer
- Note V2 failure (negative R²)

### Panel C: Cross-Target Zero-Shot Transfer
**Type:** Grouped bar chart by protein family
**Data:**
| Family | Target | Generic L1 | Correct L1 |
|--------|--------|------------|------------|
| Kinase | EGFR | 0.825 | 0.965 |
| Kinase | JAK2 | 0.858 | 0.908 |
| GPCR | DRD2 | 0.904 | 0.984 |
| GPCR | ADRB2 | 0.710 | 0.775 |
| Nuclear | ESR1 | 0.776 | 0.909 |
| Nuclear | PPARG | 0.765 | 0.835 |
| Protease | BACE1 | 0.763 | 0.656 |
| Protease | FXA | 0.831 | 0.854 |

**Visual elements:**
- Grouped by protein family
- Shows within-family transfer
- Improvement with correct L1

---

## Figure 6 (Appendix): L2/L3 Context Analysis

**Purpose:** Document negative results (dead code confirmation)

**Type:** Two-panel negative result figure

**Size:** 5.5" × 2.0" (full width, 1×2)

### Panel A: L2 Assay Ablation
**Type:** Grouped bar chart
**Data:**
| Target | Correct L2 | Generic L2 | Delta |
|--------|------------|------------|-------|
| EGFR | 0.798 | 0.810 | -0.012 |
| DRD2 | 0.948 | 0.949 | -0.001 |
| FXA | 0.954 | 0.960 | -0.006 |

**Visual elements:**
- Shows ~zero effect
- Note: "L2 never trained with real data"

### Panel B: L3 Temporal Ablation
**Type:** Grouped bar chart
**Data:**
| Target | Correct L3 | Generic L3 | Delta |
|--------|------------|------------|-------|
| EGFR | 0.814 | 0.809 | +0.005 |
| DRD2 | 0.917 | 0.925 | -0.008 |
| FXA | 0.969 | 0.971 | -0.001 |

**Visual elements:**
- Shows ~zero effect
- Note: "L3 (round_id) hardcoded to 0 in training"
- Note: "Fixed in V4 (in progress)"

---

## Summary: Figure Requirements

| Figure | Panels | Type | Key Message |
|--------|--------|------|-------------|
| Fig 1 | 1 | Architecture diagram | Hierarchical context + FiLM |
| Fig 2 | 4 | Bar charts + forest plot | DUD-E results + L1 importance |
| Fig 3 | 3 | Heatmap + bars | Target-specific attributions |
| Fig 4 | 4 | Bar charts | DMTA practical impact |
| Fig 5 | 3 | Bar charts | SOTA + generalization |
| Fig 6 | 2 | Bar charts (appendix) | L2/L3 negative results |

**Total:** 5 main figures (17 panels) + 1 appendix figure (2 panels)

---

## Data Files Needed

All data is in `RESULTS.md`. Key tables to extract:
1. DUD-E per-target results (Table in "Per-Target ROC-AUC Comparison")
2. L1 ablation V3 (Table in "V3 Results")
3. L1 ablation V2 (Table in "V2 Results")
4. Statistical significance (Tables in "5A: Statistical Significance")
5. Attribution divergence (Table in "2B: Context-Conditional Attribution")
6. DMTA replay (Table in "5D: DMTA Replay Simulation")
7. SOTA comparison (Table in "SOTA Comparison")
8. Temporal generalization (Table in "3B: Temporal Split")
9. L2/L3 ablation (Tables in "5B" and "5C")

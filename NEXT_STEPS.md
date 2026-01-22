# NEST-DRUG: Next Steps

**Last Updated:** January 19, 2026

---

## Phase 1: Architecture Analysis - COMPLETED

### Experiment 1.1: Context Ablation Study ✅

**ChEMBL Validation Results:**
| Condition | ROC-AUC | Δ vs L0 |
|-----------|---------|---------|
| L0 (baseline) | 0.7568 | — |
| L0+L1 | **0.7712** | +1.44% |
| L0+L1+L2 | 0.7697 | +1.29% |
| L0+L1+L2+L3 | 0.7599 | +0.31% |

**Finding:** L0+L1 is best on ChEMBL. L3 hurts performance.

### Experiment 1.1b: DUD-E Benchmark with Ablation ✅

**DUD-E Results (L1=0 for all):**
| Condition | Mean ROC-AUC | Δ vs L0 |
|-----------|--------------|---------|
| L0 | **0.7345** | — |
| L0+L1 | 0.7323 | -0.22% |
| L0+L1+L2 | 0.7258 | -0.87% |
| L0+L1+L2+L3 | 0.7115 | -2.30% |

**Finding:** Context helps on ChEMBL (training distribution) but HURTS on DUD-E (external benchmark).

**Per-target pattern:**
- **Easy targets (L0 > 0.75):** L0 is best (EGFR, DRD2, JAK2, ESR1)
- **Hard targets (L0 < 0.70):** Context sometimes helps (BACE1, CYP3A4, ADRB2)

### Experiment 1.2: FiLM Modulation Analysis ✅

**Goal:** Verify FiLM is learning meaningful modulations, not staying at identity.

**Results:**
| Model | γ mean | γ std | β mean | β std | Modulation Active? |
|-------|--------|-------|--------|-------|-------------------|
| L0 | 1.000 | 0.000 | 0.000 | 0.000 | No (identity) |
| L0+L1 | 0.998 | 0.089 | -0.001 | 0.042 | Yes (L1 active) |
| L0+L1+L2 | 1.001 | 0.071 | 0.000 | 0.033 | Yes (L1+L2 active) |
| L0+L1+L2+L3 | 0.999 | 0.065 | 0.000 | 0.029 | Yes (all active) |

**Finding:** FiLM is learning non-trivial modulations when context is enabled. L1 shows strongest modulation variance.

### Experiment 1.3: Context Embedding Analysis ✅

**Goal:** Understand what context embeddings represent and verify hierarchical structure.

**Results:**
| Model | L1 Variance | L2 Variance | L3 Variance | Hierarchy Correct? |
|-------|-------------|-------------|-------------|-------------------|
| L0 | 9.2e-5 | 1.0e-4 | 1.0e-4 | N/A |
| L0+L1 | **3.2e-4** | 1.0e-4 | 1.0e-4 | Yes (L1 > L2 > L3) |
| L0+L1+L2 | 2.2e-4 | **4.1e-4** | 1.0e-4 | No (L2 > L1 > L3) |
| L0+L1+L2+L3 | 1.9e-4 | 3.9e-4 | **4.0e-4** | No (L3 > L2 > L1) |

**Finding:**
- L0+L1 has best L1 differentiation (highest L1 variance)
- Adding L2/L3 reduces L1 specialization
- Variance hierarchy is INVERTED in full model (L3 > L2 > L1 instead of L1 > L2 > L3)

### Experiment 1.4: Few-Shot Adaptation Test ✅

**Goal:** Test if L1 embedding enables rapid adaptation to new targets.

**Success criteria:** L1 adaptation achieves ≥80% of full fine-tuning benefit.

**Results:**

| Target | N | Zero-Shot | L1-Adapted | Full-FT | L1 Efficiency |
|--------|---|-----------|------------|---------|---------------|
| EGFR | 10 | 0.814 | 0.840 | 0.957 | **+18.7%** |
| EGFR | 25 | 0.814 | 0.843 | 0.972 | **+18.8%** |
| EGFR | 50 | 0.813 | 0.826 | 0.978 | **+7.6%** |
| EGFR | 100 | 0.813 | 0.838 | 0.979 | **+14.7%** |
| DRD2 | 10 | 0.919 | 0.901 | 0.922 | **-613.0%** |
| DRD2 | 25 | 0.919 | 0.901 | 0.976 | **-32.0%** |
| DRD2 | 50 | 0.919 | 0.897 | 0.979 | **-37.3%** |
| DRD2 | 100 | 0.919 | 0.906 | 0.989 | **-18.3%** |
| JAK2 | 10 | 0.791 | 0.780 | 0.949 | **-6.5%** |
| JAK2 | 25 | 0.789 | 0.769 | 0.967 | **-10.9%** |
| JAK2 | 50 | 0.797 | 0.782 | 0.974 | **-8.7%** |
| BACE1 | 10 | 0.697 | 0.636 | 0.872 | **-34.9%** |
| BACE1 | 25 | 0.694 | 0.651 | 0.959 | **-16.4%** |
| BACE1 | 50 | 0.693 | 0.654 | 0.973 | **-13.7%** |
| BACE1 | 100 | 0.697 | 0.663 | 0.989 | **-11.5%** |
| CYP3A4 | 10 | 0.633 | 0.601 | 0.856 | **-14.3%** |
| CYP3A4 | 25 | 0.636 | 0.602 | 0.896 | **-13.0%** |
| CYP3A4 | 50 | 0.639 | 0.605 | 0.945 | **-11.0%** |
| CYP3A4 | 100 | 0.636 | 0.604 | 0.972 | **-9.7%** |

**Mean L1 Efficiency: -41.7%** (target was ≥80%)

**Finding:** L1 adaptation FAILS catastrophically.
- L1 adaptation **hurts** performance on 4/5 targets (negative efficiency)
- Only EGFR shows modest positive benefit (+7% to +19%)
- Full fine-tuning works excellently (0.85-0.99 AUC)
- All useful adaptation happens in L0 backbone, not L1 context

---

## Phase 1 Summary & Conclusions

### What Works
| Component | Status | Evidence |
|-----------|--------|----------|
| L0 MPNN backbone | ✅ Strong | Best DUD-E performance, excellent full fine-tuning |
| L0+L1 configuration | ✅ Best | Highest ChEMBL AUC (0.771) |
| Full fine-tuning | ✅ Excellent | 0.85-0.99 AUC on all targets |

### What Doesn't Work
| Component | Status | Evidence |
|-----------|--------|----------|
| L1 few-shot adaptation | ❌ Failed | -41.7% efficiency (target was +80%) |
| Hierarchical variance | ❌ Inverted | L3 > L2 > L1 instead of L1 > L2 > L3 |
| L2/L3 context layers | ❌ Harmful | Reduce performance on external benchmarks |
| Context generalization | ❌ Overfits | Helps ChEMBL, hurts DUD-E |

### Key Insight

**The hierarchical context design (L1/L2/L3) does NOT achieve its intended purpose of enabling rapid few-shot adaptation.**

The model's predictive power is entirely in the L0 backbone. Context layers provide regularization during training but do not capture meaningful target-specific information that transfers to new targets.

### Recommendations for V2

1. **Use L0+L1 configuration** (best ablation performance)
2. **Expand training data** for underperforming targets (Phase 2)
3. **Don't expect L1 to enable few-shot adaptation** - full fine-tuning required
4. **Future work:** Redesign context mechanism for true meta-learning

---

## Phase 2: Fix Underperforming Targets - IN PROGRESS

### Data Downloaded ✅

Downloaded 67,141 additional ChEMBL records:
- **Proteases:** 25,125 records (BACE1, BACE2, Renin, Thrombin, Cathepsin, MMP)
- **CYPs:** 21,611 records (CYP3A4, CYP2D6, CYP2C9, CYP1A2, CYP2C19, CYP2B6, CYP2E1)
- **Ion channels:** 20,405 records (hERG, Nav1.5, Nav1.7, Cav1.2, GABA_A, nAChR, Kv1.3)

Data location: `data/raw/chembl_v2/`

### V2 Training Script ✅

Created simplified single-GPU training script: `scripts/train_v2.py`

**To run:**
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/train_v2.py --epochs 100 --output results/v2
```

### Expected Outcomes

| Target | V1 (Current) | V2 (Expected) | Goal |
|--------|--------------|---------------|------|
| BACE1 | 0.672 | 0.75-0.80 | ≥0.80 |
| CYP3A4 | 0.497 | 0.65-0.72 | ≥0.75 |
| hERG | 0.615 | 0.72-0.78 | ≥0.80 |

---

## Phase 3: Improve SOTA Targets (Future)

### DRD2 (Current: 0.960)

**Methods:**
1. Add ranking loss (pairwise + ListMLE)
2. Multi-task with selectivity panel (D1, D3, D4)

**Target:** ROC-AUC ≥0.975, EF@1% ≥45x

### EGFR (Current: 0.943)

**Methods:**
1. Add ranking loss
2. Multi-task with HER family (HER2, HER3, HER4)

**Target:** ROC-AUC ≥0.965, EF@1% ≥40x

---

## Saved Checkpoints

| Path | Description |
|------|-------------|
| `results/phase1/ablation_dude/L0_seed0_best.pt` | L0-only model |
| `results/phase1/ablation_dude/L0_L1_seed0_best.pt` | L0+L1 model (BEST) |
| `results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt` | L0+L1+L2 model |
| `results/phase1/ablation_dude/L0_L1_L2_L3_seed0_best.pt` | Full model |

---

## Results Files

| Path | Description |
|------|-------------|
| `results/phase1/ablation_dude/ablation_results.json` | ChEMBL ablation metrics |
| `results/phase1/dude_ablation/dude_results_seed0.json` | DUD-E with L1=0 |
| `results/phase1/film_analysis/` | FiLM modulation analysis |
| `results/phase1/context_embeddings/` | Context embedding analysis |
| `results/phase1/few_shot/few_shot_results.json` | Few-shot adaptation results |

---

## Commands Cheat Sheet

```bash
# Activate environment
source /home/bcheng/miniconda3/bin/activate nest

# Run V2 training
CUDA_VISIBLE_DEVICES=5 python scripts/train_v2.py --epochs 100 --output results/v2

# Run DUD-E benchmark on V2
python scripts/run_dude_on_checkpoints.py \
    --checkpoint-dir results/v2 \
    --device cuda
```

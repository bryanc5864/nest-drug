# NEST-DRUG Benchmark Results

**Date:** January 19, 2026
**Model:** NEST-DRUG (6.3M parameters)
**Checkpoints:** `results/phase1/ablation_dude/`

---

## Executive Summary

NEST-DRUG was evaluated on virtual screening benchmarks and context ablation experiments. Key finding: **context helps on training distribution but hurts on external benchmarks**.

| Benchmark | Model | Mean ROC-AUC | Notes |
|-----------|-------|--------------|-------|
| **ChEMBL Validation** | L0+L1 | 0.771 | Best on training distribution |
| **DUD-E** | L0 | 0.735 | Best on external benchmark |
| **HTS Comparison** | L0 | 0.899 | EGFR only |
| **hERG Safety** | L0 | 0.615 | Needs specialized training |

---

## 1. Context Ablation Study (NEW)

### 1.1 ChEMBL Validation Results

Models trained with different context configurations, evaluated on held-out ChEMBL test set.

| Condition | ROC-AUC | Pearson | R² | Δ vs L0 |
|-----------|---------|---------|-----|---------|
| **L0** (backbone only) | 0.757 | 0.534 | 0.278 | — |
| **L0+L1** (+ program) | **0.771** | 0.558 | 0.300 | **+1.44%** |
| **L0+L1+L2** (+ assay) | 0.770 | 0.560 | 0.300 | +1.29% |
| **L0+L1+L2+L3** (full) | 0.760 | 0.541 | 0.279 | +0.31% |

**Finding:** L0+L1 is best. Adding L2 doesn't help much. L3 hurts performance.

### 1.2 DUD-E Benchmark with Ablation

Same models evaluated on DUD-E (external benchmark).

| Condition | Mean ROC-AUC | Δ vs L0 |
|-----------|--------------|---------|
| **L0** (backbone only) | **0.735** | — |
| **L0+L1** | 0.732 | -0.22% |
| **L0+L1+L2** | 0.726 | -0.87% |
| **L0+L1+L2+L3** | 0.712 | **-2.30%** |

**Finding:** Context **hurts** on external benchmark. L0 is best.

### 1.3 Per-Target Analysis

| Target | L0 | Best Config | Best AUC | Context Helped? |
|--------|-----|-------------|----------|-----------------|
| DRD2 | 0.923 | L0 | 0.923 | No |
| EGFR | 0.833 | L0 | 0.833 | No |
| JAK2 | 0.798 | L0 | 0.798 | No |
| ESR1 | 0.790 | L0 | 0.790 | No |
| FXA | 0.755 | L0+L1 | 0.790 | Yes (+3.5%) |
| HDAC2 | 0.697 | L0+L1+L2 | 0.705 | Yes (+0.9%) |
| BACE1 | 0.675 | L0+L1+L2 | 0.706 | Yes (+3.1%) |
| CYP3A4 | 0.663 | L0+L1+L2+L3 | 0.717 | Yes (+5.4%) |
| PPARG | 0.644 | L0+L1 | 0.675 | Yes (+3.1%) |
| ADRB2 | 0.568 | L0+L1+L2 | 0.662 | Yes (+9.4%) |

**Pattern:**
- **Easy targets (L0 > 0.75):** L0 is always best
- **Hard targets (L0 < 0.70):** Context sometimes helps

### 1.4 Key Insights

1. **Context overfits to training distribution** - helps on ChEMBL, hurts on DUD-E
2. **L0 backbone is surprisingly strong** - best for external benchmarks
3. **Hard targets benefit from context** - regularization effect with L1=0
4. **Per-target L1 assignment doesn't help** - L1 embeddings are ChEMBL-specific

---

## 2. DUD-E Benchmark (10 Targets)

Best results using L0 model (no context).

### Overall Performance

| Metric | Mean | Best | Worst |
|--------|------|------|-------|
| ROC-AUC | 0.735 | 0.923 (DRD2) | 0.568 (ADRB2) |
| EF@1% | 7.2x | 18.9x (DRD2) | 3.3x (ADRB2) |

### Per-Target Results

| Target | Type | ROC-AUC | EF@1% | Notes |
|--------|------|---------|-------|-------|
| **DRD2** | GPCR | **0.923** | 18.9x | Excellent |
| **EGFR** | Kinase | **0.833** | 6.3x | Good |
| **JAK2** | Kinase | 0.798 | 5.5x | Good |
| **ESR1** | Nuclear Receptor | 0.790 | 5.0x | Good |
| **FXA** | Protease | 0.755 | 4.2x | Moderate |
| **HDAC2** | Enzyme | 0.697 | 3.8x | Moderate |
| **BACE1** | Protease | 0.675 | 3.5x | Weak |
| **CYP3A4** | CYP | 0.663 | 3.2x | Weak |
| **PPARG** | Nuclear Receptor | 0.644 | 2.9x | Weak |
| **ADRB2** | GPCR | 0.568 | 3.3x | Poor |

### Performance by Target Class

| Target Class | Targets | Mean ROC-AUC |
|--------------|---------|--------------|
| **GPCRs** | DRD2, ADRB2 | 0.746 |
| **Kinases** | EGFR, JAK2 | 0.816 |
| **Nuclear Receptors** | ESR1, PPARG | 0.717 |
| **Proteases** | FXA, BACE1 | 0.715 |
| **Enzymes** | HDAC2 | 0.697 |
| **CYPs** | CYP3A4 | 0.663 |

---

## 3. HTS Comparison (EGFR)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.899 |
| **EF@1%** | 23.1x |
| **EF@0.1%** | 43.4x |
| **Recovery@5%** | 54.8% |

---

## 4. hERG Safety Prediction

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.615 |
| **Note** | Not trained for hERG; needs specialized model |

---

## 5. Summary & Recommendations

### Use Cases

| Use Case | Recommended Model | Expected Performance |
|----------|-------------------|---------------------|
| Virtual screening (new targets) | L0 | ROC-AUC 0.70-0.85 |
| ChEMBL-like targets | L0+L1 | ROC-AUC 0.75-0.80 |
| Hard targets (BACE1, CYP3A4) | L0+L1+L2 | May help +3-9% |

### Strengths
- Strong DRD2 performance (0.923)
- Good kinase coverage (EGFR, JAK2)
- Robust L0 backbone

### Weaknesses
- Context doesn't generalize to external benchmarks
- Poor ADRB2 performance (0.568)
- CYP3A4 near-random (0.663)

### Next Steps
1. FiLM modulation analysis - verify FiLM is learning
2. Context embedding analysis - understand what embeddings represent
3. Few-shot adaptation test - prove NEST can adapt quickly
4. Expand pretraining data for BACE1, CYP3A4, hERG

---

## Appendix: Saved Checkpoints

| Path | Description |
|------|-------------|
| `results/phase1/ablation_dude/L0_seed0_best.pt` | L0-only model |
| `results/phase1/ablation_dude/L0_L1_seed0_best.pt` | L0+L1 model |
| `results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt` | L0+L1+L2 model |
| `results/phase1/ablation_dude/L0_L1_L2_L3_seed0_best.pt` | Full model |

---

*Report updated: January 19, 2026*

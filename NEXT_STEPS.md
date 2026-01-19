# NEST-DRUG: Next Steps

**Last Updated:** January 19, 2026

---

## Completed Experiments

### Phase 1.1: Context Ablation Study ✅

**ChEMBL Validation Results:**
| Condition | ROC-AUC | Δ vs L0 |
|-----------|---------|---------|
| L0 (baseline) | 0.7568 | — |
| L0+L1 | **0.7712** | +1.44% |
| L0+L1+L2 | 0.7697 | +1.29% |
| L0+L1+L2+L3 | 0.7599 | +0.31% |

**Finding:** L0+L1 is best on ChEMBL. L3 hurts performance.

### Phase 1.1b: DUD-E Benchmark with Ablation ✅

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

### Per-Target L1 Assignment Test ✅

Using different L1 IDs per DUD-E target made results **worse** (-0.3% to -0.5%).
L1 embeddings learned during ChEMBL training don't transfer to arbitrary DUD-E targets.

---

## Key Insights

1. **Context overfits to training distribution** - helps on ChEMBL, hurts on DUD-E
2. **L0 backbone is surprisingly strong** - best for external benchmarks
3. **Hard targets benefit from context** - but only with L1=0 (regularization effect)
4. **Per-target L1 doesn't help** - embeddings are ChEMBL-specific

---

## Next Experiments

### Experiment 1.2: FiLM Modulation Analysis (PRIORITY: HIGH)

**Goal:** Verify FiLM is learning meaningful modulations, not staying at identity.

**Key questions:**
- Are γ values ≠ 1?
- Are β values ≠ 0?
- Do different contexts produce different modulations?

**Command:**
```bash
python experiments/phase1_architecture/film_analysis.py \
    --checkpoint results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt
```

### Experiment 1.3: Context Embedding Analysis (PRIORITY: HIGH)

**Goal:** Understand what context embeddings represent.

**Expected findings:**
- L1 embeddings should cluster by target class
- Similar targets should have similar embeddings

**Command:**
```bash
python experiments/phase1_architecture/context_embeddings.py \
    --checkpoint results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt
```

### Experiment 1.4: Few-Shot Adaptation Test (PRIORITY: CRITICAL)

**Goal:** Prove nested architecture enables fast adaptation to new targets.

**Design:**
1. Hold out target during pretraining
2. Provide small support set (N = 10, 25, 50, 100)
3. Adapt L1 context only (freeze everything else)
4. Compare to zero-shot and full fine-tuning

**Expected results:**
| Support Size | Zero-Shot | L1-Adapted | Improvement |
|--------------|-----------|------------|-------------|
| N=10 | 0.65 | 0.72 | +0.07 |
| N=25 | 0.65 | 0.78 | +0.13 |
| N=50 | 0.65 | 0.82 | +0.17 |
| N=100 | 0.65 | 0.85 | +0.20 |

**Success criteria:** L1 adaptation achieves ≥80% of full fine-tuning with 10x fewer steps.

**Additional DUD-E targets to download:**
```bash
cd data/dude
for target in ace ache cdk2 hivpr pde5 src vegfr2; do
    wget http://dude.docking.org/targets/${target}/actives_final.mol2.gz
    wget http://dude.docking.org/targets/${target}/decoys_final.mol2.gz
done
```

---

## Phase 2: Fix Underperforming Targets

| Target | Current | Root Cause | Solution |
|--------|---------|------------|----------|
| BACE1 | 0.672 | Proteases underrepresented | Add BACE1/2, Renin, Thrombin to pretraining |
| CYP3A4 | 0.497 | CYPs not in pretraining | Add CYP family to pretraining |
| hERG | 0.615 | Ion channels missing | Add hERG, Nav1.5, Cav1.2 to pretraining |

### Data to Download

```python
targets_to_download = {
    # Proteases (for BACE1)
    'BACE1': 'CHEMBL4072',
    'BACE2': 'CHEMBL4073',
    'Renin': 'CHEMBL286',
    'Thrombin': 'CHEMBL204',
    'Cathepsin_D': 'CHEMBL3837',

    # CYPs (for CYP3A4)
    'CYP3A4': 'CHEMBL340',
    'CYP2D6': 'CHEMBL289',
    'CYP2C9': 'CHEMBL3397',
    'CYP1A2': 'CHEMBL1951',
    'CYP2C19': 'CHEMBL3356',

    # Ion channels (for hERG)
    'hERG': 'CHEMBL240',
    'Nav1.5': 'CHEMBL1988',
    'Cav1.2': 'CHEMBL4441',
    'GABA_A': 'CHEMBL2093872',
}
```

### Expected Outcomes

| Target | Before | After Retraining | After Adaptation | Goal |
|--------|--------|------------------|------------------|------|
| BACE1 | 0.672 | 0.75-0.80 | 0.82-0.86 | ≥0.80 |
| CYP3A4 | 0.497 | 0.65-0.72 | 0.75-0.82 | ≥0.75 |
| hERG | 0.615 | 0.72-0.78 | 0.80-0.85 | ≥0.80 |

---

## Phase 3: Improve SOTA Targets

### DRD2 (Current: 0.960)

**Methods:**
1. Add ranking loss (pairwise + ListMLE)
2. Multi-task with selectivity panel (D1, D3, D4)

**Data:**
- DRD1: CHEMBL2056 (~3,000 records)
- DRD2: CHEMBL217 (~12,000 records)
- DRD3: CHEMBL234 (~5,000 records)
- DRD4: CHEMBL219 (~4,000 records)

**Target:** ROC-AUC ≥0.975, EF@1% ≥45x

### EGFR (Current: 0.943)

**Methods:**
1. Add ranking loss
2. Multi-task with HER family (HER2, HER3, HER4)

**Data:**
- EGFR: CHEMBL203 (~15,000 records)
- HER2: CHEMBL1824 (~5,000 records)
- HER3: CHEMBL5838 (~1,000 records)
- HER4: CHEMBL3009 (~500 records)

**Target:** ROC-AUC ≥0.965, EF@1% ≥40x

---

## Saved Checkpoints

| Path | Description |
|------|-------------|
| `results/phase1/ablation_dude/L0_seed0_best.pt` | L0-only model |
| `results/phase1/ablation_dude/L0_L1_seed0_best.pt` | L0+L1 model |
| `results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt` | L0+L1+L2 model |
| `results/phase1/ablation_dude/L0_L1_L2_L3_seed0_best.pt` | Full model |
| `checkpoints/pretrain/best_model.pt` | Original pretrained model |

---

## Results Files

| Path | Description |
|------|-------------|
| `results/phase1/ablation_dude/ablation_results.json` | ChEMBL ablation metrics |
| `results/phase1/dude_ablation/dude_results_seed0.json` | DUD-E with L1=0 |
| `results/phase1/dude_l1_per_target/dude_results_seed0.json` | DUD-E with per-target L1 |

---

## Priority Order

1. **FiLM Analysis** - Quick diagnostic
2. **Context Embedding Analysis** - Quick diagnostic
3. **Few-Shot Adaptation** - Core value proposition test
4. **Phase 2 data download** - Can run in background
5. **Retrain with expanded data** - After Phase 1 complete

---

## Commands Cheat Sheet

```bash
# Activate environment
source /home/bcheng/miniconda3/bin/activate nest

# Run FiLM analysis
python experiments/phase1_architecture/film_analysis.py \
    --checkpoint results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt

# Run context embedding analysis
python experiments/phase1_architecture/context_embeddings.py \
    --checkpoint results/phase1/ablation_dude/L0_L1_L2_seed0_best.pt

# Run DUD-E benchmark on checkpoints
python scripts/run_dude_on_checkpoints.py \
    --checkpoint-dir results/phase1/ablation_dude \
    --device cuda

# Download additional DUD-E targets
cd data/dude
for target in ace ache cdk2 hivpr pde5 src vegfr2; do
    wget http://dude.docking.org/targets/${target}/actives_final.mol2.gz
    wget http://dude.docking.org/targets/${target}/decoys_final.mol2.gz
done
```

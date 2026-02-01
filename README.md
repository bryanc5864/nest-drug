# NEST-DRUG: Context-Conditioned Molecular Property Prediction via FiLM

A graph neural network framework for drug discovery that uses Feature-wise Linear Modulation (FiLM) to condition molecular property predictions on target context. Trained on ChEMBL, evaluated on DUD-E and LIT-PCBA virtual screening benchmarks.

## Key Findings

- **Target context (L1) improves virtual screening by +5.7% AUC** (9/10 DUD-E targets, p < 0.01 after Bonferroni correction)
- **FiLM conditioning outperforms alternatives**: +12.5% over no-context, +8.6% over additive-only conditioning
- **L1 benefit transfers to unseen compounds**: +0.018 AUC on non-leaked actives (8/10 targets positive)
- **Multi-task transfer for data-scarce targets**: NEST-DRUG achieves 0.782 AUC on CYP3A4 where per-target RF collapses to 0.238
- **Honest DUD-E assessment**: Morgan FP+RF achieves 0.998 AUC; the benchmark has severe structural bias and 50% active leakage from ChEMBL

## Architecture

```
Molecule (SMILES) ──→ Graph ──→ MPNN (6 layers) ──→ h_mol (512-dim)
                                                        │
Context:                                                ↓
  L1: Target/Program (128-dim embedding)  ──→  FiLM: h_mod = γ(ctx) ⊙ h_mol + β(ctx)
  L2: Assay Type (64-dim embedding)*                    │
  L3: Temporal Round (32-dim embedding)*                ↓
                                              Multi-Task Prediction Heads
                                                        │
                                                   pChEMBL, ADMET
```

*L2/L3 are implemented but empirically provide no benefit (see RESULTS.md).

## Model Versions

| Version | Description | DUD-E AUC | Notes |
|---------|-------------|-----------|-------|
| V1 | Original pretrain (5 generic programs) | 0.803 | Best generalization (temporal split, TDC) |
| V2 | Trained from scratch (5123 programs) | 0.850* | Collapses to 0.553 without correct L1 |
| V3 | Fine-tuned from V1 (5123 programs) | **0.849*** | Best overall; primary model |
| V4 | V1 + real L2/L3 data | 0.809 | L2/L3 hurt performance |

*With correct target-specific L1 context.

## Installation

```bash
git clone <repo-url> && cd NEST
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch >= 2.0 with CUDA
- PyTorch Geometric >= 2.4
- RDKit >= 2023.9
- See `requirements.txt` for full list

### Data Setup

```bash
# Download TDC benchmark datasets
python scripts/download_tdc.py

# Download and process ChEMBL (~24 GB)
python scripts/download_chembl.py --method sqlite
python scripts/process_chembl.py --extract
python scripts/process_chembl.py --process

# Download DUD-E benchmark data
python scripts/download_benchmark_data.py --datasets dude

# Download LIT-PCBA (optional, ~52 MB)
# See scripts/download_benchmark_data.py or download from:
# https://drugdesign.unistra.fr/LIT-PCBA/
```

## Usage

### Training

```bash
# Train V2-style model from scratch with expanded ChEMBL data
python scripts/train_v2.py \
    --data data/raw/chembl_v2/ \
    --output results/v2/ \
    --epochs 100 \
    --gpu 0

# Fine-tune from V1 checkpoint (V3-style)
python scripts/train_v2.py \
    --data data/raw/chembl_v2/ \
    --output results/v3/ \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --epochs 100 \
    --gpu 0
```

### DUD-E Benchmark Evaluation

```bash
# Evaluate model on DUD-E with L1 context ablation
python scripts/benchmarks/run_dude.py \
    --checkpoint results/v3/best_model.pt \
    --output results/v3/dude_benchmark/ \
    --gpu 0
```

### Running Experiments

All supplementary experiments are in `scripts/experiments/`:

```bash
# FiLM ablation (FiLM vs No Context vs Additive)
python scripts/experiments/film_ablation.py --checkpoint results/v3/best_model.pt --gpu 0

# L1 context ablation with statistical significance
python scripts/experiments/statistical_significance.py --checkpoint results/v3/best_model.pt --gpu 0

# Data leakage analysis
python scripts/experiments/data_leakage_check.py --checkpoint results/v3/best_model.pt

# LIT-PCBA benchmark (real inactives)
python scripts/experiments/litpcba_benchmark.py --checkpoint results/v3/best_model.pt --gpu 0

# Leaked vs non-leaked DUD-E ablation
python scripts/experiments/leaked_vs_nonleaked_ablation.py --checkpoint results/v3/best_model.pt --gpu 0

# Morgan FP + RF baseline
python scripts/experiments/morgan_rf_baseline.py

# BACE1 error analysis
python scripts/experiments/bace1_analysis.py --checkpoint results/v3/best_model.pt --gpu 0

# ESM-2 protein embedding analysis
python scripts/experiments/esm2_embedding_analysis.py --checkpoint results/v3/best_model.pt --gpu 0
```

## Repository Structure

```
NEST/
├── src/                          # Core library
│   ├── models/
│   │   ├── nest_drug.py          # Main NESTDRUG model class
│   │   ├── mpnn.py               # Message-passing neural network backbone
│   │   ├── context.py            # L1/L2/L3 context modules + FiLM layer
│   │   ├── heads.py              # Multi-task prediction heads
│   │   └── ensemble.py           # Deep ensemble wrapper
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   ├── data_utils.py         # SMILES-to-graph conversion, batching
│   │   └── schedulers.py         # Learning rate schedulers
│   ├── data/
│   │   ├── datasets.py           # Portfolio/Program/DMTA datasets
│   │   └── standardize.py        # SMILES canonicalization, unit harmonization
│   ├── evaluation/
│   │   ├── dmta_replay.py        # DMTA cycle simulation
│   │   └── metrics.py            # Evaluation metrics
│   └── benchmarks/
│       ├── data_loaders.py       # DUD-E, LIT-PCBA, TDC data loading
│       └── metrics.py            # Benchmark-specific metrics
├── scripts/
│   ├── train_v2.py               # Main training script
│   ├── benchmarks/               # Benchmark runners (DUD-E, hERG, LIT-PCBA)
│   ├── experiments/              # All supplementary experiment scripts
│   ├── download_*.py             # Data download scripts
│   └── process_chembl.py         # ChEMBL processing pipeline
├── configs/
│   └── default.yaml              # Model and training hyperparameters
├── data/
│   ├── raw/                      # Downloaded datasets (ChEMBL, TDC, BindingDB)
│   ├── processed/                # Processed parquet files
│   └── external/                 # Benchmark datasets (DUD-E, LIT-PCBA)
├── results/                      # All experiment outputs and model checkpoints
│   ├── v1_pretrain_dude/         # V1 results
│   ├── v3/                       # V3 model + DUD-E results
│   ├── v4/                       # V4 model + DUD-E results
│   ├── experiments/              # Supplementary experiment outputs
│   └── figures/                  # Publication figures
├── docs/
│   ├── DATA_SOURCES.md           # Detailed data requirements
│   └── TECHNICAL_IMPLEMENTATION.md
├── RESULTS.md                    # Comprehensive experimental results
└── requirements.txt
```

## Experiment Index

Full results are documented in [RESULTS.md](RESULTS.md). Key experiments:

| ID | Experiment | Section |
|----|-----------|---------|
| 1D | L1 Context Ablation (correct vs generic) | Phase 1 |
| 2B | Context-Conditional Attribution | Phase 2 |
| 5A | Statistical Significance (5 seeds, Bonferroni) | Phase 5 |
| 5D | DMTA Replay Simulation | Phase 5 |
| 7A | FiLM Ablation (FiLM vs No Context vs Additive) | Phase 7 |
| 7B | BACE1 Error Analysis (4-part investigation) | Phase 7 |
| 7E | Morgan FP + RF Baseline | Phase 7 |
| 7F | Data Leakage Check (ChEMBL vs DUD-E overlap) | Phase 7 |
| 7G | DUD-E Structural Bias Analysis | Phase 7 |
| 7H | Per-Target ChEMBL RF Baseline | Phase 7 |
| 7I | ESM-2 Protein Embedding Analysis | Phase 7 |
| 7J | Leaked vs Non-Leaked DUD-E Ablation | Phase 7 |
| 7K | LIT-PCBA Benchmark (real inactives) | Phase 7 |

## Known Limitations

- **DUD-E benchmark is structurally biased**: Decoys are trivially separable by 2D fingerprints (0.998 AUC). All reported DUD-E numbers should be interpreted with this caveat.
- **LIT-PCBA performance is near-random** (0.517 generic AUC): The model does not generalize well to realistic screening conditions without target context.
- **L2/L3 context levels are ineffective**: Despite architectural support, assay-type and temporal-round context hurt rather than help.
- **BACE1 is a consistent failure case**: Correct L1 hurts performance due to embedding bias (see Section 7B).
- **50% active leakage from ChEMBL**: DUD-E AUC numbers are inflated for all ChEMBL-pretrained models.

## References

- Ghiandoni et al. (2024). Augmenting DMTA using predictive AI modelling at AstraZeneca
- Retchin et al. (2024). DrugGym: A testbed for the economics of autonomous drug discovery
- Huang et al. (2021). Therapeutics Data Commons
- Tran-Nguyen et al. (2020). LIT-PCBA: An Unbiased Evaluation of Machine Learning Virtual Screening
- Mysinger et al. (2012). Directory of Useful Decoys, Enhanced (DUD-E)

## License

MIT License. See [LICENSE](LICENSE) for details. ChEMBL data is under CC BY-SA 3.0. TDC and LIT-PCBA data follow respective dataset licenses.

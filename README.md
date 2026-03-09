# NEST-DRUG: Context-Conditioned Molecular Property Prediction via FiLM

A graph neural network framework for drug discovery that uses Feature-wise Linear Modulation (FiLM) to condition molecular property predictions on target context. Trained on ChEMBL bioactivity data, evaluated on DUD-E and LIT-PCBA virtual screening benchmarks.

## Key Findings

- **Target context improves virtual screening by +5.7% AUC** (9/10 DUD-E targets, p < 0.01 after Bonferroni correction across 5 seeds)
- **FiLM conditioning outperforms alternatives**: +12.5% over no-context, +8.6% over additive-only conditioning
- **L1 benefit transfers to unseen compounds**: +0.018 AUC on non-leaked actives (8/10 targets positive)
- **Multi-task transfer for data-scarce targets**: NEST-DRUG achieves 0.782 AUC on CYP3A4 where per-target RF collapses to 0.238
- **Honest DUD-E assessment**: Morgan FP+RF achieves 0.998 AUC; the benchmark has severe structural bias and 50% active leakage from ChEMBL

## Architecture

```
Molecule (SMILES) --> Graph --> MPNN (6 layers) --> h_mol (512-dim)
                                                       |
Context:                                               v
  L1: Target/Program (128-dim embedding)  -->  FiLM: h_mod = gamma(ctx) * h_mol + beta(ctx)
                                                       |
                                                       v
                                              Multi-Task Prediction Heads
                                                       |
                                                  pChEMBL, ADMET
```

The model architecture supports three context levels (L1: target, L2: assay type, L3: temporal round), but empirically only L1 provides benefit. L2/L3 are implemented but showed negative effects when trained with real data (V4 experiments).

### Model Versions

| Version | Description | DUD-E AUC | Notes |
|---------|-------------|-----------|-------|
| V1 | Portfolio pretrain (5 generic programs) | 0.803 | Best generalization (temporal split, TDC) |
| V2 | From scratch (5123 programs) | 0.850* | Collapses to 0.553 without correct L1 |
| V3 | Fine-tuned from V1 (5123 programs) | **0.849*** | Best overall; primary model |
| V4 | V1 + real L2/L3 data | 0.809 | L2/L3 hurt performance |

*With correct target-specific L1 context.

## Installation

### Requirements

- Python 3.9+
- PyTorch >= 2.0 with CUDA
- PyTorch Geometric >= 2.4
- RDKit >= 2023.9

```bash
git clone https://github.com/anonymoussubmitter-167/NEST-DRUG.git && cd NEST-DRUG
pip install -r requirements.txt
```

### Data Setup

All large datasets are excluded from the repository and must be downloaded separately.

#### ChEMBL (required for training)

ChEMBL 35 SQLite database (~4.7 GB compressed, 25 GB extracted):

```bash
# Automated download and extraction
python scripts/download_chembl.py --method sqlite

# Process into training format
python scripts/process_chembl.py --extract
python scripts/process_chembl.py --process
```

This produces `data/processed/portfolio/chembl_potency_all.parquet` (~2.4M bioactivity records across 5123 programs).

Manual alternative: download from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz

#### Expanded ChEMBL families (required for V2/V3 training)

Additional target families (proteases, CYPs, ion channels):

```bash
python scripts/download_chembl_v2.py
```

Produces enriched parquet files in `data/raw/chembl_v2_enriched/`.

#### DUD-E benchmark (required for evaluation)

Directory of Useful Decoys, Enhanced — 10 targets with property-matched decoys:

```bash
python scripts/download_benchmark_data.py --datasets dude
```

Targets: EGFR, DRD2, ADRB2, BACE1, ESR1, HDAC2, JAK2, PPARG, CYP3A4, FXA. Download source: http://dude.docking.org/

#### LIT-PCBA benchmark (optional, for unbiased evaluation)

15 targets with real experimental inactives from PubChem (~52 MB compressed):

```bash
mkdir -p data/external/litpcba
cd data/external
wget https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz
tar -xzf full_data.tgz -C litpcba/
rm full_data.tgz
```

#### TDC datasets (optional, for generalization benchmarks)

```bash
python scripts/download_tdc.py
```

#### Complete setup (all data)

```bash
python scripts/download_all.py
```

## Reproducing Results

### Training

All models are trained with `scripts/train_v2.py`. Training produces `best_model.pt` (highest validation AUC) and `final_model.pt`.

```bash
# V2: Train from scratch with expanded ChEMBL data (5123 programs)
python scripts/train_v2.py \
    --data data/raw/chembl_v2/ \
    --output results/v2/ \
    --epochs 100 \
    --batch-size 256 \
    --gpu 0

# V3: Fine-tune from V1 pretrained checkpoint (recommended)
python scripts/train_v2.py \
    --data data/raw/chembl_v2/ \
    --output results/v3/ \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --epochs 100 \
    --gpu 0
```

Training hyperparameters (from `configs/default.yaml`):
- Learning rate: 1e-4 with warmup (10% of steps) + cosine annealing
- Batch size: 256
- Weight decay: 1e-5
- Gradient clipping: norm 1.0
- Mixed precision (AMP) enabled
- Optimizer: Adam

The V1 pretrained checkpoint is produced by the full pipeline (`scripts/run_pipeline.py`) which trains on 5 generic programs from the original ChEMBL portfolio data.

### Benchmark Evaluation

#### DUD-E (virtual screening)

```bash
python scripts/benchmarks/run_dude.py \
    --checkpoint results/v3/best_model.pt \
    --output results/v3/dude_benchmark/ \
    --gpu 0
```

This evaluates per-target ROC-AUC under two conditions: correct L1 (target-specific program ID) and generic L1 (program_id=0). Expected V3 results: 0.849 mean AUC with correct L1, 0.790 with generic L1.

#### LIT-PCBA (unbiased evaluation)

```bash
python scripts/benchmarks/run_litpcba.py \
    --checkpoint results/v3/best_model.pt \
    --output results/litpcba/ \
    --gpu 0
```

Expected: 0.517 mean AUC with generic L1 (near random — confirms LIT-PCBA is genuinely hard).

#### hERG safety (cardiac toxicity)

```bash
python scripts/benchmarks/run_herg.py \
    --checkpoint results/v3/best_model.pt \
    --output results/herg/ \
    --gpu 0
```

### Ablation Studies and Experiments

All experiment scripts are in `scripts/experiments/`. Each produces JSON/CSV outputs in `results/experiments/`.

#### Core ablation experiments

```bash
# FiLM ablation: FiLM vs No Context vs Additive vs Concatenation
python scripts/experiments/film_ablation.py \
    --checkpoint results/v3/best_model.pt --gpu 0

# L1 context ablation with statistical significance (5 seeds, Bonferroni correction)
python scripts/experiments/statistical_significance.py \
    --checkpoint results/v3/best_model.pt --gpu 0

# Leaked vs non-leaked DUD-E ablation (addresses train-test contamination)
python scripts/experiments/leaked_vs_nonleaked_ablation.py \
    --checkpoint results/v3/best_model.pt --gpu 0
```

#### Benchmark bias analysis

```bash
# Morgan FP + RF baseline (demonstrates DUD-E structural bias)
python scripts/experiments/morgan_rf_baseline.py

# DUD-E structural bias analysis (NN Tanimoto, cross-target RF, similarity gap)
python scripts/experiments/dude_bias_analysis.py

# Data leakage check (ChEMBL training vs DUD-E evaluation overlap)
python scripts/experiments/data_leakage_check.py \
    --checkpoint results/v3/best_model.pt

# Per-target ChEMBL RF baseline
python scripts/experiments/per_target_rf_baseline.py
```

#### Model analysis

```bash
# BACE1 error analysis (4-part investigation of negative L1 effect)
python scripts/experiments/bace1_analysis.py \
    --checkpoint results/v3/best_model.pt --gpu 0

# ESM-2 protein embedding analysis (protein sequence vs learned L1)
python scripts/experiments/esm2_embedding_analysis.py \
    --checkpoint results/v3/best_model.pt --gpu 0

# LIT-PCBA benchmark (real inactives, no structural bias)
python scripts/experiments/litpcba_benchmark.py \
    --checkpoint results/v3/best_model.pt --gpu 0

# Context-conditional attribution (same molecule, different target attributions)
python scripts/experiments/context_attribution_fixed.py \
    --checkpoint results/v3/best_model.pt --gpu 0

# DMTA replay simulation (compound selection enrichment)
python scripts/experiments/dmta_replay.py \
    --checkpoint results/v3/best_model.pt --gpu 0

# Context benefit predictor (when does L1 help vs hurt?)
python scripts/experiments/context_benefit_predictor.py \
    --checkpoint results/v3/best_model.pt --gpu 0
```

#### Run all experiments

```bash
python scripts/experiments/run_all_experiments.py \
    --checkpoint results/v3/best_model.pt --gpu 0
```

### Expected Results Summary

#### DUD-E Virtual Screening (V3, ROC-AUC)

| Target | Correct L1 | Generic L1 | Delta |
|--------|------------|------------|-------|
| EGFR   | 0.961      | 0.826      | +0.135 |
| DRD2   | 0.987      | 0.905      | +0.082 |
| ADRB2  | 0.786      | 0.715      | +0.072 |
| BACE1  | 0.656      | 0.776      | -0.120 |
| ESR1   | 0.899      | 0.775      | +0.124 |
| HDAC2  | 0.929      | 0.830      | +0.099 |
| JAK2   | 0.908      | 0.855      | +0.053 |
| PPARG  | 0.842      | 0.761      | +0.081 |
| CYP3A4 | 0.689      | 0.638      | +0.051 |
| FXA    | 0.846      | 0.826      | +0.021 |
| **Mean** | **0.850** | **0.791** | **+0.060** |

Statistical significance (5 seeds): 9/10 targets significant after Bonferroni correction (p < 0.01). BACE1 is a consistent negative outlier explained by embedding bias (see BACE1 analysis).

#### FiLM Ablation (V3)

| Condition | Mean AUC | vs FiLM |
|-----------|----------|---------|
| FiLM (correct L1) | 0.849 | baseline |
| Additive only | 0.763 | -0.086 |
| No context | 0.724 | -0.125 |
| Concatenation* | 0.607 | -0.242 |

*Inference-time ablation with random-init projection; not a retrained architecture comparison.

#### Baselines (DUD-E)

| Method | Mean AUC | Notes |
|--------|----------|-------|
| Morgan FP + RF | 0.998 | DUD-E is structurally biased |
| 1-NN Tanimoto | 0.991 | No ML needed |
| Per-target ChEMBL RF | 0.875 | Collapses on CYP3A4 (0.238) |
| Global RF + One-Hot | 0.882 | Simple target encoding |
| **NEST-DRUG V3** | **0.849** | Context-conditional predictions |

#### Data Leakage

50% of DUD-E actives overlap with ChEMBL training data (vs 0.08% of decoys). Within-model L1 ablations remain valid since both conditions see identical compounds. On non-leaked actives, L1 improves 8/10 targets (mean delta +0.018).

## Repository Structure

```
NEST-DRUG/
├── src/                              # Core library
│   ├── models/
│   │   ├── nest_drug.py              # Main NESTDRUG model class
│   │   ├── mpnn.py                   # Message-passing neural network (6 layers)
│   │   ├── context.py                # L1/L2/L3 context modules + FiLM layer
│   │   ├── heads.py                  # Multi-task prediction heads
│   │   └── ensemble.py               # Deep ensemble wrapper (M=5)
│   ├── training/
│   │   ├── trainer.py                # Training loop with multi-stage support
│   │   ├── data_utils.py             # SMILES-to-graph (69-dim atoms, 9-dim bonds)
│   │   └── schedulers.py             # Warmup cosine + multi-timescale LR
│   ├── data/
│   │   ├── datasets.py               # Portfolio, Program, DMTA datasets
│   │   └── standardize.py            # SMILES canonicalization, unit harmonization
│   ├── evaluation/
│   │   ├── dmta_replay.py            # DMTA cycle simulation
│   │   └── metrics.py                # AUC, enrichment factor, hit rate
│   └── benchmarks/
│       ├── data_loaders.py           # DUD-E, LIT-PCBA, hERG data loading
│       └── metrics.py                # BEDROC, RIE, enrichment factors
├── scripts/
│   ├── train_v2.py                   # Main training script
│   ├── run_pipeline.py               # Full 3-stage training pipeline
│   ├── benchmarks/                   # DUD-E, hERG, LIT-PCBA evaluation
│   ├── experiments/                  # All ablation and analysis scripts
│   ├── download_*.py                 # Data download scripts
│   └── process_chembl.py             # ChEMBL extraction and processing
├── configs/
│   └── default.yaml                  # All hyperparameters
├── requirements.txt
└── LICENSE
```

## Known Limitations

- **DUD-E benchmark is structurally biased**: Decoys are trivially separable by 2D fingerprints (0.998 AUC). All reported DUD-E numbers should be interpreted with this caveat.
- **LIT-PCBA performance is near-random** (0.517 generic AUC): The model does not generalize well to realistic screening conditions without target context.
- **L2/L3 context levels are ineffective**: Despite architectural support, assay-type and temporal-round context hurt rather than help (V4 experiments).
- **BACE1 is a consistent failure case**: Correct L1 hurts performance due to embedding bias (low train-test similarity, outlier embedding, learned downward score shift).
- **50% active leakage from ChEMBL**: DUD-E AUC numbers are inflated for all ChEMBL-pretrained models.

## References

- Ghiandoni et al. (2024). Augmenting DMTA using predictive AI modelling at AstraZeneca
- Retchin et al. (2024). DrugGym: A testbed for the economics of autonomous drug discovery
- Huang et al. (2021). Therapeutics Data Commons
- Tran-Nguyen et al. (2020). LIT-PCBA: An Unbiased Evaluation of Machine Learning Virtual Screening
- Mysinger et al. (2012). Directory of Useful Decoys, Enhanced (DUD-E)

## License

MIT License. See [LICENSE](LICENSE) for details. ChEMBL data is under CC BY-SA 3.0. TDC and LIT-PCBA data follow respective dataset licenses.

# NEST-DRUG: Nested-Learning Platform for Accelerated Drug Discovery

A hierarchical machine learning framework for accelerating Design-Make-Test-Analyze (DMTA) cycles in pharmaceutical development.

## Overview

NEST-DRUG addresses key limitations in current ML approaches to drug discovery:

1. **Temporal Distribution Shift**: Standard benchmarks ignore that real DMTA programs evolve over time
2. **Hierarchical Structure**: Current models collapse organizational context (portfolio → program → assay → round)
3. **Multi-Objective Optimization**: Drug discovery requires balancing potency, ADMET, and safety endpoints
4. **Active Learning**: Principled compound selection balancing exploitation vs exploration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NEST-DRUG Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  L0: Portfolio Context (Shared MPNN Backbone)                    │
│  ├── ~2.5M parameters, update every 4 rounds                     │
│                                                                  │
│  L1: Program Context (128-dim embedding per program)             │
│  ├── Target biology, medicinal chemistry strategy                │
│                                                                  │
│  L2: Assay Context (64-dim embedding per assay)                  │
│  ├── Lab-specific noise, calibration, platform effects           │
│                                                                  │
│  L3: Round Context (32-dim embedding per round)                  │
│  └── Current SAR regime, local distribution shifts               │
├─────────────────────────────────────────────────────────────────┤
│  FiLM Conditioning: h_mod = γ ⊙ h_mol + β                        │
│  Multi-Task Heads: Potency, Solubility, LogD, Clearance, hERG... │
│  Deep Ensembles: M=5 for calibrated uncertainty                  │
│  D-Score: Weighted geometric mean of desirabilities              │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
cd /home/bcheng/NEST

# Install dependencies
pip install -r requirements.txt

# Download data (TDC only - fastest)
python scripts/download_tdc.py

# Download ChEMBL (large, ~5GB)
python scripts/download_chembl.py --method sqlite

# Process ChEMBL after download
python scripts/process_chembl.py --extract
python scripts/process_chembl.py --process
```

## Data Status

| Source | Status | Size | Records |
|--------|--------|------|---------|
| ChEMBL 35 | ✓ Ready | 24.25 GB | 21.1M activities |
| TDC ADMET | ✓ Ready | 6.6 MB | 108K samples |
| BindingDB | ⚠ Manual | ~1.5 GB | ~2.9M records |
| Historical Programs | ⚠ Required | User-provided | 5 programs |

### BindingDB Manual Download

BindingDB requires manual download from:
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

Download `BindingDB_All_*_tsv.zip` and place in `data/raw/bindingdb/`

### Historical Program Data

To run DMTA replay experiments, you need internal program data with:
- SMILES structures with stereochemistry
- Endpoint measurements (potency, ADMET)
- Test dates for round assignment
- Assay identifiers

See `docs/DATA_SOURCES.md` for detailed format requirements.

## Project Structure

```
NEST/
├── data/
│   ├── raw/              # Downloaded datasets
│   │   ├── chembl/       # ChEMBL SQLite (24GB)
│   │   ├── tdc/          # TDC ADMET CSVs (6.6MB)
│   │   └── bindingdb/    # BindingDB TSV (manual)
│   └── processed/        # Processed parquet files
├── src/
│   ├── data/             # Data loading and processing
│   ├── models/           # MPNN backbone, context modules
│   ├── training/         # Training loops
│   ├── evaluation/       # Metrics and replay
│   └── design/           # Molecular design module
├── scripts/              # Data download and processing
├── configs/              # Model and training configs
├── checkpoints/          # Saved model weights
└── results/              # Experiment outputs
```

## Key Components

### Data Processing (`src/data/`)

- **standardize.py**: SMILES canonicalization, unit harmonization, replicate aggregation
- **datasets.py**: PortfolioDataset, ProgramDataset, DMTAReplayDataset

### Models (`src/models/`)

- MPNN backbone with configurable message passing
- FiLM-based context modulation (L1-L3)
- Multi-task prediction heads
- Deep ensemble uncertainty estimation

### Training (`src/training/`)

- Phase 1: Global pretraining on portfolio data
- Phase 2: Program-specific initialization
- Phase 3: Continual nested updates during replay

### Evaluation (`src/evaluation/`)

- Time-forward prediction metrics
- DMTA replay efficiency (time-to-candidate, retrieval curves)
- Drift robustness assessment
- Continual learning forgetting analysis

## Usage

### 1. Pretraining

```python
from src.data import PortfolioDataset
from src.models import NESTDRUG

# Load portfolio data
dataset = PortfolioDataset(
    data_dir='data/processed/portfolio',
    endpoints=['pIC50', 'solubility', 'logD', 'clearance', 'hERG']
)

# Initialize model
model = NESTDRUG(
    atom_features=70,
    hidden_dim=256,
    num_mpnn_layers=6,
    context_dims={'L1': 128, 'L2': 64, 'L3': 32},
    endpoints=dataset.endpoint_columns
)

# Train
trainer = Trainer(model, dataset)
trainer.pretrain(epochs=50)
```

### 2. DMTA Replay

```python
from src.data import ProgramDataset, DMTAReplayDataset
from src.evaluation import replay_experiment

# Load program
program = ProgramDataset('data/raw/programs/fxa.csv', program_id='fxa')

# Initialize replay
replay = DMTAReplayDataset(program, seed_rounds=3)

# Run experiment
results = replay_experiment(
    model=model,
    replay=replay,
    selection_policy='ucb',  # Upper Confidence Bound
    exploration_weight=0.5
)
```

## References

- Ghiandoni et al. (2024). Augmenting DMTA using predictive AI modelling at AstraZeneca
- Retchin et al. (2024). DrugGym: A testbed for the economics of autonomous drug discovery
- Huang et al. (2021). Therapeutics Data Commons
- Behrouz et al. (2025). Nested Learning: The Illusion of Deep Learning Architectures

## License

This project is for research purposes. ChEMBL data is under CC BY-SA 3.0. TDC data follows respective dataset licenses.

## Contact

Bryan Cheng, Jasper Zhang
Cold Spring Harbor Laboratory

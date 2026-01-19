# NEST-DRUG Technical Implementation Guide

**Version:** 1.0
**Date:** January 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Model Components](#3-model-components)
4. [Training Pipeline](#4-training-pipeline)
5. [Benchmark Experiments](#5-benchmark-experiments)
6. [Data Processing](#6-data-processing)
7. [Configuration & Hyperparameters](#7-configuration--hyperparameters)
8. [Usage Examples](#8-usage-examples)

---

## 1. Overview

### 1.1 What is NEST-DRUG?

NEST-DRUG (Nested-Learning Platform for Drug Discovery) is a hierarchical deep learning framework designed to accelerate Design-Make-Test-Analyze (DMTA) cycles in pharmaceutical drug discovery. It addresses a key challenge: **distribution shift** across different stages of drug discovery programs.

### 1.2 Key Innovation

The model implements a **four-level nested architecture**:

```
L0: Shared MPNN Backbone (Portfolio-level, slow learning)
    ↓
L1: Program Context (Target-specific, moderate learning)
    ↓
L2: Assay Context (Lab-specific calibration)
    ↓
L3: Round Context (Current SAR regime, fast learning)
    ↓
Multi-Task Prediction Heads
```

Each level learns at different timescales, enabling:
- **Knowledge transfer** from large portfolio data
- **Rapid adaptation** to new DMTA rounds
- **Continual learning** without catastrophic forgetting

### 1.3 Project Structure

```
NEST/
├── src/
│   ├── models/
│   │   ├── nest_drug.py      # Main model class
│   │   ├── mpnn.py           # Message Passing Neural Network
│   │   ├── context.py        # Context embeddings & FiLM
│   │   └── heads.py          # Prediction heads
│   ├── training/
│   │   ├── trainer.py        # Three-phase trainer
│   │   ├── data_utils.py     # Data loading utilities
│   │   └── schedulers.py     # Learning rate schedulers
│   ├── benchmarks/
│   │   ├── metrics.py        # VS metrics (EF, BEDROC, etc.)
│   │   └── data_loaders.py   # Benchmark data loaders
│   └── evaluation/
│       └── dmta_replay.py    # DMTA simulation engine
├── scripts/
│   ├── run_pipeline.py       # Main training entry point
│   ├── run_all_benchmarks.py # Benchmark coordinator
│   └── benchmarks/           # Individual benchmark scripts
├── data/
│   ├── processed/            # Training data
│   └── external/             # Benchmark datasets
├── checkpoints/              # Saved models
└── results/                  # Experiment outputs
```

---

## 2. Architecture

### 2.1 High-Level Design

```
                    ┌─────────────────────────────────────┐
                    │         SMILES Input                │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │      RDKit → Graph Conversion       │
                    │  (69-dim atoms, 9-dim bonds)        │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        L0: MPNN Backbone                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │
│  │Node Encoder │ →  │ 6× Message  │ →  │ Mean + Max  │ → h_mol       │
│  │ (69→256)    │    │  Passing    │    │   Pooling   │   (512-dim)   │
│  └─────────────┘    └─────────────┘    └─────────────┘               │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    L1-L3: Context Modulation                          │
│                                                                       │
│  program_id ──→ [L1 Embedding: 128-dim] ──┐                          │
│  assay_id   ──→ [L2 Embedding:  64-dim] ──┼──→ concat (224-dim)      │
│  round_id   ──→ [L3 Embedding:  32-dim] ──┘         │                │
│                                                      ▼                │
│                                              ┌──────────────┐         │
│                                              │     FiLM     │         │
│  h_mol (512) ──────────────────────────────→│  γ⊙h + β    │         │
│                                              └──────────────┘         │
│                                                      │                │
│                                                      ▼                │
│                                              h_mod (512-dim)          │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     Multi-Task Prediction Heads                       │
│                                                                       │
│  h_mod ──→ [MLP: 512→256→128→1] ──→ pchembl_median                   │
│        ──→ [MLP: 512→256→128→1] ──→ pchembl_std                      │
│        ──→ [MLP: 512→256→128→1] ──→ n_measurements                   │
│        ──→ ...                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.2 Parameter Count

| Component | Parameters | % of Total |
|-----------|------------|------------|
| MPNN Backbone | ~2.5M | 48% |
| Context Embeddings | ~150K | 3% |
| FiLM Networks | ~300K | 6% |
| Prediction Heads | ~2.2M | 43% |
| **Total** | **~5.17M** | 100% |

---

## 3. Model Components

### 3.1 MPNN Backbone (`src/models/mpnn.py`)

The Message Passing Neural Network encodes molecular graphs into fixed-size vectors.

#### Atom Features (69 dimensions)
```python
ATOM_FEATURES = {
    'element': 45,        # One-hot for 44 elements + other
    'degree': 7,          # Valence 0-6
    'formal_charge': 5,   # -2 to +2
    'hybridization': 5,   # SP, SP2, SP3, SP3D, SP3D2
    'aromaticity': 1,     # Binary
    'ring_membership': 1, # Binary
    'num_hs': 5,          # 0-4 hydrogens
}
# Total: 69 features
```

#### Bond Features (9 dimensions)
```python
BOND_FEATURES = {
    'bond_type': 4,       # Single, Double, Triple, Aromatic
    'conjugated': 1,      # Binary
    'in_ring': 1,         # Binary
    'stereo': 3,          # None, Z, E
}
# Total: 9 features
```

#### Message Passing
```python
# For each layer t = 1..6:
for edge (i,j) with features e_ij:
    # 1. Compute message
    msg_ij = MLP([h_i || h_j || e_ij])  # 769-dim → 256-dim

    # 2. Aggregate messages
    agg_i = sum(msg_ij for j in neighbors(i))

    # 3. Update node state with GRU
    h_i = LayerNorm(h_i + GRU(agg_i, h_i))
```

#### Graph Readout
```python
# Global pooling over all nodes
h_mean = global_mean_pool(h_nodes, batch)  # 256-dim
h_max = global_max_pool(h_nodes, batch)    # 256-dim
h_mol = Linear(concat(h_mean, h_max))      # 512-dim
```

### 3.2 Context Module (`src/models/context.py`)

#### Hierarchical Context Embeddings
```python
class HierarchicalContext(nn.Module):
    def __init__(self, num_programs, num_assays, num_rounds):
        self.program_emb = nn.Embedding(num_programs, 128)  # L1
        self.assay_emb = nn.Embedding(num_assays, 64)       # L2
        self.round_emb = nn.Embedding(num_rounds, 32)       # L3

    def forward(self, program_ids, assay_ids, round_ids):
        z_program = self.program_emb(program_ids)  # [B, 128]
        z_assay = self.assay_emb(assay_ids)        # [B, 64]
        z_round = self.round_emb(round_ids)        # [B, 32]
        return torch.cat([z_program, z_assay, z_round], dim=-1)  # [B, 224]
```

#### FiLM (Feature-wise Linear Modulation)
```python
class FiLM(nn.Module):
    def __init__(self, context_dim=224, feature_dim=512):
        self.gamma_net = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        # Initialize to identity: γ=1, β=0

    def forward(self, h_mol, context):
        gamma = self.gamma_net(context)  # Scale
        beta = self.beta_net(context)    # Shift
        return gamma * h_mol + beta      # FiLM modulation
```

### 3.3 Prediction Heads (`src/models/heads.py`)

Each endpoint has a dedicated MLP head:

```python
class PredictionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[256, 128], task_type='regression'):
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)
        self.task_type = task_type

    def forward(self, h_mod):
        out = self.mlp(h_mod)
        if self.task_type == 'classification':
            out = torch.sigmoid(out)
        return out
```

---

## 4. Training Pipeline

### 4.1 Three-Phase Training Protocol

#### Phase 1: Portfolio Pretraining

**Purpose:** Learn general molecular representations from large diverse dataset

```python
# Run pretraining
python scripts/run_pipeline.py --mode pretrain \
    --data data/processed/portfolio/chembl_potency_all.parquet \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-4 \
    --gpus 0,1,2,3
```

**Configuration:**
- Learning rate: 1e-4 (single rate for all parameters)
- Batch size: 256
- Epochs: 100
- Optimizer: AdamW with weight decay 1e-5
- Scheduler: Warmup (10%) + Cosine decay
- Loss: Multi-task MSE + BCE

**Output:** `checkpoints/pretrain/best_model.pt`

#### Phase 2: Program Initialization

**Purpose:** Initialize context embeddings for a specific drug discovery program

```python
# Initialize program
python scripts/run_pipeline.py --mode program_init \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --program-data data/processed/programs/program_egfr.parquet \
    --program-id 0 \
    --seed-rounds 0,1,2
```

**Multi-timescale Learning Rates:**
```python
param_groups = [
    {'params': backbone,  'lr': 1e-5},   # Slow (preserve pretraining)
    {'params': context,   'lr': 1e-3},   # Fast (learn new context)
    {'params': heads,     'lr': 1e-4},   # Moderate
]
```

#### Phase 3: Continual Learning (Per-Round Updates)

**Purpose:** Adapt to each new DMTA round while preventing catastrophic forgetting

```python
# DMTA replay simulation
python scripts/run_pipeline.py --mode replay \
    --checkpoint checkpoints/program/program_0/initialized_model.pt \
    --program-data data/processed/programs/program_egfr.parquet \
    --epochs-per-round 20
```

**Per-level Learning Rates:**
```python
param_groups = [
    {'params': L3_embeddings, 'lr': 1e-3},   # Fastest (current round)
    {'params': L2_embeddings, 'lr': 5e-4},   # Moderate
    {'params': L1_embeddings, 'lr': 5e-4},   # Slow
    {'params': backbone,      'lr': 1e-6},   # Almost frozen
    {'params': heads,         'lr': 5e-5},
]
```

**Continual Learning Mechanisms:**
1. **Drift Regularization:** Penalize large changes to context embeddings
2. **Experience Replay:** Mix 20% historical data with current round
3. **LR Decay:** Reduce learning rates by 0.95× each round

### 4.2 Loss Function

```python
def compute_loss(predictions, targets, masks, endpoint_weights):
    total_loss = 0
    for endpoint in predictions:
        pred = predictions[endpoint]
        target = targets[endpoint]
        mask = masks[endpoint]  # 1 = valid, 0 = missing

        if endpoint_type == 'regression':
            loss = F.mse_loss(pred, target, reduction='none')
        else:
            loss = F.binary_cross_entropy(pred, target, reduction='none')

        # Masked mean
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss += endpoint_weights[endpoint] * masked_loss

    return total_loss
```

---

## 5. Benchmark Experiments

### 5.1 Benchmark Suite Overview

| Benchmark | Script | Targets | Metrics |
|-----------|--------|---------|---------|
| DUD-E | `run_dude.py` | 10 | ROC-AUC, BEDROC, EF |
| hERG Safety | `run_herg.py` | 1 | ROC-AUC, Sensitivity/Specificity |
| HTS Comparison | `run_hts_comparison.py` | 1 (EGFR) | EF, Recovery curves |
| LIT-PCBA | `run_litpcba.py` | 15 | ROC-AUC, BEDROC, EF |

### 5.2 Running Benchmarks

```bash
# Run all available benchmarks
python scripts/run_all_benchmarks.py \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --gpu 0 \
    --skip-download

# Run specific benchmarks
python scripts/run_all_benchmarks.py \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --benchmarks dude herg \
    --gpu 0
```

### 5.3 Virtual Screening Metrics

#### Enrichment Factor (EF)
```python
def enrichment_factor(y_true, y_score, percentage):
    """
    EF = (hits in top X%) / (expected hits in random X%)
    """
    n_select = int(len(y_true) * percentage / 100)
    top_indices = np.argsort(y_score)[-n_select:]
    hits = y_true[top_indices].sum()
    expected = y_true.sum() * percentage / 100
    return hits / expected
```

#### BEDROC (Boltzmann-Enhanced Discrimination of ROC)
```python
def bedroc(y_true, y_score, alpha=20.0):
    """
    Early enrichment metric emphasizing top-ranked compounds.
    alpha=20 focuses on top ~8% of ranked list.
    """
    n = len(y_true)
    n_actives = y_true.sum()

    ranks = np.argsort(np.argsort(-y_score)) + 1
    active_ranks = ranks[y_true == 1]

    s = np.sum(np.exp(-alpha * active_ranks / n))
    r_a = n_actives / n

    rand = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha/n) - 1)
    best = (1 - np.exp(-alpha * r_a)) / (1 - np.exp(-alpha/n))

    return (s - rand) / (best - rand)
```

#### ROC-AUC and Partial AUC
```python
from sklearn.metrics import roc_auc_score, roc_curve

def calculate_roc_metrics(y_true, y_score):
    roc_auc = roc_auc_score(y_true, y_score)

    # Partial AUC at 10% FPR
    fpr, tpr, _ = roc_curve(y_true, y_score)
    partial_auc_10 = np.trapz(tpr[fpr <= 0.1], fpr[fpr <= 0.1]) / 0.1

    return roc_auc, partial_auc_10
```

### 5.4 DUD-E Benchmark Implementation

```python
# scripts/benchmarks/run_dude.py

def run_dude_benchmark(model, device, output_dir, targets=None):
    """Run NEST-DRUG on DUD-E virtual screening benchmark."""

    # Load DUD-E data (actives + property-matched decoys)
    dude_data = load_all_dude(targets=targets)

    all_results = {}
    for target_name, df in dude_data.items():
        # Score all compounds
        smiles_list = df['smiles'].tolist()
        scores, valid_indices = score_compounds(model, smiles_list, device)

        # Get labels
        y_true = df.iloc[valid_indices]['is_active'].values
        y_score = np.array(scores)

        # Calculate comprehensive metrics
        metrics = calculate_all_vs_metrics(y_true, y_score, name=target_name)
        all_results[target_name] = metrics

    return all_results
```

### 5.5 HTS Comparison Implementation

```python
# scripts/run_hts_comparison.py

def run_hts_comparison(model, device, output_dir):
    """Simulate HTS campaign with NEST-DRUG prioritization."""

    # Build library: actives + ZINC decoys
    actives = load_egfr_actives()  # ~2,500 actives
    decoys = load_zinc_decoys()     # ~250,000 decoys

    library = pd.concat([actives, decoys])

    # Score entire library
    scores = score_compounds(model, library['smiles'].tolist(), device)

    # Calculate enrichment at various screening depths
    for budget in [50, 100, 500, 1000, 5000, 10000]:
        top_indices = np.argsort(scores)[-budget:]
        hits = library.iloc[top_indices]['is_active'].sum()
        hit_rate = hits / budget
        ef = hit_rate / baseline_rate
        print(f"Budget {budget}: {hits} hits, EF={ef:.1f}x")
```

### 5.6 hERG Safety Benchmark

```python
# scripts/benchmarks/run_herg.py

def run_herg_benchmark(model, device, output_dir):
    """Evaluate hERG cardiac safety prediction."""

    # Load hERG data (blockers vs non-blockers)
    herg_df = load_herg()  # IC50 < 10 µM = blocker

    # Score compounds
    scores = score_compounds(model, herg_df['smiles'].tolist(), device)
    y_true = herg_df['is_blocker'].values

    # Classification at multiple thresholds
    for threshold_pct in [10, 20, 30, 50]:
        threshold = np.percentile(scores, 100 - threshold_pct)
        y_pred = (scores >= threshold).astype(int)

        # Calculate sensitivity (catch all blockers) and specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)  # Critical for safety
        specificity = tn / (tn + fp)
```

---

## 6. Data Processing

### 6.1 SMILES to Graph Conversion

```python
# src/training/data_utils.py

def smiles_to_graph(smiles: str) -> Dict[str, torch.Tensor]:
    """Convert SMILES to PyTorch Geometric-style graph."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Extract atom features
    node_features = []
    for atom in mol.GetAtoms():
        features = get_atom_features(atom)  # 69-dim
        node_features.append(features)

    # Extract bond features and connectivity
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = get_bond_features(bond)  # 9-dim
        # Add both directions
        edge_index.extend([[i, j], [j, i]])
        edge_features.extend([features, features])

    return {
        'node_features': torch.tensor(node_features, dtype=torch.float),
        'edge_index': torch.tensor(edge_index, dtype=torch.long).T,
        'edge_features': torch.tensor(edge_features, dtype=torch.float),
        'num_atoms': mol.GetNumAtoms(),
    }
```

### 6.2 Batch Collation

```python
def collate_batch(samples: List[Dict]) -> Dict:
    """Collate variable-size graphs into batched tensors."""

    # Concatenate node features
    node_features = torch.cat([s['node_features'] for s in samples])

    # Offset edge indices for batching
    edge_indices = []
    edge_features = []
    batch_indices = []
    offset = 0

    for i, sample in enumerate(samples):
        edge_indices.append(sample['edge_index'] + offset)
        edge_features.append(sample['edge_features'])
        batch_indices.extend([i] * sample['num_atoms'])
        offset += sample['num_atoms']

    return {
        'node_features': node_features,
        'edge_index': torch.cat(edge_indices, dim=1),
        'edge_features': torch.cat(edge_features),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
    }
```

### 6.3 Data Sources

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| ChEMBL Portfolio | ChEMBL 33 | ~2M compounds | Pretraining |
| Program Data | ChEMBL + Internal | ~10K per program | Program training |
| DUD-E | dude.docking.org | ~500K total | VS benchmark |
| hERG | ChEMBL/TDC | ~8K | Safety benchmark |
| ZINC | ZINC15 | ~250K | Decoys |

---

## 7. Configuration & Hyperparameters

### 7.1 Model Hyperparameters

```yaml
# Model Architecture
mpnn:
  node_input_dim: 69
  edge_input_dim: 9
  hidden_dim: 256
  num_layers: 6
  output_dim: 512
  dropout: 0.1

context:
  program_dim: 128
  assay_dim: 64
  round_dim: 32
  num_programs: 5
  num_assays: 50
  num_rounds: 150

heads:
  hidden_dims: [256, 128]
  dropout: 0.1
```

### 7.2 Training Hyperparameters

```yaml
# Phase 1: Pretraining
pretraining:
  learning_rate: 1e-4
  batch_size: 256
  num_epochs: 100
  warmup_ratio: 0.1
  weight_decay: 1e-5
  gradient_clip: 1.0

# Phase 2: Program Init
program_init:
  backbone_lr: 1e-5
  context_lr: 1e-3
  head_lr: 1e-4
  batch_size: 64
  num_epochs: 50
  drift_weight: 0.01

# Phase 3: Continual
continual:
  l1_lr: 5e-4
  l2_lr: 5e-4
  l3_lr: 1e-3
  backbone_lr: 1e-6
  head_lr: 5e-5
  epochs_per_round: 20
  batch_size: 32
  replay_fraction: 0.2
  lr_decay_per_round: 0.95
```

---

## 8. Usage Examples

### 8.1 Training from Scratch

```bash
# Step 1: Process data
python scripts/process_chembl.py --output data/processed/portfolio/

# Step 2: Pretrain backbone
python scripts/run_pipeline.py --mode pretrain \
    --data data/processed/portfolio/chembl_potency_all.parquet \
    --epochs 100 --gpus 0,1,2,3

# Step 3: Initialize program
python scripts/run_pipeline.py --mode program_init \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --program-data data/processed/programs/program_egfr.parquet \
    --program-id 0

# Step 4: Run DMTA replay
python scripts/run_pipeline.py --mode replay \
    --checkpoint checkpoints/program/program_0/initialized_model.pt \
    --program-data data/processed/programs/program_egfr.parquet
```

### 8.2 Running Benchmarks

```bash
# Download benchmark data
python scripts/download_benchmark_data.py --datasets dude herg

# Run all benchmarks
python scripts/run_all_benchmarks.py \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --gpu 0 --skip-download

# Run specific benchmark
python scripts/benchmarks/run_dude.py \
    --checkpoint checkpoints/pretrain/best_model.pt \
    --targets egfr drd2 --gpu 0
```

### 8.3 Making Predictions

```python
import torch
from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph

# Load model
checkpoint = torch.load('checkpoints/pretrain/best_model.pt')
model = create_nest_drug(
    num_programs=5,
    num_assays=50,
    num_rounds=150,
    endpoints={'pchembl_median': {'type': 'regression', 'weight': 1.0}}
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Convert SMILES to graph
smiles = "CCOc1ccc(NC(=O)c2ccc(C)cc2)cc1"
graph = smiles_to_graph(smiles)

# Prepare input tensors
node_features = graph['node_features'].unsqueeze(0).to(device)
edge_index = graph['edge_index'].to(device)
edge_features = graph['edge_features'].to(device)
batch = torch.zeros(graph['num_atoms'], dtype=torch.long, device=device)

# Context IDs (zeros for inference without specific context)
program_ids = torch.zeros(1, dtype=torch.long, device=device)
assay_ids = torch.zeros(1, dtype=torch.long, device=device)
round_ids = torch.zeros(1, dtype=torch.long, device=device)

# Predict
with torch.no_grad():
    predictions = model(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        batch=batch,
        program_ids=program_ids,
        assay_ids=assay_ids,
        round_ids=round_ids
    )

print(f"Predicted pActivity: {predictions['pchembl_median'].item():.2f}")
```

---

## Appendix: Key Files Reference

| File | Description |
|------|-------------|
| `src/models/nest_drug.py` | Main NESTDRUG class combining all components |
| `src/models/mpnn.py` | Message Passing Neural Network backbone |
| `src/models/context.py` | Context embeddings and FiLM conditioning |
| `src/models/heads.py` | Multi-task prediction heads |
| `src/training/trainer.py` | Three-phase training orchestration |
| `src/training/data_utils.py` | Data loading and graph conversion |
| `src/benchmarks/metrics.py` | Virtual screening metrics |
| `scripts/run_pipeline.py` | Main training entry point |
| `scripts/run_all_benchmarks.py` | Benchmark coordinator |

---

*Document generated for NEST-DRUG v1.0*

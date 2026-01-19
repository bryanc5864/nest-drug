#!/usr/bin/env python3
"""
Phase 1.1: Context Ablation Study

Critical experiment to validate that L1-L3 contexts actually improve performance.

Key insight: Our current results used ALL ZEROS for context IDs, meaning
the nested architecture was never actually tested. This experiment:
1. Trains models with varying context configurations
2. Actually USES different context IDs during training
3. Compares performance across ablation conditions

Uses the same comprehensive logging as the main trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, r2_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nest_drug import NESTDRUG, create_nest_drug
from src.training.data_utils import smiles_to_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""
    name: str = "full"
    use_l1: bool = True  # Program context
    use_l2: bool = True  # Assay context
    use_l3: bool = True  # Round context

    # Training
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Data
    n_programs: int = 10   # Number of L1 contexts
    n_assays: int = 50     # Number of L2 contexts
    n_rounds: int = 20     # Number of L3 contexts

    # Validation
    validate_every: int = 1
    log_every: int = 50
    log_gradients: bool = True

    # Mixed precision
    use_amp: bool = True

    def condition_name(self) -> str:
        parts = ['L0']
        if self.use_l1:
            parts.append('L1')
        if self.use_l2:
            parts.append('L2')
        if self.use_l3:
            parts.append('L3')
        return '_'.join(parts)


class AblationLogger:
    """
    Comprehensive logging for ablation experiments.
    Mirrors the main TrainingLogger functionality.
    """

    def __init__(self, log_dir: Path, condition_name: str, seed: int):
        self.log_dir = log_dir
        self.condition_name = condition_name
        self.seed = seed

        # Create log files
        self.log_file = log_dir / f"{condition_name}_seed{seed}.log"
        self.metrics_file = log_dir / f"{condition_name}_seed{seed}_metrics.json"

        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(file_handler)
        self.file_handler = file_handler

        # Metrics storage
        self.batch_metrics: List[Dict] = []
        self.epoch_metrics: List[Dict] = []
        self.validation_metrics: List[Dict] = []
        self.gradient_metrics: List[Dict] = []

        # Timing
        self.epoch_start_time = None
        self.training_start_time = time.time()

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        lr: float,
        batch_size: int,
    ):
        """Log batch-level statistics."""
        metrics = {
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss,
            'lr': lr,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat(),
        }
        self.batch_metrics.append(metrics)

        logger.debug(
            f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{total_batches} | "
            f"Loss: {loss:.4f} | LR: {lr:.2e}"
        )

    def log_gradients(self, model: nn.Module, epoch: int, batch_idx: int):
        """Log gradient statistics per module."""
        grad_stats = {}
        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                param_norm = grad.norm(2).item()
                total_norm += param_norm ** 2

                # Group by top-level module
                parts = name.split('.')
                if len(parts) >= 2:
                    module_name = f"{parts[0]}.{parts[1]}"
                else:
                    module_name = parts[0]

                if module_name not in layer_norms:
                    layer_norms[module_name] = []
                layer_norms[module_name].append(param_norm)

        total_norm = total_norm ** 0.5

        # Aggregate
        module_stats = {}
        for module, norms in layer_norms.items():
            module_stats[module] = {
                'mean': sum(norms) / len(norms),
                'max': max(norms),
            }

        grad_stats = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_norm': total_norm,
            'module_stats': module_stats,
        }
        self.gradient_metrics.append(grad_stats)

        # Warn on issues
        if total_norm > 100:
            logger.warning(f"  Large gradient norm: {total_norm:.2f}")
        if total_norm < 1e-7:
            logger.warning(f"  Vanishing gradients: {total_norm:.2e}")

        return total_norm

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = time.time()
        logger.info("=" * 70)
        logger.info(f"EPOCH {epoch + 1}/{total_epochs} | Condition: {self.condition_name} | Seed: {self.seed}")
        logger.info("=" * 70)

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
    ):
        """Log epoch end with validation metrics."""
        epoch_time = time.time() - self.epoch_start_time

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat(),
        }
        self.epoch_metrics.append(metrics)

        logger.info(f"Epoch {epoch + 1} Complete | Time: {epoch_time:.1f}s")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
        logger.info(f"  Val Pearson: {val_metrics.get('pearson', 0):.4f}")
        logger.info(f"  Val R²: {val_metrics.get('r2', 0):.4f}")
        logger.info(f"  Val MSE: {val_metrics.get('mse', 0):.4f}")

    def log_validation(self, epoch: int, metrics: Dict[str, Any]):
        """Log detailed validation metrics."""
        self.validation_metrics.append({
            'epoch': epoch,
            **metrics,
            'timestamp': datetime.now().isoformat(),
        })

        # Log per-program metrics if available
        if 'per_program_auc' in metrics and metrics['per_program_auc']:
            logger.info("  Per-Program ROC-AUC:")
            for prog_id, auc in sorted(metrics['per_program_auc'].items()):
                logger.info(f"    Program {prog_id}: {auc:.4f}")

    def save_metrics(self):
        """Save all metrics to JSON."""
        all_metrics = {
            'condition': self.condition_name,
            'seed': self.seed,
            'total_time': time.time() - self.training_start_time,
            'batch_metrics': self.batch_metrics[-100:],  # Keep last 100
            'epoch_metrics': self.epoch_metrics,
            'validation_metrics': self.validation_metrics,
            'gradient_summary': self._summarize_gradients(),
        }

        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Metrics saved to {self.metrics_file}")

    def _summarize_gradients(self):
        """Summarize gradient statistics."""
        if not self.gradient_metrics:
            return {}

        norms = [g['total_norm'] for g in self.gradient_metrics]
        return {
            'mean_norm': sum(norms) / len(norms),
            'max_norm': max(norms),
            'min_norm': min(norms),
        }

    def cleanup(self):
        """Remove file handler."""
        logger.removeHandler(self.file_handler)
        self.file_handler.close()


class AblatedNESTDRUG(nn.Module):
    """
    NEST-DRUG with configurable context levels for ablation.
    """

    def __init__(self, base_model: NESTDRUG, config: AblationConfig):
        super().__init__()
        self.base_model = base_model
        self.use_l1 = config.use_l1
        self.use_l2 = config.use_l2
        self.use_l3 = config.use_l3

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
        program_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Zero out disabled contexts
        if not self.use_l1:
            program_ids = torch.zeros_like(program_ids)
        if not self.use_l2:
            assay_ids = torch.zeros_like(assay_ids)
        if not self.use_l3:
            round_ids = torch.zeros_like(round_ids)

        return self.base_model(
            node_features, edge_index, edge_features, batch,
            program_ids, assay_ids, round_ids
        )


class AblationDataset(Dataset):
    """
    Dataset with context ID assignment for ablation study.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = 'pchembl_median',
        smiles_col: str = 'smiles',
        target_id_col: str = 'target_chembl_id',
        assay_id_col: str = 'assay_chembl_id',
        n_programs: int = 10,
        n_assays: int = 50,
        n_rounds: int = 20,
    ):
        self.data = data.copy().reset_index(drop=True)
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.n_programs = n_programs
        self.n_assays = n_assays
        self.n_rounds = n_rounds

        # Assign context IDs
        self._assign_contexts(target_id_col, assay_id_col)

        # Pre-filter valid SMILES
        logger.info("Validating SMILES...")
        valid_mask = []
        for idx in tqdm(range(len(self.data)), desc="Checking SMILES"):
            smi = self.data.iloc[idx][self.smiles_col]
            g = smiles_to_graph(str(smi))
            valid_mask.append(g is not None)

        self.data = self.data[valid_mask].reset_index(drop=True)
        logger.info(f"Valid molecules: {len(self.data)} / {len(valid_mask)}")

    def _assign_contexts(self, target_id_col: str, assay_id_col: str):
        """Assign context IDs based on data properties."""
        # L1 (Program): Based on target
        if target_id_col in self.data.columns:
            targets = self.data[target_id_col].unique()
            target_to_id = {t: i % self.n_programs for i, t in enumerate(targets)}
            self.data['program_id'] = self.data[target_id_col].map(target_to_id)
        else:
            self.data['program_id'] = np.random.randint(0, self.n_programs, len(self.data))

        # L2 (Assay): Based on assay ID
        if assay_id_col in self.data.columns:
            assays = self.data[assay_id_col].unique()
            assay_to_id = {a: i % self.n_assays for i, a in enumerate(assays)}
            self.data['assay_id'] = self.data[assay_id_col].map(assay_to_id)
        else:
            self.data['assay_id'] = np.random.randint(0, self.n_assays, len(self.data))

        # L3 (Round): Based on index (simulating temporal order)
        self.data['round_id'] = (
            np.arange(len(self.data)) // (len(self.data) // self.n_rounds + 1)
        ) % self.n_rounds

        logger.info(f"Context distribution:")
        logger.info(f"  L1 (Programs): {self.data['program_id'].nunique()} unique")
        logger.info(f"  L2 (Assays): {self.data['assay_id'].nunique()} unique")
        logger.info(f"  L3 (Rounds): {self.data['round_id'].nunique()} unique")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        graph = smiles_to_graph(str(row[self.smiles_col]))

        if graph is None:
            # Return dummy (should not happen after filtering)
            return None

        target = row[self.target_col]
        if pd.isna(target):
            target = 0.0

        return {
            'graph': graph,
            'target': float(target),
            'program_id': int(row['program_id']),
            'assay_id': int(row['assay_id']),
            'round_id': int(row['round_id']),
        }


def collate_fn(batch):
    """Collate batch of molecules."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    node_features = []
    edge_indices = []
    edge_features = []
    batch_indices = []
    targets = []
    program_ids = []
    assay_ids = []
    round_ids = []

    node_offset = 0
    for i, item in enumerate(batch):
        g = item['graph']
        n_nodes = g['node_features'].shape[0]

        node_features.append(g['node_features'])
        edge_indices.append(g['edge_index'] + node_offset)
        edge_features.append(g['edge_features'])
        batch_indices.extend([i] * n_nodes)

        targets.append(item['target'])
        program_ids.append(item['program_id'])
        assay_ids.append(item['assay_id'])
        round_ids.append(item['round_id'])

        node_offset += n_nodes

    return {
        'node_features': torch.cat(node_features, dim=0),
        'edge_index': torch.cat(edge_indices, dim=1),
        'edge_features': torch.cat(edge_features, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'program_ids': torch.tensor(program_ids, dtype=torch.long),
        'assay_ids': torch.tensor(assay_ids, dtype=torch.long),
        'round_ids': torch.tensor(round_ids, dtype=torch.long),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    config: AblationConfig,
    epoch: int,
    device: torch.device,
    ablation_logger: AblationLogger,
) -> float:
    """Train for one epoch with comprehensive logging."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)):
        if batch is None:
            continue

        # Move to device
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        batch_tensor = batch['batch'].to(device)
        targets = batch['targets'].to(device)
        program_ids = batch['program_ids'].to(device)
        assay_ids = batch['assay_ids'].to(device)
        round_ids = batch['round_ids'].to(device)

        optimizer.zero_grad()

        # Forward with mixed precision
        with autocast(enabled=config.use_amp):
            predictions = model(
                node_features, edge_index, edge_features, batch_tensor,
                program_ids, assay_ids, round_ids
            )

            # Get pActivity prediction
            if 'pActivity' in predictions:
                pred = predictions['pActivity'].squeeze()
            else:
                pred = list(predictions.values())[0].squeeze()

            # Mask NaN targets
            mask = ~torch.isnan(targets)
            if mask.sum() == 0:
                continue

            loss = F.mse_loss(pred[mask], targets[mask])

        # Backward
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Logging
        total_loss += loss.item()
        n_batches += 1

        if batch_idx % config.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            ablation_logger.log_batch(
                epoch, batch_idx, len(dataloader),
                loss.item(), lr, len(targets)
            )

            if config.log_gradients:
                ablation_logger.log_gradients(model, epoch, batch_idx)

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 6.0,
) -> Dict[str, Any]:
    """Comprehensive evaluation."""
    model.eval()
    all_preds = []
    all_targets = []
    all_programs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None:
                continue

            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            batch_tensor = batch['batch'].to(device)
            program_ids = batch['program_ids'].to(device)
            assay_ids = batch['assay_ids'].to(device)
            round_ids = batch['round_ids'].to(device)

            predictions = model(
                node_features, edge_index, edge_features, batch_tensor,
                program_ids, assay_ids, round_ids
            )

            if 'pActivity' in predictions:
                pred = predictions['pActivity'].squeeze()
            else:
                pred = list(predictions.values())[0].squeeze()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch['targets'].numpy())
            all_programs.extend(batch['program_ids'].numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_programs = np.array(all_programs)

    # Remove NaN
    valid = ~np.isnan(all_targets)
    all_preds = all_preds[valid]
    all_targets = all_targets[valid]
    all_programs = all_programs[valid]

    # Regression metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(mse)

    # Pearson correlation
    if len(all_preds) > 1 and np.std(all_preds) > 0 and np.std(all_targets) > 0:
        pearson = np.corrcoef(all_preds, all_targets)[0, 1]
        r2 = r2_score(all_targets, all_preds)
    else:
        pearson = 0.0
        r2 = 0.0

    # Classification metrics (active if >= threshold)
    y_true = (all_targets >= threshold).astype(int)
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        roc_auc = roc_auc_score(y_true, all_preds)
    else:
        roc_auc = 0.5

    # Per-program metrics
    per_program_auc = {}
    for prog_id in np.unique(all_programs):
        mask = all_programs == prog_id
        y_true_prog = (all_targets[mask] >= threshold).astype(int)
        if y_true_prog.sum() > 0 and y_true_prog.sum() < len(y_true_prog) and len(y_true_prog) >= 10:
            per_program_auc[int(prog_id)] = roc_auc_score(y_true_prog, all_preds[mask])

    # Enrichment factor at 1%
    n_total = len(all_preds)
    n_actives = y_true.sum()
    top_1pct_idx = np.argsort(all_preds)[-max(1, n_total // 100):]
    top_1pct_hits = y_true[top_1pct_idx].sum()
    ef_1pct = (top_1pct_hits / len(top_1pct_idx)) / (n_actives / n_total) if n_actives > 0 else 0

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'pearson': float(pearson),
        'r2': float(r2),
        'roc_auc': float(roc_auc),
        'ef_1pct': float(ef_1pct),
        'n_samples': len(all_preds),
        'n_actives': int(n_actives),
        'per_program_auc': per_program_auc,
    }


def run_single_condition(
    data: pd.DataFrame,
    config: AblationConfig,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Run ablation for a single condition and seed."""

    condition_name = config.condition_name()
    logger.info(f"\n{'='*70}")
    logger.info(f"CONDITION: {condition_name} | SEED: {seed}")
    logger.info(f"{'='*70}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize logger
    ablation_logger = AblationLogger(output_dir, condition_name, seed)

    # Create dataset (using subset for speed)
    n_samples = min(100000, len(data))  # Use up to 100K samples
    data_subset = data.sample(n_samples, random_state=seed).reset_index(drop=True)

    dataset = AblationDataset(
        data_subset,
        n_programs=config.n_programs,
        n_assays=config.n_assays,
        n_rounds=config.n_rounds,
    )

    # Split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create dataloaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    logger.info(f"Dataset: {n_train} train, {n_val} val, {len(test_indices)} test")

    # Create model
    base_model = create_nest_drug(
        num_programs=config.n_programs,
        num_assays=config.n_assays,
        num_rounds=config.n_rounds,
    )
    model = AblatedNESTDRUG(base_model, config)
    model = model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
    )

    scaler = GradScaler(enabled=config.use_amp)

    # Training loop
    best_val_auc = 0.0
    best_model_state = None
    best_epoch = 0

    for epoch in range(config.epochs):
        ablation_logger.log_epoch_start(epoch, config.epochs)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch, device, ablation_logger
        )

        # Validate
        if (epoch + 1) % config.validate_every == 0:
            val_metrics = evaluate(model, val_loader, device)
            ablation_logger.log_epoch_end(epoch, train_loss, val_metrics)
            ablation_logger.log_validation(epoch, val_metrics)

            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                logger.info(f"  New best model! ROC-AUC: {best_val_auc:.4f}")

    # Load best model and test
    logger.info(f"\nLoading best model from epoch {best_epoch}")
    model.load_state_dict(best_model_state)
    model = model.to(device)

    # Save checkpoint
    checkpoint_path = output_dir / f"{condition_name}_seed{seed}_best.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'config': {
            'use_l1': config.use_l1,
            'use_l2': config.use_l2,
            'use_l3': config.use_l3,
            'n_programs': config.n_programs,
            'n_assays': config.n_assays,
            'n_rounds': config.n_rounds,
        },
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    test_metrics = evaluate(model, test_loader, device)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"  Pearson: {test_metrics['pearson']:.4f}")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  EF@1%: {test_metrics['ef_1pct']:.2f}x")

    # Save metrics
    ablation_logger.save_metrics()
    ablation_logger.cleanup()

    return {
        'condition': condition_name,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
    }


def run_ablation_experiment(
    data_path: str,
    output_dir: str,
    n_seeds: int = 3,
    device: str = 'cuda',
):
    """Run full ablation experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    main_log = output_dir / "ablation_main.log"
    file_handler = logging.FileHandler(main_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("NEST-DRUG ABLATION STUDY")
    logger.info("=" * 80)
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Seeds: {n_seeds}")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("\nLoading data...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} records")

    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Define ablation conditions
    conditions = [
        AblationConfig(name="L0_only", use_l1=False, use_l2=False, use_l3=False),
        AblationConfig(name="L0_L1", use_l1=True, use_l2=False, use_l3=False),
        AblationConfig(name="L0_L1_L2", use_l1=True, use_l2=True, use_l3=False),
        AblationConfig(name="full", use_l1=True, use_l2=True, use_l3=True),
    ]

    all_results = {}

    for config in conditions:
        condition_name = config.condition_name()
        condition_results = []

        for seed in range(n_seeds):
            result = run_single_condition(
                data, config, seed, output_dir, device
            )
            condition_results.append(result)

        # Aggregate
        test_aucs = [r['test_metrics']['roc_auc'] for r in condition_results]
        test_pearsons = [r['test_metrics']['pearson'] for r in condition_results]
        test_r2s = [r['test_metrics']['r2'] for r in condition_results]

        all_results[condition_name] = {
            'config': {
                'use_l1': config.use_l1,
                'use_l2': config.use_l2,
                'use_l3': config.use_l3,
            },
            'results': condition_results,
            'mean_roc_auc': float(np.mean(test_aucs)),
            'std_roc_auc': float(np.std(test_aucs)),
            'mean_pearson': float(np.mean(test_pearsons)),
            'std_pearson': float(np.std(test_pearsons)),
            'mean_r2': float(np.mean(test_r2s)),
            'std_r2': float(np.std(test_r2s)),
        }

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Condition':<15} {'ROC-AUC':<20} {'Pearson':<20} {'R²':<20}")
    logger.info("-" * 80)

    baseline_auc = all_results['L0']['mean_roc_auc']

    for name, results in all_results.items():
        improvement = results['mean_roc_auc'] - baseline_auc
        logger.info(
            f"{name:<15} "
            f"{results['mean_roc_auc']:.4f} ± {results['std_roc_auc']:.4f}  "
            f"{results['mean_pearson']:.4f} ± {results['std_pearson']:.4f}  "
            f"{results['mean_r2']:.4f} ± {results['std_r2']:.4f}  "
            f"({'+'if improvement >= 0 else ''}{improvement:.4f})"
        )

    # Save results
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        # Make serializable
        serializable = {}
        for name, res in all_results.items():
            serializable[name] = {
                'config': res['config'],
                'mean_roc_auc': res['mean_roc_auc'],
                'std_roc_auc': res['std_roc_auc'],
                'mean_pearson': res['mean_pearson'],
                'std_pearson': res['std_pearson'],
                'mean_r2': res['mean_r2'],
                'std_r2': res['std_r2'],
                'test_metrics': [r['test_metrics'] for r in res['results']],
            }
        json.dump(serializable, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NEST-DRUG Context Ablation Study')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='results/phase1/ablation',
                        help='Output directory')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    run_ablation_experiment(
        data_path=args.data,
        output_dir=args.output,
        n_seeds=args.seeds,
        device=args.device,
    )

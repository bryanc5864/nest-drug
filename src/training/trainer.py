#!/usr/bin/env python3
"""
NEST-DRUG Trainer

Implements the three-phase training protocol with comprehensive logging:
- Phase 1: Global pretraining on portfolio data (L0 backbone)
- Phase 2: Program-specific initialization (seed window)
- Phase 3: Continual nested updates during DMTA replay

Logging includes:
- Gradient norms (total, per-layer)
- Per-batch and per-endpoint losses
- Epoch-level training/validation metrics
- Comprehensive end-of-training validation suite
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
import copy

from .schedulers import WarmupCosineScheduler, MultiTimescaleScheduler, ContinualLearningScheduler
from .data_utils import PortfolioDataLoader, ProgramDataLoader

# Import metrics for validation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import (
    compute_rmse, compute_mae, compute_r2, compute_ranking_correlation,
    compute_enrichment_factor, compute_hit_rate, compute_auc, compute_pr_auc,
    compute_calibration_error, MetricsTracker
)


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class PretrainingConfig:
    """Configuration for Phase 1: Portfolio Pretraining."""
    # Data
    data_path: str = ""
    batch_size: int = 256
    num_workers: int = 4
    max_samples: Optional[int] = None

    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints/pretrain"
    save_every: int = 10

    # Logging
    log_every: int = 50  # Log every N batches
    log_gradients: bool = True
    validate_every: int = 1  # Validate every N epochs


@dataclass
class ProgramConfig:
    """Configuration for Phase 2: Program-Specific Initialization."""
    # Data
    data_path: str = ""
    seed_rounds: List[int] = field(default_factory=lambda: [0, 1, 2])
    batch_size: int = 64
    num_workers: int = 2

    # Training
    num_epochs: int = 50
    backbone_lr: float = 1e-5  # Slow backbone updates
    context_lr: float = 1e-3   # Fast context adaptation
    head_lr: float = 1e-4      # Moderate head updates
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Regularization
    drift_weight: float = 0.01

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints/program"

    # Logging
    log_every: int = 25
    log_gradients: bool = True
    validate_every: int = 1


@dataclass
class ContinualConfig:
    """Configuration for Phase 3: Continual Nested Updates."""
    # Per-round training
    num_epochs_per_round: int = 20
    batch_size: int = 32

    # Learning rates (different timescales)
    backbone_lr: float = 1e-6   # Very slow
    l1_lr: float = 5e-4         # Program level
    l2_lr: float = 5e-4         # Assay level
    l3_lr: float = 1e-3         # Round level (fastest)
    head_lr: float = 5e-5

    # Learning rate decay over rounds
    lr_decay_per_round: float = 0.95

    # Regularization
    drift_weight: float = 0.1
    replay_fraction: float = 0.2  # Fraction of old data to replay

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints/continual"
    save_every_round: int = 5

    # Logging
    log_every: int = 10
    log_gradients: bool = True


class TrainingLogger:
    """
    Comprehensive logging utility for training.

    Logs:
    - Gradient statistics (norm, min, max, per-layer)
    - Loss values (total, per-endpoint)
    - Batch/epoch progress
    - Validation metrics (5-10 per epoch, 10+ at end)
    """

    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # Create log file
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"

        # Configure file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(file_handler)

        # Metrics storage
        self.batch_metrics: List[Dict] = []
        self.epoch_metrics: List[Dict] = []
        self.validation_metrics: List[Dict] = []
        self.gradient_metrics: List[Dict] = []

        # Timing
        self.epoch_start_time = None
        self.batch_start_time = None

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        endpoint_losses: Dict[str, float],
        learning_rates: Dict[str, float],
        batch_size: int,
    ):
        """Log batch-level training statistics."""
        metrics = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches': total_batches,
            'loss': loss,
            'endpoint_losses': endpoint_losses,
            'learning_rates': learning_rates,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat(),
        }
        self.batch_metrics.append(metrics)

        # Format endpoint losses for display
        ep_str = " | ".join([f"{k}:{v:.4f}" for k, v in list(endpoint_losses.items())[:5]])
        lr_str = " | ".join([f"{k}:{v:.2e}" for k, v in learning_rates.items()])

        logger.info(
            f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{total_batches} | "
            f"Loss: {loss:.4f} | {ep_str}"
        )
        logger.debug(f"  Learning rates: {lr_str}")

    def log_gradients(
        self,
        model: nn.Module,
        epoch: int,
        batch_idx: int,
    ):
        """Log gradient statistics."""
        grad_stats = {}
        total_norm = 0.0
        param_count = 0

        # Per-layer gradient stats
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                param_norm = grad.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

                # Group by module
                module_name = name.split('.')[0]
                if module_name not in layer_norms:
                    layer_norms[module_name] = []
                layer_norms[module_name].append(param_norm)

        total_norm = total_norm ** 0.5

        # Aggregate per-module stats
        module_stats = {}
        for module, norms in layer_norms.items():
            module_stats[module] = {
                'mean': sum(norms) / len(norms),
                'max': max(norms),
                'min': min(norms),
            }

        grad_stats = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_norm': total_norm,
            'param_count': param_count,
            'module_stats': module_stats,
            'timestamp': datetime.now().isoformat(),
        }
        self.gradient_metrics.append(grad_stats)

        # Log summary
        module_str = " | ".join([f"{k}:{v['mean']:.4f}" for k, v in list(module_stats.items())[:4]])
        logger.debug(
            f"  Gradients | Total norm: {total_norm:.4f} | {module_str}"
        )

        # Warn on gradient issues
        if total_norm > 100:
            logger.warning(f"  ⚠ Large gradient norm: {total_norm:.2f}")
        if total_norm < 1e-7:
            logger.warning(f"  ⚠ Vanishing gradients: {total_norm:.2e}")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = time.time()
        logger.info("=" * 70)
        logger.info(f"EPOCH {epoch + 1}/{total_epochs} STARTED")
        logger.info("=" * 70)

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_endpoint_losses: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log epoch end with training and validation summary."""
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_endpoint_losses': train_endpoint_losses,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
        }

        if val_metrics:
            metrics['validation'] = val_metrics

        self.epoch_metrics.append(metrics)

        logger.info("-" * 70)
        logger.info(f"EPOCH {epoch + 1} SUMMARY")
        logger.info("-" * 70)
        logger.info(f"  Training Loss: {train_loss:.4f}")
        logger.info(f"  Elapsed Time: {elapsed:.1f}s")

        # Log top endpoint losses
        sorted_losses = sorted(train_endpoint_losses.items(), key=lambda x: x[1], reverse=True)
        logger.info("  Endpoint Losses (highest first):")
        for name, loss in sorted_losses[:5]:
            logger.info(f"    {name}: {loss:.4f}")

        if val_metrics:
            logger.info("  Validation Metrics:")
            for name, value in val_metrics.items():
                logger.info(f"    {name}: {value:.4f}")

    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "epoch",
    ):
        """Log validation metrics."""
        val_record = {
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        self.validation_metrics.append(val_record)

        logger.info(f"  Validation ({phase}):")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"    {name}: {value:.4f}")
            else:
                logger.info(f"    {name}: {value}")

    def save_metrics(self):
        """Save all metrics to JSON file."""
        all_metrics = {
            'experiment': self.experiment_name,
            'batch_metrics': self.batch_metrics[-1000:],  # Keep last 1000
            'epoch_metrics': self.epoch_metrics,
            'validation_metrics': self.validation_metrics,
            'gradient_metrics': self.gradient_metrics[-1000:],  # Keep last 1000
        }

        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Saved metrics to {self.metrics_file}")


class ValidationSuite:
    """
    Comprehensive validation metrics suite.

    Per-epoch validation (5-10 metrics):
    - RMSE, MAE, R² for regression
    - AUC, Hit Rate for ranking
    - Per-endpoint losses

    End validation (10+ metrics):
    - All per-epoch metrics
    - Spearman correlation
    - Enrichment factors (1%, 5%, 10%)
    - PR-AUC
    - Calibration error
    - Temporal metrics (if applicable)
    """

    def __init__(self, endpoint_configs: Dict[str, Dict]):
        """
        Args:
            endpoint_configs: Dict mapping endpoint names to their config
                             (type: 'regression' or 'classification')
        """
        self.endpoint_configs = endpoint_configs
        self.regression_endpoints = [
            name for name, cfg in endpoint_configs.items()
            if cfg.get('type', 'regression') == 'regression'
        ]
        self.classification_endpoints = [
            name for name, cfg in endpoint_configs.items()
            if cfg.get('type') == 'classification'
        ]

    def compute_epoch_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute per-epoch validation metrics (5-10 metrics).
        """
        metrics = {}

        # Aggregate regression metrics
        all_preds = []
        all_targets = []

        for endpoint in self.regression_endpoints:
            if endpoint not in predictions:
                continue

            pred = predictions[endpoint].squeeze()
            target = targets[endpoint].squeeze()
            mask = masks.get(endpoint) if masks else None

            if mask is not None:
                mask = mask.squeeze().bool()
                pred = pred[mask]
                target = target[mask]

            if len(pred) == 0:
                continue

            all_preds.append(pred)
            all_targets.append(target)

            # Per-endpoint RMSE
            metrics[f'{endpoint}_rmse'] = compute_rmse(pred, target)

        # Global regression metrics
        if all_preds:
            preds_cat = torch.cat(all_preds)
            targets_cat = torch.cat(all_targets)

            metrics['global_rmse'] = compute_rmse(preds_cat, targets_cat)
            metrics['global_mae'] = compute_mae(preds_cat, targets_cat)
            metrics['global_r2'] = compute_r2(preds_cat, targets_cat)
            metrics['global_spearman'] = compute_ranking_correlation(preds_cat, targets_cat)

        # Classification metrics
        for endpoint in self.classification_endpoints:
            if endpoint not in predictions:
                continue

            pred = predictions[endpoint].squeeze()
            target = targets[endpoint].squeeze()
            mask = masks.get(endpoint) if masks else None

            if mask is not None:
                mask = mask.squeeze().bool()
                pred = pred[mask]
                target = target[mask]

            if len(pred) < 2:
                continue

            metrics[f'{endpoint}_auc'] = compute_auc(pred, target)

        return metrics

    def compute_end_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        uncertainties: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive end-of-training metrics (10+ metrics).
        """
        # Start with epoch metrics
        metrics = self.compute_epoch_metrics(predictions, targets, masks)

        # Additional regression metrics
        all_preds = []
        all_targets = []
        all_uncert = []

        for endpoint in self.regression_endpoints:
            if endpoint not in predictions:
                continue

            pred = predictions[endpoint].squeeze()
            target = targets[endpoint].squeeze()
            mask = masks.get(endpoint) if masks else None
            uncert = uncertainties.get(endpoint).squeeze() if uncertainties and endpoint in uncertainties else None

            if mask is not None:
                mask = mask.squeeze().bool()
                pred = pred[mask]
                target = target[mask]
                if uncert is not None:
                    uncert = uncert[mask]

            if len(pred) == 0:
                continue

            all_preds.append(pred)
            all_targets.append(target)
            if uncert is not None:
                all_uncert.append(uncert)

            # Additional per-endpoint metrics
            metrics[f'{endpoint}_mae'] = compute_mae(pred, target)
            metrics[f'{endpoint}_r2'] = compute_r2(pred, target)
            metrics[f'{endpoint}_spearman'] = compute_ranking_correlation(pred, target)

        # Global metrics
        if all_preds:
            preds_cat = torch.cat(all_preds)
            targets_cat = torch.cat(all_targets)

            # Ranking as binary (above/below median)
            actives = (targets_cat > targets_cat.median()).float()

            metrics['global_hit_rate_100'] = compute_hit_rate(preds_cat, actives, 100)
            metrics['global_hit_rate_50'] = compute_hit_rate(preds_cat, actives, 50)
            metrics['global_ef_1pct'] = compute_enrichment_factor(preds_cat, actives, 0.01)
            metrics['global_ef_5pct'] = compute_enrichment_factor(preds_cat, actives, 0.05)
            metrics['global_ef_10pct'] = compute_enrichment_factor(preds_cat, actives, 0.10)
            metrics['global_auc_ranking'] = compute_auc(preds_cat, actives)
            metrics['global_pr_auc'] = compute_pr_auc(preds_cat, actives)

            # Calibration (if uncertainties available)
            if all_uncert:
                uncert_cat = torch.cat(all_uncert)
                metrics['calibration_error'] = compute_calibration_error(
                    preds_cat, uncert_cat, targets_cat
                )

        # Classification metrics
        for endpoint in self.classification_endpoints:
            if endpoint not in predictions:
                continue

            pred = predictions[endpoint].squeeze()
            target = targets[endpoint].squeeze()
            mask = masks.get(endpoint) if masks else None

            if mask is not None:
                mask = mask.squeeze().bool()
                pred = pred[mask]
                target = target[mask]

            if len(pred) < 2:
                continue

            metrics[f'{endpoint}_pr_auc'] = compute_pr_auc(pred, target)
            metrics[f'{endpoint}_hit_rate_50'] = compute_hit_rate(pred, target, 50)
            metrics[f'{endpoint}_ef_5pct'] = compute_enrichment_factor(pred, target, 0.05)

        return metrics


class NESTDRUGTrainer:
    """
    Main trainer for NEST-DRUG implementing three-phase training
    with comprehensive logging.

    Supports multi-GPU training via DataParallel.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        log_dir: str = "logs",
        experiment_name: str = None,
        gpu_ids: Optional[List[int]] = None,
        distributed: bool = False,
    ):
        self.model = model
        self.gpu_ids = gpu_ids
        self.distributed = distributed
        self.num_gpus = 1

        # Setup device(s)
        # NOTE: DataParallel doesn't work with GNNs due to graph batching
        # For multi-GPU, use DistributedDataParallel instead
        if gpu_ids is not None and len(gpu_ids) > 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
            self.num_gpus = 1  # Use single GPU for GNN (DataParallel incompatible)

            if len(gpu_ids) > 1:
                logger.warning(f"Multi-GPU requested but DataParallel incompatible with GNNs. Using single GPU: {gpu_ids[0]}")
                logger.warning("For multi-GPU training, use DistributedDataParallel (--distributed flag)")

            self.model.to(self.device)
            self._base_model = model
        else:
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self._base_model = model

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.current_round = 0
        self.best_val_loss = float('inf')

        # History tracking
        self.training_history = {
            'pretrain': [],
            'program': [],
            'continual': [],
        }

        # Mixed precision
        self.scaler = GradScaler() if torch.cuda.is_available() else None

        # Logging
        self.logger = TrainingLogger(log_dir, experiment_name)

        # Validation suite
        self.validation_suite = None

        logger.info(f"Trainer initialized: device={self.device}, num_gpus={self.num_gpus}")

    def _setup_validation_suite(self):
        """Setup validation suite from model endpoint configs."""
        # Handle DataParallel wrapped model
        base_model = self._base_model if hasattr(self, '_base_model') else self.model
        if hasattr(base_model, 'prediction_heads'):
            endpoints = {}
            for name in base_model.prediction_heads.endpoint_names:
                # Infer type from head
                if name in ['herg', 'ames', 'bbb']:
                    endpoints[name] = {'type': 'classification'}
                else:
                    endpoints[name] = {'type': 'regression'}
            self.validation_suite = ValidationSuite(endpoints)

    # =========================================================================
    # Phase 1: Portfolio Pretraining
    # =========================================================================

    def pretrain(
        self,
        config: PretrainingConfig,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 1: Pretrain on portfolio data with comprehensive logging.
        """
        logger.info("=" * 70)
        logger.info("PHASE 1: PORTFOLIO PRETRAINING")
        logger.info("=" * 70)
        logger.info(f"Config: epochs={config.num_epochs}, lr={config.learning_rate}, "
                   f"batch_size={config.batch_size}")

        # Setup validation suite
        self._setup_validation_suite()

        # Create data loader
        train_loader = PortfolioDataLoader(
            data_path=config.data_path,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            max_samples=config.max_samples,
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # Create checkpoint directory
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        for epoch in range(config.num_epochs):
            self.current_epoch = epoch
            self.logger.log_epoch_start(epoch, config.num_epochs)

            # Train epoch
            train_loss, train_endpoint_losses = self._train_epoch(
                train_loader, optimizer, scheduler, config
            )
            history['train_loss'].append(train_loss)

            # Validation
            val_metrics = None
            if val_dataloader is not None and (epoch + 1) % config.validate_every == 0:
                val_loss, val_preds, val_targets, val_masks = self._validate_with_predictions(
                    val_dataloader, config.use_amp
                )
                history['val_loss'].append(val_loss)

                # Compute validation metrics
                if self.validation_suite:
                    val_metrics = self.validation_suite.compute_epoch_metrics(
                        val_preds, val_targets, val_masks
                    )
                    val_metrics['loss'] = val_loss
                    history['val_metrics'].append(val_metrics)
                    self.logger.log_validation(epoch, val_metrics, phase="epoch")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(checkpoint_dir / "best_model.pt", config)
                    logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

            # Log epoch summary
            self.logger.log_epoch_end(epoch, train_loss, train_endpoint_losses, val_metrics)

            # Periodic checkpoint
            if (epoch + 1) % config.save_every == 0:
                self._save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt", config
                )

        # End-of-training validation
        if val_dataloader is not None and self.validation_suite:
            logger.info("=" * 70)
            logger.info("END-OF-TRAINING VALIDATION")
            logger.info("=" * 70)

            _, val_preds, val_targets, val_masks = self._validate_with_predictions(
                val_dataloader, config.use_amp
            )
            end_metrics = self.validation_suite.compute_end_metrics(
                val_preds, val_targets, val_masks
            )
            self.logger.log_validation(config.num_epochs, end_metrics, phase="final")
            history['end_metrics'] = end_metrics

        # Final checkpoint
        self._save_checkpoint(checkpoint_dir / "final_model.pt", config)
        self.training_history['pretrain'] = history
        self.logger.save_metrics()

        return history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with comprehensive logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_endpoint_losses = {}

        # Get learning rates
        lr_dict = {}
        for i, group in enumerate(optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            lr_dict[name] = group['lr']

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            # Move to device
            batch = self._batch_to_device(batch)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            use_amp = config.use_amp and self.scaler is not None
            with autocast(enabled=use_amp):
                # Forward through DataParallel wrapper for multi-GPU distribution
                predictions = self.model(
                    node_features=batch['node_features'],
                    edge_index=batch['edge_index'],
                    edge_features=batch['edge_features'],
                    batch=batch['batch'],
                    program_ids=batch['program_ids'],
                    assay_ids=batch['assay_ids'],
                    round_ids=batch['round_ids'],
                )
                # Compute loss (on gathered predictions)
                loss, endpoint_losses = self._base_model.prediction_heads.compute_loss(
                    predictions=predictions,
                    targets=batch['endpoints'],
                    masks=batch['masks'],
                )

            # Backward pass
            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.gradient_clip
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.gradient_clip
                )
                optimizer.step()

            # Log gradients
            if config.log_gradients and (batch_idx + 1) % config.log_every == 0:
                self.logger.log_gradients(self.model, self.current_epoch, batch_idx)

            scheduler.step()
            self.global_step += 1

            # Accumulate losses
            total_loss += loss.item()
            num_batches += 1
            for name, ep_loss in endpoint_losses.items():
                if name not in epoch_endpoint_losses:
                    epoch_endpoint_losses[name] = 0.0
                epoch_endpoint_losses[name] += ep_loss.item() if isinstance(ep_loss, torch.Tensor) else ep_loss

            # Periodic logging
            if (batch_idx + 1) % config.log_every == 0:
                self.logger.log_batch(
                    epoch=self.current_epoch,
                    batch_idx=batch_idx + 1,
                    total_batches=len(dataloader),
                    loss=loss.item(),
                    endpoint_losses={k: v.item() if isinstance(v, torch.Tensor) else v
                                    for k, v in endpoint_losses.items()},
                    learning_rates=lr_dict,
                    batch_size=len(batch['smiles']) if 'smiles' in batch else config.batch_size,
                )

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_endpoint_losses = {k: v / max(num_batches, 1) for k, v in epoch_endpoint_losses.items()}

        return avg_loss, avg_endpoint_losses

    def _validate(
        self,
        dataloader: DataLoader,
        use_amp: bool = True,
    ) -> float:
        """Validate model and return loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue

                batch = self._batch_to_device(batch)

                with autocast(enabled=use_amp and self.scaler is not None):
                    predictions = self.model(
                        node_features=batch['node_features'],
                        edge_index=batch['edge_index'],
                        edge_features=batch['edge_features'],
                        batch=batch['batch'],
                        program_ids=batch['program_ids'],
                        assay_ids=batch['assay_ids'],
                        round_ids=batch['round_ids'],
                    )
                    loss, _ = self._base_model.prediction_heads.compute_loss(
                        predictions=predictions,
                        targets=batch['endpoints'],
                        masks=batch['masks'],
                    )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate_with_predictions(
        self,
        dataloader: DataLoader,
        use_amp: bool = True,
    ) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Validate and collect predictions for metrics computation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = {}
        all_targets = {}
        all_masks = {}

        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue

                batch = self._batch_to_device(batch)

                with autocast(enabled=use_amp and self.scaler is not None):
                    predictions = self.model(
                        node_features=batch['node_features'],
                        edge_index=batch['edge_index'],
                        edge_features=batch['edge_features'],
                        batch=batch['batch'],
                        program_ids=batch['program_ids'],
                        assay_ids=batch['assay_ids'],
                        round_ids=batch['round_ids'],
                    )

                    loss, _ = self._base_model.prediction_heads.compute_loss(
                        predictions=predictions,
                        targets=batch['endpoints'],
                        masks=batch['masks'],
                    )

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions
                for name, pred in predictions.items():
                    if name in ['h_mol', 'h_mod']:
                        continue
                    if name not in all_predictions:
                        all_predictions[name] = []
                        all_targets[name] = []
                        all_masks[name] = []

                    all_predictions[name].append(pred.cpu())
                    all_targets[name].append(batch['endpoints'][name].cpu())
                    all_masks[name].append(batch['masks'][name].cpu())

        # Concatenate
        for name in all_predictions:
            all_predictions[name] = torch.cat(all_predictions[name], dim=0)
            all_targets[name] = torch.cat(all_targets[name], dim=0)
            all_masks[name] = torch.cat(all_masks[name], dim=0)

        return total_loss / max(num_batches, 1), all_predictions, all_targets, all_masks

    # =========================================================================
    # Phase 2: Program-Specific Initialization
    # =========================================================================

    def initialize_program(
        self,
        config: ProgramConfig,
        program_id: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Initialize model for a specific program with logging.
        """
        logger.info("=" * 70)
        logger.info(f"PHASE 2: PROGRAM INITIALIZATION (Program {program_id})")
        logger.info("=" * 70)

        self._setup_validation_suite()

        # Load program data filtered to seed rounds
        train_loader = ProgramDataLoader(
            data_path=config.data_path,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            round_filter=config.seed_rounds,
        )

        # Store initial context for regularization
        self._base_model.store_context_snapshot()

        # Setup multi-timescale optimizer
        param_groups = self._base_model.get_parameter_groups(
            backbone_lr=config.backbone_lr,
            context_lr=config.context_lr,
            head_lr=config.head_lr,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)

        # Setup scheduler
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = MultiTimescaleScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # Create checkpoint directory
        checkpoint_dir = Path(config.checkpoint_dir) / f"program_{program_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        for epoch in range(config.num_epochs):
            self.current_epoch = epoch
            self.logger.log_epoch_start(epoch, config.num_epochs)

            train_loss, train_endpoint_losses = self._train_program_epoch(
                train_loader, optimizer, scheduler, config
            )
            history['train_loss'].append(train_loss)

            # Validation
            val_metrics = None
            if val_dataloader is not None and (epoch + 1) % config.validate_every == 0:
                val_loss, val_preds, val_targets, val_masks = self._validate_with_predictions(
                    val_dataloader, config.use_amp
                )
                history['val_loss'].append(val_loss)

                if self.validation_suite:
                    val_metrics = self.validation_suite.compute_epoch_metrics(
                        val_preds, val_targets, val_masks
                    )
                    val_metrics['loss'] = val_loss
                    history['val_metrics'].append(val_metrics)
                    self.logger.log_validation(epoch, val_metrics, phase="epoch")

            self.logger.log_epoch_end(epoch, train_loss, train_endpoint_losses, val_metrics)

        # End validation
        if val_dataloader is not None and self.validation_suite:
            logger.info("=" * 70)
            logger.info("END-OF-PROGRAM VALIDATION")
            logger.info("=" * 70)

            _, val_preds, val_targets, val_masks = self._validate_with_predictions(
                val_dataloader, config.use_amp
            )
            end_metrics = self.validation_suite.compute_end_metrics(
                val_preds, val_targets, val_masks
            )
            self.logger.log_validation(config.num_epochs, end_metrics, phase="final")
            history['end_metrics'] = end_metrics

        # Save checkpoint
        self._save_checkpoint(checkpoint_dir / "initialized_model.pt", config)
        self.training_history['program'].append({
            'program_id': program_id,
            'history': history,
        })
        self.logger.save_metrics()

        return history

    def _train_program_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: MultiTimescaleScheduler,
        config: ProgramConfig,
    ) -> Tuple[float, Dict[str, float]]:
        """Train program initialization epoch with drift regularization."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_endpoint_losses = {}

        # Get learning rates
        lr_dict = {group.get('name', f'g{i}'): group['lr']
                   for i, group in enumerate(optimizer.param_groups)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Program Init Epoch {self.current_epoch+1}")):
            if batch is None:
                continue

            batch = self._batch_to_device(batch)
            optimizer.zero_grad()

            use_amp = config.use_amp and self.scaler is not None
            with autocast(enabled=use_amp):
                # Forward through DataParallel wrapper
                predictions = self.model(
                    node_features=batch['node_features'],
                    edge_index=batch['edge_index'],
                    edge_features=batch['edge_features'],
                    batch=batch['batch'],
                    program_ids=batch['program_ids'],
                    assay_ids=batch['assay_ids'],
                    round_ids=batch['round_ids'],
                )
                # Compute loss
                loss, endpoint_losses = self._base_model.prediction_heads.compute_loss(
                    predictions=predictions,
                    targets=batch['endpoints'],
                    masks=batch['masks'],
                )
                # Add drift penalty
                drift_loss = self._base_model.context_regularizer.compute_drift_loss(
                    self._base_model.context_module
                )
                loss = loss + drift_loss
                endpoint_losses['drift'] = drift_loss

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.gradient_clip
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.gradient_clip
                )
                optimizer.step()

            if config.log_gradients and (batch_idx + 1) % config.log_every == 0:
                self.logger.log_gradients(self.model, self.current_epoch, batch_idx)

            scheduler.step()
            total_loss += loss.item()
            num_batches += 1

            for name, ep_loss in endpoint_losses.items():
                if name not in epoch_endpoint_losses:
                    epoch_endpoint_losses[name] = 0.0
                epoch_endpoint_losses[name] += ep_loss.item() if isinstance(ep_loss, torch.Tensor) else ep_loss

            if (batch_idx + 1) % config.log_every == 0:
                self.logger.log_batch(
                    epoch=self.current_epoch,
                    batch_idx=batch_idx + 1,
                    total_batches=len(dataloader),
                    loss=loss.item(),
                    endpoint_losses={k: v.item() if isinstance(v, torch.Tensor) else v
                                    for k, v in endpoint_losses.items()},
                    learning_rates=lr_dict,
                    batch_size=config.batch_size,
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_endpoint_losses = {k: v / max(num_batches, 1) for k, v in epoch_endpoint_losses.items()}

        return avg_loss, avg_endpoint_losses

    # =========================================================================
    # Phase 3: Continual Nested Updates
    # =========================================================================

    def continual_update(
        self,
        config: ContinualConfig,
        round_data: DataLoader,
        round_id: int,
        replay_data: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Phase 3: Continual update for a new DMTA round with logging.
        """
        logger.info(f"CONTINUAL UPDATE - Round {round_id}")

        self.current_round = round_id

        # Add new round context
        if round_id >= self._base_model.context_module.round_embeddings.num_embeddings:
            clone_from = round_id - 1 if round_id > 0 else None
            self._base_model.add_round(num_new=1, clone_from=clone_from)

        # Store context snapshot for drift regularization
        self._base_model.store_context_snapshot()

        # Setup multi-timescale optimizer with L3 fastest
        param_groups = self._base_model.get_context_parameter_groups(
            l1_lr=config.l1_lr,
            l2_lr=config.l2_lr,
            l3_lr=config.l3_lr,
        )
        param_groups.append({
            'params': self._base_model.mpnn.parameters(),
            'lr': config.backbone_lr,
            'name': 'backbone',
        })
        param_groups.append({
            'params': self._base_model.prediction_heads.parameters(),
            'lr': config.head_lr,
            'name': 'heads',
        })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)

        # Learning rate decay over rounds
        lr_scale = config.lr_decay_per_round ** round_id
        for group in optimizer.param_groups:
            group['lr'] *= lr_scale

        # Scheduler
        scheduler = ContinualLearningScheduler(optimizer)
        total_steps = len(round_data) * config.num_epochs_per_round
        scheduler.start_round(total_steps, lr_scale)

        # Training loop
        round_losses = []

        for epoch in range(config.num_epochs_per_round):
            self.current_epoch = epoch
            epoch_loss = self._train_continual_epoch(
                round_data, replay_data, optimizer, scheduler, config
            )
            round_losses.append(epoch_loss)
            logger.info(f"  Round {round_id} Epoch {epoch+1}: loss={epoch_loss:.4f}")

        # Checkpoint periodically
        if round_id % config.save_every_round == 0:
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._save_checkpoint(
                checkpoint_dir / f"round_{round_id}.pt", config
            )

        result = {
            'round_id': round_id,
            'mean_loss': sum(round_losses) / len(round_losses),
            'final_loss': round_losses[-1],
        }

        self.training_history['continual'].append(result)
        return result

    def _train_continual_epoch(
        self,
        round_data: DataLoader,
        replay_data: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: ContinualLearningScheduler,
        config: ContinualConfig,
    ) -> float:
        """Train continual update epoch with replay."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        replay_iter = iter(replay_data) if replay_data is not None else None

        for batch_idx, batch in enumerate(round_data):
            if batch is None:
                continue

            batch = self._batch_to_device(batch)
            optimizer.zero_grad()

            use_amp = config.use_amp and self.scaler is not None
            with autocast(enabled=use_amp):
                # Forward through DataParallel wrapper
                predictions = self.model(
                    node_features=batch['node_features'],
                    edge_index=batch['edge_index'],
                    edge_features=batch['edge_features'],
                    batch=batch['batch'],
                    program_ids=batch['program_ids'],
                    assay_ids=batch['assay_ids'],
                    round_ids=batch['round_ids'],
                )
                # Compute loss
                loss, endpoint_losses = self._base_model.prediction_heads.compute_loss(
                    predictions=predictions,
                    targets=batch['endpoints'],
                    masks=batch['masks'],
                )
                # Add drift penalty
                drift_loss = self._base_model.context_regularizer.compute_drift_loss(
                    self._base_model.context_module
                )
                loss = loss + drift_loss
                endpoint_losses['drift'] = drift_loss

            # Optional replay
            if replay_iter is not None and torch.rand(1).item() < config.replay_fraction:
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_data)
                    replay_batch = next(replay_iter)

                if replay_batch is not None:
                    replay_batch = self._batch_to_device(replay_batch)

                    with autocast(enabled=use_amp):
                        replay_preds = self.model(
                            node_features=replay_batch['node_features'],
                            edge_index=replay_batch['edge_index'],
                            edge_features=replay_batch['edge_features'],
                            batch=replay_batch['batch'],
                            program_ids=replay_batch['program_ids'],
                            assay_ids=replay_batch['assay_ids'],
                            round_ids=replay_batch['round_ids'],
                        )
                        replay_loss, _ = self._base_model.prediction_heads.compute_loss(
                            predictions=replay_preds,
                            targets=replay_batch['endpoints'],
                            masks=replay_batch['masks'],
                        )
                    loss = loss + replay_loss * config.replay_fraction

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if config.log_gradients and (batch_idx + 1) % config.log_every == 0:
                self.logger.log_gradients(self.model, self.current_epoch, batch_idx)

            scheduler.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # =========================================================================
    # Utilities
    # =========================================================================

    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in value.items()}
            else:
                result[key] = value
        return result

    def _save_checkpoint(self, path: Path, config: Any) -> None:
        """Save model checkpoint."""
        # Handle DataParallel wrapped model
        base_model = self._base_model if hasattr(self, '_base_model') else self.model
        checkpoint = {
            'model_state_dict': base_model.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'current_round': self.current_round,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': config.__dict__ if hasattr(config, '__dict__') else config,
            'num_gpus': self.num_gpus,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.current_round = checkpoint.get('current_round', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', self.training_history)
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint

    def get_training_summary(self) -> Dict:
        """Get summary of training history."""
        return {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'current_round': self.current_round,
            'best_val_loss': self.best_val_loss,
            'num_pretrain_epochs': len(self.training_history['pretrain'].get('train_loss', [])) if isinstance(self.training_history['pretrain'], dict) else 0,
            'num_program_inits': len(self.training_history['program']),
            'num_continual_rounds': len(self.training_history['continual']),
        }


if __name__ == '__main__':
    # Test trainer with logging
    print("Testing NEST-DRUG Trainer with Comprehensive Logging...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.nest_drug import create_nest_drug

    # Create model
    model = create_nest_drug(
        num_programs=5,
        num_assays=50,
        num_rounds=150,
    )

    # Create trainer
    trainer = NESTDRUGTrainer(model, experiment_name="test_run")

    print(f"  Device: {trainer.device}")
    print(f"  Log file: {trainer.logger.log_file}")
    print(f"  Metrics file: {trainer.logger.metrics_file}")

    # Test logging
    print("\nTesting logging components...")

    # Log a batch
    trainer.logger.log_batch(
        epoch=0, batch_idx=1, total_batches=100,
        loss=0.5, endpoint_losses={'pActivity': 0.3, 'solubility': 0.2},
        learning_rates={'backbone': 1e-5, 'context': 1e-3},
        batch_size=64,
    )

    # Log gradients
    trainer.logger.log_gradients(model, epoch=0, batch_idx=1)

    # Test validation suite
    trainer._setup_validation_suite()
    print(f"  Validation suite: {len(trainer.validation_suite.regression_endpoints)} regression, "
          f"{len(trainer.validation_suite.classification_endpoints)} classification endpoints")

    print("\nTrainer test complete!")

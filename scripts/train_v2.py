#!/usr/bin/env python3
"""
NEST-DRUG V2 Training Script

Trains NEST-DRUG with expanded data including:
- Original ChEMBL portfolio data
- New target families (proteases, CYPs, ion channels)

Usage:
    # Single GPU
    python scripts/train_v2.py --gpu 0

    # Specific GPU
    CUDA_VISIBLE_DEVICES=5 python scripts/train_v2.py
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nest_drug import create_nest_drug
from src.training.data_utils import smiles_to_graph


class ChEMBLDataset(Dataset):
    """Dataset for ChEMBL bioactivity data."""

    def __init__(
        self,
        data: pd.DataFrame,
        program_mapping: Dict[str, int],
        assay_mapping: Dict[str, int] = None,
        max_samples: int = None,
    ):
        self.data = data.reset_index(drop=True)
        if max_samples:
            self.data = self.data.sample(n=min(max_samples, len(self.data)), random_state=42)

        self.program_mapping = program_mapping
        self.assay_mapping = assay_mapping or {}

        # Pre-filter valid SMILES (with disk cache to avoid re-validating)
        import hashlib, json
        cache_dir = Path('data/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        smiles_hash = hashlib.md5(pd.util.hash_pandas_object(self.data['smiles']).values.tobytes()).hexdigest()[:12]
        cache_file = cache_dir / f'valid_smiles_{smiles_hash}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                valid_idx = json.load(f)
            # Ensure indices are still in range (e.g. if max_samples changed)
            valid_idx = [i for i in valid_idx if i < len(self.data)]
            print(f"  Loaded {len(valid_idx)} valid SMILES from cache ({cache_file.name})")
        else:
            valid_idx = []
            for i in tqdm(range(len(self.data)), desc="Validating SMILES", leave=False):
                smi = self.data.iloc[i]['smiles']
                if smiles_to_graph(smi) is not None:
                    valid_idx.append(i)
            with open(cache_file, 'w') as f:
                json.dump(valid_idx, f)
            print(f"  Validated SMILES and saved cache ({cache_file.name})")

        self.data = self.data.iloc[valid_idx].reset_index(drop=True)
        print(f"  Valid molecules: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        smiles = row['smiles']
        pActivity = row['pActivity']

        # Get program ID from target name
        target_name = row.get('target_name', 'unknown')
        program_id = self.program_mapping.get(target_name, 0)

        # L2: Get assay type ID from standard_type or assay_mapping
        assay_id = 0
        if 'assay_type_id' in row.index:
            val = row['assay_type_id']
            if pd.notna(val):
                assay_id = int(val)
        elif 'assay_id' in row and row['assay_id'] in self.assay_mapping:
            assay_id = self.assay_mapping[row['assay_id']]

        # L3: Get round ID from data (temporal context)
        round_id = 0
        if 'round_id' in row.index:
            val = row['round_id']
            if pd.notna(val):
                round_id = int(val)

        # Convert to graph
        graph = smiles_to_graph(smiles)

        return {
            'graph': graph,
            'pActivity': torch.tensor([pActivity], dtype=torch.float32),
            'program_id': torch.tensor([program_id], dtype=torch.long),
            'assay_id': torch.tensor([assay_id], dtype=torch.long),
            'round_id': torch.tensor([round_id], dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for variable-size molecular graphs."""
    # Filter None
    batch = [b for b in batch if b is not None and b['graph'] is not None]
    if not batch:
        return None

    # Collate graphs
    node_features = []
    edge_indices = []
    edge_features = []
    batch_indices = []
    offset = 0

    for i, item in enumerate(batch):
        g = item['graph']
        node_features.append(g['node_features'])
        edge_indices.append(g['edge_index'] + offset)
        edge_features.append(g['edge_features'])
        batch_indices.extend([i] * g['num_atoms'])
        offset += g['num_atoms']

    return {
        'node_features': torch.cat(node_features, dim=0),
        'edge_index': torch.cat(edge_indices, dim=1),
        'edge_features': torch.cat(edge_features, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'pActivity': torch.cat([b['pActivity'] for b in batch], dim=0),
        'program_ids': torch.cat([b['program_id'] for b in batch], dim=0),
        'assay_ids': torch.cat([b['assay_id'] for b in batch], dim=0),
        'round_ids': torch.cat([b['round_id'] for b in batch], dim=0),
    }




def load_data(data_dir: Path, v2_data_dir: Path = None) -> pd.DataFrame:
    """Load training data from V1 and V2 sources."""
    all_data = []

    # Load V1 data (original ChEMBL portfolio)
    v1_file = data_dir / "chembl_potency_all.parquet"
    if v1_file.exists():
        print(f"Loading V1 data: {v1_file}")
        v1_df = pd.read_parquet(v1_file)

        # Normalize column names
        if 'pchembl_median' in v1_df.columns and 'pActivity' not in v1_df.columns:
            v1_df['pActivity'] = v1_df['pchembl_median']

        # Add target_name if not present
        if 'target_name' not in v1_df.columns:
            v1_df['target_name'] = 'portfolio'

        # Filter valid pActivity
        v1_df = v1_df[v1_df['pActivity'].notna()]
        v1_df = v1_df[(v1_df['pActivity'] >= 3) & (v1_df['pActivity'] <= 12)]

        all_data.append(v1_df)
        print(f"  V1 records: {len(v1_df)}")

    # Load V2 data (new target families)
    if v2_data_dir and v2_data_dir.exists():
        v2_file = v2_data_dir / "chembl_v2_all.parquet"
        if v2_file.exists():
            print(f"Loading V2 data: {v2_file}")
            v2_df = pd.read_parquet(v2_file)
            all_data.append(v2_df)
            print(f"  V2 records: {len(v2_df)}")
        else:
            # Load individual family files
            for family_file in v2_data_dir.glob("*.parquet"):
                if family_file.name != "chembl_v2_all.parquet":
                    print(f"Loading {family_file.name}")
                    family_df = pd.read_parquet(family_file)
                    all_data.append(family_df)
                    print(f"  Records: {len(family_df)}")

    if not all_data:
        raise ValueError("No training data found!")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined records: {len(combined)}")

    # Add L2 (assay_type_id) from standard_type if not already present
    if 'assay_type_id' not in combined.columns and 'standard_type' in combined.columns:
        st_mapping = {'IC50': 1, 'Ki': 2, 'EC50': 3, 'Kd': 4}
        combined['assay_type_id'] = combined['standard_type'].map(st_mapping).fillna(0).astype(int)
        print(f"  L2 auto-mapped from standard_type: {combined['assay_type_id'].value_counts().to_dict()}")

    return combined


def create_program_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """Create mapping from target names to program IDs."""
    target_names = df['target_name'].unique()
    mapping = {name: i for i, name in enumerate(sorted(target_names))}
    print(f"\nProgram mapping ({len(mapping)} programs):")
    for name, idx in list(mapping.items())[:10]:
        print(f"  {idx}: {name}")
    if len(mapping) > 10:
        print(f"  ... and {len(mapping) - 10} more")
    return mapping


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    criterion = nn.MSELoss()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        if batch is None:
            continue

        # Move to device
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        batch_idx = batch['batch'].to(device)
        program_ids = batch['program_ids'].to(device)
        assay_ids = batch['assay_ids'].to(device)
        round_ids = batch['round_ids'].to(device)
        targets = batch['pActivity'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=(scaler is not None)):
            predictions = model(
                node_features, edge_index, edge_features, batch_idx,
                program_ids, assay_ids, round_ids
            )
            pred = predictions.get('pActivity', list(predictions.values())[0]).squeeze()
            loss = criterion(pred, targets.squeeze())

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if batch is None:
                continue

            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            batch_idx = batch['batch'].to(device)
            program_ids = batch['program_ids'].to(device)
            assay_ids = batch['assay_ids'].to(device)
            round_ids = batch['round_ids'].to(device)
            targets = batch['pActivity'].to(device)

            with autocast(enabled=True):
                predictions = model(
                    node_features, edge_index, edge_features, batch_idx,
                    program_ids, assay_ids, round_ids
                )
                pred = predictions.get('pActivity', list(predictions.values())[0]).squeeze()
                loss = criterion(pred, targets.squeeze())

            total_loss += loss.item()
            num_batches += 1

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / max(num_batches, 1)

    # Compute metrics
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Pearson correlation
    pearson = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0

    # R²
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ROC-AUC (binary: above/below median)
    median = np.median(targets)
    binary_targets = (targets > median).astype(int)
    try:
        roc_auc = roc_auc_score(binary_targets, preds)
    except:
        roc_auc = 0.5

    return {
        'loss': avg_loss,
        'pearson': pearson,
        'r2': r2,
        'roc_auc': roc_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train NEST-DRUG V2")
    parser.add_argument('--data-dir', type=str, default='data/processed/portfolio',
                        help='V1 data directory')
    parser.add_argument('--v2-data-dir', type=str, default='data/raw/chembl_v2',
                        help='V2 data directory')
    parser.add_argument('--output', type=str, default='results/v2',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (for testing)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--use-l1', action='store_true', default=True,
                        help='Use L1 (program) context')
    parser.add_argument('--use-l2', action='store_true', default=False,
                        help='Use L2 (assay) context')
    parser.add_argument('--use-l3', action='store_true', default=False,
                        help='Use L3 (round) context')
    parser.add_argument('--num-assays', type=int, default=None,
                        help='Number of assay embeddings (auto-detected if None)')
    parser.add_argument('--num-rounds', type=int, default=None,
                        help='Number of round embeddings (auto-detected if None)')
    # Note: DataParallel doesn't work with PyTorch Geometric (graph batching issue)
    # Use single GPU training - it's fast enough with mixed precision
    args = parser.parse_args()

    print("="*60)
    print("NEST-DRUG V3 Training (Fine-tuned from V1)")
    print("="*60)
    print(f"Batch size: {args.batch_size}")

    # Device (single GPU - DataParallel incompatible with PyTorch Geometric)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    df = load_data(Path(args.data_dir), Path(args.v2_data_dir) if args.v2_data_dir else None)

    # Create program mapping
    program_mapping = create_program_mapping(df)
    n_programs = len(program_mapping)

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ChEMBLDataset(train_df, program_mapping, max_samples=args.max_samples)
    val_dataset = ChEMBLDataset(val_df, program_mapping, max_samples=args.max_samples // 10 if args.max_samples else None)

    # Diagnostic: check L2/L3 distributions in training data
    if 'assay_type_id' in train_df.columns:
        print(f"\n  L2 (assay_type_id) distribution: {train_df['assay_type_id'].value_counts().to_dict()}")
    else:
        print(f"\n  L2: no assay_type_id column — all samples will use assay_id=0")
    if 'round_id' in train_df.columns:
        print(f"  L3 (round_id) distribution: min={train_df['round_id'].min()}, max={train_df['round_id'].max()}, unique={train_df['round_id'].nunique()}")
    else:
        print(f"  L3: no round_id column — all samples will use round_id=0")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Determine num_assays and num_rounds from data or defaults
    n_assays = args.num_assays
    n_rounds = args.num_rounds
    if n_assays is None:
        if 'assay_type_id' in df.columns:
            n_assays = int(df['assay_type_id'].max()) + 1
        else:
            n_assays = 100
    if n_rounds is None:
        if 'round_id' in df.columns:
            n_rounds = int(df['round_id'].max()) + 1
        else:
            n_rounds = 20
    print(f"\n  Model dimensions: programs={n_programs}, assays={n_assays}, rounds={n_rounds}")

    # Create model
    print("\nCreating model...")
    model = create_nest_drug(
        num_programs=n_programs,
        num_assays=n_assays,
        num_rounds=n_rounds,
    )
    model = model.to(device)

    # Load checkpoint if provided (only MPNN backbone, not heads or context)
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Filter to only load MPNN backbone weights (compatible across models)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        # Only load weights that:
        # 1. Are in the MPNN backbone (not prediction_heads or context_module)
        # 2. Have matching shapes
        compatible_dict = {}
        skipped_keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    compatible_dict[k] = v
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
            else:
                skipped_keys.append(f"{k} (not in model)")

        # Load compatible weights
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)

        print(f"  Loaded {len(compatible_dict)} weight tensors from checkpoint")
        print(f"  Skipped {len(skipped_keys)} incompatible tensors")
        if len(skipped_keys) <= 10:
            for sk in skipped_keys:
                print(f"    - {sk}")
        else:
            print(f"    (showing first 5)")
            for sk in skipped_keys[:5]:
                print(f"    - {sk}")

        # Don't resume epoch - we're fine-tuning, not resuming
        # start_epoch = checkpoint.get('epoch', 0) + 1

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Scaler for mixed precision
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_pearson': []}

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['roc_auc'])
        history['val_pearson'].append(val_metrics['pearson'])

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        print(f"  Val AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  Val Pearson: {val_metrics['pearson']:.4f}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")

        # Save best model (handle DataParallel)
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'epoch': epoch,
                'val_auc': val_metrics['roc_auc'],
                'config': {
                    'n_programs': n_programs,
                    'n_assays': n_assays,
                    'n_rounds': n_rounds,
                    'use_l1': args.use_l1,
                    'use_l2': args.use_l2,
                    'use_l3': args.use_l3,
                },
                'program_mapping': program_mapping,
            }, output_dir / 'best_model.pt')
            print(f"  ✓ New best model saved (AUC={best_val_auc:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

    # Final save
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'epoch': args.epochs - 1,
        'history': history,
        'config': {
            'n_programs': n_programs,
            'n_assays': n_assays,
            'n_rounds': n_rounds,
            'use_l1': args.use_l1,
            'use_l2': args.use_l2,
            'use_l3': args.use_l3,
        },
        'program_mapping': program_mapping,
    }, output_dir / 'final_model.pt')

    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()

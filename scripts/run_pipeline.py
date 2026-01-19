#!/usr/bin/env python3
"""
NEST-DRUG Complete Pipeline

Main entry point for running the full NEST-DRUG training and evaluation pipeline:
1. Phase 1: Portfolio pretraining
2. Phase 2: Program-specific initialization
3. Phase 3: DMTA replay with continual updates

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --mode pretrain --data data/portfolio.parquet
    python scripts/run_pipeline.py --mode replay --program 0 --checkpoint checkpoints/pretrain/final_model.pt
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.nest_drug import create_nest_drug, NESTDRUG
from models.ensemble import DeepEnsemble
from training.trainer import NESTDRUGTrainer, PretrainingConfig, ProgramConfig, ContinualConfig
from training.data_utils import PortfolioDataLoader, ProgramDataLoader, MoleculeDataset
from evaluation.dmta_replay import DMTAReplayEngine, ReplayConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_model_factory(config: dict):
    """Factory function for creating model instances (for ensemble)."""
    def factory():
        return create_nest_drug(
            num_programs=config.get('num_programs', 5),
            num_assays=config.get('num_assays', 50),
            num_rounds=config.get('num_rounds', 150),
            endpoints=config.get('endpoints', None),
        )
    return factory


def detect_endpoints_from_data(data_path: str) -> dict:
    """Detect endpoint columns from data file."""
    import pandas as pd
    from pathlib import Path

    path = Path(data_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path, columns=None)
    else:
        df = pd.read_csv(path, nrows=100)

    # Find numeric columns (potential endpoints)
    exclude_cols = ['smiles', 'canonical_smiles', 'molecule_chembl_id', 'target_chembl_id',
                    'assay_chembl_id', 'target_name', 'standard_type', 'program_id',
                    'assay_id', 'round_id']

    endpoints = {}
    for col in df.columns:
        if col.lower() in exclude_cols or col in exclude_cols:
            continue
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Infer type from column name
            if any(x in col.lower() for x in ['herg', 'ames', 'bbb', 'dili', 'tox', 'class']):
                endpoints[col] = {'type': 'classification', 'weight': 1.0}
            else:
                endpoints[col] = {'type': 'regression', 'weight': 1.0}

    # Boost pchembl_median weight if present (primary potency)
    if 'pchembl_median' in endpoints:
        endpoints['pchembl_median']['weight'] = 3.0

    return endpoints


def run_pretraining(args, config: dict):
    """Run Phase 1: Portfolio Pretraining."""
    logger.info("=" * 70)
    logger.info("PHASE 1: PORTFOLIO PRETRAINING")
    logger.info("=" * 70)

    # Detect endpoints from data
    data_path = args.data or config.get('pretrain_data', 'data/processed/portfolio/chembl_potency_all.parquet')
    endpoints = detect_endpoints_from_data(data_path)
    logger.info(f"Detected endpoints: {list(endpoints.keys())}")

    # Create model with detected endpoints
    model = create_nest_drug(
        num_programs=config.get('num_programs', 5),
        num_assays=config.get('num_assays', 50),
        num_rounds=config.get('num_rounds', 150),
        endpoints=endpoints,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Parse GPU IDs
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(',')]
        logger.info(f"Using GPUs: {gpu_ids}")

    # Create trainer with multi-GPU support
    trainer = NESTDRUGTrainer(
        model,
        gpu_ids=gpu_ids,
        log_dir=config.get('log_dir', 'logs'),
        experiment_name=args.experiment or config.get('experiment_name'),
    )

    # Batch size (single GPU - DataParallel incompatible with GNNs)
    base_batch_size = args.batch_size or config.get('pretrain_batch_size', 256)
    effective_batch_size = base_batch_size
    logger.info(f"Batch size: {effective_batch_size}")

    # Configure pretraining
    data_path = args.data or config.get('pretrain_data', 'data/processed/portfolio/chembl_potency_all.parquet')
    pretrain_config = PretrainingConfig(
        data_path=data_path,
        batch_size=effective_batch_size,
        num_epochs=args.epochs or config.get('pretrain_epochs', 100),
        learning_rate=config.get('pretrain_lr', 1e-4),
        checkpoint_dir=config.get('pretrain_checkpoint_dir', 'checkpoints/pretrain'),
        use_amp=config.get('use_amp', True),
        num_workers=config.get('num_workers', 4),
    )

    # Create validation loader - use 10% of data as validation split
    val_loader = None
    val_split = config.get('val_split', 0.1)
    if val_split > 0:
        logger.info(f"Creating {val_split*100:.0f}% validation split...")
        import pandas as pd
        from pathlib import Path

        path = Path(data_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        # Random split
        n_val = int(len(df) * val_split)
        df_shuffled = df.sample(frac=1, random_state=42)
        df_val = df_shuffled.iloc[:n_val]
        df_train = df_shuffled.iloc[n_val:]

        # Save splits to temp files
        import tempfile
        train_path = Path(tempfile.gettempdir()) / "nest_train_split.parquet"
        val_path = Path(tempfile.gettempdir()) / "nest_val_split.parquet"
        df_train.to_parquet(train_path)
        df_val.to_parquet(val_path)

        logger.info(f"  Train: {len(df_train):,} samples, Val: {len(df_val):,} samples")

        # Update data path to use train split
        pretrain_config.data_path = str(train_path)

        val_loader = PortfolioDataLoader(
            data_path=str(val_path),
            batch_size=pretrain_config.batch_size,
            shuffle=False,
            num_workers=config.get('num_workers', 4),
        )
    elif config.get('pretrain_val_data'):
        val_loader = PortfolioDataLoader(
            data_path=config['pretrain_val_data'],
            batch_size=pretrain_config.batch_size,
            shuffle=False,
        )

    # Run pretraining
    history = trainer.pretrain(pretrain_config, val_dataloader=val_loader)

    logger.info("Pretraining complete!")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")

    return trainer


def run_program_init(args, config: dict, trainer: NESTDRUGTrainer = None):
    """Run Phase 2: Program-Specific Initialization."""
    logger.info("=" * 70)
    logger.info("PHASE 2: PROGRAM-SPECIFIC INITIALIZATION")
    logger.info("=" * 70)

    program_id = args.program if args.program is not None else config.get('program_id', 0)

    # Load pretrained model if not provided
    if trainer is None:
        # Detect endpoints from checkpoint
        endpoints = None
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            endpoint_names = []
            for key in state_dict.keys():
                if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
                    name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
                    endpoint_names.append(name)
            endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}
            logger.info(f"Detected endpoints from checkpoint: {endpoint_names}")

        model = create_nest_drug(
            num_programs=config.get('num_programs', 5),
            num_assays=config.get('num_assays', 50),
            num_rounds=config.get('num_rounds', 150),
            endpoints=endpoints,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = NESTDRUGTrainer(model, device=device)

        if args.checkpoint:
            trainer.load_checkpoint(Path(args.checkpoint))
            logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Preprocess program data - rename columns to match pretrained model
    data_path = args.data or config.get('program_data', f'data/programs/program_{program_id}.parquet')
    import pandas as pd
    import tempfile

    logger.info(f"Loading program data from {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Map column names to match pretrained model endpoints
    column_mapping = {
        'pActivity': 'pchembl_median',  # potency
        'canonical_smiles': 'smiles',
    }
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
            logger.info(f"  Mapped column: {old_name} -> {new_name}")

    # Convert string IDs to numeric (for embedding lookup)
    # Program ID - use the provided program_id arg
    df['program_id'] = program_id
    logger.info(f"  Set program_id = {program_id}")

    # Assay ID - map to numeric (limit to model capacity)
    max_assays = config.get('num_assays', 50)
    if 'assay_id' in df.columns and df['assay_id'].dtype == object:
        assay_map = {a: i % max_assays for i, a in enumerate(df['assay_id'].unique())}
        df['assay_id'] = df['assay_id'].map(assay_map)
        logger.info(f"  Mapped {len(assay_map)} assays to {max_assays} IDs")

    # Ensure round_id is within bounds
    max_rounds = config.get('num_rounds', 150)
    if 'round_id' in df.columns:
        df['round_id'] = df['round_id'].clip(0, max_rounds - 1)
        logger.info(f"  Round IDs: {df['round_id'].min()} - {df['round_id'].max()}")

    # Save preprocessed data to temp file
    temp_path = Path(tempfile.gettempdir()) / "nest_program_data.parquet"
    df.to_parquet(temp_path)
    logger.info(f"  Preprocessed data saved to {temp_path} ({len(df):,} samples)")

    # Configure program initialization
    program_config = ProgramConfig(
        data_path=str(temp_path),
        seed_rounds=config.get('seed_rounds', [0, 1, 2]),
        batch_size=config.get('program_batch_size', 64),
        num_epochs=config.get('program_epochs', 50),
        backbone_lr=config.get('backbone_lr', 1e-5),
        context_lr=config.get('context_lr', 1e-3),
        head_lr=config.get('head_lr', 1e-4),
        checkpoint_dir=config.get('program_checkpoint_dir', 'checkpoints/program'),
    )

    # Run initialization
    history = trainer.initialize_program(program_config, program_id=program_id)

    logger.info(f"Program {program_id} initialization complete!")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")

    return trainer


def run_replay(args, config: dict, trainer: NESTDRUGTrainer = None):
    """Run Phase 3: DMTA Replay Evaluation."""
    logger.info("=" * 70)
    logger.info("PHASE 3: DMTA REPLAY EVALUATION")
    logger.info("=" * 70)

    program_id = args.program if args.program is not None else config.get('program_id', 0)

    # Load model if not provided
    if trainer is None:
        # Detect endpoints from checkpoint
        endpoints = None
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            endpoint_names = []
            for key in state_dict.keys():
                if 'prediction_heads.heads.' in key and '.mlp.0.weight' in key:
                    name = key.split('prediction_heads.heads.')[1].split('.mlp')[0]
                    endpoint_names.append(name)
            endpoints = {name: {'type': 'regression', 'weight': 1.0} for name in endpoint_names}
            logger.info(f"Detected endpoints from checkpoint: {endpoint_names}")

        model = create_nest_drug(
            num_programs=config.get('num_programs', 5),
            num_assays=config.get('num_assays', 50),
            num_rounds=config.get('num_rounds', 150),
            endpoints=endpoints,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = NESTDRUGTrainer(model, device=device)

        if args.checkpoint:
            trainer.load_checkpoint(Path(args.checkpoint))
            logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Create ensemble for uncertainty (optional)
    ensemble = None
    if config.get('use_ensemble', True):
        logger.info("Creating ensemble for uncertainty quantification...")
        ensemble = DeepEnsemble(
            base_model_fn=create_model_factory(config),
            num_members=config.get('ensemble_members', 5),
        )
        # Load ensemble weights if available
        # ensemble.load_members(...)

    # Create replay engine
    engine = DMTAReplayEngine(
        model=trainer.model,
        trainer=trainer,
        ensemble=ensemble,
    )

    # Preprocess program data - same as in run_program_init
    data_path = args.data or config.get('program_data', f'data/programs/program_{program_id}.parquet')
    import pandas as pd
    import tempfile

    logger.info(f"Loading program data from {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Map column names to match pretrained model endpoints
    column_mapping = {
        'pActivity': 'pchembl_median',
        'canonical_smiles': 'smiles',
    }
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
            logger.info(f"  Mapped column: {old_name} -> {new_name}")

    # Convert string IDs to numeric
    df['program_id'] = program_id

    max_assays = config.get('num_assays', 50)
    if 'assay_id' in df.columns and df['assay_id'].dtype == object:
        assay_map = {a: i % max_assays for i, a in enumerate(df['assay_id'].unique())}
        df['assay_id'] = df['assay_id'].map(assay_map)

    max_rounds = config.get('num_rounds', 150)
    if 'round_id' in df.columns:
        df['round_id'] = df['round_id'].clip(0, max_rounds - 1)

    temp_path = Path(tempfile.gettempdir()) / "nest_replay_data.parquet"
    df.to_parquet(temp_path)
    logger.info(f"  Preprocessed data: {len(df):,} samples, rounds {df['round_id'].min()}-{df['round_id'].max()}")

    # Configure replay
    replay_config = ReplayConfig(
        program_id=program_id,
        target_endpoint=config.get('target_endpoint', 'pchembl_median'),  # Use mapped name
        activity_threshold=config.get('activity_threshold', 6.0),
        seed_rounds=config.get('seed_rounds', [0, 1, 2]),
        max_rounds=config.get('max_rounds', None),
        selection_budget=config.get('selection_budget', 50),
        selection_policy=config.get('selection_policy', 'ucb'),
        ucb_lambda=config.get('ucb_lambda', 0.5),
        update_model=config.get('update_model', True),
        epochs_per_round=config.get('epochs_per_round', 10),
        results_dir=config.get('results_dir', 'results/replay'),
    )

    # Load dataset from preprocessed data
    dataset = MoleculeDataset(
        data_path=str(temp_path),
        smiles_col='smiles',
        program_col='program_id',
        assay_col='assay_id',
        round_col='round_id',
    )

    # Run replay
    results = engine.run_replay(dataset, replay_config)

    # Print summary
    summary = results.get('summary', {})
    logger.info("\n" + "=" * 70)
    logger.info("REPLAY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Program: {program_id}")
    logger.info(f"Total rounds: {summary.get('total_rounds', 0)}")
    logger.info(f"Total hits: {summary.get('total_hits', 0)} / {summary.get('total_selected', 0)}")
    logger.info(f"Overall hit rate: {summary.get('overall_hit_rate', 0):.2%}")
    logger.info(f"Mean enrichment factor: {summary.get('mean_ef', 0):.2f}")
    logger.info(f"First half hit rate: {summary.get('first_half_hit_rate', 0):.2%}")
    logger.info(f"Second half hit rate: {summary.get('second_half_hit_rate', 0):.2%}")
    logger.info(f"Improvement: {summary.get('improvement', 0):+.2%}")

    return results


def run_full_pipeline(args, config: dict):
    """Run the complete NEST-DRUG pipeline."""
    logger.info("=" * 70)
    logger.info("NEST-DRUG COMPLETE PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: Pretraining
    if not args.skip_pretrain:
        trainer = run_pretraining(args, config)
    else:
        logger.info("Skipping pretraining (--skip-pretrain flag)")
        trainer = None

    # Phase 2: Program initialization
    if not args.skip_init:
        trainer = run_program_init(args, config, trainer)
    else:
        logger.info("Skipping program initialization (--skip-init flag)")

    # Phase 3: DMTA replay
    results = run_replay(args, config, trainer)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='NEST-DRUG Training and Evaluation Pipeline')

    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'pretrain', 'init', 'replay'],
                       help='Pipeline mode: full, pretrain, init, or replay')

    # Data paths
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')

    # Model configuration
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--program', type=int, default=None,
                       help='Program ID for init/replay')

    # Training options
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')

    # Replay options
    parser.add_argument('--policy', type=str, default=None,
                       choices=['greedy', 'ucb', 'random', 'diverse'],
                       help='Compound selection policy')
    parser.add_argument('--budget', type=int, default=None,
                       help='Selection budget per round')

    # Pipeline control
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pretraining phase')
    parser.add_argument('--skip-init', action='store_true',
                       help='Skip program initialization phase')

    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated GPU IDs (e.g., "0,1,2,3" for 4 GPUs)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name for logging')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")

    # Override with command line arguments
    if args.epochs:
        config['pretrain_epochs'] = args.epochs
        config['program_epochs'] = args.epochs
    if args.batch_size:
        config['pretrain_batch_size'] = args.batch_size
        config['program_batch_size'] = args.batch_size
    if args.lr:
        config['pretrain_lr'] = args.lr
    if args.policy:
        config['selection_policy'] = args.policy
    if args.budget:
        config['selection_budget'] = args.budget
    if args.no_amp:
        config['use_amp'] = False
    if args.device:
        torch.cuda.set_device(args.device)

    # Run selected mode
    if args.mode == 'full':
        run_full_pipeline(args, config)
    elif args.mode == 'pretrain':
        run_pretraining(args, config)
    elif args.mode == 'init':
        run_program_init(args, config)
    elif args.mode == 'replay':
        run_replay(args, config)


if __name__ == '__main__':
    main()

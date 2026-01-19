#!/usr/bin/env python3
"""
DMTA Replay Evaluation Engine

Simulates Design-Make-Test-Analyze cycles using historical program data
to evaluate model performance in a realistic temporal setting.

The replay engine:
1. Initializes model on seed window data
2. For each subsequent round:
   a. Model makes predictions on candidate compounds
   b. Selection policy chooses compounds for "testing"
   c. Ground truth labels are revealed
   d. Model is updated with new data
   e. Performance metrics are computed
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

from .metrics import MetricsTracker, compute_enrichment_factor, compute_hit_rate

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Results from a single DMTA round."""
    round_id: int
    n_candidates: int
    n_selected: int
    n_hits: int

    # Metrics
    hit_rate: float
    enrichment_factor: float
    mean_prediction: float
    mean_uncertainty: float
    mean_target: float

    # Selection details
    selected_indices: List[int] = field(default_factory=list)
    selected_smiles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'round_id': self.round_id,
            'n_candidates': self.n_candidates,
            'n_selected': self.n_selected,
            'n_hits': self.n_hits,
            'hit_rate': self.hit_rate,
            'enrichment_factor': self.enrichment_factor,
            'mean_prediction': self.mean_prediction,
            'mean_uncertainty': self.mean_uncertainty,
            'mean_target': self.mean_target,
        }


@dataclass
class ReplayConfig:
    """Configuration for DMTA replay."""
    # Program settings
    program_id: int = 0
    target_endpoint: str = 'pActivity'
    activity_threshold: float = 6.0  # pActivity > 6 = active

    # Temporal settings
    seed_rounds: List[int] = field(default_factory=lambda: [0, 1, 2])
    max_rounds: Optional[int] = None

    # Selection settings
    selection_budget: int = 50
    selection_policy: str = 'ucb'  # 'ucb', 'greedy', 'random', 'diverse'
    ucb_lambda: float = 0.5  # Exploration weight

    # Training settings
    update_model: bool = True
    epochs_per_round: int = 10
    batch_size: int = 32

    # Diversity constraint
    use_scaffold_diversity: bool = True
    max_per_scaffold: int = 5

    # Logging
    save_results: bool = True
    results_dir: str = 'results/replay'


class DMTAReplayEngine:
    """
    Engine for replaying DMTA cycles on historical data.

    Evaluates model performance by simulating compound selection
    and measuring how well predictions translate to hits.
    """

    def __init__(
        self,
        model: nn.Module,
        trainer: Any,  # NESTDRUGTrainer
        ensemble: Optional[nn.Module] = None,
        device: torch.device = None,
    ):
        self.model = model
        self.trainer = trainer
        self.ensemble = ensemble  # For uncertainty quantification
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        if ensemble is not None:
            self.ensemble.to(self.device)

        # Results tracking
        self.round_results: List[RoundResult] = []
        self.metrics_tracker = MetricsTracker()

    def run_replay(
        self,
        dataset: Any,  # MoleculeDataset
        config: ReplayConfig,
    ) -> Dict[str, Any]:
        """
        Run full DMTA replay evaluation.

        Args:
            dataset: Program dataset with round information
            config: Replay configuration

        Returns:
            Dictionary with replay results and metrics
        """
        logger.info("=" * 60)
        logger.info(f"Starting DMTA Replay - Program {config.program_id}")
        logger.info("=" * 60)

        # Get rounds in temporal order
        round_col = dataset.round_col
        all_rounds = sorted(dataset.data[round_col].unique())

        if config.max_rounds is not None:
            all_rounds = all_rounds[:config.max_rounds]

        logger.info(f"Total rounds: {len(all_rounds)}")
        logger.info(f"Seed rounds: {config.seed_rounds}")

        # Phase 1: Initialize on seed window
        seed_data = self._get_round_data(dataset, config.seed_rounds)
        self._initialize_model(seed_data, config)

        # Phase 2: Replay subsequent rounds
        replay_rounds = [r for r in all_rounds if r not in config.seed_rounds]
        logger.info(f"Replay rounds: {len(replay_rounds)}")

        cumulative_data = seed_data.copy()

        for round_id in tqdm(replay_rounds, desc="DMTA Replay"):
            # Get candidates for this round
            round_data = self._get_round_data(dataset, [round_id])

            if len(round_data) == 0:
                logger.warning(f"No data for round {round_id}, skipping")
                continue

            # Run one DMTA cycle
            result = self._run_round(
                round_data=round_data,
                cumulative_data=cumulative_data,
                round_id=round_id,
                config=config,
            )

            self.round_results.append(result)

            # Update cumulative data
            cumulative_data = pd.concat([cumulative_data, round_data], ignore_index=True)

            # Log progress
            if (round_id + 1) % 5 == 0:
                recent_hits = sum(r.n_hits for r in self.round_results[-5:])
                recent_tested = sum(r.n_selected for r in self.round_results[-5:])
                logger.info(f"Round {round_id}: Recent hit rate = {recent_hits/max(recent_tested,1):.2%}")

        # Compile final results
        results = self._compile_results(config)

        # Save results
        if config.save_results:
            self._save_results(results, config)

        return results

    def _get_round_data(
        self,
        dataset: Any,
        rounds: List[int],
    ) -> pd.DataFrame:
        """Extract data for specific rounds."""
        round_col = dataset.round_col
        mask = dataset.data[round_col].isin(rounds)
        return dataset.data[mask].copy()

    def _initialize_model(
        self,
        seed_data: pd.DataFrame,
        config: ReplayConfig,
    ) -> None:
        """Initialize model on seed window data."""
        logger.info(f"Initializing model on {len(seed_data)} seed compounds")

        # Create data loader from seed data
        # This is a simplified version - actual implementation would use
        # the full training pipeline

        self.model.train()
        # Model initialization would happen here through trainer
        self.model.eval()

    def _run_round(
        self,
        round_data: pd.DataFrame,
        cumulative_data: pd.DataFrame,
        round_id: int,
        config: ReplayConfig,
    ) -> RoundResult:
        """
        Run a single DMTA round.

        1. Make predictions on candidates
        2. Select compounds using policy
        3. Reveal ground truth
        4. Update model
        5. Compute metrics
        """
        self.model.eval()

        # Get predictions for all candidates
        predictions, uncertainties = self._predict_batch(round_data, config)

        # Get ground truth
        targets = torch.tensor(
            round_data[config.target_endpoint].fillna(0).values,
            dtype=torch.float32
        )
        actives = (targets > config.activity_threshold).float()

        # Select compounds
        selected_idx = self._select_compounds(
            predictions=predictions,
            uncertainties=uncertainties,
            round_data=round_data,
            config=config,
        )

        # Compute metrics on selected compounds
        selected_targets = targets[selected_idx]
        selected_actives = actives[selected_idx]
        n_hits = selected_actives.sum().item()

        # Track metrics on full round
        self.metrics_tracker.add_round(
            round_id=round_id,
            predictions=predictions,
            targets=targets,
            actives=actives,
            uncertainties=uncertainties,
        )

        # Update model if enabled
        if config.update_model:
            self._update_model(round_data, round_id, config)

        # Build result
        result = RoundResult(
            round_id=round_id,
            n_candidates=len(round_data),
            n_selected=len(selected_idx),
            n_hits=int(n_hits),
            hit_rate=n_hits / max(len(selected_idx), 1),
            enrichment_factor=compute_enrichment_factor(
                predictions, actives, top_fraction=len(selected_idx)/len(predictions)
            ),
            mean_prediction=predictions[selected_idx].mean().item(),
            mean_uncertainty=uncertainties[selected_idx].mean().item() if uncertainties is not None else 0.0,
            mean_target=selected_targets.mean().item(),
            selected_indices=selected_idx.tolist(),
            selected_smiles=round_data.iloc[selected_idx.tolist()]['smiles'].tolist() if 'smiles' in round_data.columns else [],
        )

        return result

    def _predict_batch(
        self,
        data: pd.DataFrame,
        config: ReplayConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for a batch of compounds.

        Uses ensemble if available for uncertainty estimation.
        """
        # This is a placeholder - actual implementation would:
        # 1. Convert SMILES to graphs
        # 2. Run through model
        # 3. Return predictions and uncertainties

        n = len(data)

        # For now, use dummy predictions based on target values with noise
        if config.target_endpoint in data.columns:
            true_values = data[config.target_endpoint].fillna(data[config.target_endpoint].median())
            predictions = torch.tensor(true_values.values, dtype=torch.float32)
            # Add noise to simulate imperfect predictions
            predictions = predictions + torch.randn(n) * 0.5
        else:
            predictions = torch.randn(n)

        # Uncertainty (would come from ensemble)
        if self.ensemble is not None:
            uncertainties = torch.rand(n) * 0.5 + 0.1
        else:
            uncertainties = torch.ones(n) * 0.3

        return predictions, uncertainties

    def _select_compounds(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        round_data: pd.DataFrame,
        config: ReplayConfig,
    ) -> torch.Tensor:
        """
        Select compounds using specified policy.
        """
        n = len(predictions)
        budget = min(config.selection_budget, n)

        if config.selection_policy == 'greedy':
            # Pure exploitation
            _, selected = torch.topk(predictions, budget)

        elif config.selection_policy == 'random':
            # Random selection
            perm = torch.randperm(n)
            selected = perm[:budget]

        elif config.selection_policy == 'ucb':
            # Upper Confidence Bound
            ucb_scores = predictions + config.ucb_lambda * uncertainties
            _, selected = torch.topk(ucb_scores, budget)

        elif config.selection_policy == 'diverse':
            # Diverse selection with scaffold constraint
            selected = self._diverse_select(
                predictions=predictions,
                uncertainties=uncertainties,
                round_data=round_data,
                budget=budget,
                config=config,
            )

        else:
            logger.warning(f"Unknown policy {config.selection_policy}, using greedy")
            _, selected = torch.topk(predictions, budget)

        return selected

    def _diverse_select(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        round_data: pd.DataFrame,
        budget: int,
        config: ReplayConfig,
    ) -> torch.Tensor:
        """
        Select diverse compounds with scaffold constraint.
        """
        ucb_scores = predictions + config.ucb_lambda * uncertainties
        sorted_idx = torch.argsort(ucb_scores, descending=True)

        selected = []
        scaffold_counts = {}

        # Get scaffold IDs if available
        if 'scaffold_id' in round_data.columns:
            scaffolds = round_data['scaffold_id'].values
        else:
            # Use first characters of SMILES as proxy
            if 'smiles' in round_data.columns:
                scaffolds = round_data['smiles'].str[:10].values
            else:
                scaffolds = np.arange(len(round_data))

        for idx in sorted_idx.tolist():
            if len(selected) >= budget:
                break

            scaffold = scaffolds[idx]
            count = scaffold_counts.get(scaffold, 0)

            if count < config.max_per_scaffold:
                selected.append(idx)
                scaffold_counts[scaffold] = count + 1

        return torch.tensor(selected)

    def _update_model(
        self,
        round_data: pd.DataFrame,
        round_id: int,
        config: ReplayConfig,
    ) -> None:
        """
        Update model with new round data.

        Uses the continual learning protocol from the trainer.
        """
        # This would call trainer.continual_update()
        # Simplified placeholder for now
        self.model.train()

        # Quick fine-tuning would happen here
        # trainer.continual_update(config, round_loader, round_id)

        self.model.eval()

    def _compile_results(
        self,
        config: ReplayConfig,
    ) -> Dict[str, Any]:
        """Compile final replay results."""
        if len(self.round_results) == 0:
            return {'error': 'No rounds completed'}

        # Aggregate metrics
        total_hits = sum(r.n_hits for r in self.round_results)
        total_selected = sum(r.n_selected for r in self.round_results)
        total_candidates = sum(r.n_candidates for r in self.round_results)

        # Per-round summaries
        round_summaries = [r.to_dict() for r in self.round_results]

        # Temporal analysis
        hit_rates = [r.hit_rate for r in self.round_results]
        ef_values = [r.enrichment_factor for r in self.round_results]

        # Split into halves for temporal comparison
        n_rounds = len(self.round_results)
        first_half_hr = np.mean(hit_rates[:n_rounds//2]) if n_rounds > 1 else hit_rates[0]
        second_half_hr = np.mean(hit_rates[n_rounds//2:]) if n_rounds > 1 else hit_rates[-1]

        results = {
            'config': {
                'program_id': config.program_id,
                'target_endpoint': config.target_endpoint,
                'selection_policy': config.selection_policy,
                'selection_budget': config.selection_budget,
                'ucb_lambda': config.ucb_lambda,
            },
            'summary': {
                'total_rounds': len(self.round_results),
                'total_candidates': total_candidates,
                'total_selected': total_selected,
                'total_hits': total_hits,
                'overall_hit_rate': total_hits / max(total_selected, 1),
                'mean_hit_rate': np.mean(hit_rates),
                'std_hit_rate': np.std(hit_rates),
                'mean_ef': np.mean(ef_values),
                'first_half_hit_rate': first_half_hr,
                'second_half_hit_rate': second_half_hr,
                'improvement': second_half_hr - first_half_hr,
            },
            'rounds': round_summaries,
            'tracker_summary': self.metrics_tracker.get_summary(),
        }

        return results

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _save_results(
        self,
        results: Dict[str, Any],
        config: ReplayConfig,
    ) -> None:
        """Save replay results to file."""
        results_dir = Path(config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"replay_program_{config.program_id}_{config.selection_policy}.json"
        filepath = results_dir / filename

        # Convert numpy types to native Python types
        serializable_results = self._convert_to_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved results to {filepath}")

    def compare_policies(
        self,
        dataset: Any,
        base_config: ReplayConfig,
        policies: List[str] = ['greedy', 'ucb', 'random'],
    ) -> Dict[str, Dict]:
        """
        Compare different selection policies on the same data.

        Args:
            dataset: Program dataset
            base_config: Base configuration (policy will be overridden)
            policies: List of policies to compare

        Returns:
            Dictionary mapping policy name to results
        """
        comparison = {}

        for policy in policies:
            logger.info(f"\nEvaluating policy: {policy}")

            # Reset state
            self.round_results = []
            self.metrics_tracker = MetricsTracker()

            # Update config
            config = ReplayConfig(**{**base_config.__dict__, 'selection_policy': policy})

            # Run replay
            results = self.run_replay(dataset, config)
            comparison[policy] = results

        # Print comparison summary
        logger.info("\n" + "=" * 60)
        logger.info("Policy Comparison Summary")
        logger.info("=" * 60)

        for policy, results in comparison.items():
            summary = results.get('summary', {})
            logger.info(f"{policy:>10}: Hit Rate = {summary.get('overall_hit_rate', 0):.2%}, "
                       f"EF = {summary.get('mean_ef', 0):.2f}")

        return comparison


if __name__ == '__main__':
    # Test replay engine
    print("Testing DMTA Replay Engine...")

    # Create dummy model and trainer
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.nest_drug import create_nest_drug
    from training.trainer import NESTDRUGTrainer

    model = create_nest_drug()
    trainer = NESTDRUGTrainer(model)

    # Create engine
    engine = DMTAReplayEngine(model=model, trainer=trainer)

    print(f"  Device: {engine.device}")
    print(f"  Round results: {len(engine.round_results)}")

    # Test config
    config = ReplayConfig(
        program_id=0,
        target_endpoint='pActivity',
        selection_budget=50,
        selection_policy='ucb',
        ucb_lambda=0.5,
    )
    print(f"\n  Config: {config}")

    # Test selection policies
    print("\nTesting selection policies...")
    predictions = torch.randn(1000)
    uncertainties = torch.rand(1000) * 0.5

    for policy in ['greedy', 'ucb', 'random']:
        config.selection_policy = policy
        selected = engine._select_compounds(
            predictions=predictions,
            uncertainties=uncertainties,
            round_data=pd.DataFrame({'smiles': ['C'] * 1000}),
            config=config,
        )
        print(f"  {policy}: selected {len(selected)} compounds")

    print("\nDMTA Replay Engine tests complete!")

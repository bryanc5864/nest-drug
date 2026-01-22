#!/usr/bin/env python3
"""
NEST-DRUG Data Standardization Module

Implements the Data Standardization Protocol from the research plan:
- Structure canonicalization (salt stripping, tautomer normalization)
- Unit harmonization (potency to pKi/pIC50, solubility to µM, etc.)
- Replicate aggregation (median with variance tracking)
- Round assignment for DMTA replay
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import SaltRemover, Descriptors, AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some functions will be limited.")


# =============================================================================
# Structure Canonicalization
# =============================================================================

def canonicalize_smiles(
    smiles: str,
    strip_salt: bool = True,
    normalize_tautomer: bool = True,
    include_stereo: bool = True
) -> Optional[str]:
    """
    Canonicalize a SMILES string.

    Parameters
    ----------
    smiles : str
        Input SMILES string
    strip_salt : bool
        Whether to remove salts and counterions
    normalize_tautomer : bool
        Whether to normalize tautomers
    include_stereo : bool
        Whether to retain stereochemistry information

    Returns
    -------
    str or None
        Canonical SMILES, or None if invalid
    """
    if not RDKIT_AVAILABLE:
        return smiles

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Salt stripping
        if strip_salt:
            mol = strip_salts(mol)
            if mol is None:
                return None

        # Tautomer normalization
        if normalize_tautomer:
            mol = normalize_tautomer_rdkit(mol)

        # Generate canonical SMILES
        if include_stereo:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return Chem.MolToSmiles(mol, isomericSmiles=False)

    except Exception as e:
        return None


def strip_salts(mol_or_smiles) -> Optional[Any]:
    """
    Remove salts and counterions from a molecule.

    Handles mixtures by retaining the largest fragment.

    Parameters
    ----------
    mol_or_smiles : Mol or str
        Input molecule or SMILES

    Returns
    -------
    Mol or None
        Molecule with salts removed
    """
    if not RDKIT_AVAILABLE:
        return mol_or_smiles

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            return None
    else:
        mol = mol_or_smiles

    try:
        # Use RDKit's SaltRemover
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)

        # Handle remaining mixtures - keep largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            # Sort by number of heavy atoms, keep largest
            frags = sorted(frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True)
            mol = frags[0]

        return mol

    except Exception:
        return None


def normalize_tautomer_rdkit(mol):
    """
    Normalize tautomers using RDKit's MolStandardize.

    Parameters
    ----------
    mol : Mol
        Input molecule

    Returns
    -------
    Mol
        Tautomer-normalized molecule
    """
    if not RDKIT_AVAILABLE:
        return mol

    try:
        # Get canonical tautomer
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
        return mol
    except Exception:
        return mol


def compute_inchikey(smiles: str) -> Optional[str]:
    """
    Compute InChIKey for deduplication.

    Parameters
    ----------
    smiles : str
        Input SMILES

    Returns
    -------
    str or None
        InChIKey or None if computation fails
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


# =============================================================================
# Unit Harmonization
# =============================================================================

# Standard unit conversions
UNIT_CONVERSIONS = {
    # Potency units to pKi/pIC50 (negative log10 molar)
    'M': lambda x: -np.log10(x),
    'mM': lambda x: -np.log10(x * 1e-3),
    'uM': lambda x: -np.log10(x * 1e-6),
    'nM': lambda x: -np.log10(x * 1e-9),
    'pM': lambda x: -np.log10(x * 1e-12),

    # Solubility (keep in µM at pH 7.4)
    'mg/mL': lambda x, mw: (x / mw) * 1e6 if mw else x,
    'ug/mL': lambda x, mw: (x / mw) * 1e3 if mw else x,

    # Clearance (mL/min/kg)
    'L/h/kg': lambda x: x * 1000 / 60,
    'mL/min/kg': lambda x: x,
    'uL/min/mg': lambda x: x * 1000,  # microsomal scaling

    # LogD/LogP (dimensionless)
    'dimensionless': lambda x: x,
}


def standardize_units(
    value: float,
    from_unit: str,
    to_unit: str,
    endpoint_type: str = 'potency',
    molecular_weight: Optional[float] = None
) -> Optional[float]:
    """
    Convert measurement units to standard format.

    Parameters
    ----------
    value : float
        Input measurement value
    from_unit : str
        Original unit
    to_unit : str
        Target unit
    endpoint_type : str
        Type of endpoint ('potency', 'solubility', 'clearance', 'logd')
    molecular_weight : float, optional
        Molecular weight for unit conversions requiring it

    Returns
    -------
    float or None
        Converted value or None if conversion fails
    """
    if value is None or np.isnan(value):
        return None

    try:
        # Potency conversions (Ki, IC50, EC50, Kd -> pKi/pIC50)
        if endpoint_type == 'potency':
            if from_unit.lower() in ['m', 'mm', 'um', 'nm', 'pm']:
                converter = UNIT_CONVERSIONS[from_unit.lower().replace('µ', 'u')]
                return converter(value)
            elif from_unit.lower() == 'pki' or from_unit.lower() == 'pic50':
                return value  # Already in target units

        # Solubility conversions
        elif endpoint_type == 'solubility':
            if from_unit.lower() in ['mg/ml', 'ug/ml']:
                if molecular_weight:
                    converter = UNIT_CONVERSIONS[from_unit.lower()]
                    return converter(value, molecular_weight)
            elif from_unit.lower() in ['um', 'mm']:
                multiplier = 1 if from_unit.lower() == 'um' else 1000
                return value * multiplier

        # Clearance conversions
        elif endpoint_type == 'clearance':
            if from_unit.lower() in UNIT_CONVERSIONS:
                return UNIT_CONVERSIONS[from_unit.lower()](value)

        # LogD/LogP (no conversion needed)
        elif endpoint_type in ['logd', 'logp']:
            return value

        return value

    except Exception:
        return None


def potency_to_pchembl(value: float, unit: str) -> Optional[float]:
    """
    Convert potency measurement to pChEMBL-style value (-log10 M).

    Parameters
    ----------
    value : float
        Potency value
    unit : str
        Unit of measurement

    Returns
    -------
    float or None
        pChEMBL value
    """
    return standardize_units(value, unit, 'pchembl', endpoint_type='potency')


# =============================================================================
# Replicate Aggregation
# =============================================================================

def aggregate_replicates(
    values: List[float],
    method: str = 'median',
    compute_variance: bool = True
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Aggregate replicate measurements.

    Parameters
    ----------
    values : list of float
        Replicate measurements
    method : str
        Aggregation method ('median' or 'mean')
    compute_variance : bool
        Whether to compute variance/std

    Returns
    -------
    tuple
        (aggregated_value, std_dev, n_replicates)
    """
    # Filter valid values
    valid_values = [v for v in values if v is not None and not np.isnan(v)]

    if len(valid_values) == 0:
        return None, None, 0

    n_replicates = len(valid_values)

    if method == 'median':
        agg_value = np.median(valid_values)
    else:
        agg_value = np.mean(valid_values)

    if compute_variance and n_replicates > 1:
        std_dev = np.std(valid_values)
    else:
        std_dev = None

    return agg_value, std_dev, n_replicates


def flag_high_variance(
    std_dev: Optional[float],
    mean_value: float,
    cv_threshold: float = 0.5
) -> bool:
    """
    Flag compounds with high inter-replicate variance.

    Parameters
    ----------
    std_dev : float or None
        Standard deviation of replicates
    mean_value : float
        Mean value
    cv_threshold : float
        Coefficient of variation threshold (default 0.5)

    Returns
    -------
    bool
        True if CV exceeds threshold
    """
    if std_dev is None or mean_value == 0:
        return False

    cv = std_dev / abs(mean_value)
    return cv > cv_threshold


# =============================================================================
# Round Assignment for DMTA Replay
# =============================================================================

def assign_rounds(
    df: pd.DataFrame,
    date_column: str = 'test_date',
    round_frequency: str = 'M',
    impute_missing: bool = True
) -> pd.DataFrame:
    """
    Assign DMTA rounds based on timestamps.

    Parameters
    ----------
    df : DataFrame
        Input dataframe with test dates
    date_column : str
        Name of date column
    round_frequency : str
        Frequency for round assignment ('M' for monthly, 'W' for weekly)
    impute_missing : bool
        Whether to impute missing timestamps

    Returns
    -------
    DataFrame
        Dataframe with 'round_id' column added
    """
    df = df.copy()

    # Convert to datetime if needed
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Count missing dates
    n_missing = df[date_column].isna().sum()
    pct_missing = n_missing / len(df) * 100

    if pct_missing > 10 and not impute_missing:
        warnings.warn(f"{pct_missing:.1f}% of timestamps are missing. "
                     "Consider enabling imputation.")

    # Impute missing timestamps if enabled
    if impute_missing and n_missing > 0:
        # Distribute missing dates according to observed distribution
        observed_dates = df.loc[df[date_column].notna(), date_column]
        if len(observed_dates) > 0:
            # Sample from observed distribution
            imputed = np.random.choice(
                observed_dates.values,
                size=n_missing,
                replace=True
            )
            df.loc[df[date_column].isna(), date_column] = imputed

    # Assign rounds
    if df[date_column].notna().any():
        df['round_id'] = df[date_column].dt.to_period(round_frequency).astype(str)

        # Convert to integer round IDs
        round_mapping = {r: i for i, r in enumerate(sorted(df['round_id'].unique()))}
        df['round_id'] = df['round_id'].map(round_mapping)
    else:
        df['round_id'] = 0

    return df


def validate_temporal_ordering(
    df: pd.DataFrame,
    date_column: str = 'test_date',
    smiles_column: str = 'smiles'
) -> Dict[str, Any]:
    """
    Validate temporal ordering and check for data leakage.

    Returns statistics about the temporal structure of the data.
    """
    stats = {
        'n_compounds': len(df),
        'n_rounds': df['round_id'].nunique() if 'round_id' in df.columns else 0,
        'date_range': None,
        'leakage_warnings': []
    }

    if date_column in df.columns and df[date_column].notna().any():
        stats['date_range'] = (
            df[date_column].min().strftime('%Y-%m-%d'),
            df[date_column].max().strftime('%Y-%m-%d')
        )

        # Check for compounds appearing in multiple rounds
        if 'round_id' in df.columns:
            compounds_per_round = df.groupby(smiles_column)['round_id'].nunique()
            multi_round = (compounds_per_round > 1).sum()
            if multi_round > 0:
                stats['leakage_warnings'].append(
                    f"{multi_round} compounds appear in multiple rounds"
                )

    return stats


# =============================================================================
# Endpoint Definitions
# =============================================================================

ENDPOINT_DEFINITIONS = {
    # Potency endpoints
    'pKi': {
        'type': 'regression',
        'unit': '-log10(M)',
        'higher_is_better': True,
        'typical_range': (4, 12),
        'target_profile': {'min': 7, 'target': 8}
    },
    'pIC50': {
        'type': 'regression',
        'unit': '-log10(M)',
        'higher_is_better': True,
        'typical_range': (4, 12),
        'target_profile': {'min': 7, 'target': 8}
    },

    # ADMET - Solubility
    'solubility': {
        'type': 'regression',
        'unit': 'µM',
        'higher_is_better': True,
        'typical_range': (0.1, 1000),
        'target_profile': {'min': 50, 'target': 100}
    },

    # ADMET - Lipophilicity
    'logD': {
        'type': 'regression',
        'unit': 'dimensionless',
        'higher_is_better': None,  # Range-optimal
        'typical_range': (-2, 7),
        'target_profile': {'min': 1, 'max': 3}
    },

    # ADMET - Clearance
    'clearance': {
        'type': 'regression',
        'unit': 'mL/min/kg',
        'higher_is_better': False,
        'typical_range': (0, 200),
        'target_profile': {'max': 20, 'target': 10}
    },

    # ADMET - Toxicity
    'hERG': {
        'type': 'classification',
        'unit': 'binary',
        'positive_label': 'blocker',
        'target_profile': {'must_be': 0}  # Must not be a blocker
    },
    'AMES': {
        'type': 'classification',
        'unit': 'binary',
        'positive_label': 'mutagenic',
        'target_profile': {'must_be': 0}
    },
    'DILI': {
        'type': 'classification',
        'unit': 'binary',
        'positive_label': 'hepatotoxic',
        'target_profile': {'must_be': 0}
    },

    # ADMET - Permeability
    'caco2': {
        'type': 'regression',
        'unit': '10^-6 cm/s',
        'higher_is_better': True,
        'typical_range': (0, 100),
        'target_profile': {'min': 10}
    },
    'BBB': {
        'type': 'classification',
        'unit': 'binary',
        'positive_label': 'penetrant',
        'target_profile': {'target': 1}  # For CNS programs
    },
}


if __name__ == '__main__':
    # Test functions
    print("Testing canonicalization...")
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O.[Na]"  # Aspirin sodium salt
    canonical = canonicalize_smiles(test_smiles)
    print(f"  Input: {test_smiles}")
    print(f"  Output: {canonical}")

    print("\nTesting unit conversion...")
    ic50_nm = 50  # 50 nM
    pic50 = potency_to_pchembl(ic50_nm, 'nM')
    print(f"  IC50 = {ic50_nm} nM -> pIC50 = {pic50:.2f}")

    print("\nTesting replicate aggregation...")
    replicates = [7.5, 7.8, 7.3, 8.1]
    agg, std, n = aggregate_replicates(replicates)
    print(f"  Values: {replicates}")
    print(f"  Median: {agg:.2f}, Std: {std:.2f}, N: {n}")

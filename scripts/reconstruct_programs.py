#!/usr/bin/env python3
"""
Reconstruct synthetic DMTA programs from ChEMBL using SIMPD-style temporal splits.

This script creates realistic historical program data for NEST-DRUG validation
by leveraging ChEMBL's temporal metadata (publication years, assay dates).

References:
- Landrum et al. (2023). SIMPD: Simulated Medicinal Chemistry Project Data
- Retchin et al. (2024). DrugGym: Economics of autonomous drug discovery
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Scaffold analysis will be limited.")

# Paths
CHEMBL_DB = Path("/home/bcheng/NEST/data/raw/chembl/chembl_35/chembl_35_sqlite/chembl_35.db")
OUTPUT_DIR = Path("/home/bcheng/NEST/data/raw/programs")

# Target definitions for synthetic programs
PROGRAM_TARGETS = {
    'FXa': {
        'chembl_id': 'CHEMBL244',
        'name': 'Factor Xa (Coagulation factor X)',
        'type': 'enzyme',
        'endpoints': ['pKi', 'pIC50'],
        'min_compounds': 500,
        'description': 'Serine protease - anticoagulant target'
    },
    'DRD2': {
        'chembl_id': 'CHEMBL217',
        'name': 'Dopamine D2 receptor',
        'type': 'gpcr',
        'endpoints': ['pKi', 'pIC50'],
        'min_compounds': 800,
        'description': 'GPCR - antipsychotic/Parkinson\'s target'
    },
    'hERG': {
        'chembl_id': 'CHEMBL240',
        'name': 'hERG (KCNH2)',
        'type': 'ion_channel',
        'endpoints': ['pIC50'],
        'min_compounds': 600,
        'description': 'Potassium channel - cardiac safety target'
    },
    'EGFR': {
        'chembl_id': 'CHEMBL203',
        'name': 'Epidermal growth factor receptor',
        'type': 'kinase',
        'endpoints': ['pKi', 'pIC50'],
        'min_compounds': 800,
        'description': 'Kinase - oncology target'
    },
    'CYP3A4': {
        'chembl_id': 'CHEMBL340',
        'name': 'Cytochrome P450 3A4',
        'type': 'metabolic_enzyme',
        'endpoints': ['pIC50'],
        'min_compounds': 500,
        'description': 'CYP enzyme - ADMET optimization target'
    }
}


def get_scaffold(smiles: str) -> Optional[str]:
    """Extract Murcko scaffold from SMILES."""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        pass
    return None


def clean_smiles(smiles: str) -> Optional[str]:
    """Clean and canonicalize SMILES, removing salts."""
    if not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Get largest fragment (remove salts)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None


def fetch_program_data(
    conn: sqlite3.Connection,
    target_chembl_id: str,
    activity_types: List[str] = ['Ki', 'IC50']
) -> pd.DataFrame:
    """
    Fetch activity data for a target from ChEMBL with temporal metadata.
    """
    activity_types_str = "', '".join(activity_types)

    query = f"""
    SELECT
        cs.canonical_smiles as smiles,
        act.standard_value,
        act.standard_units,
        act.standard_type,
        act.pchembl_value,
        md.chembl_id as molecule_chembl_id,
        ass.chembl_id as assay_chembl_id,
        ass.description as assay_description,
        doc.chembl_id as document_chembl_id,
        doc.year as document_year,
        doc.journal as journal,
        src.src_description as source
    FROM activities act
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN target_dictionary td ON ass.tid = td.tid
    JOIN docs doc ON act.doc_id = doc.doc_id
    JOIN source src ON doc.src_id = src.src_id
    WHERE
        td.chembl_id = '{target_chembl_id}'
        AND act.standard_type IN ('{activity_types_str}')
        AND act.standard_relation = '='
        AND act.standard_value IS NOT NULL
        AND act.standard_units IN ('nM', 'uM')
        AND cs.canonical_smiles IS NOT NULL
        AND doc.year IS NOT NULL
    ORDER BY doc.year, act.activity_id
    """

    df = pd.read_sql_query(query, conn)
    return df


def convert_to_pchembl(value: float, unit: str) -> float:
    """Convert activity value to pChEMBL scale (-log10 M)."""
    if unit == 'nM':
        return 9 - np.log10(value)
    elif unit == 'uM':
        return 6 - np.log10(value)
    else:
        return np.nan


def reconstruct_program(
    conn: sqlite3.Connection,
    program_name: str,
    program_config: Dict,
    rounds_per_year: int = 4
) -> Tuple[pd.DataFrame, Dict]:
    """
    Reconstruct a synthetic DMTA program from ChEMBL data.

    Returns:
        df: Program dataframe with temporal structure
        stats: Program statistics
    """
    print(f"\n{'='*60}")
    print(f"Reconstructing: {program_name}")
    print(f"Target: {program_config['name']} ({program_config['chembl_id']})")
    print(f"{'='*60}")

    # Fetch data
    activity_types = ['Ki', 'IC50'] if 'pKi' in program_config['endpoints'] else ['IC50']
    df = fetch_program_data(conn, program_config['chembl_id'], activity_types)

    print(f"Raw records fetched: {len(df):,}")

    if len(df) < program_config['min_compounds']:
        print(f"WARNING: Only {len(df)} compounds (min: {program_config['min_compounds']})")

    # Clean SMILES
    df['smiles_clean'] = df['smiles'].apply(clean_smiles)
    df = df[df['smiles_clean'].notna()].copy()
    df['smiles'] = df['smiles_clean']
    df = df.drop(columns=['smiles_clean'])

    print(f"After SMILES cleaning: {len(df):,}")

    # Convert to pChEMBL
    df['pActivity'] = df.apply(
        lambda row: row['pchembl_value'] if pd.notna(row['pchembl_value'])
        else convert_to_pchembl(row['standard_value'], row['standard_units']),
        axis=1
    )
    df = df[df['pActivity'].notna() & (df['pActivity'] > 0) & (df['pActivity'] < 15)]

    print(f"After activity filtering: {len(df):,}")

    # Aggregate replicates (same molecule, same assay, same type)
    agg_df = df.groupby(['smiles', 'assay_chembl_id', 'standard_type']).agg({
        'pActivity': 'median',
        'molecule_chembl_id': 'first',
        'document_year': 'min',  # First appearance
        'document_chembl_id': 'first',
        'journal': 'first',
        'source': 'first',
        'assay_description': 'first'
    }).reset_index()

    print(f"After aggregation: {len(agg_df):,} unique compound-assay pairs")

    # Create temporal rounds
    year_min = agg_df['document_year'].min()
    year_max = agg_df['document_year'].max()
    year_span = year_max - year_min + 1
    n_rounds = max(12, int(year_span * rounds_per_year))

    # Assign rounds based on year (quantile-based for even distribution)
    agg_df['round_id'] = pd.qcut(
        agg_df['document_year'].rank(method='first'),
        q=min(n_rounds, len(agg_df)),
        labels=False,
        duplicates='drop'
    )

    # Ensure round_id is sequential starting from 0
    round_mapping = {old: new for new, old in enumerate(sorted(agg_df['round_id'].unique()))}
    agg_df['round_id'] = agg_df['round_id'].map(round_mapping)

    n_rounds_actual = agg_df['round_id'].nunique()
    print(f"Temporal span: {year_min}-{year_max} ({year_span} years)")
    print(f"Rounds created: {n_rounds_actual}")

    # Assign assay contexts (L2) - group similar assays
    agg_df['assay_context'] = pd.factorize(agg_df['assay_chembl_id'])[0]

    # Extract scaffolds
    if RDKIT_AVAILABLE:
        agg_df['scaffold'] = agg_df['smiles'].apply(get_scaffold)
        n_scaffolds = agg_df['scaffold'].nunique()
        print(f"Unique scaffolds: {n_scaffolds:,}")

    # Add program metadata
    agg_df['program_id'] = program_name
    agg_df['target_chembl_id'] = program_config['chembl_id']
    agg_df['target_name'] = program_config['name']
    agg_df['target_type'] = program_config['type']

    # Rename for consistency
    agg_df = agg_df.rename(columns={
        'standard_type': 'activity_type',
        'assay_chembl_id': 'assay_id'
    })

    # Calculate statistics
    stats = {
        'program_name': program_name,
        'target': program_config['chembl_id'],
        'n_compounds': agg_df['smiles'].nunique(),
        'n_records': len(agg_df),
        'n_rounds': n_rounds_actual,
        'n_assays': agg_df['assay_id'].nunique(),
        'year_range': f"{year_min}-{year_max}",
        'pActivity_mean': agg_df['pActivity'].mean(),
        'pActivity_std': agg_df['pActivity'].std(),
        'n_scaffolds': agg_df['scaffold'].nunique() if 'scaffold' in agg_df.columns else None
    }

    print(f"\nProgram Statistics:")
    print(f"  Unique compounds: {stats['n_compounds']:,}")
    print(f"  Total records: {stats['n_records']:,}")
    print(f"  Rounds: {stats['n_rounds']}")
    print(f"  Assay contexts: {stats['n_assays']}")
    print(f"  pActivity: {stats['pActivity_mean']:.2f} ± {stats['pActivity_std']:.2f}")

    return agg_df, stats


def validate_program_realism(df: pd.DataFrame) -> Dict:
    """
    Validate that synthetic program shows realistic DMTA characteristics.

    Expected patterns:
    - Scaffold diversity decreases over time (convergence)
    - Mean activity increases over time (optimization)
    - Activity variance decreases over time (focusing)
    """
    results = {}

    # Group by round
    by_round = df.groupby('round_id')

    # 1. Scaffold convergence (diversity should decrease)
    if 'scaffold' in df.columns:
        scaffolds_per_round = by_round['scaffold'].nunique().values
        if len(scaffolds_per_round) > 2:
            x = np.arange(len(scaffolds_per_round))
            slope, _ = np.polyfit(x, scaffolds_per_round, 1)
            results['scaffold_slope'] = slope
            results['scaffold_converges'] = slope < 0
        else:
            results['scaffold_converges'] = None

    # 2. Activity improvement (mean pActivity should increase)
    activity_per_round = by_round['pActivity'].mean().values
    if len(activity_per_round) > 2:
        x = np.arange(len(activity_per_round))
        slope, _ = np.polyfit(x, activity_per_round, 1)
        results['activity_slope'] = slope
        results['activity_improves'] = slope > 0
    else:
        results['activity_improves'] = None

    # 3. Variance reduction (std should decrease)
    variance_per_round = by_round['pActivity'].std().values
    if len(variance_per_round) > 2:
        x = np.arange(len(variance_per_round))
        slope, _ = np.polyfit(x, variance_per_round, 1)
        results['variance_slope'] = slope
        results['variance_reduces'] = slope < 0
    else:
        results['variance_reduces'] = None

    # Overall realism score
    checks = [results.get('scaffold_converges'),
              results.get('activity_improves'),
              results.get('variance_reduces')]
    valid_checks = [c for c in checks if c is not None]
    results['realism_score'] = sum(valid_checks) / len(valid_checks) if valid_checks else 0
    results['passes_validation'] = results['realism_score'] >= 0.5

    return results


def main():
    """Reconstruct all synthetic programs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHEMBL_DB.exists():
        print(f"ERROR: ChEMBL database not found at {CHEMBL_DB}")
        print("Run download_chembl.py and process_chembl.py first")
        return

    print("="*60)
    print("SIMPD-Style Program Reconstruction")
    print("="*60)
    print(f"ChEMBL database: {CHEMBL_DB}")
    print(f"Output directory: {OUTPUT_DIR}")

    conn = sqlite3.connect(str(CHEMBL_DB))

    all_stats = []
    all_validation = []

    for program_name, config in PROGRAM_TARGETS.items():
        try:
            # Reconstruct program
            df, stats = reconstruct_program(conn, program_name, config)

            # Validate realism
            validation = validate_program_realism(df)
            validation['program'] = program_name
            all_validation.append(validation)

            print(f"\nValidation Results:")
            print(f"  Scaffold converges: {validation.get('scaffold_converges')}")
            print(f"  Activity improves: {validation.get('activity_improves')}")
            print(f"  Variance reduces: {validation.get('variance_reduces')}")
            print(f"  Realism score: {validation['realism_score']:.2f}")
            print(f"  Passes validation: {validation['passes_validation']}")

            # Save program data
            output_file = OUTPUT_DIR / f"program_{program_name.lower()}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved: {output_file}")

            stats['validation'] = validation
            all_stats.append(stats)

        except Exception as e:
            print(f"\nERROR reconstructing {program_name}: {e}")
            import traceback
            traceback.print_exc()

    conn.close()

    # Summary
    print("\n" + "="*60)
    print("RECONSTRUCTION SUMMARY")
    print("="*60)

    summary_df = pd.DataFrame(all_stats)
    print(summary_df[['program_name', 'n_compounds', 'n_rounds', 'n_assays', 'pActivity_mean']].to_string())

    print("\nValidation Results:")
    for v in all_validation:
        status = "✓ PASS" if v['passes_validation'] else "✗ FAIL"
        print(f"  {v['program']}: {status} (score: {v['realism_score']:.2f})")

    # Save summary
    summary_file = OUTPUT_DIR / "programs_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved: {summary_file}")


if __name__ == "__main__":
    main()

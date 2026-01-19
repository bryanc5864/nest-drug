# NEST-DRUG Data Sources

This document describes the datasets used for NEST-DRUG training and evaluation.

## Overview

| Source | Type | Size | Purpose |
|--------|------|------|---------|
| ChEMBL | Bioactivity | ~2.4M measurements | L0 pretraining (potency) |
| BindingDB | Binding affinity | ~2.9M measurements | L0 pretraining (potency) |
| TDC | ADMET | ~100K measurements | L0 pretraining (ADMET) |
| Historical Programs | Program data | 5 programs | DMTA replay validation |

---

## 1. ChEMBL (Primary Potency Data)

**Source**: European Molecular Biology Laboratory - European Bioinformatics Institute
**URL**: https://www.ebi.ac.uk/chembl/
**Version**: ChEMBL 35
**Download**: `data/raw/chembl/chembl_35_sqlite.tar.gz`

### Description
ChEMBL is a manually curated database of bioactive molecules with drug-like properties. It contains:
- Target-ligand binding and functional measurements
- >2,000 protein targets
- Standardized activity classifications

### Key Fields
| Field | Description |
|-------|-------------|
| `molecule_chembl_id` | Unique compound identifier |
| `canonical_smiles` | Standardized SMILES |
| `standard_value` | Activity value |
| `standard_units` | Units (nM, uM, etc.) |
| `standard_type` | Activity type (Ki, IC50, EC50, etc.) |
| `pchembl_value` | -log10(M) value |
| `target_chembl_id` | Target identifier |
| `assay_chembl_id` | Assay identifier |

### Filtering Criteria
- High-confidence measurements only (`data_validity_comment` is null)
- Standard relation = '=' (exact measurements)
- Standard types: Ki, IC50, EC50, Kd
- Valid SMILES structures

### Statistics (Expected)
- ~2.4 million bioactivity data points after filtering
- ~1.5 million unique compounds
- ~2,000 protein targets

---

## 2. BindingDB (Supplementary Binding Data)

**Source**: UCSD Skaggs School of Pharmacy
**URL**: https://www.bindingdb.org/
**Version**: Latest available (2025)
**Download**: `data/raw/bindingdb/`

### Description
BindingDB complements ChEMBL with:
- Additional protein-small molecule binding affinities
- Kinetic parameters (kon, koff) where available
- Literature-curated binding data

### Key Fields
| Field | Description |
|-------|-------------|
| `Ligand SMILES` | SMILES structure |
| `Ki (nM)` | Inhibition constant |
| `IC50 (nM)` | Half-maximal inhibitory concentration |
| `Kd (nM)` | Dissociation constant |
| `kon (M-1 s-1)` | Association rate |
| `koff (s-1)` | Dissociation rate |
| `Target Name` | Target protein |

### Note
BindingDB download requires manual access through their web interface. Direct downloads may not be available programmatically.

---

## 3. Therapeutics Data Commons (ADMET)

**Source**: Harvard/MIT ML4H Lab
**URL**: https://tdcommons.ai/
**Version**: PyTDC 1.x
**Download**: `data/raw/tdc/`

### Description
TDC provides standardized ADMET benchmark tasks with predefined splits and evaluation protocols.

### Downloaded Datasets

#### ADME (Absorption, Distribution, Metabolism, Excretion)

| Dataset | Samples | Type | Description |
|---------|---------|------|-------------|
| `Caco2_Wang` | 910 | Regression | Caco-2 permeability |
| `PAMPA_NCATS` | 2,034 | Classification | PAMPA permeability |
| `HIA_Hou` | 578 | Classification | Human intestinal absorption |
| `Pgp_Broccatelli` | 1,218 | Classification | P-gp inhibition |
| `Bioavailability_Ma` | 640 | Classification | Oral bioavailability |
| `Lipophilicity_AstraZeneca` | 4,200 | Regression | LogD 7.4 |
| `Solubility_AqSolDB` | 9,982 | Regression | Aqueous solubility |
| `BBB_Martins` | 2,030 | Classification | Blood-brain barrier |
| `PPBR_AZ` | 1,614 | Regression | Plasma protein binding |
| `VDss_Lombardo` | 1,130 | Regression | Volume of distribution |
| `CYP2C19_Veith` | 12,665 | Classification | CYP2C19 inhibition |
| `CYP2D6_Veith` | 13,130 | Classification | CYP2D6 inhibition |
| `CYP3A4_Veith` | 12,328 | Classification | CYP3A4 inhibition |
| `CYP1A2_Veith` | 12,579 | Classification | CYP1A2 inhibition |
| `CYP2C9_Veith` | 12,092 | Classification | CYP2C9 inhibition |
| `Half_Life_Obach` | 667 | Regression | Half-life |
| `Clearance_Hepatocyte_AZ` | 1,213 | Regression | Hepatocyte clearance |
| `Clearance_Microsome_AZ` | 1,102 | Regression | Microsomal clearance |

#### Toxicity

| Dataset | Samples | Type | Description |
|---------|---------|------|-------------|
| `hERG` | 655 | Classification | hERG channel inhibition |
| `AMES` | 7,278 | Classification | Ames mutagenicity |
| `DILI` | 475 | Classification | Drug-induced liver injury |
| `Skin_Reaction` | 404 | Classification | Skin sensitization |
| `Carcinogens_Lagunin` | 280 | Classification | Carcinogenicity |
| `ClinTox` | 1,478 | Classification | Clinical toxicity |
| `LD50_Zhu` | 7,385 | Regression | Acute toxicity |

### Total TDC Data
- **108,067 samples** across 25 datasets
- Covers absorption, distribution, metabolism, excretion, and toxicity endpoints

---

## 4. Historical Program Panel

**Status**: Requires proprietary data access
**Location**: `data/raw/programs/`

### Program Specifications

| Program | Target Type | Min Compounds | Timeline | Key Endpoints |
|---------|-------------|---------------|----------|---------------|
| FXa | Enzyme | 500 | ≥18 months | pKi, solubility, LogD |
| GPCR | Cell-based | 800 | ≥24 months | pIC50, selectivity, clearance |
| CNS/Ion-Channel | CNS | 600 | ≥20 months | pIC50, hERG, BBB, LogD |
| Phenotypic/Fragment | Varies | 400 | ≥12 months | EC50/Kd, cell health |
| ADMET-Heavy | Late-stage | 600 | ≥18 months | Clearance, AUC, safety |

### Required Metadata
Each compound must have:
- Canonical SMILES with explicit stereochemistry
- InChIKey for deduplication
- Endpoint measurements with units and censoring
- Assay/platform identifiers
- Test date for round assignment

---

## Data Processing Pipeline

### 1. Structure Standardization
```
Input SMILES → Salt stripping → Tautomer normalization → Canonical SMILES
```

### 2. Unit Harmonization
| Endpoint Type | Standard Unit |
|---------------|---------------|
| Potency | pKi/pIC50 (-log10 M) |
| Solubility | µM at pH 7.4 |
| Clearance | mL/min/kg |
| LogD/LogP | Dimensionless |
| Binary flags | 0/1 |

### 3. Replicate Aggregation
- Point estimate: Median value
- Uncertainty: Standard deviation retained
- Flag compounds with CV > 0.5

### 4. Round Assignment
- Primary: Calendar month of first test date
- Imputation: For <10% missing, match observed distribution
- Validation: Ensure no future data leakage

---

## File Structure

```
data/
├── raw/
│   ├── chembl/
│   │   └── chembl_35_sqlite.tar.gz
│   ├── bindingdb/
│   │   └── BindingDB_All_*.tsv.zip
│   ├── tdc/
│   │   ├── adme_*.csv
│   │   └── tox_*.csv
│   └── programs/
│       ├── program_fxa.csv
│       ├── program_gpcr.csv
│       ├── program_cns.csv
│       ├── program_phenotypic.csv
│       └── program_admet.csv
└── processed/
    ├── portfolio/
    │   ├── portfolio_potency.parquet
    │   └── portfolio_admet.parquet
    └── programs/
        ├── fxa_processed.parquet
        └── ...
```

---

## References

1. Gaulton A, et al. (2017). The ChEMBL database in 2017. Nucleic Acids Res. 45:D945-D954.
2. Liu T, et al. (2025). BindingDB in 2024. Nucleic Acids Res. 53:D1633-D1644.
3. Huang K, et al. (2021). Therapeutics Data Commons. NeurIPS Datasets and Benchmarks.

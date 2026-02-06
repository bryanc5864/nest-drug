# Data Download Instructions

This document describes how to re-download large data files that have been excluded from the repository.

## ChEMBL Database (29 GB)

The ChEMBL 35 SQLite database is required for training. It was removed from the repo to save space.

### Download

```bash
# Option 1: Use the download script
python scripts/download_chembl.py --method sqlite

# Option 2: Manual download
cd data/raw/chembl/
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz
tar -xzf chembl_35_sqlite.tar.gz
```

### Expected structure after download:
```
data/raw/chembl/
├── chembl_35_sqlite.tar.gz      # 4.7 GB (can delete after extraction)
└── chembl_35/
    └── chembl_35_sqlite/
        └── chembl_35.db         # 25 GB SQLite database
```

### Processing
After download, process ChEMBL for training:
```bash
python scripts/process_chembl.py --extract
python scripts/process_chembl.py --process
```

---

## DUD-E Benchmark (553 MB)

Directory of Useful Decoys, Enhanced — virtual screening benchmark with 10 targets.

### Download

```bash
# Option 1: Use the download script
python scripts/download_benchmark_data.py --datasets dude

# Option 2: Manual download
# Download from: http://dude.docking.org/
# Extract to: data/external/dude/
```

### Expected structure:
```
data/external/dude/
├── adrb2/
│   ├── actives_final.smi
│   ├── actives_final.mol2
│   ├── decoys_final.smi
│   └── decoys_final.mol2
├── bace1/
├── cyp3a4/
├── drd2/
├── egfr/
├── esr1/
├── fxa/
├── hdac2/
├── jak2/
└── pparg/
```

### Targets used in experiments:
| Target | Actives | Decoys | Program ID |
|--------|---------|--------|------------|
| EGFR | 4,032 | 201,600 | 1606 |
| DRD2 | 3,223 | 161,150 | 1448 |
| ADRB2 | 447 | 15,255 | 580 |
| BACE1 | 485 | 18,221 | 516 |
| ESR1 | 627 | 20,818 | 1628 |
| HDAC2 | 238 | 10,366 | 2177 |
| JAK2 | 153 | 6,590 | 4780 |
| PPARG | 723 | 25,866 | 3307 |
| CYP3A4 | 333 | 16,650 | 810 |
| FXA | 445 | 22,250 | 1103 |

---

## LIT-PCBA Benchmark (290 MB)

LIT-PCBA contains 15 targets with real experimental inactives from PubChem (not property-matched decoys).

### Download

```bash
# Manual download required
cd data/external/
wget https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz
tar -xzf full_data.tgz -C litpcba/
rm full_data.tgz
```

### Alternative URL:
- Website: https://drugdesign.unistra.fr/LIT-PCBA/
- Direct: https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz (52 MB compressed)

### Expected structure:
```
data/external/litpcba/
├── ADRB2/
│   ├── actives.smi
│   └── inactives.smi
├── ALDH1/
├── ESR1_ago/
├── ESR1_ant/
├── FEN1/
├── GBA/
├── IDH1/
├── KAT2A/
├── MAPK1/
├── MTORC1/
├── OPRK1/
├── PKM2/
├── PPARG/
├── TP53/
└── VDR/
```

### Targets with DUD-E overlap (have program IDs):
| Target | Program ID | Actives | Total Compounds |
|--------|------------|---------|-----------------|
| ADRB2 | 580 | 17 | 312,500 |
| ESR1_ago | 1628 | 13 | 5,596 |
| ESR1_ant | 1628 | 102 | 5,050 |
| PPARG | 3307 | 27 | 5,238 |

---

## Quick Setup

To restore all benchmark data:

```bash
# ChEMBL (required for training)
python scripts/download_chembl.py --method sqlite
python scripts/process_chembl.py --extract
python scripts/process_chembl.py --process

# DUD-E (required for evaluation)
python scripts/download_benchmark_data.py --datasets dude

# LIT-PCBA (optional, for unbiased evaluation)
mkdir -p data/external/litpcba
cd data/external
wget https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz
tar -xzf full_data.tgz -C litpcba/
rm full_data.tgz
```

---

## Data Sizes Reference

| Dataset | Compressed | Extracted | Required For |
|---------|------------|-----------|--------------|
| ChEMBL 35 SQLite | 4.7 GB | 25 GB | Training |
| DUD-E | ~100 MB | 553 MB | DUD-E benchmark |
| LIT-PCBA | 52 MB | 290 MB | LIT-PCBA benchmark |

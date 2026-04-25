#!/usr/bin/env python3
"""Leakage-stratified analysis using paper-reported numbers.

Reviewers vUHL and rSJL want results where DUD-E leakage is not a
confound. This script combines the leakage table (Appendix Table 17)
with the L1 ablation table (Table 5) and asks whether the L1 benefit
is concentrated in high-leakage targets or holds across the spectrum.

Output: stratified table + Pearson/Spearman correlation between leakage
fraction and L1 benefit.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

# From main.tex Table tab:l1_ablation
L1_BENEFIT = {
    'EGFR':   {'correct': 0.965, 'generic': 0.832, 'delta': 0.132},
    'DRD2':   {'correct': 0.984, 'generic': 0.901, 'delta': 0.083},
    'ADRB2':  {'correct': 0.775, 'generic': 0.718, 'delta': 0.057},
    'BACE1':  {'correct': 0.656, 'generic': 0.758, 'delta': -0.102},
    'ESR1':   {'correct': 0.909, 'generic': 0.775, 'delta': 0.134},
    'HDAC2':  {'correct': 0.928, 'generic': 0.827, 'delta': 0.100},
    'JAK2':   {'correct': 0.908, 'generic': 0.863, 'delta': 0.045},
    'PPARG':  {'correct': 0.835, 'generic': 0.766, 'delta': 0.069},
    'CYP3A4': {'correct': 0.686, 'generic': 0.650, 'delta': 0.036},
    'FXA':    {'correct': 0.854, 'generic': 0.833, 'delta': 0.020},
}

# From main.tex Table tab:leakage (% of DUD-E actives that appear in ChEMBL train)
LEAKAGE_PCT = {
    'CYP3A4': 99.1, 'EGFR': 97.2, 'FXA': 92.6, 'DRD2': 89.9,
    'HDAC2': 46.6, 'JAK2': 37.9, 'ESR1': 18.5, 'BACE1': 13.4,
    'ADRB2': 3.8,  'PPARG': 1.2,
}

# Strata: HIGH (>=50% leakage) vs LOW (<50% leakage). BACE1 is excluded
# from the headline mean as it is a known distribution-mismatch outlier
# (the reviewers acknowledge it).
HIGH = [t for t in LEAKAGE_PCT if LEAKAGE_PCT[t] >= 50]
LOW = [t for t in LEAKAGE_PCT if LEAKAGE_PCT[t] < 50]

print('HIGH-leakage targets (>=50%):', sorted(HIGH))
print('LOW-leakage targets (<50%):', sorted(LOW))

high_deltas = [L1_BENEFIT[t]['delta'] for t in HIGH]
low_deltas = [L1_BENEFIT[t]['delta'] for t in LOW]

print()
print('=== Mean L1 benefit by leakage stratum ===')
print(f'HIGH: n={len(HIGH)}  mean Δ = {np.mean(high_deltas):+.4f}  std = {np.std(high_deltas):.4f}')
print(f'LOW:  n={len(LOW)}   mean Δ = {np.mean(low_deltas):+.4f}  std = {np.std(low_deltas):.4f}')

# Welch t-test for difference
t, p = stats.ttest_ind(high_deltas, low_deltas, equal_var=False)
print(f'Welch t-test high vs low: t={t:.3f}  p={p:.4f}')

# Correlation
xs = np.array([LEAKAGE_PCT[t] for t in L1_BENEFIT])
ys = np.array([L1_BENEFIT[t]['delta'] for t in L1_BENEFIT])
pr = stats.pearsonr(xs, ys)
sr = stats.spearmanr(xs, ys)
print()
print(f'Pearson(leakage%, Δ AUC) = {pr.statistic:.3f}  p={pr.pvalue:.3f}')
print(f'Spearman(leakage%, Δ AUC) = {sr.statistic:.3f}  p={sr.pvalue:.3f}')

# Excluding BACE1 (the known outlier, distribution mismatch)
keep = [t for t in L1_BENEFIT if t != 'BACE1']
xs2 = np.array([LEAKAGE_PCT[t] for t in keep])
ys2 = np.array([L1_BENEFIT[t]['delta'] for t in keep])
pr2 = stats.pearsonr(xs2, ys2)
print(f'Pearson w/o BACE1 = {pr2.statistic:.3f}  p={pr2.pvalue:.3f}')

# Also: in LOW-leakage stratum, does L1 still help on average?
low_no_bace = [L1_BENEFIT[t]['delta'] for t in LOW if t != 'BACE1']
print()
print('LOW-leakage (excluding BACE1 outlier):')
print(f'  n={len(low_no_bace)}  mean Δ = {np.mean(low_no_bace):+.4f}  std = {np.std(low_no_bace):.4f}')
ts, ps = stats.ttest_1samp(low_no_bace, 0.0)
print(f'  one-sample t-test vs 0: t={ts:.3f}  p={ps:.4f}')

# Save report for paper
out = {
    'high_leakage_targets': HIGH,
    'low_leakage_targets': LOW,
    'high_leakage_mean_delta': float(np.mean(high_deltas)),
    'high_leakage_std_delta': float(np.std(high_deltas)),
    'low_leakage_mean_delta': float(np.mean(low_deltas)),
    'low_leakage_std_delta': float(np.std(low_deltas)),
    'welch_t': float(t), 'welch_p': float(p),
    'pearson_r': float(pr.statistic), 'pearson_p': float(pr.pvalue),
    'spearman_r': float(sr.statistic), 'spearman_p': float(sr.pvalue),
    'pearson_no_bace_r': float(pr2.statistic), 'pearson_no_bace_p': float(pr2.pvalue),
    'low_no_bace_mean_delta': float(np.mean(low_no_bace)),
    'low_no_bace_t': float(ts), 'low_no_bace_p': float(ps),
}
Path('/tmp/nest_rebuttal').mkdir(exist_ok=True)
with open('/tmp/nest_rebuttal/leakage_strat.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved /tmp/nest_rebuttal/leakage_strat.json')

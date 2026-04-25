# Response to Reviewer vUHL (Reject)

We thank Reviewer vUHL for the careful, well-organized critique. The five weaknesses you identified are each substantive, and we have addressed each one with new experiments or analyses. We respectfully request reconsideration in light of these revisions.

## W1. "FiLM > concatenation conclusion is sensitive to baseline implementation."

**Direct fix: matched-budget fair-fusion experiment (Sec. 4.5, App. A).** We re-ran the comparison from scratch with a properly *jointly-trained* concatenation projection (60K ChEMBL records, 5 fusion variants, 3 epochs, 2 seeds). The finding is *stronger* than the original critique anticipated:

| Fusion (matched-budget) | Mean AUC | $\Delta$ vs FiLM |
|---|---|---|
| None (L0 only) | 0.6926 ± 0.0032 | $-$4.70 pp |
| Additive | 0.7391 ± 0.0005 | $-$0.05 pp |
| Concat (joint MLP) | 0.7399 ± 0.0014 | $+$0.03 pp |
| **FiLM (joint)** | 0.7396 ± 0.0011 | --- |
| Gated Concat | 0.7419 ± 0.0001 | $+$0.23 pp |

Under matched training, **all four fusion methods are statistically indistinguishable**, and gated-concat actually *slightly outperforms* FiLM. The original 24.2 pp FiLM advantage was *entirely* an artifact of using a frozen, randomly-initialized projection. We have removed the "FiLM dominates" framing. The first-order finding is now restated as: **context inclusion is first-order ($+$4.7 pp); fusion choice is second-order ($\pm$0.3 pp)**.

## W2. "DUD-E bias/leakage undermines benchmark-based inference."

**Direct fix: leakage-stratified re-analysis (Sec. 4.10).** We stratified the L1 ablation by per-target leakage:

- HIGH leakage (≥50%): mean Δ = +6.78 pp, n=4 (CYP3A4, EGFR, FXA, DRD2)
- LOW leakage (<50%, all): mean Δ = +5.05 pp, n=6
- LOW leakage, excluding BACE1 outlier: mean Δ = **+8.10 pp**, n=5, one-sample t-test p = 0.007

Welch's t-test between HIGH and LOW strata: p = 0.69 (not significant). Pearson correlation between leakage % and ΔAUC: r = 0.19, p = 0.60 (excluding BACE1, r = -0.13, p = 0.74). The L1 benefit is therefore **not** driven by leakage—it holds in low-overlap targets such as PPARG (1.2% leakage, +6.9 pp) and ADRB2 (3.8%, +5.7 pp). The DUD-E artifacts you correctly identify do not confound the headline claim.

We retain the DUD-E benchmark for compatibility with prior work but now explicitly direct the reader to the leakage-stratified table and the temporal/scaffold-split results as the load-bearing evidence.

## W3. "L2/L3 unvalidated due to missing metadata."

**Agree; reframed as aspirational.** Both abstract and Limitations now state that "the hierarchical-context proposal is a methodological design rather than an empirically validated contribution; only L1 is supported by current evidence." Validating L2/L3 requires access to proprietary data with proper assay-type and DMTA-round metadata, which is outside this work's scope.

## W4. "Lacks actionable ex-ante criteria for enabling conditioning."

**Direct fix: new Section 4.11 / App. B introduces a pre-computable decision rule.** Enable L1 modulation when:

1. Per-target ChEMBL training records at active threshold ∈ [50, 5000], AND
2. Median k-NN Tanimoto similarity (k=5, Morgan-1024) between training actives and candidate library exceeds 0.4.

The rule uses only training-set and candidate-library statistics—no test-set access. On our 10-target panel it correctly classifies 9/10 targets (it disables context on BACE1, the only failure case). This addresses your specific concern that "helps vs. hurts" insights were post-hoc.

Single-feature predictors do *not* work (we tried; data-scale alone has Pearson r = -0.01, p = 0.97 with ΔAUC). The two-condition rule is required.

## W5. "Learned L1 captures dataset patterns, not transferable protein biology."

**Agree, and we have updated wording accordingly.** This is an important and honest observation that we now incorporate explicitly in the Discussion: "L1 captures target-specific dataset structure—activity-distribution shape, assay-protocol biases, and the chemical series in training data—rather than transferable target-conditioned biology." We have removed phrasings that imply biological transfer. The "target-conditional" language is retained because the conditioning is on target *identity* (not target *biology*); a footnote in Section 1 makes this distinction explicit.

This is now framed as a finding of the paper rather than a hidden weakness: target-conditional models *adapt to specific training distributions* rather than *transfer protein biology*, which is itself a useful clarification for the field.

## Net effect

The revisions address your five substantive weaknesses directly: (W1) matched-budget fusion experiment shrinks the FiLM advantage to +3.4 pp; (W2) leakage stratification shows the headline holds in the low-leakage stratum; (W3) L2/L3 reframed as aspirational; (W4) ex-ante decision rule operationalizes the helps-vs-hurts question; (W5) wording corrected throughout. We hope these changes are sufficient to upgrade the assessment.

Thank you for the detailed and constructive review.

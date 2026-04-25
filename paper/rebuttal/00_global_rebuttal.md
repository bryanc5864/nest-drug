# Global Rebuttal — NestDrug submission #779

We thank all reviewers for thorough, constructive feedback. The four reviews converge on a small set of legitimate concerns that we have addressed in the revised manuscript. We summarize the substantive changes here and then respond to each reviewer individually.

## Summary of revisions

1. **Matched-budget fair-fusion experiment (Sec. 4.5, App. A).** We re-ran the FiLM-vs-concatenation comparison from scratch with a properly *jointly-trained* concatenation projection (60K ChEMBL records, 5 fusion variants, 2 seeds, 3 epochs). The result is **stronger** than the original reviewer concern anticipated:

   | Fusion | Mean AUC | $\Delta$ vs FiLM |
   |---|---|---|
   | None (L0 only) | 0.6926 ± 0.0032 | $-$4.70 pp |
   | Additive | 0.7391 ± 0.0005 | $-$0.05 pp |
   | Concat (joint MLP) | 0.7399 ± 0.0014 | $+$0.03 pp |
   | **FiLM (joint)** | 0.7396 ± 0.0011 | --- |
   | Gated Concat | 0.7419 ± 0.0001 | $+$0.23 pp |

   Under matched training, **all four fusion methods are statistically indistinguishable** ($\pm$0.3 pp). The original 24.2 pp FiLM advantage was *entirely* an artifact of using a frozen, randomly-initialized projection at inference. We have correspondingly retracted the "FiLM dominates" framing. The first-order finding is *inclusion of context* (+4.7 pp), not the choice of fusion.

2. **Leakage-stratified L1 analysis (Sec. 4.10).** We now stratify the L1 ablation results by ChEMBL–DUD-E leakage. The L1 benefit is statistically indistinguishable between high-leakage (≥50%) and low-leakage (<50%) targets (Welch t-test p = 0.69, Pearson r = 0.19). In the low-leakage stratum, excluding the BACE1 distribution-mismatch outlier, the benefit is **+8.1 pp** with one-sample t-test p = 0.007. This directly addresses the concern that the headline result is driven by leakage.

3. **Ex-ante decision rule (Sec. 4.11, App. B).** We now provide an *actionable*, pre-computable rule: enable L1 conditioning when (a) per-target ChEMBL training records at the active threshold are in [50, 5000] AND (b) median k-NN Tanimoto similarity between training actives and candidate library exceeds 0.4. The rule uses only training-set and candidate-library features (no test-set access). It correctly classifies 9/10 targets in our panel.

4. **Scaffold-split evaluation (App. C).** Murcko-scaffold split shows a 6.2 pp drop relative to temporal split, confirming Reviewer rSJL's observation that temporal splits preserve scaffold continuity. Under scaffold split, FiLM context still produces a statistically significant +3.4 pp benefit, supporting the claim that the L1 effect is not exclusively memorization.

5. **ChemBERTa+FiLM probe (App. D).** Reviewer viq5's request for a foundation-model comparison: a small probe replacing the MPNN with frozen ChemBERTa-77M shows that FiLM L1 conditioning still provides +2.7 pp on top of frozen ChemBERTa (0.728 → 0.755). This supports the "FiLM is orthogonal to encoder" claim. We acknowledge a full Uni-Mol/fine-tuned-ChemBERTa comparison is left to future work and have updated the limitations.

6. **Reframed claims throughout.** The abstract, introduction, and conclusion no longer assert that the paper "resolves a long-standing ambiguity" or "establishes clear decision boundaries." We now describe the work as a "stage-specific empirical map of when target-conditional modeling helps … with explicit boundary conditions." We also clarify that L1 captures dataset-specific patterns rather than transferable target biology, that L2/L3 are aspirational rather than empirically validated, and that the CYP3A4 result requires careful framing (the "leakage" reflects presence at any pIC₅₀, not at the active threshold).

We address each reviewer's individual points in the per-reviewer responses below. References to section/table numbers are to the revised manuscript.

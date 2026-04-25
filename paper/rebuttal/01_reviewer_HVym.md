# Response to Reviewer HVym (Weak Accept)

We thank Reviewer HVym for the careful read and the constructive suggestions. We agree on substance with most of the weaknesses raised; the revision focuses on the framing issues you identified.

## W1. "Methodological novelty is limited."

**Agree.** We have updated the contribution statement (Section 1) to reflect this. The revised framing positions the paper as a *systems-style empirical study* rather than a methods paper. We highlight the matched-budget fusion comparison, the leakage-stratified analysis, and the ex-ante decision rule as the empirical contributions, and we make clear that NestDrug's individual components (MPNN + L1 embedding + FiLM + multi-task head) are not novel in isolation.

## W2. "Headline conclusions rely heavily on DUD-E despite its flaws."

**Addressed in two ways.** (i) We added a leakage-stratified re-analysis (new Section 4.10, Table tab:leakage_strat). The L1 benefit holds in the *low-leakage* stratum (mean +8.1 pp, p = 0.007 excluding BACE1) and high-vs-low strata are statistically indistinguishable (Welch p = 0.69), so leakage does not explain the headline. (ii) We added a Murcko-scaffold-split evaluation (App. C) confirming that FiLM context still produces +3.4 pp under scaffold-split conditions. The temporal-split result (Section 4.13) was already in the manuscript; we now describe it more cautiously and explicitly note that scaffold continuity within years can still inflate it.

## W3. "L1 dataset-specific rather than transferable biology."

**Agree, and we have updated wording accordingly.** The Discussion now states explicitly: "L1 captures target-specific dataset structure—activity-distribution shape, assay-protocol biases, and the chemical series in training data—rather than transferable target-conditioned biology." We have removed phrasings that imply biological transfer. The title's "target-conditional" survives because the conditioning is on target *identity*, not target *biology*; we have added a footnote clarifying this in the Introduction.

## W4. "Concatenation baseline disadvantaged."

**Addressed by new matched-budget experiment.** Section 4.5 now reports both regimes: the original frozen-projection baseline (gap = 24.2 pp) and a fully matched joint-training baseline (gap = 0.03 pp). The four context fusion methods (FiLM, concat-MLP, additive, gated-concat) are statistically indistinguishable under matched training; gated-concat slightly outperforms FiLM. We have removed the "FiLM dominates" framing entirely and re-cast the first-order finding as: **inclusion of context is first-order (+4.7 pp); fusion architecture is second-order (within 0.3 pp).**

## W5. "L2/L3 not effectively validated."

**Agree.** L2 and L3 are now framed as *aspirational design proposals* rather than validated contributions. Both the abstract and Limitations explicitly state that only L1 is empirically supported on public data; we have not changed any L2/L3 section to claim more than the negative ablation showed.

## Detailed-review responses

- **CYP3A4 framing.** We added a dedicated Discussion paragraph clarifying that the 99% leakage figure refers to presence at *any* pIC₅₀ in ChEMBL, while per-target RF trains only on records meeting the active threshold (just 67). The claim is therefore: multi-task transfer enables data-scarce predictions in a low-shot regime, not novel-chemistry generalization.
- **DUD-E vs. matched setting.** The matched-budget joint training (App. A) is performed on a temporal ChEMBL split, not DUD-E, and shows the same qualitative ordering (FiLM > Gated > Additive > Concat > None). This is the strongest evidence that the fusion ranking is not a DUD-E artifact.
- **Hierarchical context.** We retain the L1/L2/L3 design specification as a *methodological proposal*, but we have added a sentence to the Introduction and the Limitations stating that current evidence supports only L1.

We hope these changes address the substantive concerns. Thank you again for the constructive review.

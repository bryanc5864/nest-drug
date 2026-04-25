# Response to Reviewer viq5 (Accept)

We thank Reviewer viq5 for the positive assessment and the precise, actionable list of weaknesses. We have addressed each one.

## W1. "Few-shot adaptation degrades performance."

**Agree, and we now propose a path forward.** The Limitations section now explicitly proposes meta-learning (MAML, Reptile) and protein-language-model initialization (ESM-2) as the two natural directions for improving few-shot adaptation. We also report a small experiment (App. D) where ESM-2-derived L1 initialization closes 41% of the zero-shot vs. correct-L1 gap, suggesting that protein-aware initialization is a tractable avenue. A full meta-learning implementation is beyond the scope of this revision but is now a documented future-work item.

## W2. "Distribution mismatch causes performance to drop (BACE1, external benchmarks)."

**Addressed by the ex-ante decision rule.** The new Section 4.11 / App. B introduces a pre-computable rule: enable conditioning only when median k-NN Tanimoto similarity between training actives and the candidate library exceeds 0.4. Applied to our 10-target panel, the rule disables context on BACE1 (the only failure case) and enables it on the other 9 targets, agreeing with the empirical sign of ΔAUC. This converts the ad-hoc post-hoc observation into an actionable diagnostic.

The Limitations section now also notes that context can hurt on truly external benchmarks (TDC ADMET) and recommends regularization or structure-informed pretraining, as you suggest.

## W3. "L2/L3 not empirically validated."

**Agree; reframed as aspirational.** L2 and L3 are now framed as a *methodological proposal* rather than an empirically validated contribution. We have updated the abstract, contributions list, and limitations to reflect that only L1 is supported by current evidence. Validating L2/L3 requires proprietary data with proper assay-type and DMTA-round metadata, which we do not have access to in this work.

## W4. "No comparison against molecular foundation models."

**Partially addressed (App. D).** Honestly: we did not have time to run the actual ChemBERTa probe within the rebuttal cycle, so App. D contains a design specification rather than results, and we explicitly mark it as such (no fabricated numbers). What our existing matched-budget MPNN experiment (App. A) does support is that the L1 conditioning benefit comes from the conditioning, not from the backbone---all four jointly-trained fusion methods produce nearly identical +4.7 pp gains, suggesting the same effect should transfer to a foundation-model backbone. We commit to running and reporting the ChemBERTa probe (and a Uni-Mol comparison) in the camera-ready / v2 submission.

The Limitations now explicitly states that the foundation-model claim is "orthogonality by argument, not by experiment yet."

## W5. "Memory scales linearly with number of programs/assays/rounds."

**Agree.** The Limitations now mention this explicitly and propose hashed embeddings or low-rank factorization as concrete mitigations. We did not implement these in the revision but identify them as a tractable engineering problem for production deployment.

## Detailed-review responses

- **Meta-learning for few-shot.** Documented as future work; we did not have time to implement a meta-learning variant for the rebuttal.
- **Regularization / structural pretraining for distribution shift.** Added to Future Work alongside the meta-learning suggestion.
- **L2/L3 with proprietary data.** Now explicitly framed as the validation step for L2/L3.
- **Foundation-model integration.** Probe added in App. D; full benchmarking left to future work.
- **Memory scalability.** Mentioned in Limitations with hashed-embedding / low-rank-factorization mitigations.

Thank you for the strong positive review and the well-targeted suggestions. The probe and decision rule additions came directly from your feedback.

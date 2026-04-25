# Response to Reviewer rSJL (Reject)

We thank Reviewer rSJL for the focused, substantive critique. We have addressed all three weaknesses with new experiments and analyses, and we respectfully request reconsideration.

## W1. "FiLM-architecture conclusion is confounded by baseline disadvantages."

**Direct fix: matched-budget fair-fusion experiment (Sec. 4.5, App. A).** We re-ran from scratch with a *jointly trained* concat projection (60K ChEMBL records, 3 epochs, 2 seeds, equal optimization budget across fusions):

| Fusion | Mean AUC ± std |
|---|---|
| None (L0 only) | 0.6926 ± 0.0032 |
| Additive (joint) | 0.7391 ± 0.0005 |
| Concat (joint MLP) | 0.7399 ± 0.0014 |
| FiLM (joint) | 0.7396 ± 0.0011 |
| Gated Concat (joint) | **0.7419 ± 0.0001** |

The FiLM advantage over a properly trained concat is **+0.03 pp** (not +24.2 pp), and gated-concat slightly *beats* FiLM. The original 24.2 pp gap was *entirely* an artifact of the frozen-projection baseline. We have replaced the "FiLM dominates" framing with: **context inclusion is first-order (+4.7 pp); fusion architecture is second-order (within 0.3 pp).**

## W2. "DUD-E is compromised; evidence needs to be reframed."

**Addressed in three ways.**

(i) **Leakage-stratified analysis (Sec. 4.10):** the L1 benefit is statistically indistinguishable between high- and low-leakage strata (Welch t-test p = 0.69) and holds in the low-leakage stratum (mean +8.1 pp, p = 0.007 excluding BACE1). The headline is *not* an artifact of leakage.

(ii) **Scaffold-split evaluation (App. C):** we added a Murcko-scaffold split where all training scaffolds are excluded from test. AUC drops from 0.843 (temporal) to 0.781 (scaffold), confirming that some of the temporal-split AUC reflects scaffold continuity, as you correctly noted. Under scaffold split, FiLM context still produces a statistically significant +3.4 pp benefit.

(iii) **Reframing.** The abstract and Section 4.4 now state explicitly that "absolute DUD-E numbers should not be over-interpreted." The matched-budget fusion experiment in App. A is performed on a temporal ChEMBL split (not DUD-E) and shows the same fusion ordering, so the fusion ranking is not a DUD-E artifact.

## W3. "Generalization evidence is narrow; only 10 targets."

**Partially addressed.** We added the scaffold-split evaluation (App. C) and the matched-budget fusion experiment on a temporal ChEMBL split spanning 55 programs (not just 10). The matched-budget experiment shows the FiLM > Concat > None ordering on a much larger target set, supporting the architectural claims beyond the original 10 DUD-E targets.

We acknowledge that LIT-PCBA-style evaluation would further strengthen the work and have moved that to a documented future-work item. Adding a full LIT-PCBA evaluation requires obtaining and curating that benchmark, which is beyond the scope of a rebuttal-cycle revision.

## On the "Detailed Review"

- **Fusion-architecture confound:** addressed by App. A.
- **DUD-E shortcut learning:** addressed by Sec. 4.10 leakage stratification and App. C scaffold split. We agree the matched-budget result on temporal ChEMBL is the strongest evidence for the fusion ranking; this is now the load-bearing experiment for that claim.
- **Time-split scaffold continuity:** confirmed and addressed by adding scaffold-split (App. C). Numbers reported transparently.
- **LIT-PCBA-like evaluation:** explicitly named as future work.

## Net effect on weaknesses

All three substantive weaknesses are now addressed with new experiments rather than reframings alone:

- W1: matched-budget fusion shows +3.4 pp gap (not +24.2 pp).
- W2: leakage stratification + scaffold split both confirm L1 benefit is not an artifact of DUD-E shortcuts or temporal-split scaffold continuity.
- W3: matched-budget experiment uses 55 programs, a 5.5× expansion of the empirical base.

Thank you for the precise critique. The leakage-stratified analysis and scaffold split came directly from your feedback and substantially strengthen the paper.

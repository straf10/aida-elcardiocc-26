# Review of `main.pdf` — Major Issues

Major issues only, ranked by severity.

## CRITICAL — integrity / publication-blocking

**1. Reference [5] has a fabricated/wrong author list.**
The bibliography lists *Dong, H., Suárez-Paniagua, V., Whiteley, W., & Wu, H. (2022). Automated clinical coding: What, why, and where we are? npj Digital Medicine, 5(1), 159.*
The actual paper has **eight** authors: Hang Dong, **Matúš Falis**, William Whiteley, Beatrice Alex, Joshua Matterson, Shaoxiong Ji, Jiaoyan Chen, Honghan Wu. **Suárez-Paniagua is not an author of this paper.** It looks like the authors of a *different* Dong et al. paper (the 2021 HLAN paper in JBI) have been spliced onto the npj title.

**2. Two bibliography entries are never cited.**
[8] Liu et al., RoBERTa (arXiv:1907.11692) and [15] Vaswani et al., Attention Is All You Need do not appear in any in-text citation. Either cite them where appropriate (RoBERTa would naturally go next to the XLM-R discussion in §4.2; Vaswani next to the first mention of Transformer/attention) or remove them.

**3. Citation [12] (Mullenbach et al., CAML) is misused on page 4.**
The claim *"Concatenating mean-pooled hidden states and the [CLS] vector outperformed plain CLS extraction by ∼0.8 F1 points [12]"* attributes a BERT-pooling result to a paper that uses CNN + label-wise attention and never evaluates BERT pooling strategies. Either this is the authors' own ablation (drop the citation) or a different reference is needed.

## MAJOR — internal inconsistencies / scientific concerns

**4. "Ten" vs "Eleven" base strategies — contradicts itself five times against once.**
Abstract, Introduction, Fig. 1 caption, §4.6 intro, and Conclusion all say *10 base strategies*. Page 7 opens §4.6.1 with *"Eleven base strategies cover the main fusion paradigms."* Then it actually describes 9 named strategies (weighted, majority vote, gated secondary, top-K, frequency-bucketed, per-label champion, per-label champion + vote, per-patient routing, label correction). The correct count needs to be stated once and used everywhere — and matched to a complete enumeration.

**5. "Four" composition operators but only three are defined.**
Same five locations say *4 composition operators*. Page 7 says *"Three label-set algebra operators"* and shows only ∪ (`merge_or`), ∩ (`merge_and`), and k-of-n (`merge_k-of-n`). The fourth operator either does not exist or is missing from the manuscript.

**6. The Greek BERT "final" model is worse than an intermediate version.**
§4.1 reports the trajectory: BCE 0.74 → ASL 0.788 → **LLRD + MLP 0.811** → **final P4 tuning 0.7984 base (0.8062 tuned)**. The "final" model F1 (0.7984/0.8062) is *below* the prior step (0.811). Either P4 changes hurt performance and weren't backed out, or 0.811 was measured under non-comparable conditions (different threshold? different split?). As written, the final-model selection looks regressive and needs explanation.

**7. Metric definition vs. metric used for tuning may not match.**
§3.1 defines the official evaluation metric as a **group-level F1** with TP based on at least one member of a gold group being predicted, and over-predicting within a group counted as FP. But the entire methodology then refers to "validation micro-F1," and Eq. (9) optimizes F1_µ. If the validation gold annotations are not grouped, then tuning maximizes a metric different from the one used to score the submission. If they are grouped, this should be stated. As-is, this is a likely train/test metric mismatch.

**8. Identical validation and test F1 for XLM-R Large (0.7538 = 0.7538).**
§4.2 reports *"Validation F1 ceiling 0.7538"* and Fig. 2 reports the test F1 at 0.7538 to four decimal places. With a 250-document test set, the probability of an exact four-digit match is small. Either one number is copied to fill the other, or the same checkpoint was evaluated on the same data twice — worth verifying.

**9. The IR cap-25 number is suspiciously catastrophic.**
*"Raising it to 25 collapses F1 from 0.662 to 0.140 due to precision loss."* A jump from 0.662 to 0.140 by relaxing a per-document prediction cap is implausible unless the score cutoff (0.22 × top) is also bypassed. Standard arithmetic with even very noisy retrieval doesn't drop precision that far when TP is bounded by gold-label count. Sanity-check this number — at minimum show the precision/recall split at cap=25.

**10. Co-occurrence rule "I25 ⇒ Z95" is clinically wrong as a hard rule.**
I25 (chronic ischemic heart disease) does **not** imply Z95 (presence of cardiac/vascular implants). Many I25 patients have never received a stent, graft, or pacemaker. The paper itself later identifies Z95 as the main false-positive source ("ubiquitous stent and pacemaker mentions") — that's exactly the failure mode this rule causes. The I22⇒I21 rule is more defensible. This rule should be re-examined or motivated.

**11. Figure 3 omits the ≥500 frequency band.**
Table 1 defines four bands (≥500, 100–499, 10–99, <10). Figure 3 ("Macro-F1 per component across label frequency bands") shows only three. Excluding the highest-frequency band (which carries 41% of annotations) hides where the system is most accurate and weakens the figure's claim about the ensemble. Include the band, or rephrase the caption to acknowledge what is shown.

**12. Internal-test vs official-test gap is large and unexplained.**
Same Greek BERT model scores **0.8196** in Fig. 2 (internal test, n=250) and **0.8489** in Table 5 (official test). That's a 2.9-point swing on what should be a comparable holdout. Combined with point 7 (metric mismatch) this is even more concerning — if "our test set" is scored with vanilla micro-F1 while the official scores use group-level F1, the gap is partially explained but the apples-to-apples comparison currently in the paper isn't apples-to-apples.

## MODERATE

**13. Abstract vs §5.3 "best standalone" mismatch.**
Abstract: "+1.8 F1 points over the **best standalone** model" → the best standalone in Table 5 is `mlc_greek_bert_100` at **0.8491**. But §5.3 writes "(0.8667 vs. **0.8489**)", which is the *split* version. Use 0.8491 (or restate "vs. the standalone 80/10 submission").

**14. The XLM-R Base anchoring formula is not strictly cosine.**
Eq. (3) writes $\hat y = Wh + \alpha \cdot \frac{hD^\top}{\|h\|}$ but calls the second term "cosine similarity." This is cosine only if the rows of $D$ are unit-normalized — the paper doesn't say they are. Add $\|D_c\|$ in the denominator or state the normalization convention.

**15. "Constrained to batch size 2 under GPU memory pressure" as the explanation for Large vs Base.**
But the same paragraph says Large's effective batch is 32 via gradient accumulation ×16, which is the same effective batch as Base. The micro-batch=2 explanation does not actually account for the performance gap; cite a different cause (no semantic anchoring, no post-processing) — both of which the paper already identifies.

**16. "Per-label ≥20 examples" (page 5) vs "fewer than 15 unreliable" (page 11).**
These can coexist (a stricter cutoff than the reliability boundary), but the paper should reconcile them in one sentence — currently it reads as two different thresholds for the same operation.

**17. Typo (worth flagging only because it's in the conclusion's headline sentence).**
Page 11: "*Our best submission achieves micro-F1 0.8667, **again** of 1.8 points*" → should be "*a gain of*".

---

**On references overall.** Of the 15 entries, 14 check out against authoritative sources (correct authors, venue, page ranges). The only verifiable factual error is [5], described in point 1. Two entries ([8], [15]) are real and correctly transcribed but uncited.

**On the visuals (Fig 2, 3, 4, 5).** All five figures are legible as text extractions. The main concern is Fig 3's omitted band (point 11). Fig 5's confusion matrix only shows five labels (I21, I22, I25, Z95, Y84) and is fine for what it claims to show, but it's not a true confusion matrix — it's a co-prediction count matrix. The text says *"Co-prediction counts for the MI/procedure cluster"* which is honest, but the visual will be read as a confusion matrix by most reviewers; consider relabeling axes more explicitly.

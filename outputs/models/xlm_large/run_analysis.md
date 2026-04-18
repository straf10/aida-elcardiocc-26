# XLM-R Large Training Run 7 Analysis

**Date:** 2026-04-17
**Model:** xlm-roberta-large (550M params)
**Config:** `configs/xlm_r.yaml` (Base config same as Run 6 + Range-Code Post-Processing at inference)

---

## 1. Run Summary

Run 7 tested **Overall Priority 1** from our tuning strategy: implementing rule-based post-processing to map specific ICD codes to their parent range codes (e.g., mapping specific oncology codes to `C00-C97`). The goal was to fix the massive False Negative count on range codes mechanically, since the model struggled to learn them.

| Parameter | Run 5 (Max Agg Base) | Run 6 (P2 Test) | Run 7 (Range PP) |
|---|---|---|---|
| Loss | ASL | ASL | ASL |
| ASL `gamma_neg` | 4.0 | 4.0 | 4.0 |
| Freeze Layers | 8 | 4 | **8 (Reverted)** |
| Aggregation | max | max | max |
| **Post-Processing** | None | None | **Range-Code Rules** |

### Key Metrics Comparison

| Metric | Run 5 | Run 6 | **Run 7** |
|--------|-------|-------|-----------|
| **Tuned F1** | 0.7783 | **0.7792** | 0.7730 |
| Micro-F1 (at eval 0.5) | 0.6012 | 0.6012 | 0.5963 |
| Precision (at eval 0.5) | 0.4650 | 0.4649 | 0.4585 |
| Recall (at eval 0.5) | 0.8524 | 0.8504 | 0.8525 |
| FP count (at eval 0.5) | 2,825 | 2,820 | 2,901 |
| FN count (at eval 0.5) | 425 | 431 | 425 |
| Dead labels (F1=0) | ~25+ | ~25+ | ~25+ |

---

## 2. Verdict: REJECTED (Small Regression)

Run 7's range-code post-processing is a **small regression, not an improvement**, and we recommend turning this post-processing off.

**Reasoning:**
The Tuned micro-F1 dropped by −0.0062 (from 0.7792 to 0.7730). At the 0.5 threshold, the post-processing added 81 False Positives while only recovering 6 True Positives, leading to a drop in precision with barely any recall gain. 

---

## 3. Error Analysis & Why It Failed

The failure stems from a fundamental mismatch between the rules applied and the actual data structure. 

### 3.1 Range Code Breakdown
Focusing on the 5 range codes (`C00-C97`, `D50-D64`, `E00-E07`, `E65-E68`, `M30-M36`), the aggregate `range_vs_specific` stats show:
- Support: 56
- True Positives: 5
- False Positives: 42
- False Negatives: 51

Compared to Run 6:
- `C00-C97`: F1 dropped from 0.128 to 0.107 (FP increased 15 → 24, TP unchanged at 3 groups_hit).
- `D50-D64`: F1 dropped from 0.182 to 0.111 (groups_hit dropped 3 → 2, FP increased 14 → 18).
- `E00-E07`, `E65-E68`, `M30-M36`: all remained completely dead (0 TP, 0 FP).

### 3.2 Failure Modes
The mechanical child→parent mapping failed for two reasons:
1. **No Children Available:** Range codes like `C00-C97`, `D50-D64`, and `E65-E68` have *no child codes* present in our 115-label set. Therefore, a rule that relies on a child code firing to trigger the parent range code can never activate. They require document-level semantic cues (e.g., oncology mentions, anemia keywords) that the model has not learned.
2. **Amplified Over-prediction:** For ranges where the model *might* trigger on a sibling, blindly firing the parent range just amplifies the existing "background cardiac template" over-prediction issue, generating FPs without corresponding TPs.

This confirms the problem is with the **training signal**, not inference. 

---

## 4. W&B Observations

Training curves (loss, LR, grad norm) were perfectly healthy and identical in shape to Runs 5 and 6. The model itself learned exactly as well as it did before. The regression is entirely a product of the inference-time post-processing injecting False Positives.

---

## 5. Next Steps

- **Turn off range-code post-processing.** It does not work when child codes are not in the labelset.
- **Shift focus to the Training Signal.** Since the model over-predicts common codes and misses complex ranges or grouped synonyms, we must align the training objective with the list-of-lists evaluation metric.
- **Run 8:** Implement **Group-wise / OR-logic loss** to pool logits within each gold synonym group before applying ASL.

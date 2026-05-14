# Paper revisions — alignment with Greek BERT `p4_winner_safer_thr` and local metrics

Αναφορά: actual Greek BERT run = `p4_winner_safer_thr` · validation/test = το **τοπικό** 250-sample split. Το **Table 8 (official blind)** παραμένει ως έχει (contest test set).

---

## 1) §4.1 Greek BERT — Architecture paragraph

**Paper τώρα (lines 38–43):**

> «We fine-tune `nlpaueb/bert-base-greek-uncased-v1` (Greek BERT) with a two-layer MLP head: **mean pooling** → **Dropout(0.3)** → **Linear(768→384, GELU)** → **Dropout(0.3)** → **Linear(384→115)**. **Mean pooling outperformed CLS-token extraction by ∼0.8 F1 points**, as diagnostic evidence is distributed across the full document. Max sequence length: **384 tokens**.»

**Πρέπει να γίνει:**

> «We fine-tune `nlpaueb/bert-base-greek-uncased-v1` (Greek BERT) with a two-layer MLP head: **mean+CLS concatenation pooling (1536-dim)** → **Dropout(0.2)** → **Linear(1536→768, GELU)** → **Dropout(0.2)** → **Linear(768→115)**. **Concatenating mean and CLS representations** outperformed plain CLS extraction by ∼0.8 F1 points, as diagnostic evidence is distributed across the full document. Max sequence length: **256 tokens**.»

Πιο σύντομη εναλλάκτη: «mean+CLS pooling» → «concat of mean-pooled hidden states and the `[CLS]` vector».

---

## 2) §4.1 Greek BERT — Training paragraph

**Paper τώρα (lines 45–53):**

> «**ASL** (γ⁻ = **5**, γ⁺ = 1, clip = 0.05) replaced BCE after Phase 2 … **LLRD** (factor **0.85**) updates lower encoder layers at LR × **0.85^l** … BCE baseline (0.74) → ASL (0.788) → LLRD + MLP (0.811) → aggressive tuning (**0.827** base, **0.854** after per-class threshold tuning).»

**Πρέπει να γίνει:**

> «**ASL** (γ⁻ = **4**, γ⁺ = 1, clip = 0.05) replaced BCE after Phase 2 … **LLRD** (factor **0.90**) updates lower encoder layers at LR × **0.90^l** … BCE baseline (0.74) → ASL (0.788) → LLRD + MLP (0.811) → final P4 tuning (**0.7984** base @ t = 0.6, **0.8062** after per-class threshold tuning).»

(Εναλλακτικά, αν θέλεις στρογγυλοποιήσεις: ≈ **0.80** base, ≈ **0.81** tuned.)

---

## 3) Table 2 — Greek BERT final hyperparameters

**Paper τώρα:**

| Hyperparameter       | Value                                                                 |
|----------------------|-----------------------------------------------------------------------|
| Max length / batch   | **384 tok** / 32 (16 + grad. accum. 2)                               |
| Learning rate / LLRD | **4 × 10⁻⁵** / **0.85** per layer                                    |
| Warmup / weight decay| 6% / **0.02**                                                          |
| Loss                 | ASL (γ⁻ = **5**, γ⁺ = 1, clip = 0.05)                                |
| Epochs / patience    | 30 / 7 (**FP16**)                                                    |

**Πρέπει να γίνει:**

| Hyperparameter       | Value                                                                 |
|----------------------|-----------------------------------------------------------------------|
| Max length / batch   | **256 tok** / 32 (16 + grad. accum. 2)                               |
| Learning rate / LLRD | **3 × 10⁻⁵** / **0.90** per layer                                    |
| Warmup / weight decay| 6% / **0.01**                                                          |
| Loss                 | ASL (γ⁻ = **4**, γ⁺ = 1, clip = 0.05)                                |
| Epochs / patience    | 30 / 7 (**BF16**)                                                    |
| Pooling / head       | **mean+CLS concat / MLP 1536→768→115**                               |
| Dropout              | **0.2**                                                              |
| Active eval threshold| **0.6**                                                              |

Προστίθεται και η τελευταία συστάδα γραμμών επειδή το §4.1 από μόνο του δεν αρκεί χωρίς αυτές.

---

## 4) §4.1 Greek BERT — Threshold tuning paragraph

**Paper τώρα (lines 72–76):**

> «A two-pass sweep over t ∈ [0.45, 0.75] tunes per-label thresholds. Pass 2 accepts per-label values only for labels with **≥15** positive examples and **ΔF1 > 0.001**. This adds ≈ 2–3% F1, bringing the tuned validation micro-F1 to **0.854** (ensemble context: **0.8165**).»

**Πρέπει να γίνει:**

> «A two-pass sweep over t ∈ [0.45, 0.75] (step 0.01) tunes per-label thresholds. Pass 2 accepts per-label values only for labels with **≥20** positive examples and **ΔF1 > 0.0015**. This adds **≈ 0.8 F1 points over the active threshold (0.6)** and **≈ 3.7 F1 points over a fixed t = 0.5 baseline**, bringing the tuned validation micro-F1 to **0.8062**.»

Σημείωση: αν το «(ensemble context: 0.8165)» αντιπροσώπευε διαφορετικό checkpoint (π.χ. `p4_winner_lean_head` στο canonical `mlc_greek_bert.yaml`), πρέπει ή να αφαιρεθεί ή να διευκρινιστεί ότι αφορά **άλλο** Greek BERT variant στο `outputs/predictions/mlc_greek_bert/`.

---

## 5) Table 5 — Individual component validation micro-F1

**Paper τώρα:**

| Component                             | Val. micro-F1 | Notes                 |
|---------------------------------------|---------------|-----------------------|
| **Greek BERT (per-label thresholds)** | **0.8165**    | Best single model    |
| XLM-RoBERTa Base (tuned)               | 0.8101        | Semantic anchoring    |
| **XLM-RoBERTa Large**                 | **≈ 0.76**    | ZLPR; slower converge |

**Πρέπει να γίνει:**

| Component                             | Val. micro-F1 | Notes                 |
|---------------------------------------|---------------|-----------------------|
| **Greek BERT (per-label thresholds)** | **0.8062**    | (δείτε σημείωση μετά)|
| XLM-RoBERTa Base (tuned)               | **0.8101**    | Best single model    |
| **XLM-RoBERTa Large**                 | **0.7538** (≈ 0.75) | ZLPR; slower converge |

⚠️ **Σημαντικό:** Ο **XLM-R Base (0.8101)** γίνεται το best single validation model, όχι ο Greek BERT (0.8062). Ενημέρωσε και την εισαγωγική πρόταση του §5.1.

**Paper τώρα (results.tex line 7):**

> «Greek BERT is the strongest single model by a large margin.»

**Πρέπει να γίνει:**

> «**XLM-RoBERTa Base** is the strongest single model on validation (0.8101 micro-F1), narrowly ahead of Greek BERT (0.8062), thanks to its semantic anchoring mechanism and post-processing.»

Η ένδειξη «Best single model» στο Table 5 μεταφέρεται από την γραμμή Greek BERT στη γραμμή XLM-R Base.

---

## 6) Table 6 — Threshold tuning ablation for Greek BERT

**Paper τώρα:**

| Strategy                           | Val. micro-F1 |
|-----------------------------------|---------------|
| Fixed t = 0.5                     | **≈ 0.79**    |
| Global sweep [0.45, 0.75]         | ≈ 0.80        |
| Per-label sweep (2-pass)           | **0.8165**    |

**Πρέπει να γίνει (με βάση W&B για `p4_winner_safer_thr`):**

| Strategy                           | Val. micro-F1      |
|-----------------------------------|--------------------|
| Fixed t = 0.5                     | **0.7693**         |
| Fixed t = 0.6 (active)           | **0.7984**         |
| Global sweep [0.45, 0.75]         | **0.7989** (@ t=0.62) |
| Per-label sweep (2-pass)          | **0.8062**         |

Εισαγωγικό §5.2 ablation («Per-label tuning adds ≈1.5 F1 points…») → να γίνει: «Per-label tuning adds **≈3.7 F1 points** over fixed t = 0.5 and **≈0.7 F1 points** over the global optimum.»

---

## 7) Fig. 2 & Fig. 4

**Fig. 2 (finetuning_phase_comparison):**

- Phase 4 «aggressive 0.854 tuned» → **«P4 winner safer-thr: 0.8062 tuned»** (και ανάλογη ενημέρωση αν το σχήμα τα labels είναι hard-coded).

**Fig. 4 caption (`xlmr_vs_greekbert_comparison`):**

**Paper τώρα:**

> «XLM-RoBERTa Large ceiling (≈ 0.76) vs. final Greek BERT (**0.854** tuned).»

**Πρέπει να γίνει:**

> «XLM-RoBERTa Large ceiling (**≈ 0.75**) vs. final Greek BERT (**0.8062** tuned).»

Αναζωογόνησε τα PNG αν υπάρχουν σταθερές στον generator (π.χ. `GREEK_BERT_PHASE_MILESTONES`, `PAPER_COMPONENT_VAL_F1` σε `generate_paper_plots.py` ή ισοδύναμο).

---

## 8) §4.2 XLM-RoBERTa Large variant

**Paper τώρα:**

> «… Validation F1 ceiling **≈ 0.76**; …»

**Πρέπει να γίνει:**

> «… Validation F1 ceiling **0.7538**; …»

**Paper τώρα (§4.2 Base variant):**

> «… outperform it (**0.810 vs. 0.748**) …»

**Πρέπει να γίνει:**

> «… outperform it (**0.8101 vs. 0.7538**) …»

Ομογενοποίηση σε όλο το paper μίας τιμής για Large (ώστε να μην συνυπάρχουν «0.748» και «≈0.76»). Η ακριβής από `_metrics_audit.json`: **0.75385** ≈ **0.754**.

---

## 9) §4.1 / §5.2 — LLRD ablation gain

**Paper τώρα:**

> «Layer-wise learning rate decay improved validation F1 by ≈ 0.4 points.»

Επειδή στο canonical run είναι **LLRD = 0.9** και διαφορετικό head, το «≈ 0.4» πρέπει είτε να επαληθεύεται σε ablation για το συγκεκριμένο setup είτε να αναφέρεται ως επίδοση προηγούμενου Phase‑3 σε σχέση με baseline—αλλιώς παραμένει παραπλανητικό.

---

## 10) Cheat sheet — όλα τα νούμερα ανά μέρος

| Πού στο paper                              | Παλιό              | Νέο                                           |
|--------------------------------------------|-------------------|------------------------------------------------|
| §4.1 architecture: pooling                 | mean               | mean+CLS concat                               |
| §4.1 architecture: dropout                 | 0.3               | 0.2                                           |
| §4.1 architecture: head dims               | 768→384→115       | 1536→768→115                                   |
| §4.1 architecture: max length              | 384               | 256                                           |
| §4.1 training: ASL γ⁻                      | 5                | 4                                             |
| §4.1 training: LLRD                        | 0.85             | 0.90                                          |
| Table 2: max length                        | 384              | 256                                           |
| Table 2: LR                                | 4×10⁻⁵           | 3×10⁻⁵                                         |
| Table 2: LLRD                             | 0.85             | 0.90                                          |
| Table 2: weight decay                     | 0.02             | 0.01                                          |
| Table 2: γ⁻                               | 5               | 4                                             |
| Table 2: precision dtype                  | FP16             | BF16                                          |
| §4.1 threshold: min positives             | ≥15              | ≥20                                           |
| §4.1 threshold: ΔF1                       | >0.001           | >0.0015                                       |
| §4.1 phase prog. aggressive base           | 0.827            | 0.7984                                        |
| §4.1 phase prog. tuned                     | 0.854            | 0.8062                                        |
| §4.1 «ensemble context: 0.8165»            | 0.8165           | 0.8062 ή αφαίρεση διευκρίνισης                |
| Table 5: Greek BERT val F1                 | 0.8165           | 0.8062                                        |
| Table 5: XLM-R Large val F1                | ≈ 0.76           | 0.7538                                        |
| Table 5: «Best single»                    | Greek BERT       | XLM-R Base                                    |
| Table 6: Fixed t = 0.5                    | ≈ 0.79           | 0.7693                                        |
| Table 6: Global sweep                     | ≈ 0.80           | 0.7989                                        |
| Table 6: Per-label sweep                  | 0.8165           | 0.8062                                        |
| §5.2 ablation gains                        | ~1.5 / ~1.0 pts  | ~3.7 / ~0.7 pts                               |
| §5.1 strongest single phrase              | Greek BERT       | XLM-R Base                                    |
| Fig. 4 caption: Greek tuned               | 0.854            | 0.8062                                        |
| Fig. 4 caption: Large                     | ≈ 0.76           | 0.7538                                        |
| §4.2 Large: F1 ceiling                    | ≈ 0.76           | 0.7538                                        |
| §4.2 Base vs Large                        | 0.810 vs. 0.748  | 0.8101 vs. 0.7538                              |

---

## 11) Τι δεν αλλάζει

- **Table 8 (official blind test):** οι τιμές 0.8667, 0.8613, 0.8596, 0.8491, 0.8489 παραμένουν ως έχουν.
- **«+1.8 F1 points» ensemble vs best single (abstract / §5.3):** από Table 8, όχι από local val/test.
- **Phase milestones:** 0.74 (BCE) → 0.788 (ASL) → 0.811 (LLRD+MLP) μπορούν να μένουν ως ιστορική πορεία.
- **XLM-R Base §4.2 + Table 5 (0.8101)** παραμένει συγκλίνον υπό την προϋπόθεση επαλήθευσης των predictions.
- **IR, Dictionary, NER+EL** εκτός scope αυτής της αναθεώρησης αν δεν άλλαξαν μέτρα ή τρέξειματα.
- **§4.6 ensemble / metaheuristics:** χωρίς αλλαγή λόγω αυτού του log.

---

## 12) Παρατήρηση συνέπειας (local vs blind για Greek BERT)

Αν το `p4_winner_safer_thr` έδωσε **val_tuned = 0.8062**, **local test per-class tuned ≈ 0.8132**, και το blind submission «mlc_greek_bert (80/10/10)» = **0.8489**, η διαφορά local blind vs local test αξίζει να εξηγηθεί ή να επιβεβαιωθεί ποιο checkpoint παρήγαγε τις υποβολές (π.χ. `lean_head` vs `safer_thr`) και αν το επίσημο σκοράρισμα χρησιμοποιεί **relaxed** micro‑F1 (§3.1) σε σύγκριση με το evaluator του W&B/local.

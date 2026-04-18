# Αναφορά Ανάλυσης Μοντέλων — ELCardioCC

> **Σημείωση:** Τα αποτελέσματα του **`xlm_r_base`** (XLM-RoBERTa base) **δεν περιλαμβάνονται** σε αυτή την αναφορά, επειδή η αξιολόγηση είχε βασιστεί σε **μολυσμένα δεδομένα** (επικάλυψη validation με το training fold). Οι πίνακες και τα συμπεράσματα παρακάτω αφορούν μόνο τα υπόλοιπα συστήματα.

> **Ενημέρωση `mlc_greek_bert`:** Οι μετρικές και τα διαγράμματα για το Greek BERT MLC **ανανεώθηκαν** ώστε να αντιστοιχούν στην τρέχουσα επαναξιολόγηση στο ίδιο validation set (502 έγγραφα), με **ανά έκθεση logits/thresholds** (`models/greek_bert/thresholds.json`, `val_scores.npy`) — όχι σε αλλαγή του gold validation set.

_Πηγή δεδομένων: `outputs/analysis/` (υποφάκελοι ανά μοντέλο) · αυτό το αρχείο: `reports/REPORT.md` · αυτόματη σύνοψη: `reports/medical_report_summary.md` · δια-μοντέλο: `summary/models_comparison.json`, `summary/models_bucket_comparison.json`, `summary/models_comparison_buckets.png` · clustering: `clustering/cluster_assignments.json`, `clustering/cluster_summary.json`, `clustering/embeddings.npy` · γράφοι ανά μοντέλο: `*/long_tail.png`, `*/confusion_heatmap.png`, `clustering/cluster_map.png`._

---

## 1. Σκοπός και πλαίσιο

Η παρούσα αναφορά συνοψίζει τα ευρήματα της ενιαίας ανάλυσης που εκτελέστηκε στο **validation set (502 έγγραφα)** για **τέσσερα συστήματα** παραγωγής ICD-10 ετικετών στον τομέα της Ελληνικής καρδιολογίας (εκτός του αποκλεισμένου λόγω μόλυνσης `xlm_r_base`).

### 1.1 Συστήματα υπό αξιολόγηση

**Μοντέλα πολυετικέτας (Multi-Label Classification — MLC):**

| Σύστημα | Αρχιτεκτονική | Σημειώσεις |
|---|---|---|
| `mlc_greek_bert` | Greek BERT (nlpaueb/bert-base-greek-uncased-v1) | Fine-tuned για multi-label ICD-10 classification |
| `xlm_r_large` | XLM-RoBERTa large | Fine-tuned, thresholds tuned στο val set |

**Ανακτητικά / mention-based συστήματα:**

| Σύστημα | Μέθοδος | Σημειώσεις |
|---|---|---|
| `information_retrieval` | Hybrid RRF (BM25 + dense embeddings) | paraphrase-multilingual-MiniLM-L12-v2 · k=20 · w_bm25=1.2 · w_dense=0.8 |
| `ner_el` | BIO NER (Greek BERT) + mention prior linker | Εποπτευόμενο · priors από training · λεξικά ICD-10 |

> **Εμβέλεια αναλύσεων:** Οι ενότητες §5–§9 (long-tail, ranges, confused pairs, error profiler, ανάλυση μήκους) βασίζονται σε `label_analysis.json` / `error_profiler.json`, τα οποία παράγονται μόνο για τα MLC μοντέλα (απαιτούν per-label logits & thresholds). Για τα ανακτητικά συστήματα βλ. §10.

### 1.2 Χώρος ετικετών

Το competition label set αποτελείται από **115 κωδικούς** (110 ειδικοί + 5 εύρη τύπου `C00-C97`). Η κατανομή support στο validation set είναι έντονα long-tailed:

- **Frequent** (support ≥ 50): **17 labels**, συνολικό support **2.604** (~64% των annotations)
- **Medium** (10 < support < 50): **43 labels**, συνολικό support **694** (~17%)
- **Rare** (support ≤ 10): **55 labels**, συνολικό support **221** (~5%)
- Υπόλοιπα labels: 0 annotations στο val set

---

## 2. Χαρτογράφηση του validation set σε clusters

Η ομαδοποίηση βασίζεται σε ενσωματώσεις Greek BERT (**k-means, k=12**) και αποτυπώνεται στον γράφο UMAP (ή PCA αν δεν είναι διαθέσιμο το UMAP):

<img src="../clustering/cluster_map.png" alt="Cluster Map UMAP (k=12)" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

### 2.1 Περιγραφή clusters

| Cluster | Μέγεθος | Μέσο μήκος (χαρ.) | Κυρίαρχη ορολογία | Ερμηνεία |
|---:|---:|---:|---|---|
| 0 | 33 | 2405 | ΕΞΕΤΕΤΑΣΗ, ΑΙΤΙΑ ΕΙΣΟΔΟΥ, ΠΑΡΑΚΛΙΝΙΚΕΣ, HR, COVID | Εισαγωγικά / παρακλινικά |
| 1 | 33 | 1855 | Tab, XX, αναμνηστικό, βαλβίδα, Εργαστηριακά | Δομημένα tabs καρδιολογικά |
| 2 | 53 | 1633 | πρωι, βραδυ, ΦAΡΜΑΚΕΥΤΙΚΗ, GLU, αγωγης | Ημερήσια ροή / φαρμακευτική αγωγή |
| 3 | 51 | 1559 | Εισαγωγής, Echo, XX, Tbs, αναμνηστικό | Εισαγωγική πορεία + Echo |
| 4 | 44 | 2332 | δισκίο, BIL, Αναμνηστικό, Καρδιολογικό, ΕΙΣΟΔΟΥ | Αναμνηστικό + απεικόνιση |
| 5 | 68 | 2013 | Πορίσματα, FiO221, Troponin, BIL, Φαρμ | Πορίσματα νοσηλείας + labs |
| 6 | 24 | 2214 | Echo, XX, Tbs, βαλβίδα, λειτουργικότητα | ECHO / tabs μικτό (μικρό cluster) |
| 7 | 60 | 2262 | Ψιθύρισμα, αδιαλείπτως, Φυσική, Εργαστηριακά | Κλινική ροή / παρακολούθηση |
| 8 | 35 | 1375 | ΠΟΡΙΣΜΑ, ΕΞΕΤΕΤΑΣΗ, ΕΙΣΟΔΟΥ, Φαρμ, ΔΕΝ | Σύντομα εισαγωγικά blocks |
| 9 | 39 | 2000 | Echo, βαλβίδα, Τριγλώχινα, λειτουργικότητα | Υπερηχοκαρδιογραφία |
| 10 | 25 | 2607 | cap, COVID, Sars, Rapid, Test | Εισαγωγή / infectious workup |
| 11 | 37 | 2225 | SO2, ΦAΡΜΑΚΕΥΤΙΚΗ, Υπερηχος, καρδιολογικό | SpO2 + αγωγή / καρδιολογική ροή |

**Ερμηνεία (k=12):** Η διαμέριση είναι πιο λεπτή από το προηγούμενο k=8: ξεχωρίζονται π.χ. clusters με **Echo-λεξιλόγιο** (3, 6, 9), **δομημένα tabs** (1), **ημερήσια φαρμακευτική ροή** (2, 11), **πορίσματα/labs** (5), **σύντομα εισαγωγικά** (8) και **COVID/Rapid** (10). Τα IDs **δεν αντιστοιχούν** στα clusters παλαιότερων εκδόσεων της αναφοράς — η αναλυτική περιγραφή προκύπτει από το `cluster_summary.json` μετά την εκτέλεση του `src.analysis`.

---

## 3. Συγκριτικές συνολικές μετρικές

### 3.1 Group-level (ICD-10 σε επίπεδο ομάδας/προθέματος)

| Μετρική | mlc_greek_bert | xlm_r_large | information_retrieval | ner_el |
|---|---:|---:|---:|---:|
| micro F1 | **0.901** | 0.775 | 0.523 | 0.732 |
| precision | **0.921** | 0.808 | 0.383 | 0.695 |
| recall | **0.883** | 0.744 | **0.826** | 0.773 |
| macro F1 (παρόντα labels) | **0.699** | 0.359 | 0.486 | 0.560 |
| macro F1 (όλα τα labels) | **0.608** | 0.312 | 0.423 | 0.487 |

### 3.2 Flat metrics (επίπεδο individual labels)

| Μετρική | mlc_greek_bert | xlm_r_large | information_retrieval | ner_el |
|---|---:|---:|---:|---:|
| micro precision | **0.930** | **0.817** | 0.404 | 0.699 |
| micro recall | **0.850** | 0.661 | **0.753** | 0.657 |
| micro F1 | **0.888** | 0.731 | 0.525 | 0.677 |
| macro precision | **0.655** | 0.346 | 0.310 | 0.410 |
| macro recall | **0.547** | 0.237 | **0.492** | 0.426 |
| macro F1 | **0.584** | 0.253 | 0.332 | 0.376 |
| weighted F1 | **0.878** | 0.671 | **0.638** | 0.636 |

### 3.3 Recall@k (logit ranking, πριν thresholds — μόνο MLC)

| k | mlc_greek_bert | xlm_r_large |
|---:|---:|---:|
| @3 | 0.483 | 0.451 |
| @5 | 0.710 | 0.642 |
| @10 | **0.927** | 0.832 |

> **Recall@k ερμηνεία για MLC:** Και τα δύο MLC επιτυγχάνουν recall@10 ≥ 0.83· το `mlc_greek_bert` φτάνει **~0.93**, άρα το ranking καλύπτει σχεδόν πλήρως τους gold κωδικούς πριν το thresholding. Για το `xlm_r_large`, η μεγάλη απόκλιση μεταξύ recall@10 και τελικού flat recall εξακολουθεί να δείχνει **υπερ-υψηλά thresholds** στο tail. Για το `mlc_greek_bert` μετά την επαναξιολόγηση, τα tail macro F1 (ενότητα 5) ευθυγραμμίζονται πολύ καλύτερα με το ranking — το bottleneck μετατοπίζεται σε λεπτή βαθμονόμηση και co-occurrence patterns, όχι σε «αόρατους» κωδικούς στο top-10.

> **Recall@k για IR/NER-EL:** Τα ανακτητικά συστήματα δεν εκπέμπουν πλήρες ranking πάνω στα 115 labels, οπότε η μετρική δεν ορίζεται με τον ίδιο τρόπο και παραλείπεται.

### 3.4 Ανάλυση precision-recall tradeoff

```
Precision / Recall (flat micro):

mlc_greek_bert  │ P=0.93 █████████░  R=0.85 ████████░░  [ισορροπημένο MLC]
xlm_r_large     │ P=0.82 ████████░░  R=0.66 ██████░░░░  [υπερ-συντηρητικό]
information_ret │ P=0.40 ████░░░░░░  R=0.75 ███████░░░  [wide-net]
ner_el          │ P=0.70 ███████░░░  R=0.66 ██████░░░░  [ισορροπημένο]
```

Μετά την ανανέωση των scores/thresholds, το **`mlc_greek_bert`** εμφανίζει **υψηλό P και υψηλό R** σε επίπεδο flat micro — δηλαδή δεν είναι πλέον το «χαμηλή precision / υψηλή recall» προφίλ της προηγούμενης έκδοσης της αναφοράς. Το **`ner_el`** παραμένει **ισορροπημένο** P/R σε σχέση με τα ανακτητικά και το `xlm_r_large`, λόγω της mention-based λογικής (πρόβλεψη μόνο όταν εντοπίζεται span).

---

## 4. Απόδοση ανά cluster

### 4.1 Micro F1 ανά cluster — όλα τα συστήματα

_Μετρικές από `metrics_engine.json` μετά την ενοποίηση σε **k=12** (`cluster_assignments.json`)._

| Cluster | docs | mlc_greek_bert | xlm_r_large | information_retrieval | ner_el |
|---:|---:|---:|---:|---:|---:|
| 0 (εισαγωγικά) | 33 | **0.890** | 0.743 | 0.492 | 0.675 |
| 1 (tabs) | 33 | **0.868** | 0.811 | 0.514 | 0.754 |
| 2 (αγωγή) | 53 | **0.903** | 0.776 | 0.502 | 0.736 |
| 3 (Echo/εισαγωγή) | 51 | **0.919** | 0.770 | 0.502 | 0.742 |
| 4 (αναμνηστικό) | 44 | **0.894** | 0.775 | 0.557 | 0.722 |
| 5 (πορίσματα) | 68 | **0.902** | 0.763 | 0.529 | 0.740 |
| 6 (ECHO/tabs) | 24 | **0.951** | 0.832 | 0.573 | 0.704 |
| 7 (κλινική ροή) | 60 | **0.934** | 0.812 | 0.578 | **0.780** |
| 8 (σύντομα blocks) | 35 | **0.926** | 0.761 | 0.496 | 0.726 |
| 9 (Echo βαλβίδα) | 39 | **0.886** | 0.763 | 0.463 | 0.735 |
| 10 (COVID/Rapid) | 25 | **0.850** | 0.771 | 0.562 | 0.736 |
| 11 (SpO2/αγωγή) | 37 | **0.849** | 0.706 | 0.476 | 0.678 |

### 4.2 Per-cluster Precision / Recall για xlm_r_large και ner_el

| Cluster | xlm_r_large P | xlm_r_large R | ner_el P | ner_el R |
|---:|---:|---:|---:|---:|
| 0 | 0.793 | 0.700 | 0.634 | 0.723 |
| 1 | 0.884 | 0.750 | 0.742 | 0.767 |
| 2 | 0.852 | 0.712 | 0.742 | 0.729 |
| 3 | 0.790 | 0.751 | 0.702 | 0.788 |
| 4 | 0.789 | 0.761 | 0.660 | 0.796 |
| 5 | 0.799 | 0.731 | 0.688 | 0.800 |
| 6 | 0.868 | 0.799 | 0.684 | 0.726 |
| 7 | 0.837 | 0.789 | 0.722 | 0.849 |
| 8 | 0.843 | 0.694 | 0.783 | 0.677 |
| 9 | 0.745 | 0.782 | 0.681 | 0.799 |
| 10 | 0.760 | 0.782 | 0.675 | 0.810 |
| 11 | 0.743 | 0.672 | 0.647 | 0.712 |

**Ευρήματα (k=12):**

- **Κορυφή `mlc_greek_bert`:** cluster **6** (micro F1 ≈ **0.95**, n=24) — συγκέντρωση Echo/tab λεξιλογίου· ακολουθούν τα **7** και **8** (F1 > 0.92).
- **`ner_el`:** υψηλότερο micro F1 στο cluster **7** (≈0.78, R≈0.85)· τα clusters **0** και **11** παραμένουν χαμηλότερα (~0.68–0.70) λόγω εισαγωγικών/labs χωρίς εμφανή mention ισοδυναμία.
- **`information_retrieval`:** χαμηλότερο F1 στο cluster **9** (~0.46)· η ευρεία ανάκληση παραμένει σε όλα τα clusters (~0.75–0.89), με precision που «πονά» ιδιαίτερα όταν το κείμενο είναι περιγραφικό Echo χωρίς άμεση ICD-10 ορολογία.
- **Κατάταξη MLC (ενδεικτικά):** για `mlc_greek_bert` η ιεράρχηση micro F1 ακολουθεί περίπου **6 > 7 > 8 > 3 > … > 11 ≈ 10** — όλα τα clusters παραμένουν >0.84, άρα η διακύμανση είναι μικρότερη από ότι σε χονδρότερο k.

---

## 5. Ανάλυση long-tail (frequent / medium / rare)

> _Μόνο για MLC μοντέλα (απαιτεί per-label logits). Ορισμός:_ `frequent` ≥ 50 annotations · `rare` ≤ 10 annotations · `medium` ενδιάμεσο.

### 5.1 Macro F1 ανά tail bucket

| Bucket | # labels | Support | mlc_greek_bert | xlm_r_large |
|---|---:|---:|---:|---:|
| frequent | 17 | 2.604 | **0.927** | 0.856 |
| medium | 43 | 694 | **0.576** | 0.290 |
| rare | 55 | 221 | **0.535** | 0.161 |

### 5.2 Mean Precision / Recall ανά tail bucket

| Bucket | mlc_greek_bert P | mlc_greek_bert R | xlm_r_large P | xlm_r_large R |
|---|---:|---:|---:|---:|
| frequent | 0.937 | 0.918 | 0.850 | 0.869 |
| medium | 0.601 | 0.559 | 0.426 | 0.267 |
| rare | 0.630 | 0.488 | 0.228 | 0.146 |

### 5.3 Γράφοι long-tail κατανομής

**mlc_greek_bert:**

<img src="../mlc_greek_bert/long_tail.png" alt="mlc_greek_bert long-tail" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

**xlm_r_large:**

<img src="../xlm_r_large/long_tail.png" alt="xlm_r_large long-tail" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

### 5.4 Ανάλυση

**`mlc_greek_bert` (ανανεωμένο):** Μετά την ευθυγράμμιση thresholds με τα logits, το μοντέλο είναι **ισχυρό σε όλα τα tail buckets**: macro F1 frequent **0.927** (υψηλότερο και από το `xlm_r_large`), medium **0.576** και rare **0.535** — δηλαδή **δεν** παραμένει «ολική κατάρρευση» στο tail όπως πριν. Η μέση recall στα rare (0.488) είναι πλέον συμβατή με recall@10 ≈0.93, άρα το ranking και το τελικό decision boundary είναι **πολύ καλύτερα συγχρονισμένα** από την προηγούμενη αναφορά.

**`xlm_r_large`:** Παραμένει ισχυρό στις frequent (macro F1 = 0.856, recall = 0.869) αλλά με **έντονη κατάρρευση στις medium (0.290) και rare (0.161)**. Η μέση recall στις rare (0.146) έναντι recall@10 0.832 εξακολουθεί να δείχνει **threshold miscalibration στο tail**, όχι έλλειψη ranking signal.

**Πρακτική σύσταση:** Για `xlm_r_large`, ειδικό threshold tuning ανά bucket (πχ. rare threshold χαμηλότερα από το global) εξακολουθεί να είναι η κύρια «γρήγορη νίκη». Για `mlc_greek_bert`, η κύρια μόχλευση μετατοπίζεται σε **co-occurrence post-processing** και σε λεπτή fine-tuning ανά κωδικό όπου εμφανίζονται ακόμη συστηματικά confused pairs (ενότητα 7).

---

## 6. Range codes έναντι Specific codes

Πέντε εύρη ICD-10 υπάρχουν στο label set: `C00-C97` (νεοπλασίες), `D50-D64` (αναιμίες), `E00-E07` (θυρεοειδής), `M30-M36` (συνδετικός ιστός), `E65-E68` (παχυσαρκία). Συνολικό support στο val set: **56 annotations**.

| Μοντέλο | Ranges: FP | Ranges: FN | Specific: FP | Specific: FN | Recall ranges | Recall specific |
|---|---:|---:|---:|---:|---:|---:|
| mlc_greek_bert | 4 | 11 | 215 | 367 | **0.804** | **0.894** |
| xlm_r_large | **0** | **55** | 510 | 799 | **0.000** | 0.769 |

**Κρίσιμο εύρημα:** Το `xlm_r_large` **αποτυγχάνει παντελώς** να προβλέψει range codes (0 FP / 55 FN = 0% recall, 100% miss rate). Οι κωδικοί τύπου εύρους (`C00-C97`, κ.λπ.) εμφανίζονται συχνά ως συνοδές διαγνώσεις μαζί με specific codes — η απόλυτη μη-πρόβλεψη στο `xlm_r_large` υποδεικνύει training/threshold bias για τα range labels.

Μετά την ανανέωση, το **`mlc_greek_bert` ανακτά σχεδόν όλα τα range annotations** (recall ranges **0.80**, με χαμηλότερα FP/FN από πριν), ενώ βελτιώνει και τα specific (recall **0.89**). Το `xlm_r_large` εξακολουθεί να εμφανίζει υψηλά FP και FN στα specific codes (510 / 799) — precision-recall mismatch στο long tail.

---

## 7. Συγχύσεις ανά ζεύγος (top confused pairs)

_Σύμβαση: `predicted → missed` = το μοντέλο είπε Α αλλά αστόχησε στον συσχετιζόμενο Β._

### 7.1 mlc_greek_bert — top 10 pairs

| predicted | missed | count | Κλινική ερμηνεία |
|---|---|---:|---|
| I50 (καρδ. ανεπ.) | Y84 (παρεν. ιατρ. πράξης) | 3 | Χαμένη ιατρογενής επιπλοκή |
| Z95 (εμφύτευμα) | R00 (αρρυθμία-NOS) | 3 | Εμφύτευμα χωρίς ρητή αρρυθμία-NOS |
| I21 (STEMI/NSTEMI) | I20 (στηθάγχη) | 3 | Οξύ στεφανιαίο vs unstable angina (fine-grained) |
| I25 (ΣΝΑ) | Z99 (εξάρτηση από μηχ.) | 2 | Χαμένη εξάρτηση από μηχανική υποστήριξη |
| I25 (ΣΝΑ) | I44 (AV block) | 2 | Χαμένη διαταραχή αγωγιμότητας |
| Z95 (εμφύτευμα) | I44 (AV block) | 2 | Συσκευή χωρίς ρητό AV block |
| I50 (καρδ. ανεπ.) | I49 (αρρυθμία) | 2 | Ανεπάρκεια χωρίς ρητή αρρυθμία |
| I50 (καρδ. ανεπ.) | I44 (AV block) | 2 | Ανεπάρκεια χωρίς ρητό AV block |
| I48 (κολπική παρεκτόπιση) | I49 (αρρυθμία) | 2 | Υπέρθεση κολπικής παρεκτόπισης vs άλλη αρρυθμία |
| R06 (αναπνευστικά) | I05 (ρευματική νόσος) | 2 | Αναπνευστικά συμπτώματα χωρίς ρητή βαλβιδοπάθεια |

### 7.2 xlm_r_large — top 10 pairs

| predicted | missed | count | Κλινική ερμηνεία |
|---|---|---:|---|
| R07 (στηθαλγία) | M54 (πόνος ράχης) | **8** | Στηθαλγία χωρίς μυοσκελετικό |
| R07 (στηθαλγία) | R10 (κοιλιακό άλγος) | **8** | Στηθαλγία χωρίς επιγαστρικό |
| I50 (καρδ. ανεπ.) | Y84 (παρεν.) | 5 | Ίδιο pattern με greek_bert |
| R07 (στηθαλγία) | C00-C97 (νεοπλασία) | 5 | Χαμένη νεοπλασία παρουσία στηθαλγίας |
| I50 (καρδ. ανεπ.) | R60 (οίδημα) | 5 | Καρδιακή ανεπάρκεια χωρίς οίδημα |
| R07 (στηθαλγία) | I10 (υπέρταση) | 5 | Χαμένη υπέρταση |
| R07 (στηθαλγία) | I11 (υπερτασική καρδιοπάθεια) | 5 | Χαμένη υπερτασική |
| R07 (στηθαλγία) | I46 (καρδιακή ανακοπή) | 4 | Χαμένη ανακοπή |
| Z95 (εμφύτευμα) | I49 (αρρυθμία) | 4 | Εμφύτευμα χωρίς αρρυθμία |
| I25 (ΣΝΑ) | I44 (AV block) | 4 | ΣΝΑ χωρίς αγωγιμότητα |

**Δομικό pattern για `xlm_r_large`:** Ο κωδικός R07 (θωρακικός πόνος) εμφανίζεται σε **7 από τα top-10 confused pairs** ως predicted code που συνοδεύεται από πολλές χαμένες συνοσηρότητες. Αυτό δεν είναι τυχαίο: το R07 έχει υψηλό support (frequent label) και το μοντέλο το προβλέπει σωστά, αλλά αδυνατεί να «ακολουθήσει» τις συσχετιζόμενες diagnosis codes. Είναι άμεσα διορθώσιμο με co-occurrence post-processing.

### 7.3 Κοινές κλινικές συνοσηρότητες — «universal miss» patterns

Ανεξάρτητα μοντέλου, τα παρακάτω ζεύγη εμφανίζονται σταθερά ως δύσκολα:

| Ζεύγος | Κλινική εξήγηση | Στρατηγική διόρθωσης |
|---|---|---|
| `I50 ↔ Y84` | Καρδιακή ανεπάρκεια + ιατρογενής επιπλοκή | co-occurrence rule |
| `Z95 ↔ I49` | Βηματοδότης/defibrillator + αρρυθμία | co-occurrence rule |
| `I21/I25 ↔ I79` | Έμφραγμα/ΣΝΑ + αγγειακές συνοσηρότητες | co-occurrence rule |
| `R07 ↔ I10/I11` | Στηθαλγία + υπέρταση | co-occurrence rule |
| `I25 ↔ I44` | ΣΝΑ + AV block | co-occurrence rule |
| `I50 ↔ R60` | Καρδιακή ανεπ. + οίδημα | co-occurrence rule |

Το repo διαθέτει αρχείο co-occurrence rules (π.χ. `cooccurrence_rules.json`) για post-processing. Η εφαρμογή τους αναμένεται να ωφελήσει κυρίως το **`xlm_r_large`** (συστηματικά misses) αλλά και το **`mlc_greek_bert`** (υπολειπόμενα ζεύγη τύπου I50↔Y84) κατά **~0.01–0.04** flat micro F1 ανάλογα την κάλυψη των rules.

---

## 8. Χάρτες σύγχυσης (Confusion Heatmaps)

Οι heatmaps δείχνουν τα ζεύγη predicted/ground-truth με τις υψηλότερες εμφανίσεις λάθους (normalized ανά GT label).

**mlc_greek_bert:**

<img src="../mlc_greek_bert/confusion_heatmap.png" alt="mlc_greek_bert confusion heatmap" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

**xlm_r_large:**

<img src="../xlm_r_large/confusion_heatmap.png" alt="xlm_r_large confusion heatmap" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

Στον heatmap του `xlm_r_large` εμφανίζεται καθαρά ο «θόρυβος» γύρω από τον R07 (σειρά με έντονα off-diagonal στοιχεία), επιβεβαιώνοντας το pattern της §7.2.

---

## 9. Προφίλ σφαλμάτων (Error Profiler)

### 9.1 Μέσα FN/FP ανά keyword — πλήρης πίνακας

| Keyword | docs | bert FN | bert FP | large FN | large FP |
|---|---:|---:|---:|---:|---:|
| ιστορικό | 410 | **0.70** | **0.46** | **1.49** | **1.03** |
| διάγνωση | 486 | **0.67** | **0.45** | **1.49** | **1.02** |
| υπέρταση | 133 | **0.67** | **0.43** | **1.38** | **1.17** |
| στεφανιαία | 225 | **0.72** | **0.51** | **1.48** | **1.15** |
| έμφραγμα | 60 | **0.85** | **0.43** | **1.42** | **1.07** |
| αρρυθμία | 16 | **0.38** | **0.00** | 1.31 | 0.50 |

**Σημαντικές παρατηρήσεις:**

1. **`mlc_greek_bert`:** Τα μέσα FN έπεσαν σε **~0.7–0.85** ανά έγγραφο για τα κύρια keywords (από ~3.2+ στην προηγούμενη έκδοση της αναφοράς), με **FP και FN στην ίδια τάξη μεγέθους** (π.χ. ιστορικό 0.70 / 0.46) — δηλαδή πλέον **όχι** καθαρά «χαμηλή recall μόνο». Το `αρρυθμία` (μικρό n=16) παραμένει δύσκολο αλλά όχι εκτός κλίματος (0.38 FN, 0 FP στο sample).

2. **`xlm_r_large`:** Ενδιαφέρον προφίλ: FP ≈ FN για τα περισσότερα keywords. Για `υπέρταση` τα FP (1.17) _υπερβαίνουν_ ελαφρώς τα FN (1.38) — δηλαδή το μοντέλο **υπερ-προβλέπει** σε έγγραφα που περιέχουν εμφανή term. Αυτό αντιφαίνεται με το «conservative» προφίλ του συνολικά, και υποδεικνύει ότι η συντηρητικότητα αφορά τις **αθέατες** συνοσηρότητες, ενώ τα **εμφανή** primary diagnoses μπορεί να over-predict.

### 9.2 Worst PIDs — έγγραφα που αποτυγχάνουν σε όλα τα μοντέλα

Τα PIDs που εμφανίζονται ως worst cases σε **≥3** από τα keywords ιστορικό / διάγνωση / υπέρταση / στεφανιαία / έμφραγμα (top-5 ανά keyword στο `error_profiler.json`):

| PID | mlc_greek_bert (εμφ.) | xlm_r_large (εμφ.) | Σχόλιο |
|---:|---:|---:|---|
| **3028** | ιστορικό, διάγνωση, στεφανιαία, έμφραγμα (4/5) | έμφραγμα | Σταθερά hard για BERT· στο XLM εμφανίζεται κυρίως στο keyword «έμφραγμα» |
| **9103** | ιστορικό, διάγνωση, στεφανιαία, έμφραγμα (4/5) | — | Ίδιο «πυρήνα» keywords με το 3028 |
| **9150** | ιστορικό, στεφανιαία, έμφραγμα (3/5) | — | Έντονο σε ιστορικό αλλά όχι στα top-5 του «διάγνωση» |
| **231** | υπέρταση, στεφανιαία, έμφραγμα (3/5) | ιστορικό, διάγνωση, υπέρταση, στεφανιαία, έμφραγμα (5/5) | Universal hard· το XLM το χάνει σε όλα τα πέντε keywords |
| **147** | διάγνωση, υπέρταση, στεφανιαία (3/5) | — | — |
| **5147** | — | ιστορικό, διάγνωση (2/5) | Χαρακτηριστικό residual για `xlm_r_large` (όχι πλέον στα top-5 του retuned BERT) |
| **3169** | — | ιστορικό, διάγνωση (2/5) | Όπως πάνω |

> **Σύσταση:** Τα PIDs **3028**, **9103**, **9150** και **231** χρήζουν χειροκίνητης επαλήθευσης (gold labels ή εξαιρετική πολυπλοκότητα). Η λίστα worst-PIDs **άλλαξε** μετά το retuning — αξίζει επανέλεγχος όσων είχαν σημανθεί μόνο στην παλιά έκδοση της αναφοράς.

### 9.3 Ανάλυση μήκους εγγράφου

Διάμεσος μήκους: **1.948 χαρακτήρες**. Ορισμός: short ≤ 1.948, long > 1.948 (n=251 κάθε group).

| Μοντέλο | Short micro F1 | Short P | Short R | Long micro F1 | Long P | Long R |
|---|---:|---:|---:|---:|---:|---:|
| mlc_greek_bert | **0.914** | **0.922** | **0.906** | **0.890** | **0.919** | **0.863** |
| xlm_r_large | 0.777 | 0.835 | 0.727 | 0.773 | 0.787 | 0.759 |

**Συμπέρασμα:** Το `xlm_r_large` παραμένει **σχεδόν αναίσθητο στο μήκος** (Δ F1 ≈ 0.004). Για το `mlc_greek_bert` μετά την ανανέωση, τα **μικρά** έγγραφα είναι οριακά καλύτερα (Δ F1 ≈ **0.024** short−long)· η διαφορά είναι μικρή και δεν υποδηλώνει truncation ως κύριο bottleneck. Και τα δύο MLC παραμένουν σε καθεστώς όπου long-context architectures δεν είναι προτεραιότητα μόνο από το μήκος.

---

## 10. Ανακτητικά και mention-based συστήματα

### 10.1 `information_retrieval` — hybrid BM25 + dense RRF

**Αρχιτεκτονική:** Ο χώρος ICD-10 μεταχειρίζεται ως corpus 115 «εγγράφων» (κωδικός + περιγραφή + mined mentions από training). Κάθε ιατρικό κείμενο αποτελεί ερώτημα. Τα αποτελέσματα BM25 και dense (MiniLM) συνδυάζονται με Reciprocal Rank Fusion (k=20, w=1.2/0.8). Τελικές παράμετροι μετά από tuning (βλ. `outputs/experiments/information_retrieval/ir_tune_summary_hybrid.json`):

| Παράμετρος | Τιμή |
|---|---|
| fraction_of_top_score | 0.22 |
| max_codes | 8 |
| include_dictionary | True |
| RRF k | 20 |
| BM25 weight | 1.2 |
| Dense weight | 0.8 |

**Εξέλιξη απόδοσης μέσω tuning:**

| Σταδιο | micro F1 | precision | recall |
|---|---:|---:|---:|
| Baseline (top-25, plain corpus) | 0.130 | 0.080 | 0.347 |
| Default expanded hybrid | 0.439 | 0.298 | 0.832 |
| Tuned (full train) | **0.525** | 0.386 | 0.823 |

Το tuning **+0.086 F1** κυρίως μέσω ανύψωσης precision (0.298 → 0.386) διατηρώντας recall (~0.83). Η αρχική baseline έχει recall 0.347, δηλαδή μόνο η mention expansion απέδωσε +0.475 σε recall.

**Per-cluster απόδοση** (k=12):

| Cluster | micro F1 | precision | recall |
|---:|---:|---:|---:|
| 0 | 0.492 | 0.363 | 0.761 |
| 1 | 0.514 | 0.372 | 0.831 |
| 2 | 0.502 | 0.369 | 0.784 |
| 3 | 0.502 | 0.358 | 0.837 |
| 4 | 0.557 | 0.415 | 0.846 |
| 5 | 0.529 | 0.382 | 0.859 |
| 6 | 0.573 | 0.437 | 0.829 |
| 7 | **0.578** | 0.429 | **0.886** |
| 8 | 0.496 | 0.371 | 0.747 |
| 9 | 0.463 | 0.324 | 0.810 |
| 10 | 0.562 | 0.419 | 0.851 |
| 11 | 0.476 | 0.341 | 0.788 |

Το recall παραμένει **υψηλό** σε όλα τα clusters (περίπου **0.75–0.89**) — «ευρύ δίχτυ». Η precision κυμαίνεται περίπου **0.32–0.44**· το χαμηλότερο F1 εμφανίζεται στο cluster **9** (Echo-λεξιλόγιο χωρίς άμεση ICD-10 ορολογία), ενώ χαμηλότερη precision (~0.32) συνδέεται με περιγραφικά Echo blocks όπου το IR πυροδοτεί πολλά false positives παρά τη διατήρηση recall.

### 10.2 `ner_el` — BIO NER + mention prior linker

**Αρχιτεκτονική:** Greek BERT εκπαιδευμένο για token-level BIO classification (O, B-MED, I-MED). Τα predicted spans linkάρονται σε ICD-10 codes μέσω mention priors (P(code|mention) από training) και λεξικά `full_dictionary.csv` / `icd10_greek_lookup.csv`. Document-level aggregation με majority voting mentions.

**Per-cluster απόδοση** (k=12):

| Cluster | micro F1 | precision | recall |
|---:|---:|---:|---:|
| 0 | 0.675 | 0.634 | 0.723 |
| 1 | 0.754 | 0.742 | 0.767 |
| 2 | 0.736 | 0.742 | 0.729 |
| 3 | 0.742 | 0.702 | 0.788 |
| 4 | 0.722 | 0.660 | 0.796 |
| 5 | 0.740 | 0.688 | 0.800 |
| 6 | 0.704 | 0.684 | 0.726 |
| 7 | **0.780** | 0.722 | **0.849** |
| 8 | 0.726 | 0.783 | 0.677 |
| 9 | 0.735 | 0.681 | 0.799 |
| 10 | 0.736 | 0.675 | 0.810 |
| 11 | 0.678 | 0.647 | 0.712 |

**Αξιοσημείωτα (k=12):**
- **Cluster 1 (tabs):** Το `ner_el` παραμένει ισχυρό (F1≈0.75) όπου υπάρχουν εμφανείς ιατρικοί όροι σε δομημένα tabs· το IR παραμένει πιο αδύναμο εκεί λόγω απόκλισης ορολογίας από τις ICD-10 περιγραφές.
- **Clusters 6, 9 (Echo-λεξιλόγιο):** Χαμηλότερο F1 για `ner_el` (~0.70–0.74) έναντι **πολύ υψηλότερου** `mlc_greek_bert` στο ίδιο υλικό — τα mention priors δεν καλύπτουν πλήρως περιγραφικές εκφράσεις βαλβίδων/κινητικότητας.
- **Clusters 0, 11:** Χαμηλότερη απόδοση `ner_el` (~0.68–0.70)· το **`mlc_greek_bert`** παραμένει >0.84 micro F1, άρα το κενό «NER vs MLC» σε εισαγωγικά/labs επιβεβαιώνεται στο νέο διαμέρισμα.

### 10.3 Συγκριτική τοποθέτηση — σύνοψη

| Προσέγγιση | Flat micro F1 | Group macro F1 (όλα τα labels) | Βέλτιστη χρήση |
|---|---:|---:|---|
| **`mlc_greek_bert`** | **0.888** | **0.608** | **Κύριο MLC · υψηλό group+flat F1** |
| `xlm_r_large` | 0.731 | 0.312 | Συμπληρωματικό όταν θέλεις υψηλό micro-P σε flat αλλά αποδέχεσαι tail/range αδυναμίες |
| `information_retrieval` | 0.525 | 0.423 | Candidate generation / pipeline stage 1 |
| `ner_el` | 0.677 | 0.487 | Interpretable mention-based · ισχυρό macro σε σχέση με IR |

---

## 11. Κεντρικά συμπεράσματα

### 11.1 Κατάταξη συστημάτων

```
mlc_greek_bert > xlm_r_large > ner_el > information_retrieval   (flat micro F1)
0.888           > 0.731        > 0.677  > 0.525
```

**Group-level micro F1:** `mlc_greek_bert` (**0.901**) > `xlm_r_large` (0.775) > `ner_el` (0.732) > `information_retrieval` (~0.523).

**Macro F1 (flat, όλα τα labels):** `mlc_greek_bert` (**0.584**) > `ner_el` (0.376) > `information_retrieval` (~0.332) > `xlm_r_large` (0.253).

Σημείωση: το **`mlc_greek_bert`** πλέον **υπερτερεί συστηματικά** σε flat micro F1, group micro F1 και macro F1 έναντι των άλλων συστημάτων της αναφοράς (εκτός του αποκλεισμένου `xlm_r_base`). Το **`ner_el`** παραμένει πολύτιμο για **ερμηνευσιμότητα** (mentions) και competitive flat F1 (0.677), αλλά δεν είναι πλέον «κορυφή» σε ουδέτερη σύγκριση αριθμών έναντι του retuned Greek BERT MLC.

### 11.2 Χαρακτηριστικά προφίλ σφαλμάτων

| Σύστημα | Κύριο σφάλμα | Αιτία | Ευθεία βελτίωση |
|---|---|---|---|
| `mlc_greek_bert` | Υπολειπόμενα co-occurrence misses (π.χ. I50↔Y84), sporadic hard PIDs | Όχι πλέον «ολική» recall κατάρρευση | `cooccurrence_rules.json`, λεπτό per-code threshold |
| `xlm_r_large` | Range codes = 0 recall, συνοσηρότητες χάνονται | Miscalibrated thresholds, tail bias | Threshold retuning per bucket, co-occurrence |
| `information_retrieval` | Χαμηλή precision (0.40) | Wide-net by design | Max_codes tuning, per-code thresholds |
| `ner_el` | Clusters χωρίς εμφανή mentions (labs/ECHO) | Prior sparsity | Prior enrichment per cluster |

### 11.3 Cluster-level ευρήματα (k=12)

- **Cluster 6 (Echo/tabs, n=24):** Κορυφαίο micro F1 για **`mlc_greek_bert`** (~0.95) και ισχυρό **`xlm_r_large`** (~0.83)· το **`ner_el`** παραμένει πιο μέτριο (~0.70) — μικρό αλλά «σκληρό» υποσύνολο με πυκνό Echo/tab λεξιλόγιο.
- **Cluster 7 (κλινική ροή / παρακολούθηση):** Ισορροπημένη υψηλή απόδοση σε MLC και **`ner_el`** (F1 ~0.78, R~0.85)· το IR ευνοείται από πιο «λεκτική» κλινική γλώσσα.
- **Clusters 2, 5, 11 (αγωγή / πορίσματα / SpO2):** Θορυβώδη κείμενα (φάρμακα, τιμές)· το **`mlc_greek_bert`** διατηρεί **>0.84** micro F1, ενώ **`xlm_r_large`** παραμένει ~0.76–0.77 — η διαφορά MLC vs tail-thresholds παραμένει δομική, όχι «ολική αποτυχία» σε κάθε cluster.
- **Cluster 9 (Echo βαλβίδα):** Το IR χτυπά χαμηλότερο F1 (~0.46) παρά υψηλό recall — ενδεικτικό wide-net προφίλ σε περιγραφικό Echo κείμενο.

---

## 12. Προτάσεις επόμενων βημάτων

### Άμεσα (χωρίς επανεκπαίδευση)

| Ενέργεια | Αναμενόμενο όφελος | Επηρεαζόμενα μοντέλα |
|---|---|---|
| Threshold retuning ανά long-tail bucket (rare: ~0.15, medium: ~0.25) | +0.03–0.06 macro F1 | **`xlm_r_large`** (κύριος ωφελούμενος)· το `mlc_greek_bert` ήδη σε καλή θέση |
| Post-processing με `cooccurrence_rules.json` | +0.01–0.04 flat F1 | **`mlc_greek_bert`**, `xlm_r_large` (κοινά patterns εν. 7.3) |
| Per-code thresholds για IR | +0.03–0.05 F1 | `information_retrieval` |
| Manual audit PIDs 3028, 9103, 9150, 231 (labeling check) | Καλύτερη baseline | Όλα |

### Βραχυπρόθεσμα (επανεκπαίδευση)

| Ενέργεια | Λεπτομέρεια |
|---|---|
| Focal loss / hard-negative mining για `mlc_greek_bert` | Προαιρετικό πλέον· προτεραιότητα έχουν co-occurrence + per-code calibration πριν νέα εκπαίδευση |
| Enrichment priors `ner_el` per cluster | Ειδικά για clusters με Echo-λεξιλόγιο (π.χ. 6, 9) και εισαγωγικά/labs (0, 11): context-level patterns |
| Mention expansion IR corpus για tab-clusters | Mining καρδιολογικής ορολογίας (Tab-based terms) για ICD-10 code documents (π.χ. cluster 1) |

### Μακροπρόθεσμα (ensemble)

Τα τέσσερα συστήματα της αναφοράς έχουν **συμπληρωματικά** error profiles. Ένα δύο-σταδιακό pipeline:

```
Stage 1 (Candidate Generation):
  IR top-8 [recall≈0.83] ∪ mlc_greek_bert top-10 logits [recall@10≈0.93]
  → Union: πολύ υψηλό recall, αυξημένα FP

Stage 2 (Re-ranking / Filtering):
  ner_el × xlm_r_large (voting/confidence threshold) ή δεύτερο pass με stricter thresholds στο BERT
  → Precision recovery

Expected: flat micro F1 σε ζώνη ~0.80+ ανάλογα το βάρος στο Stage 2
```

---

_Όλα τα αναφερόμενα νούμερα αναπαράγονται από: `outputs/analysis/*/metrics_engine.json`, `*/label_analysis.json`, `*/error_profiler.json`, `clustering/cluster_summary.json`, `summary/models_comparison.json` (όπου εφαρμόζεται), `outputs/experiments/information_retrieval/ir_tune_summary_hybrid.json`, και για το `mlc_greek_bert` επιπλέον `models/greek_bert/thresholds.json` / `val_scores.npy`._

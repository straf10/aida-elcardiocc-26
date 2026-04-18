# Αναφορά Ανάλυσης Μοντέλων — ELCardioCC

> **Σημείωση:** Τα αποτελέσματα του **`xlm_r_base`** (XLM-RoBERTa base) **δεν περιλαμβάνονται** σε αυτή την αναφορά, επειδή η αξιολόγηση είχε βασιστεί σε **μολυσμένα δεδομένα** (επικάλυψη validation με το training fold). Οι πίνακες και τα συμπεράσματα παρακάτω αφορούν μόνο τα υπόλοιπα συστήματα.

> **Ενημέρωση `mlc_greek_bert`:** Οι μετρικές και τα διαγράμματα για το Greek BERT MLC **ανανεώθηκαν** ώστε να αντιστοιχούν στην τρέχουσα επαναξιολόγηση στο ίδιο validation set (502 έγγραφα), με **ανά έκθεση logits/thresholds** (`models/greek_bert/thresholds.json`, `val_scores.npy`) — όχι σε αλλαγή του gold validation set.

_Πηγή δεδομένων: `outputs/analysis/` (υποφάκελοι ανά μοντέλο) · κοινά clustering artefacts: `clustering/cluster_assignments.json`, `clustering/cluster_summary.json`, `clustering/embeddings.npy` · γράφοι: `*/long_tail.png`, `*/confusion_heatmap.png`, `clustering/cluster_map.png`._

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

Η ομαδοποίηση βασίζεται σε ενσωματώσεις Greek BERT (cosine + k-means, k=8) και αποτυπώνεται στον γράφο UMAP:

<img src="./clustering/cluster_map.png" alt="Cluster Map UMAP" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

### 2.1 Περιγραφή clusters

| Cluster | Μέγεθος | Μέσο μήκος (χαρ.) | Κυρίαρχη ορολογία | Ερμηνεία |
|---:|---:|---:|---|---|
| 0 | 64 | 2.447 | ΕΞΕΤΑΣΗ, ΠΑΡΑΚΛΙΝΙΚΕΣ, ΑΙΤΙΑ ΕΙΣΟΔΟΥ, θεράποντα | Εισαγωγικά-παραπεμπτικά έγγραφα |
| 1 | 29 | 1.724 | Tab, Παράγοντες Κινδύνου, φλεβοκομβικός, περικαρδιακή | Δομημένα καρδιολογικά tabs |
| 2 | 88 | 1.488 | πρωι/βραδυ/μεσημερι, ΦΑΡΜΑΚΕΥΤΙΚΗ ΑΓΩΓΗ, Hb, Na | Συνταγογράφηση/αγωγή |
| 3 | 88 | 1.906 | LDL, HDL, UREA, WBC, PLT, TRIGL | Εργαστηριακές εξετάσεις |
| 4 | 46 | 2.266 | δισκίο, BIL, Αναμνηστικό, ΠΑΡΑΚΛΙΝΙΚΕΣ | Αναμνηστικό + εκτεταμένο ιστορικό |
| 5 | 75 | 2.067 | FiO221, Πορίσματα, CREAT, ALP, CHOL, ΑΙΤΙΑ ΕΙΣΟΔΟΥ | Πορίσματα νοσηλείας + labs |
| 6 | 54 | 2.091 | ECHO, Αορτική, Μιτροειδής, Τριγλώχινα | Υπερηχοκαρδιογραφικές εκθέσεις |
| 7 | 58 | 2.261 | Ψιθύρισμα, αεριομετρικά, διακομίστηκε, αδιαλείπτως | Οξέα/βαρέα περιστατικά (ΜΕΘ-like) |

**Ερμηνεία ορολογίας clusters:** Τα clusters 6 και 7 αντιστοιχούν σε πιο «τεχνική» καρδιολογική/εντατική ορολογία. Τα clusters 2 και 3 αντιστοιχούν σε πεδία φαρμακευτικής αγωγής και εργαστηριακών αναλύσεων, όπου τα ICD-10 labels είναι λιγότερο εμφανή στο κείμενο. Το cluster 1, αν και μικρό (29 έγγραφα), περιέχει δομημένη καρδιολογική ορολογία που αποκλίνει από την επίσημη ΙCD-10 ονοματολογία — εξ ου και η σχετική δυσκολία για τα ανακτητικά συστήματα.

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

| Cluster | docs | mlc_greek_bert | xlm_r_large | information_retrieval | ner_el |
|---:|---:|---:|---:|---:|---:|
| 0 (είσοδος) | 64 | **0.872** | 0.755 | 0.522 | 0.708 |
| 1 (Tabs/καρδιο) | 29 | **0.894** | 0.795 | 0.495 | 0.771 |
| 2 (αγωγή) | 88 | **0.910** | 0.775 | 0.496 | 0.734 |
| 3 (εργαστηριακά) | 88 | **0.885** | 0.746 | 0.496 | 0.705 |
| 4 (αναμνηστικό) | 46 | **0.883** | 0.774 | 0.555 | 0.718 |
| 5 (πορίσματα) | 75 | **0.911** | 0.773 | 0.534 | 0.742 |
| 6 (ECHO) | 54 | **0.920** | 0.796 | 0.512 | 0.718 |
| 7 (ΜΕΘ/οξέα) | 58 | **0.932** | **0.807** | **0.577** | **0.784** |

### 4.2 Per-cluster Precision / Recall για xlm_r_large και ner_el

| Cluster | xlm_r_large P | xlm_r_large R | ner_el P | ner_el R |
|---:|---:|---:|---:|---:|
| 0 | 0.771 | 0.740 | 0.658 | 0.766 |
| 1 | 0.880 | 0.725 | 0.783 | 0.761 |
| 2 | 0.849 | 0.712 | 0.762 | 0.708 |
| 3 | 0.781 | 0.715 | 0.665 | 0.749 |
| 4 | 0.792 | 0.757 | 0.673 | 0.770 |
| 5 | 0.809 | 0.741 | 0.689 | 0.804 |
| 6 | 0.803 | 0.789 | 0.676 | 0.765 |
| 7 | 0.826 | 0.790 | 0.716 | 0.865 |

**Ευρήματα:**

- **Cluster 7 (ΜΕΘ/οξέα):** Κορυφαία απόδοση για όλα τα συστήματα. Η ειδική ορολογία («αεριομετρικά», «αδιαλείπτως», «διακομίστηκε», «Ψιθύρισμα») χαρτογραφείται μονοσήμαντα σε κωδικούς, διευκολύνοντας και MLC και NER-EL. Το `ner_el` στο cluster 7 εμφανίζει recall = 0.865 — το υψηλότερο μεταξύ των συστημάτων της αναφοράς.
- **Cluster 6 (ECHO):** Ισχυρός για τα MLC (`mlc_greek_bert` F1≈0.920, `xlm_r_large` F1=0.796), αλλά χαμηλότερος για `ner_el` (0.718). Τα ECHO reports περιέχουν πολλές κατανεμημένες περιγραφικές φράσεις («αορτική/μιτροειδής βαλβίδα», «κινητικότητα τοιχωμάτων»), πολλές από τις οποίες δεν έχουν αντίστοιχα mention-level priors στα training δεδομένα.
- **Clusters 0, 3 (είσοδος/εργαστηριακά):** Ουραίοι για `ner_el`. Τα εργαστηριακά αποτελέσματα (UREA, WBC, PLT) και τα εισαγωγικά πεδία δεν αποτελούν «mentions» κατά την BIO-NER λογική — τα ICD-10 labels εκεί προκύπτουν από πλαίσιο (context), όχι από εμφανείς ορολογικές φράσεις.
- **Η σχετική κατάταξη clusters** για τα MLC παραμένει περίπου **7 > 6 > 5 > 2 > 1 > 3 > 4 > 0** για το `mlc_greek_bert` (όλα πλέον >0.87 micro F1), ενώ για `ner_el` / IR η ιεράρχηση 7 > … > 0 ≈ 3 παραμένει πιο έντονη — επιβεβαιώνοντας εγγενή domain δυσκολία αλλά με πολύ μικρότερη διακύμανση για το retuned Greek BERT MLC.

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

<img src="./mlc_greek_bert/long_tail.png" alt="mlc_greek_bert long-tail" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

**xlm_r_large:**

<img src="./xlm_r_large/long_tail.png" alt="xlm_r_large long-tail" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

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

<img src="./mlc_greek_bert/confusion_heatmap.png" alt="mlc_greek_bert confusion heatmap" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

**xlm_r_large:**

<img src="./xlm_r_large/confusion_heatmap.png" alt="xlm_r_large confusion heatmap" style="max-width:100%;height:auto;display:block;margin:1em auto;" />

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

**Per-cluster απόδοση:**

| Cluster | micro F1 | precision | recall |
|---:|---:|---:|---:|
| 0 (είσοδος) | 0.522 | 0.387 | 0.805 |
| 1 (Tabs/καρδιο) | 0.495 | 0.355 | 0.817 |
| 2 (αγωγή) | 0.496 | 0.366 | 0.770 |
| 3 (εργαστηριακά) | 0.496 | 0.356 | 0.817 |
| 4 (αναμνηστικό) | 0.555 | 0.418 | 0.824 |
| 5 (πορίσματα) | 0.534 | 0.387 | 0.858 |
| 6 (ECHO) | 0.512 | 0.372 | 0.820 |
| 7 (ΜΕΘ/οξέα) | **0.577** | **0.425** | **0.899** |

Το recall είναι **εξαιρετικά σταθερό** (0.77–0.90) σε όλα τα clusters — αυτό είναι η ουσία της προσέγγισης: «ευρύ δίχτυ» που πιάνει σχεδόν τα πάντα, αλλά με πολλά false positives. Η precision κυμαίνεται 0.36–0.43, με το worst cluster να είναι το **1 (Tabs/καρδιο, P=0.355)** — η εξειδικευμένη ορολογία των structured tabs αποκλίνει από τις ICD-10 περιγραφές.

### 10.2 `ner_el` — BIO NER + mention prior linker

**Αρχιτεκτονική:** Greek BERT εκπαιδευμένο για token-level BIO classification (O, B-MED, I-MED). Τα predicted spans linkάρονται σε ICD-10 codes μέσω mention priors (P(code|mention) από training) και λεξικά `full_dictionary.csv` / `icd10_greek_lookup.csv`. Document-level aggregation με majority voting mentions.

**Per-cluster απόδοση:**

| Cluster | micro F1 | precision | recall |
|---:|---:|---:|---:|
| 0 (είσοδος) | 0.708 | 0.658 | 0.766 |
| 1 (Tabs/καρδιο) | 0.771 | 0.783 | 0.761 |
| 2 (αγωγή) | 0.734 | 0.762 | 0.708 |
| 3 (εργαστηριακά) | 0.705 | 0.665 | 0.749 |
| 4 (αναμνηστικό) | 0.718 | 0.673 | 0.770 |
| 5 (πορίσματα) | 0.742 | 0.689 | 0.804 |
| 6 (ECHO) | 0.718 | 0.676 | 0.765 |
| 7 (ΜΕΘ/οξέα) | **0.784** | **0.716** | **0.865** |

**Αξιοσημείωτα:**
- **Cluster 1 (Tabs, F1=0.771):** Ενώ το IR αποτυγχάνει σε αυτό το cluster (F1=0.495), το NER-EL τα πηγαίνει εξαιρετικά. Τα structured tabs περιέχουν εμφανείς ιατρικές λέξεις-κλειδιά («φλεβοκομβικός», «περικαρδιακή») που εντοπίζονται ως mentions και linkάρονται σωστά.
- **Cluster 6 (ECHO, F1=0.718):** Χαμηλότερο σχετικά με το ότι τα ECHO reports περιέχουν πολλές «βαθμολογικές» περιγραφές (μικρού/μετρίου βαθμού αιμοδυναμική συνέπεια) χωρίς ακριβές mention-to-code match.
- **Clusters 0+3 (είσοδος/εργαστηριακά):** Για το `ner_el` παραμένουν σχετικά χαμηλότερα (0.705–0.708)· μετά την ανανέωση, το **`mlc_greek_bert`** πετυχαίνει **0.87–0.89** micro F1 στα ίδια clusters, άρα το κενό «NER vs MLC» σε εισαγωγικά/labs **αντιστράφηκε** για το Greek BERT MLC.

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

### 11.3 Cluster-level ευρήματα

- **Cluster 7 (ΜΕΘ/οξέα):** Ο «εύκολος» cluster για όλα τα συστήματα. Η ειδική ορολογία εγγυάται high-precision αντιστοίχιση.
- **Clusters 2, 3 (αγωγή/εργαστηριακά):** Παραμένουν σχετικά «θορυβώδη» κείμενα (φάρμακα, τιμές)· το **`mlc_greek_bert`** πλέον αποδίδει **0.88–0.91** micro F1, ενώ το **`xlm_r_large`** παραμένει χαμηλότερο (~0.75–0.77) — η διαφορά δεν είναι πλέον «όλα τα MLC αποτυγχάνουν εδώ».
- **Cluster 6 (ECHO):** Εξαιρετικός για MLC transformers (`mlc_greek_bert` ≈0.92, `xlm_r_large` ≈0.80) αλλά μέτριος για ner_el — αναδεικνύει τα όρια της mention-based προσέγγισης.

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
| Enrichment priors `ner_el` per cluster | Ειδικά για clusters 3 (εργαστηριακά) και 6 (ECHO): προσθήκη context-level patterns |
| Mention expansion IR corpus για cluster 1 | Mining καρδιολογικής ορολογίας (Tab-based terms) για ICD-10 code documents |

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

_Όλα τα αναφερόμενα νούμερα αναπαράγονται από: `outputs/analysis/*/metrics_engine.json`, `*/label_analysis.json`, `*/error_profiler.json`, `clustering/cluster_summary.json`, `outputs/experiments/information_retrieval/ir_tune_summary_hybrid.json`, και για το `mlc_greek_bert` επιπλέον `models/greek_bert/thresholds.json` / `val_scores.npy`._

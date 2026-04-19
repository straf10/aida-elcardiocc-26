# Medical Report Summary

## 1. Cross-Model Comparison

| Model | Micro-F1 (Group) | Micro-F1 (Flat) | Macro-F1 | Weighted-F1 | Recall@3 | Recall@5 |
|---|---|---|---|---|---|---|
| **xlm_r_large** | 0.7747 | 0.7308 | 0.2534 | 0.6711 | 0.4507 | 0.6417 |
| **mlc_greek_bert** | 0.8028 | 0.7939 | 0.3856 | 0.7723 | 0.4601 | 0.6671 |
| **xlm_r_base** | 0.8088 | 0.8192 | 0.4112 | 0.8066 | N/A | N/A |
| **information_retrieval** | 0.6913 | 0.6733 | 0.3919 | 0.6750 | N/A | N/A |
| **ner_el** | 0.7321 | 0.6774 | 0.3758 | 0.6362 | N/A | N/A |


### Long-tail (frequency bucket) comparison across models

![Model comparison by frequency bucket](../summary/models_comparison_buckets.png)

## 2. Per-Model Details

### xlm_r_large

#### Long-Tail Performance

| Bucket | N Labels | Support | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| frequent | 17 | 2604 | 0.8563 | 0.8729 |
| medium | 43 | 694 | 0.2903 | 0.4751 |
| rare | 55 | 221 | 0.1607 | 0.2345 |


#### Top 10 Confused Pairs

| Predicted (Wrong) | Missed (True) | Count |
|---|---|---|
| R07 | M54 | 8 |
| R07 | R10 | 8 |
| I50 | Y84 | 5 |
| R07 | C00-C97 | 5 |
| I50 | R60 | 5 |
| R07 | I10 | 5 |
| R07 | I11 | 5 |
| R07 | I46 | 4 |
| Z95 | I49 | 4 |
| I25 | I44 | 4 |


![Confusion Heatmap](../xlm_r_large/confusion_heatmap.png)

![Long Tail Analysis](../xlm_r_large/long_tail.png)

#### Keyword Hard Cases

| Keyword | N Docs | Mean FP | Mean FN | Worst PIDs |
|---|---|---|---|---|
| ιστορικό | 410 | 1.03 | 1.49 | 3169, 231, 5147, 1089, 1913 |
| διάγνωση | 486 | 1.02 | 1.49 | 3169, 231, 5147, 1089, 1913 |
| υπέρταση | 133 | 1.17 | 1.38 | 231, 7878, 6031, 8438, 9230 |
| στεφανιαία | 225 | 1.15 | 1.48 | 231, 1913, 7599, 2513, 6695 |
| έμφραγμα | 60 | 1.07 | 1.42 | 231, 5516, 9103, 3028, 6888 |
| αρρυθμία | 16 | 0.50 | 1.31 | 5090, 6316, 6839, 5397, 6834 |


#### Short vs Long Reports

Split Threshold: 1948.5 chars

| Split | N Docs | Micro-F1 | Precision | Recall |
|---|---|---|---|---|
| Short | 251 | 0.7773 | 0.8354 | 0.7267 |
| Long | 251 | 0.7727 | 0.7874 | 0.7585 |


---

### mlc_greek_bert

#### Long-Tail Performance

| Bucket | N Labels | Support | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| frequent | 17 | 2604 | 0.8437 | 0.8623 |
| medium | 43 | 694 | 0.4795 | 0.7376 |
| rare | 55 | 221 | 0.2746 | 0.3754 |


#### Top 10 Confused Pairs

| Predicted (Wrong) | Missed (True) | Count |
|---|---|---|
| I50 | I48 | 7 |
| I50 | I25 | 6 |
| I50 | Y84 | 5 |
| I50 | J81 | 5 |
| Z95 | R07 | 5 |
| Z95 | R06 | 4 |
| I50 | I49 | 4 |
| I21 | I44 | 4 |
| R53 | I49 | 4 |
| Z95 | I49 | 4 |


![Confusion Heatmap](../mlc_greek_bert/confusion_heatmap.png)

![Long Tail Analysis](../mlc_greek_bert/long_tail.png)

#### Keyword Hard Cases

| Keyword | N Docs | Mean FP | Mean FN | Worst PIDs |
|---|---|---|---|---|
| ιστορικό | 410 | 0.91 | 1.35 | 3169, 7257, 5644, 1913, 3028 |
| διάγνωση | 486 | 0.88 | 1.30 | 3169, 7257, 5644, 1913, 3028 |
| υπέρταση | 133 | 1.01 | 1.33 | 147, 231, 2493, 6464, 4004 |
| στεφανιαία | 225 | 1.02 | 1.33 | 1913, 3028, 147, 231, 2493 |
| έμφραγμα | 60 | 0.90 | 1.33 | 3028, 231, 9150, 5625, 245 |
| αρρυθμία | 16 | 0.75 | 1.00 | 2798, 6839, 6316, 5397, 5662 |


#### Short vs Long Reports

Split Threshold: 1948.5 chars

| Split | N Docs | Micro-F1 | Precision | Recall |
|---|---|---|---|---|
| Short | 251 | 0.8189 | 0.8363 | 0.8022 |
| Long | 251 | 0.7892 | 0.8324 | 0.7503 |


---

### xlm_r_base

#### Long-Tail Performance

| Bucket | N Labels | Support | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| frequent | 17 | 2604 | 0.8588 | 0.8774 |
| medium | 43 | 694 | 0.4803 | 0.7439 |
| rare | 55 | 221 | 0.3021 | 0.3854 |


#### Top 10 Confused Pairs

| Predicted (Wrong) | Missed (True) | Count |
|---|---|---|
| I49 | R00 | 5 |
| I22 | I50 | 5 |
| I50 | Y84 | 4 |
| I50 | I25 | 4 |
| I25 | I44 | 4 |
| Z95 | I48 | 4 |
| I50 | J81 | 4 |
| Z95 | I44 | 4 |
| I50 | I21 | 4 |
| Z95 | R07 | 4 |


![Confusion Heatmap](../xlm_r_base/confusion_heatmap.png)

![Long Tail Analysis](../xlm_r_base/long_tail.png)

#### Keyword Hard Cases

| Keyword | N Docs | Mean FP | Mean FN | Worst PIDs |
|---|---|---|---|---|
| ιστορικό | 410 | 1.18 | 1.13 | 3169, 5644, 3028, 6464, 3508 |
| διάγνωση | 486 | 1.12 | 1.09 | 3169, 5644, 3028, 6464, 3508 |
| υπέρταση | 133 | 1.14 | 1.05 | 6464, 4004, 231, 2814, 9230 |
| στεφανιαία | 225 | 1.33 | 1.04 | 3028, 6464, 6695, 9103, 9150 |
| έμφραγμα | 60 | 1.22 | 1.17 | 3028, 9103, 9150, 5516, 231 |
| αρρυθμία | 16 | 0.56 | 0.62 | 6316, 5397, 2798, 6839, 7400 |


#### Short vs Long Reports

Split Threshold: 1948.5 chars

| Split | N Docs | Micro-F1 | Precision | Recall |
|---|---|---|---|---|
| Short | 251 | 0.8224 | 0.8143 | 0.8306 |
| Long | 251 | 0.7975 | 0.8023 | 0.7927 |


---

### information_retrieval

#### Long-Tail Performance

| Bucket | N Labels | Support | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| frequent | 17 | 2604 | 0.7638 | 0.7680 |
| medium | 43 | 694 | 0.4319 | 0.6717 |
| rare | 55 | 221 | 0.4678 | 0.4791 |


#### Top 10 Confused Pairs

| Predicted (Wrong) | Missed (True) | Count |
|---|---|---|
| Z95 | I10 | 44 |
| Z95 | I11 | 44 |
| Y84 | I10 | 34 |
| Y84 | I11 | 34 |
| I25 | I10 | 33 |
| I25 | I11 | 33 |
| I35 | I10 | 19 |
| I35 | I11 | 19 |
| R53 | I10 | 19 |
| R53 | I11 | 19 |


![Confusion Heatmap](../information_retrieval/confusion_heatmap.png)

![Long Tail Analysis](../information_retrieval/long_tail.png)

#### Keyword Hard Cases

| Keyword | N Docs | Mean FP | Mean FN | Worst PIDs |
|---|---|---|---|---|
| ιστορικό | 410 | 2.99 | 1.17 | 3169, 8769, 5147, 2633, 5739 |
| διάγνωση | 486 | 3.03 | 1.12 | 3169, 8769, 9224, 5147, 2633 |
| υπέρταση | 133 | 3.49 | 0.74 | 8347, 895, 5076, 6464, 8450 |
| στεφανιαία | 225 | 3.14 | 1.05 | 9224, 2086, 9103, 9151, 9358 |
| έμφραγμα | 60 | 2.78 | 0.93 | 9224, 6904, 9103, 231, 8299 |
| αρρυθμία | 16 | 2.50 | 0.81 | 6316, 5662, 4939, 2090, 3525 |


#### Short vs Long Reports

Split Threshold: 1948.5 chars

| Split | N Docs | Micro-F1 | Precision | Recall |
|---|---|---|---|---|
| Short | 251 | 0.6907 | 0.6179 | 0.7829 |
| Long | 251 | 0.6917 | 0.5949 | 0.8262 |


---

### ner_el

#### Long-Tail Performance

| Bucket | N Labels | Support | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| frequent | 17 | 2604 | 0.7973 | 0.7972 |
| medium | 43 | 694 | 0.4211 | 0.6551 |
| rare | 55 | 221 | 0.4428 | 0.4706 |


#### Top 10 Confused Pairs

| Predicted (Wrong) | Missed (True) | Count |
|---|---|---|
| Z95 | I10 | 35 |
| Z95 | I11 | 35 |
| R53 | I10 | 19 |
| R53 | I11 | 19 |
| I25 | I10 | 18 |
| I25 | I11 | 18 |
| Y84 | I10 | 13 |
| Y84 | I11 | 13 |
| Z95 | I48 | 12 |
| R07 | I10 | 12 |


![Confusion Heatmap](../ner_el/confusion_heatmap.png)

![Long Tail Analysis](../ner_el/long_tail.png)

#### Keyword Hard Cases

| Keyword | N Docs | Mean FP | Mean FN | Worst PIDs |
|---|---|---|---|---|
| ιστορικό | 410 | 1.93 | 1.37 | 3169, 8769, 9122, 2633, 3709 |
| διάγνωση | 486 | 1.96 | 1.31 | 3169, 9224, 8769, 9122, 2633 |
| υπέρταση | 133 | 2.35 | 0.90 | 231, 5076, 8347, 6464, 895 |
| στεφανιαία | 225 | 2.26 | 1.25 | 9224, 9122, 9358, 2086, 231 |
| έμφραγμα | 60 | 2.10 | 1.22 | 9224, 231, 6904, 9103, 3028 |
| αρρυθμία | 16 | 1.56 | 1.00 | 8417, 6316, 3525, 5662, 4939 |


#### Short vs Long Reports

Split Threshold: 1948.5 chars

| Split | N Docs | Micro-F1 | Precision | Recall |
|---|---|---|---|---|
| Short | 251 | 0.7344 | 0.7211 | 0.7483 |
| Long | 251 | 0.7303 | 0.6765 | 0.7933 |


---

## 3. Medical Clusters (Global Validation Set)

| Cluster ID | Size | Mean Len | Top Terms |
|---|---|---|---|
| 0 | 33 | 2405 | ΕΞΕΤΕΤΑΣΗ, ασθενης, ΔΙΑΚΟΠΗ, ΕΠΙΣΥΝΑΠΤΟΝΤΑΙ, ΕΙΣΟΔΟΥ, ΑΙΤΙΑ, ΠΑΡΑΚΛΙΝΙΚΕΣ, θεράποντα, ενδείξεων, HR, Ro, COVID, ΑΨ, Καθημερινη, BP |
| 1 | 33 | 1855 | XX, Tab, XXXX, ΧΧ, αναμνηστικό, Επισυναπτεται, βαλβίδα, Παρούσα, κολπικής, Tabs, μαρμαρυγής, βαλβίδος, επηρεασμένη, Δεξιός, Εργαστηριακά |
| 2 | 53 | 1633 | πρωι, βραδυ, αγωγης, ΕΞΕΤΕΤΑΣΗ, Rο, μεσημερι, οξυ, Εως, ΦAΡΜΑΚΕΥΤΙΚΗ, απο, ασθενης, GLU, καλη, Επιμελης, Φερριτινη |
| 3 | 51 | 1559 | XX, Εισαγωγής, XXXX, απο, λεπτό, φάση, παρούσα, ομαλή, Στην, Tbs, υπήρξε, ασθενης, κοιλιας, Echo, αναμνηστικό |
| 4 | 44 | 2332 | δισκίο, ένα, ΚΕ, ΕΞΕΤΕΤΑΣΗ, BIL, Αναμνηστικό, ΕΞΕΤΑΣΗ, θώρακα, Νοσηλείας, Καρδιολογικό, υπερηχοκαρδιογραφικός, γνωμάτευσης, ΕΙΣΟΔΟΥ, ΑΙΤΙΑ, Εισαγωγής |
| 5 | 68 | 2013 | BIL, ΕΞΕΤΕΤΑΣΗ, Πορίσματα, FiO221, Φαρμ, ΠΑΡΑΚΛΙΝΙΚΕΣ, Dimers, Troponin, Tot, fT4, Prot, HCT, Glucose, Glob, δισκίο |
| 6 | 24 | 2214 | XX, δισκίο, ένα, Εισαγωγής, XXXX, Tbs, αυτής, λειτουργικότητα, εστάλη, επιμελή, λεπτό, Echo, ΚΕ, φυσιολογικές, βαλβίδα |
| 7 | 60 | 2262 | XX, Φυσική, Ψιθύρισμα, κλινικός, Σφ, Παρούσα, συνεχώς, αδιαλείπτως, καθημερινός, άλιπο, παράγοντες, μεταβολών, Αναπνευστικό, αναμνηστικό, Εργαστηριακά |
| 8 | 35 | 1375 | ΚΑΙ, 1X1, ΠΟΡΙΣΜΑ, ΕΠΙΣΥΝΑΠΤΕΤΑΙ, ΣΤΗΝ, ΜΕ, ΣΕ, ΕΞΕΤΕΤΑΣΗ, ΕΙΣΟΔΟΥ, ΑΙΤΙΑ, ΠΑΡΑΚΛΙΝΙΚΕΣ, ΕΠΙΣΥΝΑΠΤΟΝΤΑΙ, Φαρμ, ΚΔ, ΔΕΝ |
| 9 | 39 | 2000 | βαλβίδα, XX, Εισαγωγής, καλές, λειτουργικότητα, φυσιολογική, Echo, λεπτό, αυτής, φυσιολογικές, διαφυγή, Αριστερός, XXXX, Τριγλώχινα, Διαστολική |
| 10 | 25 | 2607 | cap, θεράποντα, εμφάνισης, ΕΞΕΤΕΤΑΣΗ, ΔΙΑΚΟΠΗ, ΑΨ, ΕΙΣΟΔΟΥ, ΑΙΤΙΑ, CoV2, Sars, ΕΠΙΣΥΝΑΠΤΟΝΤΑΙ, Ro, ΠΑΡΑΚΛΙΝΙΚΕΣ, Test, Rapid |
| 11 | 37 | 2225 | ΕΞΕΤΕΤΑΣΗ, Rο, 36, κι, ΦAΡΜΑΚΕΥΤΙΚΗ, πτωχή, SO2, Επικοινωνίας, Υπερηχος, αντιφλεγμονωδών, Καρδιάς, εξωτερικό, Σφύξεις, Επισυναπτονται, καρδιολογικό |


![Global Cluster Map](../clustering/cluster_map.png)

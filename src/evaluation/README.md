# Οδηγός Αξιολόγησης (Evaluation Guide) - ELCardioCC 2026

Αυτός ο φάκελος περιέχει τα εργαλεία αξιολόγησης για την ομάδα μας. Υλοποιεί τη λογική αξιολόγησης των διοργανωτών (document-level, list-of-lists για συνώνυμους κωδικούς). 

**Βασικός Κανόνας:** Κάθε εσωτερική λίστα στο `document_level_annotations` είναι μια κλινική οντότητα. Αν το σύστημά σας βρει *τουλάχιστον έναν* κωδικό από αυτή τη λίστα, μετράει ως True Positive (TP). Επιπλέον συνώνυμοι κωδικοί για την ίδια οντότητα ούτε προσθέτουν βαθμούς, ούτε τιμωρούνται ως False Positives (FP).

---

## 1. Τι αρχεία χρειάζομαι από την ομάδα

Για να δουλέψουν σωστά τα scripts, χρειάζομαι τα εξής από τα αντίστοιχα μέλη:

*   **Από τη Φρόσω (Data Prep):**
    *   `val.jsonl`: Το validation split μας (ground truth). Πρέπει να έχει ακριβώς την ίδια δομή με το `train_dataset.jsonl`.
    *   `label_names.json`: Μια απλή JSON λίστα με τους 115 μοναδικούς κωδικούς ICD-10 με τη σειρά που τους βλέπουν τα μοντέλα.
*   **Από Βασιλική & Δημήτρη (MLC Track):**
    *   `val_scores.npy`: Ένα numpy array με τα sigmoid scores του μοντέλου σας στο validation set (διαστάσεις: `N_docs x 115`).
    *   `val_patient_ids.json`: Μια JSON λίστα με τα `patient_id`s ακριβώς με τη σειρά που εμφανίζονται οι γραμμές στο `val_scores.npy`.
*   **Από Στέλιο (NER+EL) & Παναγιώτη (LLM):**
    *   `predictions.jsonl`: Το αρχείο με τις τελικές σας προβλέψεις σε μορφή submission (δηλαδή με `patient_id` και `document_level_annotations`).

---

## 2. Πώς να τρέξετε τα scripts

**ΠΡΟΣΟΧΗ:** Όλα τα scripts πρέπει να εκτελούνται από τον **κεντρικό φάκελο του project** (εκεί που βρίσκεται το `requirements.txt`) χρησιμοποιώντας το flag `-m`, ώστε να δουλεύουν σωστά τα imports.

### A. Αξιολόγηση ενός μεμονωμένου συστήματος (`evaluator.py`)
Υπολογίζει το Micro-F1, Precision, Recall και (προαιρετικά) τυπώνει τα λάθη ανά έγγραφο και τα στατιστικά ανά κλάση (Macro-F1).

```bash
python -m src.evaluation.evaluator \
    --ground-truth data/processed/val.jsonl \
    --pred predictions/my_model_preds.jsonl \
    --labels outputs/experiments/evaluation/label_names.json \
    --show-missing
```

### B. Εύρεση βέλτιστων κατωφλίων (`threshold_tune.py`)
Για τα μοντέλα MLC, βρίσκει το καλύτερο global threshold και μετά κάνει fine-tuning στο threshold της κάθε κλάσης (ICD-10 code) ξεχωριστά για να μεγιστοποιήσει το Micro-F1.

```bash
python -m src.evaluation.threshold_tune \
    --scores outputs/experiments/evaluation/val_scores.npy \
    --pids outputs/experiments/evaluation/val_patient_ids.json \
    --labels outputs/experiments/evaluation/label_names.json \
    --ground-truth data/processed/val.jsonl \
    --out outputs/experiments/evaluation/best_thresholds.json
```

---

## 3. Χρήση του `config.yaml` (Προαιρετικό αλλά βολικό)

Για να μην γράφετε όλα αυτά τα paths κάθε φορά στο terminal, μπορείτε να φτιάξετε ένα αρχείο `configs/run_v1.yaml` (δείτε το `src/evaluation/config.yaml` ως παράδειγμα) και να τρέχετε τα scripts απλά δίνοντας το config:

```bash
python -m src.evaluation.threshold_tune --config src/evaluation/config.yaml
```

Οποιοδήποτε flag περάσετε στο CLI (π.χ. `--pred-dir`) κάνει override την τιμή που υπάρχει μέσα στο YAML config.

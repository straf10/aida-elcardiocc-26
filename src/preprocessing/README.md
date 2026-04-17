# Data Cleaning & Preprocessing (`cleaning.py`)

## Τι κάνει το script
Το `cleaning.py` είναι το βασικό script για την προετοιμασία των δεδομένων εκπαίδευσης. Συγκεκριμένα:
1. Φορτώνει τα raw δεδομένα (JSONL) από το `data/raw/Train_Set_2026/train_dataset.jsonl`.
2. Καθαρίζει το κείμενο (`clean_text`), αφαιρώντας περιττούς χαρακτήρες και κρατώντας μόνο γράμματα (Λατινικά και Ελληνικά), αριθμούς και συγκεκριμένα σημεία στίξης.
3. Εξάγει τα labels. Διατηρεί την αρχική δομή λίστας από λίστες (`document_level_annotations`) για group-aware evaluation και δημιουργεί μια επίπεδη λίστα (`labels_flat`) για Multi-Label Classification (MLC).
4. Τρέχει βασικό Exploratory Data Analysis (EDA) και υπολογίζει τις συχνότητες των ICD-10 κωδικών.
5. Αποθηκεύει το ενιαίο `cleaned.jsonl` αρχείο στο `data/processed/`. (Το σπάσιμο σε train/val γίνεται πλέον από το module `training_validation`).

## Πώς να το τρέξετε
Για να εκτελέσετε το script, βεβαιωθείτε ότι βρίσκεστε στο root του project και έχετε εγκαταστήσει τα dependencies (`pip install -r requirements.txt`).

```bash
python src/preprocessing/cleaning.py
```

**Είσοδος:**
- `data/raw/Train_Set_2026/train_dataset.jsonl`

**Έξοδος (στο `data/processed/`):**
- `cleaned.jsonl`
- `icd10_frequencies.json`
- `icd10_frequencies.csv`

## Ποιος χρειάζεται τι
- **Froso / Data Team:** Αν αλλάξει η δομή ή η τοποθεσία του raw dataset, πρέπει να ενημερωθεί η μεταβλητή `DATA_PATH` μέσα στο `cleaning.py`.
- **Evaluation (Strafiotis):** Τα config YAML (`src/evaluation/config.yaml` και `src/evaluation/config.yaml`) πλέον δείχνουν στο `validation_set.jsonl`. Αν αλλάξει το όνομα του αρχείου εξόδου, πρέπει να ενημερωθούν και τα δύο configs.
- **Dictionary Baseline:** Χρησιμοποιεί το raw JSONL κατευθείαν και δεν εξαρτάται από τα processed αρχεία αυτού του script.
- **Modeling / MLC:** 
  - Χρησιμοποιήστε το πεδίο `labels_flat` για τυπικό Multi-Label Classification (π.χ. binary cross-entropy).
  - Χρησιμοποιήστε το πεδίο `document_level_annotations` για group-aware evaluation ή custom loss functions που λαμβάνουν υπόψη τα συνώνυμα.
# XLM-RoBERTa για Multi-Label Classification (ELCardioCC)

Όλος ο κώδικας pipeline για το XLM-R βρίσκεται στο πακέτο `src/xlm_r_large` (underscore — απαιτείται από την Python· δεν μπορεί να υπάρχει `xlm-r` ως όνομα module).

## Τι είναι το XLM-RoBERTa

Το XLM-RoBERTa-large είναι ένα πολυγλωσσικό μοντέλο Transformer με ~550 εκατομμύρια παραμέτρους. Έχει εκπαιδευτεί σε 100 γλώσσες, συμπεριλαμβανομένων των Ελληνικών, και χρησιμοποιεί τον SentencePiece tokenizer, κάνοντάς το ιδανικό για την επεξεργασία ιατρικών κειμένων με ανορθογραφίες ή συντομογραφίες.

## Πώς δουλεύει για MLC

Το μοντέλο προσαρμόζεται για Multi-Label Classification (MLC) προσθέτοντας ένα linear classification head πάνω από το `[CLS]` token. Η έξοδος είναι 115 logits (ένα για κάθε κωδικό ICD-10). Χρησιμοποιούμε τη συνάρτηση απώλειας `BCEWithLogitsLoss` (με `pos_weight` για την αντιμετώπιση του class imbalance) ή `Focal Loss`.

## Sliding Window

Επειδή τα εξιτήρια μπορεί να ξεπερνούν το όριο των 512 tokens του μοντέλου, υποστηρίζεται η χρήση **Sliding Window** (ενεργοποιημένο by default στο `src/xlm_r_large/xlm_r.yaml`):

- **Εκπαίδευση:** Αν ένα έγγραφο είναι μεγάλο, κόβεται σε επικαλυπτόμενα chunks (π.χ. μήκος 512, βήμα 256). Σε κάθε εποχή, επιλέγεται τυχαία ένα chunk για κάθε έγγραφο (data augmentation).
- **Inference:** Όλα τα chunks του εγγράφου περνούν από το μοντέλο. Τα τελικά logits προκύπτουν παίρνοντας το μέγιστο (max-pooling) για κάθε κλάση ανάμεσα σε όλα τα chunks. Αυτό αυξάνει το μέγεθος του batch στο validation, καθώς κάθε έγγραφο μπορεί να παράγει πολλαπλά chunks.

## Πώς να το τρέξω

1. **Εκπαίδευση:**

   ```bash
   python -m src.xlm_r_large.train --config src/xlm_r_large/xlm_r.yaml
   ```

   Προαιρετικά `--device cpu` ή `--device cuda`. Checkpoints και `val_scores.npy` στο `outputs/xlm_r_large/` (όπως στο YAML).

2. **Threshold Tuning (Βελτιστοποίηση Κατωφλίων):**

   ```bash
   python -m src.analysis.threshold_tune \
       --scores outputs/xlm_r_large/val_scores.npy \
       --pids outputs/xlm_r_large/val_patient_ids.json \
       --labels outputs/xlm_r_large/label_names.json \
       --ground-truth data/processed/validation_set.jsonl \
       --out outputs/xlm_r_large/thresholds.json
   ```

3. **Inference (Πρόβλεψη στο Test Set):**

   ```bash
   python -m src.xlm_r_large.predict --config src/xlm_r_large/xlm_r.yaml --split test --thresholds outputs/xlm_r_large/thresholds.json
   ```

   Επίσης υποστηρίζει `--device`.

4. **Αξιολόγηση (Validation):**

   ```bash
   python -m src.evaluation.evaluator \
       --ground-truth data/processed/validation_set.jsonl \
       --pred outputs/xlm_r_large/val_predictions.jsonl
   ```

## Πλάνο Πρόοδου (Progression Plan)

- **Run 1:** Baseline με `xlm-roberta-large`, lr=1e-5, max_length=512, 3 epochs. (Στόχος: F1 0.65-0.72)
- **Run 2:** Προσθήκη class weights (`pos_weight` BCE loss).
- **Run 3:** Μεγαλύτερη εκπαίδευση (10 epochs, warmup 10%, grad accum 4). (Στόχος: F1 >= 0.78)
- **Run 4:** Ενεργοποίηση `sliding_window: true` με `stride: 256`.
- **Run 5:** Δοκιμή με Focal Loss (`gamma=2.0`).
- **Run 6:** Εφαρμογή per-class thresholds. (Στόχος: F1 >= 0.82)

## Experiment Tracking (Weights & Biases)

Το pipeline υποστηρίζει πλήρη παρακολούθηση πειραμάτων μέσω του Weights & Biases (W&B).

1. **Αρχικό Setup (Μία φορά):**  
   Τρέξτε στο τερματικό:
   ```bash
   wandb login
   ```
   (Επικολλήστε το API key σας από το wandb.ai)

2. **Ενεργοποίηση/Απενεργοποίηση:**  
   Στο αρχείο `src/xlm_r_large/xlm_r.yaml`, αλλάξτε την τιμή:
   ```yaml
   wandb:
     enabled: true   # true για καταγραφή, false για τοπικό τρέξιμο χωρίς internet
     project: "elcardiocc-2026"
   ```

3. **Τι καταγράφεται αυτόματα:**
   - **Metrics:** Step loss, Epoch validation Micro-F1, Precision, Recall.
   - **Hyperparameters:** Όλο το config περνάει στο W&B αυτόματα.
   - **Per-Class Table:** Όταν βρίσκεται νέο best F1, καταγράφεται ένας διαδραστικός πίνακας με τα F1/Precision/Recall **ανά κωδικό ICD-10**, ώστε να βλέπουμε ποιοι κωδικοί αποτυγχάνουν.
   - **GPU Stats:** Χρήση μνήμης (αν τρέχετε σε CUDA).
   - **Artifacts:** Το checkpoint του καλύτερου μοντέλου ανεβαίνει στο W&B cloud (Model Registry).

## Δομή Αρχείων (XLM-R)

- `src/xlm_r_large/model.py`: HF sequence classification + `pos_weights`.
- `src/xlm_r_large/train.py`: Εκπαίδευση (AMP, warmup scheduler, validation).
- `src/xlm_r_large/predict.py`: Προβλέψεις / submission JSONL.
- `src/xlm_r_large/chunk_aggregate.py`: Max-pool chunks ανά ασθενή + sigmoid.
- `src/training_validation/device_utils.py`: Συσκευή και AMP μόνο σε CUDA (κοινό για εκπαίδευση).
- Το dataset splitting (train/val) γίνεται πλέον κεντρικά μέσω του `src/training_validation/`.
- `src/xlm_r_large/xlm_r.yaml`: Υπερπαράμετροι.

Κοινά με άλλα μοντέλα: `src/data/dataset.py`, `src/evaluation/*`, `src/training_validation/device_utils.py`.

## Σημείωση (Hardware)

Το `xlm-roberta-large` απαιτεί αρκετή μνήμη GPU (VRAM). Αν υπάρχει θέμα μνήμης (OOM), μειώστε το `batch_size` στο `src/xlm_r_large/xlm_r.yaml` (π.χ. σε 4 ή 2) και αυξήστε αντίστοιχα το `gradient_accumulation_steps` για να διατηρήσετε το ίδιο effective batch size.

**Σημείωση για GPU/CUDA:** Το σύστημα θα χρησιμοποιήσει αυτόματα CUDA αν είναι διαθέσιμο. Η χρήση μικτής ακρίβειας (fp16) ενεργοποιείται **μόνο** όταν τρέχετε σε CUDA GPU, για να αποφευχθούν σφάλματα σε CPU. Αν δείτε `Using device: cpu`, βεβαιωθείτε ότι έχετε εγκαταστήσει τη σωστή έκδοση του PyTorch με υποστήριξη CUDA.

<p align="center">
  <img src="assets/logo.jpeg" alt="Πανεπιστήμιο Μακεδονίας — AIDA" width="220"/>
</p>

<h1 align="center">ELCardioCC</h1>

Έργο στο πλαίσιο του shared task **BioASQ / CLEF (ELCardioCC)** από το μεταπτυχιακό πρόγραμμα **AIDA** — *AI and Data Analytics* — του [**Πανεπιστημίου Μακεδονίας**](https://www.uom.gr).

---

## Το πρόβλημα

Δίνεται **ελληνικό κείμενο εξιτηρίου** από καρδιολογική κλινική· ζητείται να προβλεφθούν οι σχετικοί **κωδικοί ICD-10** για κάθε έγγραφο. Πρόκειται για **πολυ-ετικέτα ταξινόμηση**: ένα έγγραφο μπορεί να έχει πολλές σωστές ετικέτες. Οι ετικέτες ομαδοποιούνται σε **κλινικές οντότητες** (λίστες συνώνυμων κωδικών)· η αξιολόγηση των διοργανωτών βασίζεται κυρίως στο **Micro-F1** με χαλαρή αντιστοίχιση μέσα σε κάθε ομάδα.

---

## Τεχνολογίες και τι έχει υλοποιηθεί

- **Γλώσσα & περιβάλλον:** Python 3.10+, εικονικά περιβάλλοντα, ρυθμίσεις μέσω YAML και `.env` όπου χρειάζεται.
- **Δεδομένα:** JSONL για εγγραφές· καθαρισμός και εξαγωγή ετικετών (`labels_flat`, `document_level_annotations`)· **πολυ-ετικέτα stratified split** (train/validation) με ευθυγράμμιση **ίδιων ασθενών** (`patient_id`) σε καθαρισμένα και raw κείμενα.
- **Βασική γραμμή (λεξικό):** αντιστοίχιση όρων–κωδικών από CSV, κανόνες και fuzzy matching (π.χ. FuzzyWuzzy / Levenshtein).
- **Βαθιά μάθηση για MLC:** **Hugging Face Transformers**, **PyTorch**, εκπαίδευση με mixed precision όπου υπάρχει CUDA· μοντέλα **Greek BERT** (`nlpaueb/bert-base-greek-uncased-v1`) και **XLM-RoBERTa** (base & large) με κεφαλή πολυ-ετικέτας, BCE με class weights, προαιρετικά focal loss· για μεγάλα κείμενα **sliding window** και συγχώνευση logits ανά έγγραφο (max pooling)· **K-fold** για το base track.
- **Information retrieval:** ανάκτηση κωδικών μέσω **BM25**, **TF-IDF**, **dense embeddings** (sentence-transformers) και **υβριδικές** συνδυαστικές στρατηγικές (π.χ. RRF).
- **NER & entity linking:** pipeline που συνδυάζει λεξικά/οντολογία με το κείμενο και παράγει προβλέψεις σε μορφή submission.
- **Αξιολόγηση:** υλοποίηση επίσημων μετρικών (micro precision/recall/F1)· **ρύθμιση κατωφλιών** (global και ανά κλάση) πάνω σε validation scores.
- **Πειράματα:** καταγραφή με **Weights & Biases** όπου είναι ενεργοποιημένο στα configs.

Για εγκατάσταση εξαρτήσεων: `pip install -r requirements.txt`. Λεπτομέρειες εκτέλεσης ανά υποσύστημα υπάρχουν στα `README.md` μέσα στους αντίστοιχους φακέλους του `src/`.

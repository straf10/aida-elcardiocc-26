# Training Validation

This module handles splitting the raw and cleaned data into consistent training and validation sets to ensure all models are compared fairly on the same exact patients.

It replaces the ad-hoc splitting that used to happen in `preprocessing/cleaning.py`.

## What it does

It takes two inputs:
1. `data/raw/Train_Set_2026/train_dataset.jsonl` (the raw dataset)
2. `data/processed/cleaned.jsonl` (produced by `src/preprocessing/cleaning.py`)

It performs a **Multilabel Stratified Shuffle Split** (with an 80/20 ratio by default) based on the `labels_flat` distribution found in the cleaned dataset. It also ensures that extremely rare or singleton labels are strictly placed into the training set to avoid impossible predictions during validation.

Then, it applies this exact split across *both* the raw dataset and the cleaned dataset, producing 4 output files in `data/processed/`:

- `training_set.jsonl` & `validation_set.jsonl` (the **cleaned** variant — used by transformers like XLM-R and Greek-BERT)
- `training_set_raw.jsonl` & `validation_set_raw.jsonl` (the **raw** variant — used by lexical matchers like Dictionary, NER-EL, and Information Retrieval, which benefit from keeping original text capitalization/accents).

## How to run it

After running the cleaning script, run the split module:

```bash
python -m src.training_validation --config src/training_validation/split.yaml
```

This will overwrite the files in `data/processed/` and also emit a `split_assignments.json` tracking the exact patient_ids placed in each split.

## Cross-validation support

For models that do their own cross-validation (like `xlm_r_base`), this package exposes a `make_kfold_splits()` utility function in `src/training_validation/split.py` to ensure the K-Fold process is standardized.

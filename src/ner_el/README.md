# NER + Entity Linking Pipeline (src/ner_el)

This folder contains the supervised BIO NER pipeline plus entity linking and document-level aggregation for ELCardioCC.

Important terminology:
- In this main pipeline, "dictionary" means the data resources:
  - data/external/full_dictionary.csv
  - data/external/icd10_greek_lookup.csv
- The file src/ner_el/dictionary_ner_el.py is a separate backup baseline and is not part of the main NER->EL pipeline path.

## What this pipeline does

1. Train a token classification model (BIO labels: O, B-MED, I-MED) from mention-level annotations.
2. Decode predicted mentions from token logits.
3. Link mentions to ICD-10 codes using mention priors + dictionary candidates.
4. Aggregate linked mentions to document-level predictions.
5. Evaluate both mention-level span quality and official document-level micro-F1.

## Runtime Autonomy

- Inference is artifact-first: the linker tries to load mention priors from
  `linker_prior.json` inside the model directory.
- `linker_prior.json` is now written during training into the `best` model folder.
- The best checkpoint is also exported to `outputs/models/NER_EL` for direct app inference.
- If the prior artifact is missing, you may provide `--train-path-for-linker` as an explicit fallback.

## Files

- train.py: train NER model
- predict.py: run end-to-end NER->EL inference
- evaluate.py: span-level + official document-level evaluation
- bio_dataset.py: tokenizer alignment to BIO labels
- decode.py: BIO decoding to mention spans
- linker.py: mention-to-code linker
- pipeline.py: orchestration for inference
- io_utils.py: loading/validation helpers
- config.py: CLI config definitions
- model.py: model factory

## Train

python3 -m src.ner_el.train \
  --train-path data/processed/training_set.jsonl \
  --val-path data/processed/validation_set.jsonl \
  --model-name nlpaueb/bert-base-greek-uncased-v1 \
  --output-dir outputs/experiments/ner_el/greek_bert_ner \
  --epochs 3

Training writes these inference artifacts under `outputs/experiments/ner_el/greek_bert_ner/best`:
- model/tokenizer files
- `linker_prior.json` (mention prior map for autonomous linker runtime)

The same best checkpoint is exported to `outputs/models/NER_EL` for production inference.

## Predict

python3 -m src.ner_el.predict \
  --model-dir outputs/models/NER_EL \
  --tokenizer-name nlpaueb/bert-base-greek-uncased-v1 \
  --input-path data/raw/Test_Set_2026/test_set.jsonl \
  --output-doc-path submissions/ner_el_main.jsonl \
  --output-debug-path outputs/experiments/ner_el/ner_el_main_debug.jsonl

Optional fallback if `linker_prior.json` is missing:

python3 -m src.ner_el.predict \
  --model-dir outputs/models/NER_EL \
  --tokenizer-name nlpaueb/bert-base-greek-uncased-v1 \
  --input-path data/raw/Test_Set_2026/test_set.jsonl \
  --train-path-for-linker data/processed/training_set.jsonl \
  --output-doc-path submissions/ner_el_main.jsonl \
  --output-debug-path outputs/experiments/ner_el/ner_el_main_debug.jsonl

## Main App Integration

Use the service class for in-process inference.

```python
from src.ner_el import NERELService
from src.ner_el.types import DocumentRecord

service = NERELService.from_model_dir(
  model_dir="outputs/models/NER_EL",
    tokenizer_name="nlpaueb/bert-base-greek-uncased-v1",
    use_dictionary_fusion=True,
    dictionary_doc_boost=False,
)

# Single document
single_out = service.predict_text(patient_id=101, text="...")

# Batch documents
docs = [
    DocumentRecord(patient_id=101, text="..."),
    DocumentRecord(patient_id=102, text="..."),
]
batch_out = service.predict_many(docs)
```

## Evaluate

python3 -m src.ner_el.evaluate \
  --ground-truth data/processed/validation_set.jsonl \
  --pred-doc submissions/ner_el_main_val.jsonl \
  --pred-debug outputs/experiments/ner_el/ner_el_main_val_debug.jsonl

## Notes

- The linker currently uses exact normalized mention priors and dictionary candidates.
- Dictionary candidates are loaded from data/external/full_dictionary.csv and data/external/icd10_greek_lookup.csv.
- Document-level output is emitted in competition-compatible JSONL format.
- For stronger performance, add confidence thresholds and code-specific post-processing in pipeline.py.

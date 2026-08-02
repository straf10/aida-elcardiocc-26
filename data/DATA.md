# Data Card

This project was built for the **BioASQ / CLEF ELCardioCC 2026** shared task: predicting
ICD-10 codes for Greek-language cardiology discharge summaries. The underlying clinical
text is **not included in this repository** — it was provided to registered task
participants under the organizers' data usage terms and cannot be redistributed.

This document describes the data schema, splits, and provenance so the pipeline is
reproducible by anyone who obtains the official dataset, without publishing patient text.

## Obtaining the data

The source dataset is distributed by the ELCardioCC/BioASQ task organizers to registered
participants. See the [BioASQ](http://bioasq.org/) shared task page for the current
edition and registration process. This repo does not mirror or host the dataset.

## Directory layout (not tracked in git)

```
data/
├── raw/
│   ├── train.jsonl / val.jsonl / test.jsonl / blind_test.jsonl   # gitignored, real patient text
│   ├── split_report.json    # tracked — split sizes & label coverage, no patient text
│   └── labelset.txt          # tracked — 115 target ICD-10 codes/groups
├── processed/
│   └── train.jsonl / val.jsonl / test.jsonl / all_data.jsonl     # gitignored, cleaned + flattened
└── external/
    ├── icd10_greek_lookup.csv          # tracked — code → Greek description lookup
    ├── full_dictionary.csv             # tracked — term → ICD-10 code(s) mapping (dictionary baseline)
    └── full_dictionary.train_only.csv  # tracked — same, restricted to training-derived terms
```

Once you have the official dataset, place the raw JSONL files under `data/raw/` using
the filenames above and run the split/cleaning scripts under `src/split_data/` to
populate `data/processed/`.

## Record schema

**`data/raw/*.jsonl`** — one JSON object per line:

```json
{
  "patient_id": "string",
  "text": "Greek-language discharge summary (free text)",
  "document_level_annotations": ["I21", "I25", "Z95", "..."]
}
```

**`data/processed/*.jsonl`** — same as above plus a flattened label field used by
training code:

```json
{
  "patient_id": "string",
  "text": "...",
  "document_level_annotations": ["I21", "I25"],
  "labels_flat": ["I21", "I25"]
}
```

## Splits

From `data/raw/split_report.json` (safe to keep — contains only counts, no patient text):

| Split | Documents |
|---|---|
| Train | 2000 |
| Validation | 250 |
| Test | 250 |
| **Total** | **2500** |

- 115 target ICD-10 codes/groups (`labelset.txt`); 110 present at least once in this data.
- Splitting used multi-label stratification (`iterative-stratification`) with forced
  placement rules for singleton/rare labels (e.g. codes with only 1–4 total instances
  are explicitly assigned across splits rather than left to random stratification) so
  every validation/test label also appears in training. See `split_report.json` for the
  exact per-label placement log.

## Labels

`data/raw/labelset.txt` lists the 115 target codes/groups, one per line (e.g. `C00-C97`,
`D47`, `D50-D64`). Evaluation is group-level: a gold group is satisfied by any one of its
member codes (see `report/sections/*.tex` for the full task definition).

## External lookup resources

- `icd10_greek_lookup.csv` — `code,greek_description`: human-readable Greek label for
  each ICD-10 code/group, used for semantic anchoring (XLM-RoBERTa) and error analysis.
- `full_dictionary.csv` / `full_dictionary.train_only.csv` — `term,codes_pipe_sep`:
  surface-form → code mapping mined from the training set, used by the Aho-Corasick
  dictionary baseline and NER+EL entity linking. These contain short clinical terms/abbreviations,
  not full documents, and were reviewed to exclude patient-identifying content.

## Predictions and model outputs

`outputs/predictions/**` and `outputs/models/**` contain only `patient_id` +
predicted ICD-10 codes (no source text), and are safe to keep versioned for
reproducing the paper's reported results.

# Project memory: micro-F1 upgrade (XLM-R large + NER-EL)

*Last updated: 2026-04-23*

This file records the substantive codebase changes aimed at **group-level micro-F1** for ELCardioCC, aligned with the project plan: XLM-R large (ZLPR + mean–max chunk pooling) and NER+EL (partial CRF + context-aware entity linking). **ICD-10 hierarchy post-processing was skipped** in this round (kept only existing opt-in behavior).

---

## XLM-R large (`src/xlm_r_large/`)

### W&B: pick best run → baseline YAML (one-shot script)

- Added `pick_best_wandb_run.py` (temporary, **self-deleting** after a successful run):
  - Uses `split_data.dotenv_util.load_dotenv_if_present` (`.env` with `WANDB_API_KEY`).
  - Queries W&B (defaults: entity/project from `xlm_r.yaml`).
  - Picks the best run by `summary["best_val_micro_f1"]`, tie-breaks with a small **overfitting gap** from run history.
  - Rewrites **only** `data:` and `training:` in `xlm_r.yaml` from the selected run’s `config`.
  - Writes `outputs/experiments/xlm_r_large/best_run_snapshot.json` for traceability.
  - Deletes itself on success (`Path(__file__).unlink()`).

**Run (from repo root, with `src` on `PYTHONPATH`):**

```text
$env:PYTHONPATH='src'
python -m xlm_r_large.pick_best_wandb_run
```

### Removed Optuna / tuning trial stack

- Deleted: `tune.py`, `tune.yaml`, `tuning_utils.py`.
- `train.py`: removed `optuna_trial` plumbing; inlined W&B no-upload settings; simplified run naming (no trial-specific names).

### Loss: ZLPR

- `train.py`: added `_zlpr_loss` and `training.loss: "zlpr"` branch.
- `xlm_r.yaml`: `training.loss: "zlpr"`, `training.auto_threshold_tuning: false`, `training.eval_thresholds: [0.50]`.
- `.cursorrules`: when using ZLPR, skip `src/evaluation/threshold_tune.py` (zero-bounded decision: `logit > 0` ⟺ `sigmoid > 0.5`).

### Chunk aggregation: mean–max

- `chunk_aggregate.py`: new strategy `mean_max` with explicit `alpha`.
- `xlm_r.yaml`: `data.chunk_aggregation: "mean_max"`, `data.chunk_aggregation_alpha: 0.5`.
- `train.py` / `predict.py`: read these keys instead of hard-coding `max` pooling.

### Inference thresholds

- `predict.py`: if the thresholds JSON is missing, falls back to a **global** `training.eval_threshold` vector (enables ZLPR workflow without a tuned per-class JSON).
- `predict.py` CLI help: avoid Unicode arrows in help strings (Windows `cp1253` help printing).

### Hierarchy propagation

- **Not implemented** in this round beyond existing `postprocess.py` and optional `--apply-parent-child` in `predict.py`.

### Throughput / memory quick wins (math-equivalent)

These changes aim for faster training and inference **without** altering loss, thresholds, label space, or aggregation math (micro-F1 expected unchanged within seed noise).

- **`model.py`**: Load with `attn_implementation="sdpa"` when supported; `TypeError`/`ValueError` fallback to default attention.
- **`train.py`**: `_unwrap_for_save(model).save_pretrained(...)` so `torch.compile` wrappers still save a normal HF checkpoint; BCE `pos_weight` uses `train_dataset.records` (no second `load_jsonl` of the train JSONL). Training loss accumulated on device with postfix updates on optimizer steps; validation uses `torch.inference_mode()`, tensor sum for val loss, `torch.cat` of val logits + patient ids, then **`aggregate_scores_by_patient_torch`** (see below).
- **`predict.py`**: Same batched logits + `aggregate_scores_by_patient_torch`; `os.makedirs` for each of `scores_path`, `patient_ids_path`, and `label_names_path` when exporting val artifacts.
- **`chunk_aggregate.py`**: New **`aggregate_scores_by_patient_torch`** (`scatter_reduce_` / `scatter_add_` for max/mean/mean_max on the logits’ device); empty-chunk guard; legacy **`aggregate_scores_by_patient`** kept (e.g. `logsumexp` fallback path).
- **`preprocessing/dataset.py` (`ELCardioDataset`)**: One batched tokenizer pass over all texts; **packed** per-chunk `input_ids` / `attention_mask` / labels / ids in dense tensors; `__getitem__` still returns the same dict keys (`input_ids`, `attention_mask`, `labels`, `patient_id`, `doc_idx`, optional `groups`).
- **`xlm_r.yaml`**: Comment only next to `training.num_workers` suggesting a Windows benchmark (0 vs 2 vs 4); default **`4` unchanged**.

---

## NER+EL (`src/ner_el/`)

### Partial CRF (unlabeled-entity / partial-annotation)

- New `partial_crf.py`: linear-chain CRF with **partial allow-sets** per token (`allow_mask`), forward NLL, Viterbi decode.
- `bio_dataset.py`:
  - Detects “partial” documents (default: `mention_level_annotations` empty but `labels_flat` non-empty); optional `partial_all` to widen behavior.
  - For partial docs, **ambiguous** real tokens get label `-100` and `allow_mask` all tags; gold spans still one-hot; padding stays masked out.
- `model.py`: `TokenClassifierWithCRF` wraps HF token classifier + CRF; saves CRF head as `partial_crf.pt` next to the HF save.
- `train.py`: `PartialLabelCollator`, CRF loss path, CRF-based decoding in metrics/final pass; `remove_unused_columns=False` so `allow_mask` is preserved.
- `decode.py`: `decode_mentions_from_paths(...)` for Viterbi tag sequences.
- `pipeline.py` / `service.py`: decode path uses CRF when present.

### Context-aware entity linking (re-ranker)

- New `context_reranker.py`: encodes local mention windows vs ICD-10 code descriptions; caches `reranker_code_embeddings.npy` + `reranker_meta.json` under `linker.artifact_dir`.
- `linker.py`: optional fusion `alpha * prior_softmax + (1-alpha) * cosine_semantic` when `reranker` and `context_text` are set.
- `train.py` / `service.py`: build or load reranker artifacts; train passes `context_text=doc.text` into `link_mentions`.

### Config: YAML + CLI overrides

- `ner_el.yaml` consolidates model/data/training/linker/predict/output.
- `config.py`: `--config` for train/predict; CLI overrides win over YAML; flags for partial CRF and reranker.

**Typical commands (from repo root):**

```text
$env:PYTHONPATH='src'
python -m ner_el.train --config src/ner_el/ner_el.yaml
python -m ner_el.predict --config src/ner_el/ner_el.yaml
```

---

## Environment / import layout

- Training modules expect `src` on `PYTHONPATH` (e.g. `$env:PYTHONPATH='src'` on Windows) so top-level packages like `evaluation`, `dictionary` resolve.
- Entrypoint scripts under `src/` (e.g. `xlm_r_large/train.py`, `ner_el/train.py`, `mlc_greek_bert/train.py`) also prepend `src/` to `sys.path` at import time, so running them as `python src/.../train.py` from the repo root works without setting `PYTHONPATH`. The `python -m package.module` pattern with `PYTHONPATH=src` remains the preferred/clean approach.
- `ner_el` dataclasses live in `schemas.py` (renamed from `types.py` so the stdlib `types` module is not shadowed when the script’s directory is on `sys.path`). Imports use `from ner_el.schemas import ...` inside the package and absolute `ner_el.*` imports in the train/predict entrypoints.
- `sentence-transformers` is used by the reranker (same pattern as `information_retrieval/embedding_retrieval.py`); ensure it is installed in the venv you use for NER+EL.

---

## Validation that was run in-repo

- `python -m compileall` on modified packages (including `src/xlm_r_large` and `src/preprocessing` after the throughput pass).
- CLI `--help` smoke for `xlm_r_large.train|predict` and `ner_el.train|predict` (as scripts and/or with `PYTHONPATH=src`).
- Grep: no `optuna` / `tuning_utils` / `xlm_r_large.tune` references under `src/`.

Full end-to-end training (GPU/time) and empiric micro-F1 comparison to prior baselines is left to the user’s training runs.

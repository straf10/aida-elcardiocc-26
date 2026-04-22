from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from transformers import AutoConfig, AutoModelForTokenClassification

from .bio_dataset import ID2LABEL, LABEL2ID


_LAYERNORM_LEGACY_SUFFIXES = (
    ("LayerNorm.gamma", "LayerNorm.weight"),
    ("LayerNorm.beta", "LayerNorm.bias"),
)


def _remap_legacy_layernorm_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename legacy BERT LayerNorm params (``gamma``/``beta``) to the modern
    PyTorch convention (``weight``/``bias``).

    Older Greek BERT checkpoints (e.g. ``nlpaueb/bert-base-greek-uncased-v1``)
    ship with the legacy naming. ``transformers>=5`` no longer auto-remaps
    these, which silently leaves LayerNorm parameters randomly initialized and
    significantly degrades fine-tuning quality.
    """
    remapped: Dict[str, torch.Tensor] = {}
    touched = False
    for key, value in state_dict.items():
        new_key = key
        for legacy, modern in _LAYERNORM_LEGACY_SUFFIXES:
            if legacy in new_key:
                new_key = new_key.replace(legacy, modern)
                touched = True
        remapped[new_key] = value
    if touched:
        print("[ner_el.model] Remapped legacy LayerNorm keys (gamma/beta -> weight/bias).")
    return remapped


def _load_pretrained_state_dict(model_name: str) -> Optional[Dict[str, torch.Tensor]]:
    """Best-effort fetch of a pretrained checkpoint state dict, supporting both
    ``safetensors`` and ``pytorch_model.bin`` artifacts. Returns ``None`` if no
    artifact could be resolved (e.g. offline and not cached)."""

    def _try_local_file(filename: str) -> Optional[str]:
        if os.path.isdir(model_name):
            candidate = os.path.join(model_name, filename)
            return candidate if os.path.exists(candidate) else None
        return None

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        return None

    for filename in ("model.safetensors", "pytorch_model.bin"):
        local = _try_local_file(filename)
        if local is None:
            try:
                local = hf_hub_download(model_name, filename)
            except (HfHubHTTPError, FileNotFoundError, OSError):
                continue
            except Exception:
                continue
        if filename.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError:
                continue
            return load_file(local)
        return torch.load(local, map_location="cpu", weights_only=True)
    return None


def build_ner_model(model_name: str):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    state_dict = _load_pretrained_state_dict(model_name)
    if state_dict is None:
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
        )

    remapped = _remap_legacy_layernorm_keys(state_dict)
    # transformers>=5 forbids passing `state_dict` together with a repo id, so
    # we instantiate from config and load the remapped weights explicitly. The
    # classifier head has no counterpart in the MLM checkpoint and stays on its
    # fresh random initialization (identical to the default fine-tuning flow).
    model = AutoModelForTokenClassification.from_config(config)
    load_result = model.load_state_dict(remapped, strict=False)

    def _is_classifier_key(key: str) -> bool:
        return key.startswith("classifier") or key.endswith(("classifier.weight", "classifier.bias"))

    non_classifier_missing = [k for k in load_result.missing_keys if not _is_classifier_key(k)]
    if non_classifier_missing:
        print(
            f"[ner_el.model] WARNING: {len(non_classifier_missing)} encoder params were not "
            f"found in the pretrained checkpoint (sample: {non_classifier_missing[:3]})."
        )
    return model

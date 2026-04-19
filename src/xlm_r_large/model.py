import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

DESC_RESIDUAL_STATE_FILENAME = "desc_residual_state.pt"
DESC_RESIDUAL_META_FILENAME = "desc_residual_meta.json"


class MultiSampleDropoutHead(nn.Module):
    """
    RoBERTa-style classification head with multi-sample dropout.
    Keeps dense/out_proj parameter names compatible with standard checkpoints.
    """

    def __init__(self, hidden_size, num_labels, dropout_rate=0.3, num_samples=5):
        super().__init__()
        self.num_samples = max(1, int(num_samples))
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(self.num_samples)]
        )
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        # HF sequence classifiers pass full sequence output [B, T, H].
        if features.dim() == 3:
            x = features[:, 0, :]  # CLS token
        else:
            x = features
        x = torch.tanh(self.dense(x))
        if not self.training:
            return self.out_proj(x)
        logits = torch.stack([self.out_proj(dp(x)) for dp in self.dropouts], dim=0)
        return logits.mean(dim=0)


class XLMRMeanPoolClassifier(nn.Module):
    """
    Mean-pooling classifier wrapper for XLM-R style backbones.
    """

    HEAD_STATE_FILENAME = "mean_pool_head.pt"
    MODEL_META_FILENAME = "custom_model_meta.json"

    def __init__(
        self,
        backbone,
        num_labels,
        dropout=0.3,
        multi_sample_dropout_samples=1,
    ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.num_labels = int(num_labels)
        self.config = self.backbone.config
        self.config.num_labels = self.num_labels
        self.config.problem_type = "multi_label_classification"
        self.config.pooling_strategy = "mean"
        self.config.classifier_dropout = float(dropout)
        self.config.multi_sample_dropout_samples = int(multi_sample_dropout_samples)

        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)
        self.ms_dropouts = nn.ModuleList(
            [
                nn.Dropout(dropout)
                for _ in range(max(1, int(multi_sample_dropout_samples)))
            ]
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        denom = torch.clamp(mask_expanded.sum(dim=1), min=1e-8)
        pooled = (token_embeddings * mask_expanded).sum(dim=1) / denom
        if not self.training:
            logits = self.classifier(pooled)
        else:
            logits = torch.stack(
                [self.classifier(dp(self.dropout(pooled))) for dp in self.ms_dropouts],
                dim=0,
            ).mean(dim=0)
        return SequenceClassifierOutput(logits=logits)

    def save_pretrained(self, save_directory):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(save_dir))
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "dropout_p": self.dropout.p,
                "num_labels": self.num_labels,
                "multi_sample_dropout_samples": len(self.ms_dropouts),
            },
            save_dir / self.HEAD_STATE_FILENAME,
        )
        with open(save_dir / self.MODEL_META_FILENAME, "w", encoding="utf-8") as handle:
            json.dump({"pooling_strategy": "mean"}, handle, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        num_labels=115,
        classifier_dropout=0.3,
        multi_sample_dropout_samples=1,
        local_files_only=False,
    ):
        backbone = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        model = cls(
            backbone=backbone,
            num_labels=num_labels,
            dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
        )
        head_path = Path(model_name_or_path) / cls.HEAD_STATE_FILENAME
        if head_path.exists():
            payload = torch.load(head_path, map_location="cpu")
            model.classifier.load_state_dict(payload.get("classifier", {}))
        return model


def load_label_descriptions_from_csv(csv_path: str, label_order: list) -> list[str]:
    """Map labelset order to Greek description strings from ICD-10 lookup CSV."""
    desc_by_code: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row.get("code", "").strip()
            desc = row.get("greek_description", row.get("description", "")).strip()
            if code:
                desc_by_code[code] = desc if desc else code
    return [desc_by_code.get(str(lab), str(lab)) for lab in label_order]


@torch.no_grad()
def encode_descriptions(
    model_name: str,
    descriptions: list[str],
    tokenizer,
    device: torch.device,
    max_len: int = 64,
) -> torch.Tensor:
    """
    CLS-encode each description with a freshly loaded pretrained backbone.
    Returns L2-normalized tensor [num_labels, hidden].
    """
    enc_model = AutoModel.from_pretrained(model_name)
    enc_model.eval()
    enc_model.to(device)
    hidden_size = enc_model.config.hidden_size
    out = torch.empty(len(descriptions), hidden_size, device=device)
    for i, text in enumerate(descriptions):
        batch = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        h = enc_model(**batch).last_hidden_state[:, 0, :]
        out[i] = h.squeeze(0)
    del enc_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return F.normalize(out, dim=-1)


def _classifier_out_proj(module: nn.Module):
    """Return the final Linear mapping hidden -> num_labels if present."""
    if hasattr(module, "out_proj") and isinstance(module.out_proj, nn.Linear):
        return module.out_proj
    if hasattr(module, "classifier"):
        c = module.classifier
        if hasattr(c, "out_proj") and isinstance(c.out_proj, nn.Linear):
            return c.out_proj
    return None


@torch.no_grad()
def apply_description_init_to_classifier(
    model: nn.Module,
    desc_emb: torch.Tensor,
    scale: float = 0.02,
) -> None:
    """Initialize final classifier weights with scaled per-label description embeddings."""
    out_proj = _classifier_out_proj(model)
    if out_proj is not None:
        out_proj.weight.copy_((scale * desc_emb).to(out_proj.weight.dtype))
        if out_proj.bias is not None:
            out_proj.bias.zero_()
        return
    if isinstance(model, XLMRMeanPoolClassifier):
        model.classifier.weight.copy_((scale * desc_emb).to(model.classifier.weight.dtype))
        if model.classifier.bias is not None:
            model.classifier.bias.zero_()


class DescResidualWrapper(nn.Module):
    """
    Adds logits += alpha * (normalize(pooled_doc) @ desc_emb.T).
    desc_emb rows are L2-normalized per-label description vectors (buffer).
    """

    def __init__(self, inner: nn.Module, desc_emb: torch.Tensor, alpha_init: float = 0.1):
        super().__init__()
        self.inner = inner
        self.register_buffer("desc_emb", desc_emb.clone())
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if isinstance(self.inner, XLMRMeanPoolClassifier):
            outputs = self.inner.backbone(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            denom = torch.clamp(mask_expanded.sum(dim=1), min=1e-8)
            pooled = (token_embeddings * mask_expanded).sum(dim=1) / denom
            inner_mod = self.inner
            if not self.training:
                logits = inner_mod.classifier(pooled)
            else:
                logits = torch.stack(
                    [
                        inner_mod.classifier(dp(inner_mod.dropout(pooled)))
                        for dp in inner_mod.ms_dropouts
                    ],
                    dim=0,
                ).mean(dim=0)
            h = pooled
        else:
            roberta_out = self.inner.roberta(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            sequence_output = roberta_out.last_hidden_state
            logits = self.inner.classifier(sequence_output)
            h = sequence_output[:, 0, :]

        sim = F.normalize(h, dim=-1) @ self.desc_emb.T
        logits = logits + self.alpha * sim
        return SequenceClassifierOutput(logits=logits)

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.inner.save_pretrained(str(save_dir))
        torch.save(
            {
                "alpha": self.alpha.detach().cpu(),
                "desc_emb": self.desc_emb.detach().cpu(),
            },
            save_dir / DESC_RESIDUAL_STATE_FILENAME,
        )
        with open(save_dir / DESC_RESIDUAL_META_FILENAME, "w", encoding="utf-8") as handle:
            json.dump({"desc_residual": True}, handle, indent=2)


@torch.no_grad()
def rebake_description_embeddings(
    wrapper: DescResidualWrapper,
    descriptions: list[str],
    tokenizer,
    device: torch.device,
    max_len: int = 64,
) -> None:
    """Re-encode descriptions with the wrapper's current backbone (e.g. SWA weights)."""
    if isinstance(wrapper.inner, XLMRMeanPoolClassifier):
        backbone = wrapper.inner.backbone
    else:
        backbone = wrapper.inner.roberta
    backbone.eval()
    hidden_size = backbone.config.hidden_size
    out = torch.empty(len(descriptions), hidden_size, device=device)
    for i, text in enumerate(descriptions):
        batch = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        h = backbone(**batch).last_hidden_state[:, 0, :]
        out[i] = h.squeeze(0)
    wrapper.desc_emb.copy_(F.normalize(out, dim=-1))


def _build_hf_sequence_classifier(
    model_name,
    num_labels,
    classifier_dropout=0.3,
    multi_sample_dropout_samples=1,
    local_files_only=False,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        classifier_dropout=classifier_dropout,
        local_files_only=local_files_only,
    )
    if int(multi_sample_dropout_samples) > 1:
        original_head = model.classifier
        hidden_size = model.config.hidden_size
        new_head = MultiSampleDropoutHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout_rate=classifier_dropout,
            num_samples=multi_sample_dropout_samples,
        )
        if hasattr(original_head, "dense") and hasattr(original_head, "out_proj"):
            new_head.dense.load_state_dict(original_head.dense.state_dict())
            new_head.out_proj.load_state_dict(original_head.out_proj.state_dict())
        model.classifier = new_head
    model.config.pooling_strategy = "cls"
    model.config.multi_sample_dropout_samples = int(multi_sample_dropout_samples)
    return model


def build_model(
    num_labels=115,
    model_name="xlm-roberta-large",
    classifier_dropout=0.3,
    pooling_strategy="cls",
    multi_sample_dropout_samples=1,
    init_classifier_from_descriptions: bool = False,
    label_descriptions: list | None = None,
    desc_init_scale: float = 0.02,
    use_desc_residual: bool = False,
    desc_residual_alpha_init: float = 0.1,
):
    if pooling_strategy == "mean":
        backbone = AutoModel.from_pretrained(model_name)
        inner = XLMRMeanPoolClassifier(
            backbone=backbone,
            num_labels=num_labels,
            dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
        )
    else:
        inner = _build_hf_sequence_classifier(
            model_name=model_name,
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
        )

    desc_emb = None
    if init_classifier_from_descriptions or use_desc_residual:
        if not label_descriptions:
            raise ValueError("label_descriptions required for description init or residual head.")
        # Encode on CPU to avoid doubling GPU memory during startup; caller may move model after.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = __import__(
            "transformers", fromlist=["AutoTokenizer"]
        ).AutoTokenizer.from_pretrained(model_name)
        desc_emb = encode_descriptions(
            model_name, label_descriptions, tokenizer, device=device
        )

    if init_classifier_from_descriptions and desc_emb is not None:
        apply_description_init_to_classifier(inner, desc_emb.cpu(), scale=desc_init_scale)

    if use_desc_residual:
        if desc_emb is None:
            raise ValueError("use_desc_residual requires encoded label_descriptions.")
        return DescResidualWrapper(
            inner,
            desc_emb.cpu(),
            alpha_init=desc_residual_alpha_init,
        )

    return inner


def compute_pos_weights(labelset, frequencies_path, num_train_docs):
    with open(frequencies_path, "r", encoding="utf-8") as f:
        freqs = json.load(f)

    freq_dict = {code: count for code, count in freqs}

    pos_weights = torch.zeros(len(labelset), dtype=torch.float)
    MAX_POS_WEIGHT = 50.0
    for i, code in enumerate(labelset):
        pos_count = freq_dict.get(code, 0)
        if pos_count > 0:
            pos_weights[i] = min((num_train_docs - pos_count) / pos_count, MAX_POS_WEIGHT)
        else:
            pos_weights[i] = 1.0

    return pos_weights


def _load_inner_for_inference(
    checkpoint_dir,
    num_labels,
    classifier_dropout,
    multi_sample_dropout_samples,
    local_files_only,
):
    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=local_files_only)
    pooling_strategy = getattr(config, "pooling_strategy", "cls")
    if pooling_strategy == "mean":
        return XLMRMeanPoolClassifier.from_pretrained(
            checkpoint_dir,
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
            local_files_only=local_files_only,
        )
    return _build_hf_sequence_classifier(
        model_name=checkpoint_dir,
        num_labels=num_labels,
        classifier_dropout=classifier_dropout,
        multi_sample_dropout_samples=multi_sample_dropout_samples,
        local_files_only=local_files_only,
    )


def load_model_for_inference(checkpoint_dir, num_labels=115, local_files_only=True):
    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=local_files_only)
    pooling_strategy = getattr(config, "pooling_strategy", "cls")
    classifier_dropout = float(getattr(config, "classifier_dropout", 0.3))
    multi_sample_dropout_samples = int(
        getattr(config, "multi_sample_dropout_samples", 1)
    )

    ckpt = Path(checkpoint_dir)
    meta_path = ckpt / DESC_RESIDUAL_META_FILENAME
    state_path = ckpt / DESC_RESIDUAL_STATE_FILENAME
    if meta_path.is_file() and state_path.is_file():
        inner = _load_inner_for_inference(
            checkpoint_dir,
            num_labels,
            classifier_dropout,
            multi_sample_dropout_samples,
            local_files_only,
        )
        payload = torch.load(state_path, map_location="cpu")
        desc_emb = payload["desc_emb"]
        wrap = DescResidualWrapper(inner, desc_emb, alpha_init=0.1)
        alpha_val = payload.get("alpha", 0.1)
        if isinstance(alpha_val, torch.Tensor):
            wrap.alpha.data.copy_(alpha_val.reshape_as(wrap.alpha.data))
        else:
            wrap.alpha.data.fill_(float(alpha_val))
        return wrap

    if pooling_strategy == "mean":
        return XLMRMeanPoolClassifier.from_pretrained(
            checkpoint_dir,
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            multi_sample_dropout_samples=multi_sample_dropout_samples,
            local_files_only=local_files_only,
        )
    return _build_hf_sequence_classifier(
        model_name=checkpoint_dir,
        num_labels=num_labels,
        classifier_dropout=classifier_dropout,
        multi_sample_dropout_samples=multi_sample_dropout_samples,
        local_files_only=local_files_only,
    )

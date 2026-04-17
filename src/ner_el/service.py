from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification

from .config import PredictConfig
from .dictionary_features import load_dictionary_candidates
from .io_utils import load_documents
from .linker import (
    MentionLinker,
    build_prior_map,
    default_prior_artifact_path,
    load_prior_map,
)
from .pipeline import NERELPipeline, PipelineOutput
from .types import DocumentRecord


def _load_prior_map_for_runtime(model_dir: str, train_path_for_linker: Optional[str] = None):
    prior_path = default_prior_artifact_path(model_dir)
    if Path(prior_path).exists():
        return load_prior_map(prior_path), prior_path

    if train_path_for_linker:
        train_docs = load_documents(train_path_for_linker)
        return build_prior_map(train_docs), train_path_for_linker

    raise FileNotFoundError(
        "Missing linker prior artifact and no fallback train path was provided. "
        f"Expected artifact: {prior_path}"
    )


class NERELService:
    def __init__(self, pipeline: NERELPipeline):
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, cfg: PredictConfig) -> "NERELService":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = AutoModelForTokenClassification.from_pretrained(cfg.model_dir)

        prior_map, prior_source = _load_prior_map_for_runtime(
            model_dir=cfg.model_dir,
            train_path_for_linker=cfg.train_path_for_linker,
        )
        dict_map = load_dictionary_candidates()
        linker = MentionLinker(prior_map=prior_map, dictionary_map=dict_map)

        pipeline = NERELPipeline(
            model=model,
            tokenizer_name=cfg.tokenizer_name,
            linker=linker,
            max_length=cfg.max_length,
            dictionary_map=dict_map,
            use_dictionary_fusion=cfg.use_dictionary_fusion,
            dictionary_doc_boost=cfg.dictionary_doc_boost,
            device=device,
        )

        print(f"Loaded linker priors from: {prior_source}")
        return cls(pipeline=pipeline)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        tokenizer_name: str = PredictConfig.tokenizer_name,
        max_length: int = PredictConfig.max_length,
        use_dictionary_fusion: bool = True,
        dictionary_doc_boost: bool = True,
        train_path_for_linker: Optional[str] = None,
    ) -> "NERELService":
        cfg = PredictConfig(
            model_dir=model_dir,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            use_dictionary_fusion=use_dictionary_fusion,
            dictionary_doc_boost=dictionary_doc_boost,
            train_path_for_linker=train_path_for_linker,
        )
        return cls.from_config(cfg)

    def predict_document(self, doc: DocumentRecord) -> PipelineOutput:
        return self.pipeline.predict_document(doc)

    def predict_text(self, patient_id: int, text: str) -> PipelineOutput:
        return self.pipeline.predict_document(DocumentRecord(patient_id=patient_id, text=text))

    def predict_many(self, docs: List[DocumentRecord]) -> List[PipelineOutput]:
        return self.pipeline.predict_many(docs)


def build_service_from_config(cfg: PredictConfig) -> NERELService:
    return NERELService.from_config(cfg)


def predict_documents(
    service: NERELService,
    docs: List[DocumentRecord],
) -> Tuple[List[dict], List[dict]]:
    outputs = service.predict_many(docs)
    return [o.doc_prediction for o in outputs], [o.debug_prediction for o in outputs]

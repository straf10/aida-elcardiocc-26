from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

from dictionary.config import get_config_path_default, load_dictionary_config
from dictionary.export import load_code_description_csv
from dictionary.matcher import build_automaton
from .config import PredictConfig
from .context_reranker import ContextReranker
from .dictionary_features import load_dictionary_candidates
from .io_utils import load_documents
from .linker import (
    MentionLinker,
    build_prior_map,
    default_prior_artifact_path,
    load_prior_map,
)
from .model import load_ner_model_for_inference
from .pipeline import NERELPipeline, PipelineOutput
from .schemas import DocumentRecord
from preprocessing.io_utils import LABELSET_PATH, load_labelset
from split_data.device_utils import get_device


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
        device = get_device()
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        print(f"Using device: {device}")
        
        model = load_ner_model_for_inference(
            model_dir=cfg.model_dir,
            use_partial_crf=cfg.use_partial_crf,
        )

        prior_map, prior_source = _load_prior_map_for_runtime(
            model_dir=cfg.model_dir,
            train_path_for_linker=cfg.train_path_for_linker,
        )
        dictionary_cfg = load_dictionary_config(get_config_path_default())
        labelset = load_labelset(LABELSET_PATH)
        dict_map = load_dictionary_candidates(
            labelset=labelset,
            config_path=get_config_path_default(),
        )
        dict_matcher = build_automaton(
            dict_map,
            word_boundary=bool((dictionary_cfg.matching or {}).get("word_boundary", False)),
        )
        code_desc_map = load_code_description_csv(dictionary_cfg.paths["code_description_csv"])
        reranker = None
        if cfg.use_reranker:
            try:
                reranker = ContextReranker.load(
                    artifact_dir=cfg.reranker_artifact_dir,
                    code_desc_map=code_desc_map,
                )
                print(f"Loaded context reranker artifacts from: {cfg.reranker_artifact_dir}")
            except FileNotFoundError:
                reranker = ContextReranker(
                    code_desc_map=code_desc_map,
                    model_name=cfg.reranker_model,
                    window_chars=cfg.reranker_window_chars,
                ).fit(labelset)
                reranker.save(cfg.reranker_artifact_dir)
                print(f"Built and saved context reranker to: {cfg.reranker_artifact_dir}")
        linker = MentionLinker(
            prior_map=prior_map,
            dictionary_map=dict_map,
            reranker=reranker,
            alpha=cfg.reranker_alpha,
        )

        pipeline = NERELPipeline(
            model=model,
            tokenizer_name=cfg.tokenizer_name,
            linker=linker,
            max_length=cfg.max_length,
            dictionary_map=dict_map,
            dictionary_matcher=dict_matcher,
            dictionary_config=dictionary_cfg,
            labelset=labelset,
            code_desc_map=code_desc_map,
            use_dictionary_fusion=cfg.use_dictionary_fusion,
            dictionary_doc_boost=cfg.dictionary_doc_boost,
            dictionary_word_boundary=bool((dictionary_cfg.matching or {}).get("word_boundary", False)),
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

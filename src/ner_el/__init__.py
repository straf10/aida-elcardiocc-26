"""NER + Entity Linking pipeline for ELCardioCC."""

from .pipeline import NERELPipeline
from .service import NERELService, build_service_from_config, predict_documents
from .schemas import DocumentRecord, MentionAnnotation

__all__ = [
	"NERELPipeline",
	"NERELService",
	"build_service_from_config",
	"predict_documents",
	"DocumentRecord",
	"MentionAnnotation",
]

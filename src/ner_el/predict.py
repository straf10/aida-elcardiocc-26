from __future__ import annotations

from .config import parse_predict_args
from .io_utils import load_documents, save_jsonl
from .service import NERELService


def main() -> None:
    cfg = parse_predict_args()
    cfg.validate_for_cli()

    service = NERELService.from_config(cfg)

    docs = load_documents(cfg.input_path)
    outputs = service.predict_many(docs)
    doc_preds = [o.doc_prediction for o in outputs]
    debug_preds = [o.debug_prediction for o in outputs]

    save_jsonl(cfg.output_doc_path, doc_preds)
    save_jsonl(cfg.output_debug_path, debug_preds)

    print(f"Wrote document predictions: {cfg.output_doc_path}")
    print(f"Wrote debug predictions: {cfg.output_debug_path}")


if __name__ == "__main__":
    main()

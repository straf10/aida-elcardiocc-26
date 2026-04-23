from __future__ import annotations

import os
import sys

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from ner_el.config import parse_predict_args
from ner_el.io_utils import load_documents, save_jsonl
from ner_el.service import NERELService


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

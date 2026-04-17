import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

try:
    from ..preprocessing.io_utils import load_jsonl
    from ..evaluation.config_utils import load_config, get_cfg
    from .common import clustering_output_dir
except ImportError:
    from src.preprocessing.io_utils import load_jsonl
    from src.evaluation.config_utils import load_config, get_cfg
    from src.analysis.common import clustering_output_dir


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def embed_texts(
    texts: List[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate mean-pooled embeddings using HuggingFace model."""
    print(f"Loading {model_name} for clustering...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Mean pooling (mask-aware)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def cluster_kmeans(embeddings: np.ndarray, n_clusters: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform KMeans clustering."""
    print(f"Running KMeans with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_, kmeans.inertia_


def describe_clusters(
    texts: List[str],
    patient_ids: List[int],
    cluster_labels: np.ndarray,
    n_clusters: int,
    top_k_terms: int = 15
) -> List[Dict[str, Any]]:
    """Describe clusters using TF-IDF."""
    print("Computing TF-IDF cluster descriptions...")
    # Group texts by cluster
    cluster_texts = [""] * n_clusters
    cluster_sizes = [0] * n_clusters
    cluster_doc_lens = [0.0] * n_clusters
    cluster_pids: List[List[int]] = [[] for _ in range(n_clusters)]

    for text, pid, label in zip(texts, patient_ids, cluster_labels):
        cluster_texts[label] += " " + text
        cluster_sizes[label] += 1
        cluster_doc_lens[label] += len(text)
        cluster_pids[label].append(pid)

    vectorizer = TfidfVectorizer(lowercase=False, max_df=0.8, min_df=2)
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        # Fallback if vocabulary is empty
        return [{"cluster_id": i, "size": cluster_sizes[i], "top_terms": []} for i in range(n_clusters)]

    descriptions = []
    for i in range(n_clusters):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = row.argsort()[-top_k_terms:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        
        descriptions.append({
            "cluster_id": int(i),
            "size": int(cluster_sizes[i]),
            "mean_doc_len": float(cluster_doc_lens[i] / cluster_sizes[i]) if cluster_sizes[i] > 0 else 0.0,
            "top_terms": top_terms,
            "sample_pids": [int(x) for x in cluster_pids[i][:5]]
        })

    return descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/analysis/analysis.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_path = get_cfg(cfg, "data.val_path")
    cluster_out = clustering_output_dir(cfg)

    model_name = get_cfg(cfg, "clustering.model_name", "nlpaueb/bert-base-greek-uncased-v1")
    n_clusters = get_cfg(cfg, "clustering.n_clusters", 8)
    max_length = get_cfg(cfg, "clustering.max_length", 256)
    batch_size = get_cfg(cfg, "clustering.batch_size", 16)
    default_cache = str(cluster_out / "embeddings.npy")
    cache_path = get_cfg(cfg, "clustering.embeddings_cache", default_cache)

    records = load_jsonl(val_path)
    texts = [r["text"] for r in records]
    pids = [int(r["patient_id"]) for r in records]

    # Load or compute embeddings
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
        if embeddings.shape[0] != len(texts):
            print("Cache size mismatch, recomputing...")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = embed_texts(texts, model_name, max_length, batch_size, device)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)

    labels, _, _ = cluster_kmeans(embeddings, n_clusters)
    
    descriptions = describe_clusters(texts, pids, labels, n_clusters)

    assignments = {int(pid): int(label) for pid, label in zip(pids, labels)}
    
    with open(cluster_out / "cluster_assignments.json", "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2)
        
    with open(cluster_out / "cluster_summary.json", "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    print("Clustering complete.")

if __name__ == "__main__":
    main()

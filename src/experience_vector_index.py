#!/usr/bin/env python3
"""Build and query the FAISS semantic explanation index."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from acprover_config import load_config
except ModuleNotFoundError:
    from .acprover_config import load_config


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILENAME = "semantic_explanations.faiss"
MANIFEST_FILENAME = "semantic_explanations.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _vendor_paths() -> List[str]:
    candidates = [_repo_root() / ".vendor", _repo_root() / ".vendor" / "python"]
    return [str(path) for path in candidates if path.exists()]


for candidate in reversed(_vendor_paths()):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def _load_faiss():
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FAISS is required in the configured RocSql conda environment. "
            "Install faiss-cpu in the configured vector_conda_env."
        ) from exc
    return faiss


def _load_embedding_runtime():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Transformers embedding dependencies are required in the configured RocSql vector environment. "
            "Install `torch`, `transformers`, and their dependencies, or vendor missing packages into the repo."
        ) from exc
    return torch, AutoTokenizer, AutoModel


def _embedding_cache_dir() -> Path:
    config = load_config()
    raw = str(config.embedding_cache_dir or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_repo_root() / "models" / "huggingface").resolve()


def _prepare_hf_runtime(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _load_embedding_model() -> Tuple[Any, Any, Any, str]:
    torch, AutoTokenizer, AutoModel = _load_embedding_runtime()
    config = load_config()
    model_name = str(config.embedding_model_name or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    cache_dir = _embedding_cache_dir()
    _prepare_hf_runtime(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
    model.eval()
    return torch, tokenizer, model, model_name


def _mean_pool(last_hidden_state: Any, attention_mask: Any, torch_module: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_texts(texts: List[str]) -> Tuple[np.ndarray, str]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), str(load_config().embedding_model_name or DEFAULT_MODEL_NAME)
    torch, tokenizer, model, model_name = _load_embedding_model()
    all_vectors: List[np.ndarray] = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"], torch)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(all_vectors), model_name


def _metadata_paths(experience_root: Path) -> List[Path]:
    paths = []
    for path in experience_root.rglob("metadata.json"):
        if path.parent == experience_root:
            continue
        paths.append(path)
    return sorted(paths)


def _load_metadata(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata at {path} is not an object")
    return payload


def _collect_records(experience_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for metadata_path in _metadata_paths(experience_root):
        metadata = _load_metadata(metadata_path)
        semantic_explanation = str(metadata.get("semantic_explanation", "")).strip()
        record_id = str(metadata.get("record_id", "")).strip()
        if not record_id or not semantic_explanation:
            continue
        records.append(
            {
                "record_id": record_id,
                "metadata_path": str(metadata_path),
                "semantic_explanation": semantic_explanation,
            }
        )
    return records


def build_index(experience_root: Path) -> Dict[str, Any]:
    faiss = _load_faiss()
    records = _collect_records(experience_root)
    vectors, model_name = _encode_texts([record["semantic_explanation"] for record in records])
    dimension = int(vectors.shape[1]) if len(records) > 0 else 0

    index = faiss.IndexFlatIP(dimension)
    if len(records) > 0:
        index.add(vectors)

    index_path = experience_root / INDEX_FILENAME
    manifest_path = experience_root / MANIFEST_FILENAME
    faiss.write_index(index, str(index_path))
    manifest_path.write_text(
        json.dumps(
            {
                "dimension": dimension,
                "embedding_model_name": model_name,
                "embedding_cache_dir": str(_embedding_cache_dir()),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "success": True,
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
        "record_count": len(records),
        "dimension": dimension,
        "embedding_model_name": model_name,
    }


def _load_manifest(experience_root: Path) -> Dict[str, Any]:
    manifest_path = experience_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic explanation manifest is not an object")
    return payload


def search_index(experience_root: Path, query: str, limit: int) -> Dict[str, Any]:
    faiss = _load_faiss()
    index_path = experience_root / INDEX_FILENAME
    if not index_path.exists():
        build_index(experience_root)
    manifest = _load_manifest(experience_root)
    records = manifest.get("records", [])
    dimension = int(manifest.get("dimension", 0))
    if not isinstance(records, list):
        raise ValueError("semantic explanation manifest records is not a list")

    index = faiss.read_index(str(index_path))
    if index.ntotal == 0:
        return {"success": True, "hits": [], "query": query}

    query_vectors, model_name = _encode_texts([query])
    if query_vectors.shape[1] != dimension:
        raise ValueError(
            f"query embedding dimension {query_vectors.shape[1]} does not match index dimension {dimension}"
        )
    query_vector = query_vectors.reshape(1, dimension)
    scores, indices = index.search(query_vector, max(1, min(limit, len(records))))
    hits: List[Dict[str, Any]] = []
    for score, raw_index in zip(scores[0], indices[0]):
        if raw_index < 0 or raw_index >= len(records):
            continue
        record = records[raw_index]
        metadata = _load_metadata(Path(record["metadata_path"]))
        hits.append(
            {
                "score": float(score),
                "record_id": record["record_id"],
                "metadata_path": record["metadata_path"],
                "metadata": metadata,
            }
        )
    return {"success": True, "hits": hits, "query": query, "embedding_model_name": model_name}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Build and query the experience semantic explanation FAISS index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rebuild = subparsers.add_parser("rebuild")
    rebuild.add_argument("--experience-root", required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--experience-root", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=3)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    experience_root = Path(args.experience_root).resolve()
    experience_root.mkdir(parents=True, exist_ok=True)
    if args.command == "rebuild":
        result = build_index(experience_root)
    elif args.command == "search":
        result = search_index(experience_root, query=args.query, limit=args.limit)
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

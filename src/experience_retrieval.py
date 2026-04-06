#!/usr/bin/env python3
"""Retrieve and render reusable theorem-solving experience."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from acprover_config import load_config
    from experience_extract import _extract_coq_libraries, _normalized_theorem_types, _semantic_explanation
    from experience_store import default_experience_root
    from theorem_task import TheoremTask
except ModuleNotFoundError:
    from .acprover_config import load_config
    from .experience_extract import _extract_coq_libraries, _normalized_theorem_types, _semantic_explanation
    from .experience_store import default_experience_root
    from .theorem_task import TheoremTask


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _load_metadata_records() -> List[Dict[str, Any]]:
    def is_usable(metadata: Dict[str, Any]) -> bool:
        return bool(str(metadata.get("record_id", "")).strip()) and bool(
            str(metadata.get("semantic_explanation", "")).strip()
        )

    index_path = default_experience_root() / "metadata_index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = []
            for item in payload["records"]:
                metadata_path = Path(str(item.get("metadata_path", "")))
                if not metadata_path.exists():
                    continue
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(metadata, dict) and is_usable(metadata):
                    records.append(metadata)
            return records

    records: List[Dict[str, Any]] = []
    for metadata_path in sorted(default_experience_root().rglob("metadata.json")):
        if metadata_path.parent == default_experience_root():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(metadata, dict) and is_usable(metadata):
            records.append(metadata)
    return records


def _query_metadata(task: TheoremTask) -> Dict[str, Any]:
    theorem_types = _normalized_theorem_types(task, [])
    coq_libraries = _extract_coq_libraries(task, "", "")
    semantic_explanation = _semantic_explanation(task, theorem_types, coq_libraries, "")
    return {
        "semantic_explanation": semantic_explanation,
        "normalized_theorem_types": theorem_types,
        "coq_libraries": coq_libraries,
    }


def _read_excerpt(path_text: Any, limit: int = 320) -> str:
    path = str(path_text or "").strip()
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.exists():
        return ""
    text = candidate.read_text(encoding="utf-8").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _coq_library_overlap(query_libs: Dict[str, Any], metadata_libs: Dict[str, Any]) -> int:
    query_tokens = set(map(str, query_libs.get("declared_imports", []))) | set(
        map(str, query_libs.get("referenced_namespaces", []))
    )
    metadata_tokens = set(map(str, metadata_libs.get("declared_imports", []))) | set(
        map(str, metadata_libs.get("referenced_namespaces", []))
    )
    return len(query_tokens & metadata_tokens)


def _search_from_query_metadata(
    query_semantic_explanation: str,
    query_types: List[str],
    query_libraries: Optional[Dict[str, Any]],
    *,
    limit: int,
    project: str = "",
    file_relpath: str = "",
) -> List[Dict[str, Any]]:
    query_tokens = set(_tokenize(query_semantic_explanation))
    query_type_set = set(query_types)
    query_libraries = query_libraries or {}
    scored: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for metadata in _load_metadata_records():
        metadata_text = str(metadata.get("semantic_explanation", ""))
        score = 0.0
        score += 1.0 * len(query_tokens & set(_tokenize(metadata_text)))
        score += 4.0 * len(query_type_set & set(map(str, metadata.get("normalized_theorem_types", []))))
        score += 1.5 * _coq_library_overlap(query_libraries, metadata.get("coq_libraries", {}))
        if project and metadata.get("project") == project:
            score += 12.0
        if file_relpath and metadata.get("file_relpath") == file_relpath:
            score += 10.0
        if score <= 0:
            continue
        scored.append((score, metadata, {"mode": "fallback"}))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_decorate_hit(score, metadata, breakdown) for score, metadata, breakdown in scored[:limit]]


def _fallback_search(task: TheoremTask, limit: int) -> List[Dict[str, Any]]:
    query = _query_metadata(task)
    return _search_from_query_metadata(
        query["semantic_explanation"],
        query["normalized_theorem_types"],
        query["coq_libraries"],
        limit=limit,
        project=task.project,
        file_relpath=task.file_relpath,
    )


def _run_faiss_search(query: str, limit: int) -> List[Dict[str, Any]]:
    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("`conda` is required to query the FAISS semantic index.")
    config = load_config()
    script_path = Path(__file__).resolve().parent / "experience_vector_index.py"
    command = [
        conda,
        "run",
        "-n",
        config.vector_conda_env,
        "python",
        str(script_path),
        "search",
        "--experience-root",
        str(default_experience_root()),
        "--query",
        query,
        "--limit",
        str(limit),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "FAISS search failed")
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("FAISS search returned non-object JSON")
    return payload.get("hits", []) if isinstance(payload.get("hits"), list) else []


def _decorate_hit(score: float, metadata: Dict[str, Any], score_breakdown: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "score": score,
        "record_id": metadata.get("record_id"),
        "source_theorem_id": metadata.get("source_theorem_id"),
        "project": metadata.get("project"),
        "file_relpath": metadata.get("file_relpath"),
        "source_file_path": metadata.get("source_file_path", ""),
        "semantic_explanation": metadata.get("semantic_explanation", ""),
        "normalized_theorem_types": metadata.get("normalized_theorem_types", []),
        "coq_libraries": metadata.get("coq_libraries", {}),
        "reasoning_path": metadata.get("reasoning_path", ""),
        "issues_path": metadata.get("issues_path", ""),
        "result_path": metadata.get("result_path", ""),
        "final_proof_path": metadata.get("final_proof_path", ""),
        "oracle_proof_path": metadata.get("oracle_proof_path", ""),
        "reasoning_excerpt": _read_excerpt(metadata.get("reasoning_path")),
        "issues_excerpt": _read_excerpt(metadata.get("issues_path")),
        "result_excerpt": _read_excerpt(metadata.get("result_path")),
        "score_breakdown": score_breakdown,
    }


def retrieve_relevant_experiences(task: TheoremTask, limit: int = 3) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    query = _query_metadata(task)
    try:
        faiss_hits = _run_faiss_search(query["semantic_explanation"], limit=max(limit * 3, 6))
    except Exception:
        return _fallback_search(task, limit)

    query_types = set(query["normalized_theorem_types"])
    reranked: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for hit in faiss_hits:
        metadata = hit.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        vector_score = float(hit.get("score", 0.0))
        type_overlap = len(query_types & set(map(str, metadata.get("normalized_theorem_types", []))))
        library_overlap = _coq_library_overlap(query["coq_libraries"], metadata.get("coq_libraries", {}))
        project_bonus = 1.5 if metadata.get("project") == task.project else 0.0
        file_bonus = 1.0 if metadata.get("file_relpath") == task.file_relpath else 0.0
        total = vector_score + 0.25 * type_overlap + 0.08 * library_overlap + project_bonus + file_bonus
        reranked.append(
            (
                total,
                metadata,
                {
                    "mode": "faiss",
                    "vector_score": vector_score,
                    "type_overlap": type_overlap,
                    "library_overlap": library_overlap,
                    "project_bonus": project_bonus,
                    "file_bonus": file_bonus,
                },
            )
        )
    reranked.sort(key=lambda item: item[0], reverse=True)
    return [_decorate_hit(score, metadata, breakdown) for score, metadata, breakdown in reranked[:limit]]


def query_experiences_by_description(description: str, limit: int = 5) -> List[Dict[str, Any]]:
    query = str(description).strip()
    if not query or limit <= 0:
        return []
    try:
        faiss_hits = _run_faiss_search(query, limit=max(limit * 3, 6))
    except Exception:
        return _search_from_query_metadata(query, [], {}, limit=limit)

    query_tokens = set(_tokenize(query))
    reranked: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for hit in faiss_hits:
        metadata = hit.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        vector_score = float(hit.get("score", 0.0))
        lexical_overlap = len(query_tokens & set(_tokenize(str(metadata.get("semantic_explanation", "")))))
        total = vector_score + 0.05 * lexical_overlap
        reranked.append(
            (
                total,
                metadata,
                {
                    "mode": "faiss",
                    "vector_score": vector_score,
                    "lexical_overlap": lexical_overlap,
                },
            )
        )
    reranked.sort(key=lambda item: item[0], reverse=True)
    return [_decorate_hit(score, metadata, breakdown) for score, metadata, breakdown in reranked[:limit]]


def render_experience_prompt_block(experiences: List[Dict[str, Any]]) -> str:
    if not experiences:
        return ""

    lines: List[str] = []
    lines.append("[Relevant Experience]")
    lines.append("Use these prior theorem-solving artifacts when they fit; prefer the saved reasoning/issues/result paths over guessing.")
    for index, item in enumerate(experiences, start=1):
        lines.append(
            f"{index}. {item.get('source_theorem_id')} {item.get('project')}::{item.get('file_relpath')}"
        )
        if item.get("source_file_path"):
            lines.append(f"   source_file_path: {str(item.get('source_file_path'))}")
        lines.append(f"   semantic_explanation: {str(item.get('semantic_explanation', '')).strip()}")
        theorem_types = item.get("normalized_theorem_types", [])
        if theorem_types:
            lines.append("   normalized_theorem_types: " + ", ".join(map(str, theorem_types[:6])))
        libraries = item.get("coq_libraries", {})
        declared_imports = libraries.get("declared_imports", []) if isinstance(libraries, dict) else []
        if declared_imports:
            lines.append("   coq_libraries: " + ", ".join(map(str, declared_imports[:6])))
        reasoning_excerpt = str(item.get("reasoning_excerpt", "")).strip()
        if reasoning_excerpt:
            lines.append("   reasoning: " + reasoning_excerpt.replace("\n", " ")[:280])
        issues_excerpt = str(item.get("issues_excerpt", "")).strip()
        if issues_excerpt:
            lines.append("   issues: " + issues_excerpt.replace("\n", " ")[:280])
        result_excerpt = str(item.get("result_excerpt", "")).strip()
        if result_excerpt:
            lines.append("   result: " + result_excerpt.replace("\n", " ")[:220])
        if item.get("final_proof_path"):
            lines.append("   final_proof_path: " + str(item.get("final_proof_path")))
        if item.get("oracle_proof_path"):
            lines.append("   oracle_proof_path: " + str(item.get("oracle_proof_path")))
    return "\n".join(lines).rstrip() + "\n"

#!/usr/bin/env python3
"""Retrieve standard-library records from the local semantic index."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List

try:
    from acprover_config import load_config
    from experience_store import default_experience_root, experience_domain_root
except ModuleNotFoundError:
    from .acprover_config import load_config
    from .experience_store import default_experience_root, experience_domain_root


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _load_metadata_records(experience_root: Path) -> List[Dict[str, Any]]:
    index_path = experience_root / "metadata_index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records: List[Dict[str, Any]] = []
            for item in payload["records"]:
                metadata_path = Path(str(item.get("metadata_path", "")))
                if not metadata_path.exists():
                    continue
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(metadata, dict):
                    records.append(metadata)
            return records

    records = []
    for metadata_path in sorted(experience_root.rglob("metadata.json")):
        if metadata_path.parent == experience_root:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(metadata, dict):
            records.append(metadata)
    return records


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


def _run_faiss_search(query: str, limit: int, experience_root: Path) -> List[Dict[str, Any]]:
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
        str(experience_root),
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
        "project": metadata.get("project", ""),
        "file_path": metadata.get("file_path", metadata.get("file_relpath", "")),
        "module_path": metadata.get("module_path", ""),
        "semantic_explanation": metadata.get("semantic_explanation", ""),
        "normalized_theorem_types": metadata.get("normalized_theorem_types", []),
        "context": metadata.get("context", ""),
        "proof": metadata.get("proof", ""),
        "detail_path": metadata.get("detail_path", ""),
        "reasoning_path": metadata.get("reasoning_path", ""),
        "detail_excerpt": _read_excerpt(metadata.get("detail_path")),
        "reasoning_excerpt": _read_excerpt(metadata.get("reasoning_path")),
        "score_breakdown": score_breakdown,
    }


def _fallback_search(description: str, limit: int, experience_root: Path) -> List[Dict[str, Any]]:
    query_tokens = set(_tokenize(description))
    scored: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for metadata in _load_metadata_records(experience_root):
        semantic = str(metadata.get("semantic_explanation", ""))
        lexical_overlap = len(query_tokens & set(_tokenize(semantic)))
        if lexical_overlap <= 0:
            continue
        scored.append((float(lexical_overlap), metadata, {"mode": "fallback", "lexical_overlap": lexical_overlap}))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_decorate_hit(score, metadata, breakdown) for score, metadata, breakdown in scored[:limit]]


def query_experiences_by_description(
    description: str,
    limit: int = 5,
    *,
    experience_root: Path | None = None,
) -> List[Dict[str, Any]]:
    query = str(description).strip()
    if not query or limit <= 0:
        return []
    experience_root = experience_root or default_experience_root()
    try:
        faiss_hits = _run_faiss_search(query, limit=max(limit * 3, 6), experience_root=experience_root)
    except Exception:
        return _fallback_search(query, limit, experience_root)

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


def query_metadata_sql(sql: str, *, experience_root: Path) -> Dict[str, Any]:
    statement = str(sql).strip()
    if not statement:
        raise ValueError("SQL query is empty")
    lowered = statement.lower()
    if not lowered.startswith("select "):
        raise ValueError("Only SELECT queries are allowed")
    if ";" in statement.rstrip().rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed")
    db_path = experience_root / "metadata.db"
    if not db_path.exists():
        raise FileNotFoundError(f"metadata database not found: {db_path}")
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(statement)
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()
    return {
        "sql": statement,
        "row_count": len(rows),
        "rows": rows,
        "metadata_db_path": str(db_path),
    }


def query_stdlib_by_description(description: str, limit: int = 5) -> List[Dict[str, Any]]:
    return query_experiences_by_description(description, limit=limit, experience_root=experience_domain_root("stdlib"))


def query_coqstoq_by_description(description: str, limit: int = 5) -> List[Dict[str, Any]]:
    return query_experiences_by_description(description, limit=limit, experience_root=experience_domain_root("coqstoq"))


def query_stdlib_sql(sql: str) -> Dict[str, Any]:
    return query_metadata_sql(sql, experience_root=experience_domain_root("stdlib"))


def query_coqstoq_sql(sql: str) -> Dict[str, Any]:
    return query_metadata_sql(sql, experience_root=experience_domain_root("coqstoq"))


def render_experience_prompt_block(experiences: List[Dict[str, Any]]) -> str:
    if not experiences:
        return ""
    lines: List[str] = ["[Relevant Experience]", "Prefer the saved detail and reasoning files over guessing."]
    for index, item in enumerate(experiences, start=1):
        lines.append(f"{index}. {item.get('record_id')}")
        if item.get("module_path"):
            lines.append(f"   module_path: {item.get('module_path')}")
        lines.append(f"   semantic_explanation: {str(item.get('semantic_explanation', '')).strip()}")
        if item.get("detail_excerpt"):
            lines.append("   detail: " + str(item.get("detail_excerpt")).replace("\n", " ")[:280])
        if item.get("reasoning_excerpt"):
            lines.append("   reasoning: " + str(item.get("reasoning_excerpt")).replace("\n", " ")[:280])
        if item.get("detail_path"):
            lines.append("   detail_path: " + str(item.get("detail_path")))
        if item.get("reasoning_path"):
            lines.append("   reasoning_path: " + str(item.get("reasoning_path")))
    return "\n".join(lines).rstrip() + "\n"

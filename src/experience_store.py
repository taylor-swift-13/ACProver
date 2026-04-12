#!/usr/bin/env python3
"""Persist theorem-solving experiences and rebuild retrieval indexes."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict

try:
    from acprover_config import load_config
    from logging_utils import write_json, write_text
except ModuleNotFoundError:
    from .acprover_config import load_config
    from .logging_utils import write_json, write_text


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_experience_root() -> Path:
    return _repo_root() / "experience"


def experience_domain_root(domain: str) -> Path:
    root = default_experience_root() / domain
    root.mkdir(parents=True, exist_ok=True)
    return root


def _theorem_slug(theorem_id: str) -> str:
    return theorem_id.replace(":", "_")


def _timestamp_from_log_dir(log_dir: Path) -> str:
    parts = log_dir.name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-3:-1])
    return log_dir.name


def prepare_experience_dir(theorem_id: str, status: str, log_dir: Path, *, experience_root: Path | None = None) -> Path:
    root = experience_root or experience_domain_root("coqstoq")
    experience_dir = root / _theorem_slug(theorem_id) / f"{_timestamp_from_log_dir(log_dir)}_{status}"
    experience_dir.mkdir(parents=True, exist_ok=True)
    return experience_dir


def _write_optional_text(path: Path, content: str) -> str:
    write_text(path, content.rstrip() + "\n")
    return str(path)


def _write_optional_proof(path: Path, content: str) -> str:
    if not content.strip():
        return ""
    write_text(path, content.rstrip() + "\n")
    return str(path)


def _write_metadata_index(experience_root: Path) -> Path:
    records = []
    for metadata_path in sorted(experience_root.rglob("metadata.json")):
        if metadata_path.parent == experience_root:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(metadata, dict):
            continue
        records.append(
            {
                "record_id": metadata.get("record_id"),
                "metadata_path": str(metadata_path),
                "semantic_explanation": metadata.get("semantic_explanation", ""),
                "module_path": metadata.get("module_path", ""),
            }
        )
    index_path = experience_root / "metadata_index.json"
    write_json(index_path, {"records": records})
    return index_path


def _rebuild_metadata_db(experience_root: Path) -> Path:
    db_path = experience_root / "metadata.db"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE records (
                record_id TEXT PRIMARY KEY,
                project TEXT,
                file_path TEXT,
                module_path TEXT,
                item_kind TEXT,
                item_name TEXT,
                semantic_explanation TEXT,
                normalized_theorem_types_json TEXT,
                context TEXT,
                proof TEXT,
                related_json TEXT,
                detail_path TEXT,
                reasoning_path TEXT,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX idx_records_module_path ON records(module_path)")
        connection.execute("CREATE INDEX idx_records_project ON records(project)")
        connection.execute("CREATE INDEX idx_records_file_path ON records(file_path)")
        connection.execute("CREATE INDEX idx_records_item_kind ON records(item_kind)")
        connection.execute("CREATE INDEX idx_records_item_name ON records(item_name)")
        connection.execute("CREATE INDEX idx_records_semantic ON records(semantic_explanation)")

        for metadata_path in sorted(experience_root.rglob("metadata.json")):
            if metadata_path.parent == experience_root:
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(metadata, dict):
                continue
            record_id = str(metadata.get("record_id", "")).strip()
            if not record_id:
                continue
            file_path = metadata.get("file_path", metadata.get("file_relpath", ""))
            connection.execute(
                """
                INSERT INTO records (
                    record_id, project, file_path, module_path, item_kind, item_name, semantic_explanation,
                    normalized_theorem_types_json, context, proof, related_json, detail_path, reasoning_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    str(metadata.get("project", "")),
                    str(file_path),
                    str(metadata.get("module_path", "")),
                    str(metadata.get("item_kind", "")),
                    str(metadata.get("item_name", "")),
                    str(metadata.get("semantic_explanation", "")),
                    json.dumps(metadata.get("normalized_theorem_types", []), ensure_ascii=False),
                    str(metadata.get("context", "")),
                    str(metadata.get("proof", "")),
                    json.dumps(metadata.get("related", []), ensure_ascii=False),
                    str(metadata.get("detail_path", "")),
                    str(metadata.get("reasoning_path", "")),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return db_path


def _rebuild_semantic_index(experience_root: Path) -> Dict[str, Any]:
    config = load_config()
    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("`conda` is required to rebuild the FAISS semantic index.")
    script_path = _repo_root() / "src" / "experience_vector_index.py"
    command = [
        conda,
        "run",
        "-n",
        config.vector_conda_env,
        "python",
        str(script_path),
        "rebuild",
        "--experience-root",
        str(experience_root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to rebuild FAISS semantic index via conda env "
            f"`{config.vector_conda_env}`: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("semantic index rebuild returned non-object JSON")
    return payload


def refresh_experience_indexes(experience_root: Path | None = None) -> Dict[str, Any]:
    experience_root = experience_root or default_experience_root()
    experience_root.mkdir(parents=True, exist_ok=True)
    metadata_index_path = _write_metadata_index(experience_root)
    metadata_db_path = _rebuild_metadata_db(experience_root)
    semantic_index_warning = ""
    try:
        semantic_index = _rebuild_semantic_index(experience_root)
    except Exception as exc:
        semantic_index = {}
        semantic_index_warning = str(exc)
        write_json(experience_root / "semantic_index_status.json", {"warning": semantic_index_warning})
    else:
        write_json(experience_root / "semantic_index_status.json", {"status": semantic_index})
    return {
        "metadata_index_path": str(metadata_index_path),
        "metadata_db_path": str(metadata_db_path),
        "semantic_index_warning": semantic_index_warning,
        "semantic_index": semantic_index,
    }


def write_experience_bundle(
    bundle: Dict[str, Any],
    log_dir: Path,
    *,
    rebuild_indexes: bool = True,
) -> Dict[str, Any]:
    theorem_id = str(bundle["source_theorem_id"])
    status = "proven" if str(bundle.get("final_proof_text", "")).strip() else str(
        "oracle_postmortem" if str(bundle.get("oracle_proof_text", "")).strip() else "failed"
    )
    experience_root = experience_domain_root("coqstoq")
    experience_dir = prepare_experience_dir(theorem_id, status, log_dir, experience_root=experience_root)

    reasoning_path = _write_optional_text(experience_dir / "reasoning.md", str(bundle["reasoning_md"]))
    detail_path = _write_optional_text(experience_dir / "detail.md", str(bundle["detail_md"]))
    result_path = _write_optional_text(experience_dir / "result.md", str(bundle["result_md"]))
    final_proof_path = _write_optional_proof(experience_dir / "final_proof.v", str(bundle.get("final_proof_text", "")))
    oracle_proof_path = _write_optional_proof(
        experience_dir / "oracle_proof.v",
        str(bundle.get("oracle_proof_text", "")),
    )

    metadata = {
        "record_id": bundle["record_id"],
        "source_theorem_id": theorem_id,
        "project": bundle["project"],
        "file_relpath": bundle["file_relpath"],
        "source_file_path": bundle["source_file_path"],
        "semantic_explanation": bundle["semantic_explanation"],
        "normalized_theorem_types": bundle["normalized_theorem_types"],
        "coq_libraries": bundle["coq_libraries"],
        "detail_path": detail_path,
        "reasoning_path": reasoning_path,
        "result_path": result_path,
        "final_proof_path": final_proof_path,
        "oracle_proof_path": oracle_proof_path,
    }
    metadata_path = experience_dir / "metadata.json"
    write_json(metadata_path, metadata)

    write_json(experience_dir / "attempts.json", {"attempts": bundle.get("attempts", [])})
    write_json(experience_dir / "failed_proofs.json", {"failed_proofs": bundle.get("failed_proofs", [])})
    write_json(experience_dir / "repair_chain.json", {"repair_chain": bundle.get("repair_chain", [])})
    write_json(experience_dir / "artifacts.json", bundle.get("artifacts", {}))

    if rebuild_indexes:
        refresh_info = refresh_experience_indexes(experience_root)
    else:
        refresh_info = {
            "metadata_index_path": "",
            "semantic_index_warning": "",
            "semantic_index": {},
        }

    return {
        "experience_dir": str(experience_dir),
        "metadata_path": str(metadata_path),
        "metadata_index_path": refresh_info["metadata_index_path"],
        "semantic_index_warning": refresh_info["semantic_index_warning"],
    }

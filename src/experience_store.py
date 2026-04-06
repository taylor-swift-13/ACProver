#!/usr/bin/env python3
"""Persist theorem-solving experiences and rebuild retrieval indexes."""

from __future__ import annotations

import json
import shutil
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


def _theorem_slug(theorem_id: str) -> str:
    return theorem_id.replace(":", "_")


def _timestamp_from_log_dir(log_dir: Path) -> str:
    parts = log_dir.name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-3:-1])
    return log_dir.name


def prepare_experience_dir(theorem_id: str, status: str, log_dir: Path) -> Path:
    root = default_experience_root()
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


def _write_metadata_index() -> Path:
    experience_root = default_experience_root()
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
                "source_theorem_id": metadata.get("source_theorem_id"),
                "project": metadata.get("project"),
                "file_relpath": metadata.get("file_relpath"),
            }
        )
    index_path = experience_root / "metadata_index.json"
    write_json(index_path, {"records": records})
    return index_path


def _rebuild_semantic_index() -> Dict[str, Any]:
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
        str(default_experience_root()),
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


def write_experience_bundle(bundle: Dict[str, Any], log_dir: Path) -> Dict[str, Any]:
    theorem_id = str(bundle["source_theorem_id"])
    status = "proven" if str(bundle.get("final_proof_text", "")).strip() else str(
        "oracle_postmortem" if str(bundle.get("oracle_proof_text", "")).strip() else "failed"
    )
    experience_dir = prepare_experience_dir(theorem_id, status, log_dir)

    reasoning_path = _write_optional_text(experience_dir / "reasoning.md", str(bundle["reasoning_md"]))
    issues_path = _write_optional_text(experience_dir / "issues.md", str(bundle["issues_md"]))
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
        "reasoning_path": reasoning_path,
        "issues_path": issues_path,
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

    metadata_index_path = _write_metadata_index()
    semantic_index_warning = ""
    try:
        semantic_index = _rebuild_semantic_index()
    except Exception as exc:
        semantic_index = {}
        semantic_index_warning = str(exc)
        write_json(default_experience_root() / "semantic_index_status.json", {"warning": semantic_index_warning})
    else:
        write_json(default_experience_root() / "semantic_index_status.json", {"status": semantic_index})

    return {
        "experience_dir": str(experience_dir),
        "metadata_path": str(metadata_path),
        "metadata_index_path": str(metadata_index_path),
        "semantic_index_warning": semantic_index_warning,
    }

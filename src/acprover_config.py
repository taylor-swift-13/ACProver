#!/usr/bin/env python3
"""Repository-local configuration for RocSql runtime defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RocSqlConfig:
    opam_switch: str = "qcp-8.20"
    vector_conda_env: str = "coq-py310"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: str = ""
    semantic_model: str = "gpt-5.4-nano"
    semantic_reasoning_effort: str = "low"
    llm_base_url: str = "https://yunwu.ai/v1"
    llm_api_key: str = ""
    semantic_temperature: float = 0.2


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    override = os.environ.get("ACPROVER_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "config" / "acprover.local.json"


def load_config() -> RocSqlConfig:
    path = config_path()
    env_api_key = os.environ.get("ACPROVER_LLM_API_KEY", "")
    if not path.exists():
        return RocSqlConfig(llm_api_key=env_api_key)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RocSqlConfig(
        opam_switch=str(payload.get("opam_switch", "qcp-8.20")),
        vector_conda_env=str(payload.get("vector_conda_env", "coq-py310")),
        embedding_model_name=str(payload.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        embedding_cache_dir=str(payload.get("embedding_cache_dir", "")),
        semantic_model=str(payload.get("semantic_model", "gpt-5.4-nano")),
        semantic_reasoning_effort=str(payload.get("semantic_reasoning_effort", "low")),
        llm_base_url=str(payload.get("llm_base_url", "https://yunwu.ai/v1")),
        llm_api_key=str(payload.get("llm_api_key", env_api_key)),
        semantic_temperature=float(payload.get("semantic_temperature", 0.2)),
    )


# Backward-compatible alias for older imports.
ACProverConfig = RocSqlConfig

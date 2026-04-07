#!/usr/bin/env python3
"""Repository-local configuration for ACProver runtime defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ACProverConfig:
    opam_switch: str = "qcp-8.20"
    vector_conda_env: str = "coq-py310"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    override = os.environ.get("ACPROVER_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "config" / "acprover.local.json"


def load_config() -> ACProverConfig:
    path = config_path()
    if not path.exists():
        return ACProverConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ACProverConfig(
        opam_switch=str(payload.get("opam_switch", "qcp-8.20")),
        vector_conda_env=str(payload.get("vector_conda_env", "coq-py310")),
    )

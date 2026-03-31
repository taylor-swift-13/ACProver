#!/usr/bin/env python3
"""
Agent-based LLM proof-task client.

Architecture:
- ProofAgent: encapsulates one proof loop
- ProofOrchestrator: coordinates main agent + parallel lemma sub-agents
- LemmaRegistry: thread-safe shared state for inserted lemma status
- All runtime lemma visibility is via Admitted; real lemma proofs are used only for final verification
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

try:
    from coq_print import execute_print_command
    from verify import CoqProofVerifier
except ModuleNotFoundError:
    from .coq_print import execute_print_command
    from .verify import CoqProofVerifier


# =========================
# Configuration
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://yunwu.ai/v1"
NL_PROOF_MODEL = os.environ.get("NL_PROOF_MODEL", "claude-opus-4-6")
LEMMA_MAX_STEPS = int(os.environ.get("LEMMA_MODE_MAX_STEPS", "8"))


# =========================
# Lemma Registry
# =========================
class LemmaState(Enum):
    ADMITTED = "admitted"
    PROVEN = "proven"
    FAILED = "failed"


@dataclass
class LemmaEntry:
    name: str
    declaration: str
    state: LemmaState
    proof_text: Optional[str] = None
    thread: Optional[threading.Thread] = None


class LemmaRegistry:
    """Thread-safe registry tracking inserted lemma states across agents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lemmas: Dict[str, LemmaEntry] = {}

    @staticmethod
    def _normalize_declaration(declaration: str) -> str:
        return " ".join(declaration.split())

    def register_lemma(self, name: str, declaration: str, thread: threading.Thread) -> Dict[str, Any]:
        normalized = self._normalize_declaration(declaration)
        with self._lock:
            entry = self._lemmas.get(name)
            if entry is None:
                self._lemmas[name] = LemmaEntry(
                    name=name,
                    declaration=declaration,
                    state=LemmaState.ADMITTED,
                    thread=thread,
                )
                return {"status": "registered"}
            if self._normalize_declaration(entry.declaration) != normalized:
                return {
                    "status": "conflict",
                    "error": (
                        f"lemma '{name}' already exists with a different declaration: "
                        f"{entry.declaration}"
                    ),
                }
            return {"status": "already_exists"}

    def mark_proven(self, name: str, proof_text: str) -> None:
        with self._lock:
            entry = self._lemmas.get(name)
            if entry is not None:
                entry.state = LemmaState.PROVEN
                entry.proof_text = proof_text

    def mark_failed(self, name: str) -> None:
        with self._lock:
            entry = self._lemmas.get(name)
            if entry is not None and entry.state != LemmaState.PROVEN:
                entry.state = LemmaState.FAILED

    @staticmethod
    def _build_admitted_block(entry: LemmaEntry) -> str:
        return f"{entry.declaration}\nAdmitted."

    @staticmethod
    def _build_proven_block(entry: LemmaEntry) -> str:
        if not entry.proof_text:
            return f"{entry.declaration}\nAdmitted."
        return f"{entry.declaration}\n{entry.proof_text}"

    def get_runtime_prelude_for_main(self) -> str:
        with self._lock:
            return "\n\n".join(self._build_admitted_block(entry) for entry in self._lemmas.values())

    def get_runtime_prelude_for_lemma(self, exclude_name: Optional[str] = None) -> str:
        with self._lock:
            blocks: List[str] = []
            for name, entry in self._lemmas.items():
                if exclude_name is not None and name == exclude_name:
                    continue
                blocks.append(self._build_admitted_block(entry))
            return "\n\n".join(blocks)

    def get_final_prelude_all_proven(self) -> Optional[str]:
        with self._lock:
            if not self._lemmas:
                return ""
            blocks: List[str] = []
            for entry in self._lemmas.values():
                if entry.state != LemmaState.PROVEN or not entry.proof_text:
                    return None
                blocks.append(self._build_proven_block(entry))
            return "\n\n".join(blocks)

    def all_proven(self) -> bool:
        with self._lock:
            return all(entry.state == LemmaState.PROVEN for entry in self._lemmas.values())

    def has_any(self) -> bool:
        with self._lock:
            return len(self._lemmas) > 0

    def unproven_names(self) -> List[str]:
        with self._lock:
            return [name for name, entry in self._lemmas.items() if entry.state != LemmaState.PROVEN]

    def wait_all(self, timeout: float = 300.0) -> None:
        with self._lock:
            threads = [entry.thread for entry in self._lemmas.values() if entry.thread is not None]
        for thread in threads:
            thread.join(timeout=timeout)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "state": entry.state.value,
                    "has_proof": entry.proof_text is not None,
                    "declaration": entry.declaration,
                }
                for name, entry in self._lemmas.items()
            }


# =========================
# Data structures
# =========================
@dataclass
class SourceSlices:
    before_theorem: str
    theorem_declaration: str
    after_proof: str


@dataclass
class TaskContext:
    theorem_id: str
    project: str
    file_relpath: str
    repo_path: str
    compile_args: List[str]
    theorem_statement: str
    header_context: str
    upper_context: str


@dataclass
class AgentConfig:
    max_steps: int
    agent_id: str
    is_main_agent: bool = False


# =========================
# Tool protocol & registry
# =========================
class Tool(Protocol):
    name: str

    def spec(self) -> Dict[str, Any]:
        ...

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def specs(self) -> List[Dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def validate_and_dispatch(self, action_obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        action = action_obj.get("action")
        if not isinstance(action, str):
            return None, "action 必须是字符串"
        tool = self.get(action)
        if tool is None:
            return None, f"未知 action: {action}"
        args = action_obj.get("args", {})
        if not isinstance(args, dict):
            return None, "args 必须是 JSON object"
        return tool.run(args), ""


# =========================
# Built-in tools
# =========================
class VerifyMainProofTool:
    name = "verify_proof"

    def __init__(self, client: "ProofTaskClient"):
        self.client = client

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "验证主定理证明。原主定理 proof 不可见；已声明 lemma 在运行期均按 Admitted 使用。",
            "args_schema": {
                "type": "object",
                "required": ["proof"],
                "properties": {
                    "proof": {"type": "string", "description": "主定理证明文本，可完整或未完成"},
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        proof = args.get("proof", "")
        if not isinstance(proof, str) or not proof.strip():
            return {"success": False, "error": "proof 不能为空字符串"}
        return self.client.verify_main_theorem_proof(proof)


class StepMainTacticTool:
    name = "step_tactic"

    def __init__(self, client: "ProofTaskClient"):
        self.client = client

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "单步追加 tactic 到主定理 proof，并返回当前证明状态与更新后的 proof prefix。",
            "args_schema": {
                "type": "object",
                "required": ["proof_prefix", "tactic"],
                "properties": {
                    "proof_prefix": {"type": "string", "description": "当前主定理证明前缀"},
                    "tactic": {"type": "string", "description": "单步 tactic（单行）"},
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        proof_prefix = args.get("proof_prefix", "")
        tactic = args.get("tactic", "")
        if not isinstance(proof_prefix, str) or not proof_prefix.strip():
            return {"success": False, "error": "proof_prefix 不能为空字符串"}
        if not isinstance(tactic, str) or not tactic.strip():
            return {"success": False, "error": "tactic 不能为空字符串"}
        if "\n" in tactic or "\r" in tactic:
            return {"success": False, "error": "tactic 必须是单行单步命令"}
        next_proof = proof_prefix.rstrip() + "\n" + tactic.strip()
        result = self.client.verify_main_theorem_proof(next_proof)
        result["current_proof"] = next_proof
        result["step_appended"] = tactic.strip()
        return result


class VerifyLemmaProofTool:
    name = "verify_proof"

    def __init__(self, client: "ProofTaskClient", lemma_name: str, lemma_declaration: str):
        self.client = client
        self.lemma_name = lemma_name
        self.lemma_declaration = lemma_declaration

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "验证当前辅助 lemma 的证明。主定理只以 Admitted 形式存在。",
            "args_schema": {
                "type": "object",
                "required": ["proof"],
                "properties": {
                    "proof": {"type": "string", "description": "lemma 证明文本，可完整或未完成"},
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        proof = args.get("proof", "")
        if not isinstance(proof, str) or not proof.strip():
            return {"success": False, "error": "proof 不能为空字符串"}
        return self.client.verify_lemma_proof(self.lemma_name, self.lemma_declaration, proof)


class StepLemmaTacticTool:
    name = "step_tactic"

    def __init__(self, client: "ProofTaskClient", lemma_name: str, lemma_declaration: str):
        self.client = client
        self.lemma_name = lemma_name
        self.lemma_declaration = lemma_declaration

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "单步追加 tactic 到当前 lemma proof，并返回当前证明状态与更新后的 proof prefix。",
            "args_schema": {
                "type": "object",
                "required": ["proof_prefix", "tactic"],
                "properties": {
                    "proof_prefix": {"type": "string", "description": "当前 lemma 证明前缀"},
                    "tactic": {"type": "string", "description": "单步 tactic（单行）"},
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        proof_prefix = args.get("proof_prefix", "")
        tactic = args.get("tactic", "")
        if not isinstance(proof_prefix, str) or not proof_prefix.strip():
            return {"success": False, "error": "proof_prefix 不能为空字符串"}
        if not isinstance(tactic, str) or not tactic.strip():
            return {"success": False, "error": "tactic 不能为空字符串"}
        if "\n" in tactic or "\r" in tactic:
            return {"success": False, "error": "tactic 必须是单行单步命令"}
        next_proof = proof_prefix.rstrip() + "\n" + tactic.strip()
        result = self.client.verify_lemma_proof(self.lemma_name, self.lemma_declaration, next_proof)
        result["current_proof"] = next_proof
        result["step_appended"] = tactic.strip()
        return result


class PrintInSyntheticContextTool:
    name = "print"

    def __init__(
        self,
        client: "ProofTaskClient",
        mode: str,
        lemma_name: Optional[str] = None,
        lemma_declaration: Optional[str] = None,
    ):
        self.client = client
        self.mode = mode
        self.lemma_name = lemma_name
        self.lemma_declaration = lemma_declaration

    def spec(self) -> Dict[str, Any]:
        description = (
            "在主定理重证明工作区中打印定义。"
            if self.mode == "main"
            else "在当前 lemma 工作区中打印定义；主定理以 Admitted 存在。"
        )
        return {
            "name": self.name,
            "description": description,
            "args_schema": {
                "type": "object",
                "required": ["definition"],
                "properties": {
                    "definition": {"type": "string", "description": "定义名，例如 unique_key_in"},
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        definition = args.get("definition", "")
        if not isinstance(definition, str) or not definition.strip():
            return {"success": False, "error": "definition 不能为空字符串"}
        definition_name = definition.strip().rstrip(".").strip()
        if not definition_name:
            return {"success": False, "error": "definition 不能为空字符串"}
        result = self.client.print_definition(
            definition_name,
            mode=self.mode,
            lemma_name=self.lemma_name,
            lemma_declaration=self.lemma_declaration,
        )
        result["definition_name"] = definition_name
        return result


class BM25SearchTool:
    name = "bm25_search"

    _DECL_RE = re.compile(
        r"^\s*(Theorem|Lemma|Corollary|Proposition|Fact|Remark|Definition|Fixpoint|CoFixpoint|Inductive|CoInductive|Record|Class|Instance)\s+([A-Za-z0-9_']+)?"
    )
    _TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")

    def __init__(self, repo_path: str, current_file_relpath: str):
        self.repo_path = repo_path
        self.current_file_relpath = current_file_relpath.replace("\\", "/")
        self.current_dir_relpath = os.path.dirname(self.current_file_relpath).replace("\\", "/")
        self._docs: List[Dict[str, Any]] = []
        self._df: Dict[str, int] = {}
        self._avgdl: float = 1.0
        self._built = False
        self._build_lock = threading.Lock()

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "BM25 检索相关 theorem/lemma/definition 等声明，辅助证明。",
            "args_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，例如 rewrite equality lemma"},
                    "k": {"type": "integer", "description": "返回条数（默认 8，最大 30）"},
                    "scope": {"type": "string", "description": "检索范围：current_file/current_dir/repo"},
                },
            },
        }

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in self._TOKEN_RE.findall(text)]

    def _build_index(self) -> None:
        docs: List[Dict[str, Any]] = []
        for root, _, files in os.walk(self.repo_path):
            for filename in files:
                if not filename.endswith(".v"):
                    continue
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, self.repo_path).replace("\\", "/")
                try:
                    with open(abs_path, "r", encoding="utf-8") as handle:
                        lines = handle.readlines()
                except Exception:
                    continue
                index = 0
                while index < len(lines):
                    match = self._DECL_RE.match(lines[index])
                    if not match:
                        index += 1
                        continue
                    kind = match.group(1)
                    name = match.group(2) or "(anonymous)"
                    start = index
                    block = [lines[index].rstrip("\n")]
                    next_index = index + 1
                    while "." not in block[-1] and next_index < len(lines) and next_index - start <= 8:
                        block.append(lines[next_index].rstrip("\n"))
                        next_index += 1
                    text = " ".join(line.strip() for line in block if line.strip())
                    tokens = self._tokenize(text)
                    if tokens:
                        docs.append(
                            {
                                "kind": kind,
                                "name": name,
                                "file": rel_path,
                                "line": start + 1,
                                "text": text[:500],
                                "tokens": tokens,
                                "len": len(tokens),
                            }
                        )
                    index = max(next_index, index + 1)
        df: Dict[str, int] = {}
        for doc in docs:
            for token in set(doc["tokens"]):
                df[token] = df.get(token, 0) + 1
        self._docs = docs
        self._df = df
        self._avgdl = sum(doc["len"] for doc in docs) / len(docs) if docs else 1.0
        self._built = True

    def ensure_index(self) -> None:
        with self._build_lock:
            if not self._built:
                self._build_index()

    def _in_scope(self, rel_file: str, scope: str) -> bool:
        if scope == "current_file":
            return rel_file == self.current_file_relpath
        if scope == "current_dir":
            if not self.current_dir_relpath:
                return True
            return rel_file.startswith(self.current_dir_relpath + "/") or rel_file == self.current_dir_relpath
        return True

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return {"success": False, "error": "query 不能为空字符串"}
        k = args.get("k", 8)
        if not isinstance(k, int):
            return {"success": False, "error": "k 必须是整数"}
        k = max(1, min(30, k))
        scope = args.get("scope", "repo")
        if scope not in {"current_file", "current_dir", "repo"}:
            return {"success": False, "error": "scope 必须是 current_file/current_dir/repo"}
        self.ensure_index()
        if not self._docs:
            return {"success": False, "error": "未找到可索引的 Coq 声明"}
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return {"success": False, "error": "query 解析后为空"}
        total = len(self._docs)
        k1 = 1.5
        b = 0.75
        results: List[Tuple[float, Dict[str, Any]]] = []
        for doc in self._docs:
            if not self._in_scope(doc["file"], scope):
                continue
            tf: Dict[str, int] = {}
            for token in doc["tokens"]:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            for query_token in q_tokens:
                freq = tf.get(query_token, 0)
                if freq == 0:
                    continue
                doc_freq = self._df.get(query_token, 0)
                idf = math.log((total - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                denom = freq + k1 * (1 - b + b * (doc["len"] / self._avgdl))
                score += idf * (freq * (k1 + 1) / max(denom, 1e-9))
            if score <= 0:
                continue
            if scope == "repo":
                if doc["file"] == self.current_file_relpath:
                    score *= 1.15
                elif self.current_dir_relpath and doc["file"].startswith(self.current_dir_relpath + "/"):
                    score *= 1.08
            results.append((score, doc))
        results.sort(key=lambda item: item[0], reverse=True)
        top = results[:k]
        return {
            "success": True,
            "scope": scope,
            "query": query,
            "k": k,
            "hits": [
                {
                    "score": round(score, 4),
                    "kind": doc["kind"],
                    "name": doc["name"],
                    "file": doc["file"],
                    "line": doc["line"],
                    "text": doc["text"],
                }
                for score, doc in top
            ],
        }


class NaturalLanguageProofTool:
    name = "natural_language_proof"

    def __init__(self, task: TaskContext, model: str = NL_PROOF_MODEL):
        self.task = task
        self.model = model

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "调用强模型生成当前上下文下的自然语言证明思路与步骤。",
            "args_schema": {
                "type": "object",
                "required": [],
                "properties": {
                    "proof_prefix": {"type": "string", "description": "可选：当前证明前缀"},
                    "question": {"type": "string", "description": "可选：希望重点解释的问题"},
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not OPENAI_API_KEY:
            return {"success": False, "error": "OPENAI_API_KEY is empty"}
        proof_prefix = args.get("proof_prefix", "")
        question = args.get("question", "")
        if proof_prefix is not None and not isinstance(proof_prefix, str):
            return {"success": False, "error": "proof_prefix 必须是字符串"}
        if question is not None and not isinstance(question, str):
            return {"success": False, "error": "question 必须是字符串"}
        prompt = f"""You are a Coq expert.
Given the theorem context below, produce a concise natural-language proof plan.

Task:
- theorem_id: {self.task.theorem_id}
- project: {self.task.project}
- file: {self.task.file_relpath}

[Theorem Statement]
{self.task.theorem_statement}

[Header Context]
{self.task.header_context}

[Upper Context]
{self.task.upper_context}

[Current Proof Prefix]
{proof_prefix or "(none)"}

[Focus Question]
{question or "(none)"}

Requirements:
1) Explain the proof idea in plain language.
2) Give 3-8 concrete tactic-level steps.
3) Mention likely useful lemmas/definitions to inspect.
4) Keep it brief and actionable.
"""
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            return {
                "success": True,
                "model": self.model,
                "natural_language_proof": text,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"natural language proof generation failed: {exc}",
                "model": self.model,
            }


# =========================
# Model driver
# =========================
class ModelDriver(Protocol):
    def next(self, messages: List[Dict[str, str]]) -> str:
        ...


class OpenAIModelDriver:
    """Use OpenAI SDK with key from environment variable OPENAI_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = 0.0):
        self.model = model
        self.api_key = OPENAI_API_KEY
        self.base_url = OPENAI_BASE_URL
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is empty. Set it in your shell environment before running.")

    @staticmethod
    def _build_query(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = [
            "You must return exactly one JSON object. No markdown. No explanations.",
            "Conversation history follows. Continue based on the latest user message.",
        ]
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"[{role}]\n{content}")
        parts.append(
            "再次强调：仅输出 JSON。格式如 "
            '{"action":"verify_proof","args":{"proof":"Proof. ..."}} '
            '或 {"action":"print","args":{"definition":"foo"}} '
            '或 {"action":"step_tactic","args":{"proof_prefix":"Proof. ...","tactic":"intros x."}} '
            '或 {"action":"bm25_search","args":{"query":"rewrite equality lemma","k":8,"scope":"current_dir"}} '
            '或 {"action":"natural_language_proof","args":{"proof_prefix":"Proof. ...","question":"what is the key idea?"}} '
            '或 {"action":"enter_lemma_mode","args":{"lemma_name":"aux1","lemma_statement":"forall x, ..."}}'
        )
        return "\n\n".join(parts)

    def next(self, messages: List[Dict[str, str]]) -> str:
        from openai import OpenAI  # type: ignore

        query = self._build_query(messages)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": query}],
        )
        return (response.choices[0].message.content or "").strip()


# =========================
# ProofTaskClient
# =========================
class ProofTaskClient:
    def __init__(self, theorem_id: str, context_lines: int = 80, coqstoq_path: Optional[str] = None):
        self.verifier = CoqProofVerifier(coqstoq_path=coqstoq_path)
        self.task, self._theorem_def, self.source_slices = self._build_task_context(theorem_id, context_lines)
        self.lemma_registry: Optional[LemmaRegistry] = None
        self.theorem_registry = ToolRegistry()
        self.registry = self.theorem_registry
        self._bm25 = BM25SearchTool(self.task.repo_path, self.task.file_relpath)
        self._bm25.ensure_index()
        self._register_builtin_tools()

    def _build_task_context(
        self, theorem_id: str, context_lines: int
    ) -> Tuple[TaskContext, Dict[str, Any], SourceSlices]:
        split_name, index = self.verifier._parse_theorem_id(theorem_id)  # pylint: disable=protected-access
        theorem_def = self.verifier._load_theorem_definition(split_name, index)  # pylint: disable=protected-access
        if theorem_def is None:
            raise ValueError(f"Theorem not found: {theorem_id}")

        repo_path = os.path.join(
            self.verifier.coqstoq_path,
            theorem_def["project"]["split"]["dir_name"],
            theorem_def["project"]["dir_name"],
        )
        source_file = os.path.join(repo_path, theorem_def["path"])
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")
        with open(source_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        theorem_start_line = theorem_def["theorem_start_pos"]["line"]
        compile_args = theorem_def["project"].get("compile_args", [])
        prefixes = (
            "From ",
            "Require ",
            "Import ",
            "Export ",
            "Open Scope ",
            "Local Open Scope ",
            "Set ",
            "Unset ",
        )
        headers = [line.rstrip("\n") for line in lines[:theorem_start_line] if line.lstrip().startswith(prefixes)]
        header_context = "\n".join(headers[-120:]) if headers else "(无显式头部依赖语句)"
        start = max(0, theorem_start_line - context_lines)
        upper_context = "".join(lines[start:theorem_start_line]).rstrip("\n")
        if not upper_context.strip():
            upper_context = "(No additional local context before theorem)"

        theorem_end_pos = theorem_def.get("theorem_end_pos", {})
        theorem_end_line = theorem_end_pos.get("line", theorem_start_line)
        theorem_end_column = theorem_end_pos.get("column", 0)
        theorem_decl_lines = lines[theorem_start_line:theorem_end_line + 1].copy()
        if theorem_decl_lines:
            theorem_decl_lines[-1] = theorem_decl_lines[-1][:theorem_end_column]
        theorem_declaration = "".join(theorem_decl_lines)

        theorem_statement = theorem_declaration.strip() or "(unable to extract theorem declaration)"

        proof_end_pos = theorem_def.get("proof_end_pos", {})
        proof_end_line = proof_end_pos.get("line", theorem_end_line)
        proof_end_column = proof_end_pos.get("column", 0)
        suffix_lines = lines[proof_end_line:].copy()
        if suffix_lines:
            suffix_lines[0] = suffix_lines[0][proof_end_column:]
        after_proof = "".join(suffix_lines)

        task = TaskContext(
            theorem_id=theorem_id,
            project=theorem_def["project"]["dir_name"],
            file_relpath=theorem_def["path"],
            repo_path=repo_path,
            compile_args=compile_args,
            theorem_statement=theorem_statement,
            header_context=header_context,
            upper_context=upper_context,
        )
        slices = SourceSlices(
            before_theorem="".join(lines[:theorem_start_line]),
            theorem_declaration=theorem_declaration,
            after_proof=after_proof,
        )
        return task, theorem_def, slices

    @staticmethod
    def build_lemma_declaration(lemma_name: str, lemma_statement: str) -> str:
        statement = lemma_statement.strip()
        if not statement:
            return ""
        if re.match(r"^\s*(Lemma|Theorem|Corollary|Proposition|Fact|Remark)\b", statement):
            return statement if statement.endswith(".") else statement + "."
        name = lemma_name.strip() or "aux_lemma"
        body = statement.rstrip(".").strip()
        return f"Lemma {name} : {body}."

    def get_runtime_lemma_prelude_for_main(self) -> str:
        if self.lemma_registry is None:
            return ""
        return self.lemma_registry.get_runtime_prelude_for_main()

    def get_runtime_lemma_prelude_for_lemma(self, exclude_name: Optional[str] = None) -> str:
        if self.lemma_registry is None:
            return ""
        return self.lemma_registry.get_runtime_prelude_for_lemma(exclude_name=exclude_name)

    @staticmethod
    def _append_block(base: str, block: str) -> str:
        text = block.strip("\n")
        if not text:
            return base
        if base and not base.endswith("\n"):
            base += "\n"
        base += text
        return base

    @staticmethod
    def _append_suffix(base: str, suffix: str) -> str:
        if not suffix:
            return base
        if base and not base.endswith("\n") and not suffix.startswith("\n"):
            base += "\n"
        base += suffix
        return base

    @staticmethod
    def _strip_trailing_qed(proof_content: str) -> str:
        text = proof_content.strip()
        if text.endswith("Qed."):
            return text[:-len("Qed.")].rstrip()
        return text

    @staticmethod
    def _normalize_partial_proof(proof_content: str) -> str:
        text = proof_content.strip()
        if text.startswith("Proof."):
            text = text[len("Proof."):].lstrip()
        if text.endswith("Qed."):
            text = text[:-len("Qed.")].rstrip()
        return text

    def build_synthetic_source(
        self,
        mode: str,
        theorem_proof: str = "",
        lemma_name: Optional[str] = None,
        lemma_declaration: str = "",
        lemma_proof: str = "",
        runtime_prelude: str = "",
        final_prelude: str = "",
    ) -> str:
        content = self.source_slices.before_theorem
        if mode == "theorem_runtime":
            content = self._append_block(content, runtime_prelude)
            content = self._append_block(content, self.source_slices.theorem_declaration)
            content = self._append_block(content, self._strip_trailing_qed(theorem_proof))
            content = self._append_block(content, "Qed.")
            return self._append_suffix(content, self.source_slices.after_proof)
        if mode == "theorem_final":
            content = self._append_block(content, final_prelude)
            content = self._append_block(content, self.source_slices.theorem_declaration)
            content = self._append_block(content, self._strip_trailing_qed(theorem_proof))
            content = self._append_block(content, "Qed.")
            return self._append_suffix(content, self.source_slices.after_proof)
        if mode == "lemma_runtime":
            content = self._append_block(content, runtime_prelude)
            content = self._append_block(content, lemma_declaration)
            content = self._append_block(content, self._strip_trailing_qed(lemma_proof))
            content = self._append_block(content, "Qed.")
            content = self._append_block(content, self.source_slices.theorem_declaration)
            content = self._append_block(content, "Admitted.")
            return self._append_suffix(content, self.source_slices.after_proof)
        if mode == "print_runtime":
            content = self._append_block(content, runtime_prelude)
            if lemma_name and lemma_declaration:
                content = self._append_block(content, lemma_declaration)
                content = self._append_block(content, "Admitted.")
            content = self._append_block(content, self.source_slices.theorem_declaration)
            content = self._append_block(content, "Admitted.")
            return self._append_suffix(content, self.source_slices.after_proof)
        raise ValueError(f"unsupported synthetic mode: {mode}")

    def _run_show_script(self, script: str) -> Optional[Dict[str, Any]]:
        command = [self.verifier.coqtop_path] + self.task.compile_args + ["-quiet"]
        try:
            proc = subprocess.run(
                command,
                input=script if script.endswith("\n") else script + "\n",
                text=True,
                capture_output=True,
                cwd=self.task.repo_path,
                timeout=60,
            )
        except Exception:
            return None
        raw = (proc.stdout or "") + (proc.stderr or "")
        parsed = self.verifier._parse_show_state(raw)  # pylint: disable=protected-access
        if parsed is None:
            return None
        parsed["raw_show_output"] = raw
        return parsed

    def get_main_proof_state_with_show(self, proof_content: str) -> Optional[Dict[str, Any]]:
        script = self.source_slices.before_theorem
        script = self._append_block(script, self.get_runtime_lemma_prelude_for_main())
        script = self._append_block(script, self.source_slices.theorem_declaration)
        script = self._append_block(script, "Proof.")
        partial = self._normalize_partial_proof(proof_content)
        if partial:
            script = self._append_block(script, partial)
        script = self._append_block(script, "Show.")
        script = self._append_block(script, "Quit.")
        return self._run_show_script(script)

    def get_lemma_proof_state_with_show(
        self,
        lemma_name: str,
        lemma_declaration: str,
        proof_content: str,
    ) -> Optional[Dict[str, Any]]:
        script = self.source_slices.before_theorem
        script = self._append_block(script, self.get_runtime_lemma_prelude_for_lemma(exclude_name=lemma_name))
        script = self._append_block(script, lemma_declaration)
        script = self._append_block(script, "Proof.")
        partial = self._normalize_partial_proof(proof_content)
        if partial:
            script = self._append_block(script, partial)
        script = self._append_block(script, "Show.")
        script = self._append_block(script, "Abort.")
        script = self._append_block(script, self.source_slices.theorem_declaration)
        script = self._append_block(script, "Admitted.")
        script = self._append_block(script, "Quit.")
        return self._run_show_script(script)

    def _compile_synthetic_source(self, source: str, label: str) -> Tuple[bool, str, str]:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{label}.v",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(source)
            temp_path = handle.name
        command = [self.verifier.coqc_path] + self.task.compile_args + [temp_path]
        env = os.environ.copy()
        coqc_dir = str(Path(self.verifier.coqc_path).resolve().parent) if self.verifier.coqc_path != "coqc" else ""
        if coqc_dir and coqc_dir not in env.get("PATH", ""):
            env["PATH"] = coqc_dir + ":" + env.get("PATH", "")
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
                cwd=self.task.repo_path,
            )
            combined = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode == 0, combined, temp_path
        except subprocess.TimeoutExpired:
            return False, "编译超时", temp_path
        except Exception as exc:
            return False, f"编译错误: {exc}", temp_path

    def verify_main_theorem_proof(self, proof_content: str) -> Dict[str, Any]:
        result = self.verifier.verify_proof(
            self.task.theorem_id,
            proof_content,
            injected_prelude=self.get_runtime_lemma_prelude_for_main(),
        )
        if result.get("state") == "in_progress":
            result["proof_state"] = self.get_main_proof_state_with_show(proof_content)
        return result

    def verify_main_theorem_proof_final(self, proof_content: str, final_prelude: str) -> Dict[str, Any]:
        result = self.verifier.verify_proof(
            self.task.theorem_id,
            proof_content,
            injected_prelude=final_prelude,
        )
        if result.get("state") == "in_progress":
            result["proof_state"] = self.get_main_proof_state_with_show(proof_content)
        return result

    def verify_lemma_proof(
        self,
        lemma_name: str,
        lemma_declaration: str,
        proof_content: str,
    ) -> Dict[str, Any]:
        if not lemma_declaration.strip():
            return {"success": False, "state": "error", "error": "lemma declaration is empty"}
        if not isinstance(proof_content, str) or not proof_content.strip():
            return {"success": False, "state": "error", "error": "proof 不能为空字符串"}
        source = self.build_synthetic_source(
            "lemma_runtime",
            lemma_name=lemma_name,
            lemma_declaration=lemma_declaration,
            lemma_proof=proof_content,
            runtime_prelude=self.get_runtime_lemma_prelude_for_lemma(exclude_name=lemma_name),
        )
        success, combined, temp_path = self._compile_synthetic_source(source, f"lemma_{lemma_name}")
        has_qed = "Qed" in proof_content
        if success:
            return {
                "success": True,
                "state": "proven",
                "proof_status": "lemma proven",
                "lemma_declaration": lemma_declaration,
                "proof_content": proof_content,
                "temp_file": temp_path,
                "compilation_output": combined,
            }
        if has_qed:
            return {
                "success": False,
                "state": "failed",
                "error": combined[-1200:],
                "proof_status": "lemma failed",
                "lemma_declaration": lemma_declaration,
                "proof_content": proof_content,
                "temp_file": temp_path,
                "compilation_output": combined,
            }
        proof_state = self.get_lemma_proof_state_with_show(lemma_name, lemma_declaration, proof_content)
        state = "in_progress" if proof_state is not None else "failed"
        return {
            "success": False,
            "state": state,
            "error": None if state == "in_progress" else combined[-1200:],
            "proof_status": "lemma in progress" if state == "in_progress" else "lemma step failed",
            "lemma_declaration": lemma_declaration,
            "proof_content": proof_content,
            "proof_state": proof_state,
            "temp_file": temp_path,
            "compilation_output": combined,
        }

    def print_definition(
        self,
        definition_name: str,
        mode: str,
        lemma_name: Optional[str] = None,
        lemma_declaration: Optional[str] = None,
    ) -> Dict[str, Any]:
        if mode == "main":
            source = self.build_synthetic_source(
                "print_runtime",
                runtime_prelude=self.get_runtime_lemma_prelude_for_main(),
            )
        elif mode == "lemma":
            source = self.build_synthetic_source(
                "print_runtime",
                lemma_name=lemma_name,
                lemma_declaration=lemma_declaration or "",
                runtime_prelude=self.get_runtime_lemma_prelude_for_lemma(exclude_name=lemma_name),
            )
        else:
            return {"success": False, "error": f"unknown print mode: {mode}"}
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_print_workspace.v",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(source)
            temp_path = handle.name
        escaped = temp_path.replace("\\", "\\\\").replace('"', '\\"')
        result = execute_print_command(
            query_command=f"Print {definition_name}.",
            setup_commands=[f'Load "{escaped}".'],
            compile_args=self.task.compile_args,
            cwd=self.task.repo_path,
        )
        result["temp_file"] = temp_path
        return result

    def _register_builtin_tools(self) -> None:
        self.theorem_registry.register(VerifyMainProofTool(self))
        self.theorem_registry.register(PrintInSyntheticContextTool(self, mode="main"))
        self.theorem_registry.register(StepMainTacticTool(self))
        self.theorem_registry.register(self._bm25)
        self.theorem_registry.register(NaturalLanguageProofTool(self.task))

    def build_lemma_tool_registry(self, lemma_name: str, lemma_declaration: str) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(VerifyLemmaProofTool(self, lemma_name, lemma_declaration))
        registry.register(PrintInSyntheticContextTool(self, mode="lemma", lemma_name=lemma_name, lemma_declaration=lemma_declaration))
        registry.register(StepLemmaTacticTool(self, lemma_name, lemma_declaration))
        registry.register(self._bm25)
        return registry

    def register_tool(self, tool: Tool) -> None:
        self.theorem_registry.register(tool)

    def build_system_prompt(self) -> str:
        tool_specs = json.dumps(self.theorem_registry.specs(), ensure_ascii=False, indent=2)
        return f"""You are a Coq proof assistant. You must call tools via JSON actions.

Task info:
- theorem_id: {self.task.theorem_id}
- project: {self.task.project}
- file: {self.task.file_relpath}
- repo_path: {self.task.repo_path}
- compile_args: {self.task.compile_args}

Available tools:
{tool_specs}

You must return exactly one of the following JSON objects:
{{"action":"verify_proof","args":{{"proof":"Proof. ..."}}}}
{{"action":"print","args":{{"definition":"foo"}}}}
{{"action":"step_tactic","args":{{"proof_prefix":"Proof. ...","tactic":"intros x."}}}}
{{"action":"bm25_search","args":{{"query":"rewrite equality lemma","k":8,"scope":"current_dir"}}}}
{{"action":"natural_language_proof","args":{{"proof_prefix":"Proof. ...","question":"what is the key idea?"}}}}
{{"action":"enter_lemma_mode","args":{{"lemma_name":"aux1","lemma_statement":"forall x, ..."}}}}

Do not return markdown. Do not return any extra text.

Global rules:
- You are reproving the main theorem from scratch.
- You cannot use the original theorem proof.
- Every auxiliary lemma you declare is inserted before the theorem and is immediately available as Admitted.
- Even if a background lemma prover finishes, you should still treat runtime lemmas as admitted and keep focusing on the main theorem proof.
- Do not repeat the exact same action with the same arguments in the same proof state.

Proof strategy guidance:
- You may first attempt a full proof in one shot up to `Qed.`.
- If that fails, switch to iterative exploration.
- Use `verify_proof` frequently to debug and refine.
- Use `print` whenever you need to inspect definitions/lemmas in the current synthetic workspace.
- Use `step_tactic` for single-step exploration; it returns current proof state and updated proof prefix.
- Use `bm25_search` to retrieve nearby related theorem/lemma/definition candidates before trying new tactics.
- Use `natural_language_proof` when you need a stronger model to generate a human-readable proof plan under current context.
- If a key auxiliary lemma is needed, call `enter_lemma_mode`. This spawns a background sub-agent, the lemma is immediately available to you as Admitted, and you should continue proving the main theorem without waiting.

Context:
[Theorem Statement]
{self.task.theorem_statement}

[Header Context]
{self.task.header_context}

[Upper Context: previous lines]
{self.task.upper_context}
"""

    def build_lemma_system_prompt(self, lemma_name: str, lemma_declaration: str) -> str:
        tool_specs = json.dumps(
            self.build_lemma_tool_registry(lemma_name, lemma_declaration).specs(),
            ensure_ascii=False,
            indent=2,
        )
        return f"""You are a Coq proof assistant. Your task is to prove one auxiliary lemma inserted before the main theorem.

Lemma to prove:
{lemma_declaration}

You must return exactly one JSON action per turn:
{{"action":"verify_proof","args":{{"proof":"Proof. ... Qed."}}}}
{{"action":"step_tactic","args":{{"proof_prefix":"Proof. ...","tactic":"intros x."}}}}
{{"action":"print","args":{{"definition":"foo"}}}}
{{"action":"bm25_search","args":{{"query":"rewrite equality","k":8,"scope":"current_dir"}}}}

Do not return markdown. Do not return any extra text.

Global rules:
- You are proving only this one auxiliary lemma.
- This lemma will be inserted before the main theorem.
- The main theorem exists only as an admitted declaration for context compatibility.
- You cannot use the original theorem proof.
- Do not call `enter_lemma_mode`.

Available tools:
{tool_specs}

Context from the main theorem's file:
- project: {self.task.project}
- file: {self.task.file_relpath}

[Header Context]
{self.task.header_context}

[Upper Context]
{self.task.upper_context}
"""

    def build_initial_user_prompt(self) -> str:
        return (
            "Start reproving the main theorem iteratively. Runtime lemmas are available as Admitted only. "
            "Use `step_tactic` for single-step progress, `bm25_search` for related lemmas/definitions, "
            "`natural_language_proof` for proof planning, `print` for definitions, and `verify_proof` for complete-proof validation."
        )


# =========================
# Action parsing & helpers
# =========================
THEOREM_ACTIONS: Set[str] = {
    "verify_proof",
    "print",
    "step_tactic",
    "bm25_search",
    "natural_language_proof",
    "enter_lemma_mode",
}
LEMMA_ACTIONS: Set[str] = {"verify_proof", "print", "step_tactic", "bm25_search"}


def parse_action(raw: str, allowed_actions: Optional[Set[str]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    if allowed_actions is None:
        allowed_actions = THEOREM_ACTIONS
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failed: {exc}"
    if not isinstance(obj, dict):
        return None, "Response must be a JSON object"
    extra_top = sorted(set(obj.keys()) - {"action", "args"})
    if extra_top:
        return None, f"unexpected top-level keys: {extra_top}"
    action = obj.get("action")
    if action not in allowed_actions:
        return None, f"action must be one of: {' / '.join(sorted(allowed_actions))}"
    args = obj.get("args")
    if not isinstance(args, dict):
        return None, "args must be a JSON object"
    if action == "verify_proof":
        extra = sorted(set(args.keys()) - {"proof"})
        if extra:
            return None, f"verify_proof.args has unexpected keys: {extra}"
        proof = args.get("proof")
        if not isinstance(proof, str) or not proof.strip():
            return None, "verify_proof.args.proof must be a non-empty string"
    if action == "print":
        extra = sorted(set(args.keys()) - {"definition"})
        if extra:
            return None, f"print.args has unexpected keys: {extra}"
        definition = args.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            return None, "print.args.definition must be a non-empty string"
    if action == "step_tactic":
        extra = sorted(set(args.keys()) - {"proof_prefix", "tactic"})
        if extra:
            return None, f"step_tactic.args has unexpected keys: {extra}"
        proof_prefix = args.get("proof_prefix")
        tactic = args.get("tactic")
        if not isinstance(proof_prefix, str) or not proof_prefix.strip():
            return None, "step_tactic.args.proof_prefix must be a non-empty string"
        if not isinstance(tactic, str) or not tactic.strip():
            return None, "step_tactic.args.tactic must be a non-empty string"
        if "\n" in tactic or "\r" in tactic:
            return None, "step_tactic.args.tactic must be a single-line tactic"
    if action == "bm25_search":
        extra = sorted(set(args.keys()) - {"query", "k", "scope"})
        if extra:
            return None, f"bm25_search.args has unexpected keys: {extra}"
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return None, "bm25_search.args.query must be a non-empty string"
        if "k" in args and not isinstance(args.get("k"), int):
            return None, "bm25_search.args.k must be an integer"
        if "scope" in args and args.get("scope") not in {"current_file", "current_dir", "repo"}:
            return None, "bm25_search.args.scope must be current_file/current_dir/repo"
    if action == "natural_language_proof":
        extra = sorted(set(args.keys()) - {"proof_prefix", "question"})
        if extra:
            return None, f"natural_language_proof.args has unexpected keys: {extra}"
        if "proof_prefix" in args and not isinstance(args.get("proof_prefix"), str):
            return None, "natural_language_proof.args.proof_prefix must be a string"
        if "question" in args and not isinstance(args.get("question"), str):
            return None, "natural_language_proof.args.question must be a string"
    if action == "enter_lemma_mode":
        extra = sorted(set(args.keys()) - {"lemma_name", "lemma_statement"})
        if extra:
            return None, f"enter_lemma_mode.args has unexpected keys: {extra}"
        lemma_statement = args.get("lemma_statement")
        if not isinstance(lemma_statement, str) or not lemma_statement.strip():
            return None, "enter_lemma_mode.args.lemma_statement must be a non-empty string"
        if "lemma_name" in args and not isinstance(args.get("lemma_name"), str):
            return None, "enter_lemma_mode.args.lemma_name must be a string"
    return obj, ""


def _normalize_action_signature(action_obj: Dict[str, Any]) -> str:
    action = str(action_obj.get("action") or "")
    args = action_obj.get("args") if isinstance(action_obj.get("args"), dict) else {}
    normalized_args: Dict[str, Any] = {}
    if isinstance(args, dict):
        for key, value in args.items():
            if isinstance(value, str):
                normalized_args[key] = " ".join(value.split())
            else:
                normalized_args[key] = value
    return json.dumps({"action": action, "args": normalized_args}, ensure_ascii=False, sort_keys=True)


def _compact_tool_result(action: str, result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "success",
        "state",
        "error",
        "error_message",
        "proof_status",
        "lemma_declaration",
        "proof_state",
        "definition_name",
        "current_proof",
        "step_appended",
        "command",
        "model",
        "registration_status",
    ):
        if key in result:
            out[key] = result[key]
    if action == "print":
        if result.get("success"):
            out["output"] = "Definition viewed; full body cleared from context."
        else:
            raw_error = result.get("error") or result.get("error_message") or ""
            out["output"] = str(raw_error)[:400]
    elif "output" in result:
        out["output"] = str(result.get("output", ""))[:800]
    if action == "bm25_search":
        hits = result.get("hits")
        if isinstance(hits, list):
            compact_hits = []
            for hit in hits[:10]:
                if not isinstance(hit, dict):
                    continue
                compact_hits.append(
                    {
                        "score": hit.get("score"),
                        "kind": hit.get("kind"),
                        "name": hit.get("name"),
                        "file": hit.get("file"),
                        "line": hit.get("line"),
                    }
                )
            out["hits"] = compact_hits
            out["hit_count"] = len(hits)
        if "query" in result:
            out["query"] = result.get("query")
        if "scope" in result:
            out["scope"] = result.get("scope")
    if action == "natural_language_proof":
        out["natural_language_proof"] = str(result.get("natural_language_proof", ""))[:2200]
    return out


# =========================
# ProofAgent
# =========================
class ProofAgent:
    """Generic proof agent usable for both the main theorem and lemma sub-agents."""

    def __init__(
        self,
        client: ProofTaskClient,
        driver: ModelDriver,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        allowed_actions: Set[str],
        system_prompt: str,
        initial_user_prompt: str,
        spawn_lemma_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        on_successful_proof: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.client = client
        self.driver = driver
        self.config = config
        self.tool_registry = tool_registry
        self.allowed_actions = allowed_actions
        self.spawn_lemma_fn = spawn_lemma_fn
        self.on_successful_proof = on_successful_proof
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_prompt},
        ]
        self.action_cache: Dict[str, Dict[str, Any]] = {}
        self.state_action_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.current_state_fingerprint = "__initial__"

    @staticmethod
    def _fingerprint_from_result(result: Dict[str, Any]) -> Optional[str]:
        proof_state = result.get("proof_state")
        if proof_state is not None:
            try:
                return "proof_state:" + json.dumps(proof_state, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return "proof_state:" + str(proof_state)
        for key in ("current_proof", "proof_content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return "proof:" + " ".join(value.split())
        return None

    def _record_main_action(self, state_fingerprint: str, action_sig: str, feedback: Dict[str, Any]) -> None:
        bucket = self.state_action_cache.setdefault(state_fingerprint, {})
        bucket[action_sig] = feedback

    def _main_duplicate_feedback(self, state_fingerprint: str, action_sig: str) -> Optional[Dict[str, Any]]:
        bucket = self.state_action_cache.get(state_fingerprint, {})
        if action_sig not in bucket:
            return None
        return {
            "success": False,
            "error": (
                "Duplicate action detected in the same proof state. Change tactic, "
                "inspect a definition, search for lemmas, or spawn a different auxiliary lemma."
            ),
            "state_fingerprint": state_fingerprint[:400],
            "cached_result": bucket[action_sig],
        }

    def run(self) -> Dict[str, Any]:
        for step in range(1, self.config.max_steps + 1):
            raw = self.driver.next(self.messages)
            self.messages.append({"role": "assistant", "content": raw})

            action_obj, error = parse_action(raw, self.allowed_actions)
            if action_obj is None:
                feedback = {"success": False, "error": error, "raw_model_output": raw}
                self.messages.append(
                    {"role": "user", "content": "Tool execution result:\n" + json.dumps(feedback, ensure_ascii=False)}
                )
                continue

            action_sig = _normalize_action_signature(action_obj)
            action_name = action_obj["action"]
            args = action_obj.get("args", {})

            if self.config.is_main_agent:
                duplicate_feedback = self._main_duplicate_feedback(self.current_state_fingerprint, action_sig)
                if duplicate_feedback is not None:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Tool execution result:\n" + json.dumps(duplicate_feedback, ensure_ascii=False),
                        }
                    )
                    continue
            elif action_sig in self.action_cache:
                feedback = {
                    "success": False,
                    "error": "Duplicate action detected. Do not repeat identical action+args.",
                    "cached_result": self.action_cache[action_sig],
                }
                self.messages.append(
                    {"role": "user", "content": "Tool execution result:\n" + json.dumps(feedback, ensure_ascii=False)}
                )
                continue

            if action_name == "enter_lemma_mode":
                tool_result = self._handle_enter_lemma(args)
            else:
                tool_result, dispatch_error = self.tool_registry.validate_and_dispatch(action_obj)
                if tool_result is None:
                    tool_result = {"success": False, "error": dispatch_error}

            feedback = {"tool": action_name, "result": _compact_tool_result(action_name, tool_result)}
            if self.config.is_main_agent:
                self._record_main_action(self.current_state_fingerprint, action_sig, feedback)
                next_fingerprint = self._fingerprint_from_result(tool_result)
                if next_fingerprint is not None:
                    self.current_state_fingerprint = next_fingerprint
            else:
                self.action_cache[action_sig] = feedback

            self.messages.append(
                {"role": "user", "content": "Tool execution result:\n" + json.dumps(feedback, ensure_ascii=False)}
            )

            if tool_result.get("success") and tool_result.get("state") == "proven":
                if self.on_successful_proof is not None:
                    self.on_successful_proof(tool_result)
                return {
                    "success": True,
                    "agent_id": self.config.agent_id,
                    "steps_used": step,
                    "verification": tool_result,
                    "messages": self.messages,
                }

        return {
            "success": False,
            "agent_id": self.config.agent_id,
            "error": f"Reached max steps ({self.config.max_steps}) without proven verification",
            "messages": self.messages,
        }

    def _handle_enter_lemma(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.spawn_lemma_fn is None:
            return {"success": False, "state": "error", "error": "enter_lemma_mode is unavailable in this agent"}
        lemma_name = str(args.get("lemma_name", "") or "aux_lemma")
        lemma_statement = str(args.get("lemma_statement", "") or "")
        lemma_declaration = self.client.build_lemma_declaration(lemma_name, lemma_statement)
        if not lemma_declaration:
            return {"success": False, "state": "error", "error": "invalid lemma declaration"}
        spawn_result = self.spawn_lemma_fn(lemma_name, lemma_statement)
        status = spawn_result.get("status")
        if status == "conflict":
            return {
                "success": False,
                "state": "error",
                "error": spawn_result.get("error", "lemma registration conflict"),
                "lemma_declaration": lemma_declaration,
                "registration_status": status,
            }
        proof_status = (
            f"Lemma '{lemma_name}' registered. A background sub-agent is now proving it. "
            f"The lemma is IMMEDIATELY available to you (as Admitted). Continue proving the main theorem."
            if status == "registered"
            else (
                f"Lemma '{lemma_name}' was already registered and remains available as Admitted. "
                f"Continue proving the main theorem."
            )
        )
        return {
            "success": True,
            "state": "in_progress",
            "proof_status": proof_status,
            "lemma_declaration": lemma_declaration,
            "registration_status": status,
        }


# =========================
# ProofOrchestrator
# =========================
class ProofOrchestrator:
    """Coordinates main proof agent and parallel lemma sub-agents."""

    def __init__(
        self,
        client: ProofTaskClient,
        driver_factory: Callable[[], ModelDriver],
        max_steps: int = 20,
        lemma_max_steps: int = LEMMA_MAX_STEPS,
    ):
        self.client = client
        self.driver_factory = driver_factory
        self.max_steps = max_steps
        self.lemma_max_steps = lemma_max_steps
        self.lemma_registry = LemmaRegistry()
        self.client.lemma_registry = self.lemma_registry
        self._lemma_results: Dict[str, Dict[str, Any]] = {}
        self._lemma_results_lock = threading.Lock()

    def run(self) -> Dict[str, Any]:
        main_agent = ProofAgent(
            client=self.client,
            driver=self.driver_factory(),
            config=AgentConfig(max_steps=self.max_steps, agent_id="main", is_main_agent=True),
            tool_registry=self.client.theorem_registry,
            allowed_actions=THEOREM_ACTIONS,
            system_prompt=self.client.build_system_prompt(),
            initial_user_prompt=self.client.build_initial_user_prompt(),
            spawn_lemma_fn=self._spawn_lemma_agent,
        )
        result = main_agent.run()
        if result.get("success") and self.lemma_registry.has_any():
            result = self._final_verification(result)
        self.lemma_registry.wait_all(timeout=300)
        with self._lemma_results_lock:
            if self._lemma_results:
                result["lemma_results"] = dict(self._lemma_results)
        result["lemma_summary"] = self.lemma_registry.summary()
        return result

    def _spawn_lemma_agent(self, lemma_name: str, lemma_statement: str) -> Dict[str, Any]:
        declaration = self.client.build_lemma_declaration(lemma_name, lemma_statement)
        if not declaration:
            return {"status": "conflict", "error": "invalid lemma declaration"}
        thread = threading.Thread(
            target=self._run_lemma_agent,
            args=(lemma_name, declaration),
            daemon=True,
            name=f"lemma-agent-{lemma_name}",
        )
        register_result = self.lemma_registry.register_lemma(lemma_name, declaration, thread)
        status = register_result.get("status")
        if status == "registered":
            thread.start()
        return register_result

    def _run_lemma_agent(self, lemma_name: str, declaration: str) -> None:
        try:
            agent = ProofAgent(
                client=self.client,
                driver=self.driver_factory(),
                config=AgentConfig(max_steps=self.lemma_max_steps, agent_id=f"lemma:{lemma_name}"),
                tool_registry=self.client.build_lemma_tool_registry(lemma_name, declaration),
                allowed_actions=LEMMA_ACTIONS,
                system_prompt=self.client.build_lemma_system_prompt(lemma_name, declaration),
                initial_user_prompt=f"Prove this lemma: {declaration}",
                on_successful_proof=lambda result: self.lemma_registry.mark_proven(
                    lemma_name,
                    str(result.get("proof_content", "") or result.get("current_proof", "")),
                ),
            )
            agent_result = agent.run()
            with self._lemma_results_lock:
                self._lemma_results[lemma_name] = agent_result
            if not agent_result.get("success"):
                self.lemma_registry.mark_failed(lemma_name)
        except Exception as exc:
            self.lemma_registry.mark_failed(lemma_name)
            with self._lemma_results_lock:
                self._lemma_results[lemma_name] = {
                    "success": False,
                    "error": f"lemma agent crashed: {exc}",
                    "traceback": traceback.format_exc(),
                }

    def _final_verification(self, result: Dict[str, Any]) -> Dict[str, Any]:
        self.lemma_registry.wait_all(timeout=300)
        if not self.lemma_registry.all_proven():
            result["success"] = False
            result["error"] = f"Theorem proof depends on unproven lemmas: {self.lemma_registry.unproven_names()}"
            return result
        final_prelude = self.lemma_registry.get_final_prelude_all_proven()
        if final_prelude is None:
            result["success"] = False
            result["error"] = "Failed to build final prelude with proven lemmas"
            return result
        verification = result.get("verification", {})
        proof = verification.get("proof_content", "")
        if not proof:
            return result
        final_result = self.client.verify_main_theorem_proof_final(proof, final_prelude)
        if final_result.get("success") and final_result.get("state") == "proven":
            result["verification"] = final_result
            result["final_verification"] = True
        else:
            result["success"] = False
            result["error"] = "Final verification with proven lemmas failed"
            result["final_verification_error"] = final_result
        return result


# =========================
# Logging
# =========================
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _default_log_dir(theorem_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    theorem_slug = theorem_id.replace(":", "_")
    return Path(__file__).resolve().parent.parent / "log" / f"{theorem_slug}_{stamp}_log"


def _render_readable_log(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"success: {result.get('success')}")
    if "steps_used" in result:
        lines.append(f"steps_used: {result.get('steps_used')}")
    if result.get("error"):
        lines.append(f"error: {result.get('error')}")

    verification = result.get("verification")
    if isinstance(verification, dict):
        lines.append("")
        lines.append("[verification]")
        for key in ("state", "proof_status", "error_message", "proof_state"):
            if verification.get(key):
                lines.append(f"{key}: {verification.get(key)}")
        if verification.get("proof_content"):
            lines.append("proof_content:")
            lines.append(str(verification.get("proof_content")))

    lemma_summary = result.get("lemma_summary")
    if isinstance(lemma_summary, dict) and lemma_summary:
        lines.append("")
        lines.append("[lemma_summary]")
        for name, info in lemma_summary.items():
            lines.append(f"  {name}: {info}")

    messages = result.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                continue
            lines.append("")
            lines.append(f"[message {index}] role={message.get('role', 'unknown')}")
            lines.append(str(message.get("content", "")))
    return "\n".join(lines).rstrip() + "\n"


def write_attempt_logs(theorem_id: str, result: Dict[str, Any], readable_log_file: Optional[str] = None) -> Path:
    if readable_log_file:
        readable_path = Path(readable_log_file).resolve()
        run_dir = readable_path.parent
    else:
        run_dir = _default_log_dir(theorem_id)
        readable_path = run_dir / "readable"

    run_dir.mkdir(parents=True, exist_ok=True)
    result["log_dir"] = str(run_dir)
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    readable_path.write_text(_render_readable_log(result), encoding="utf-8")
    return run_dir


# =========================
# CLI entry point
# =========================
def main() -> None:
    parser = argparse.ArgumentParser("Agent-based proof-task client.")
    parser.add_argument("--theorem-id", required=True, help="e.g. test:39")
    parser.add_argument("--context-lines", type=int, default=80, help="Number of lines before theorem as local context")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the main model")
    parser.add_argument("--readable-log-file", help="Optional path for the human-readable attempt log")
    parser.add_argument("--dump-system-prompt", action="store_true", help="Print system prompt only")
    args = parser.parse_args()

    client = ProofTaskClient(theorem_id=args.theorem_id, context_lines=args.context_lines)
    if args.dump_system_prompt:
        print(client.build_system_prompt())
        return

    def driver_factory() -> ModelDriver:
        return OpenAIModelDriver(model=args.model, temperature=args.temperature)

    try:
        orchestrator = ProofOrchestrator(
            client=client,
            driver_factory=driver_factory,
            max_steps=args.max_steps,
            lemma_max_steps=LEMMA_MAX_STEPS,
        )
        result = orchestrator.run()
    except Exception as exc:
        result = {
            "success": False,
            "error": "fatal_client_error",
            "details": str(exc),
            "traceback": traceback.format_exc(),
            "messages": [],
        }
    write_attempt_logs(args.theorem_id, result, readable_log_file=args.readable_log_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

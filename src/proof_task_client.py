#!/usr/bin/env python3
"""
Extensible LLM proof-task client.

Design goals:
- One client binds a fixed theorem_id (repo/compile_args are fixed accordingly)
- Tool registry architecture for future extensibility
- Model returns JSON actions only; client executes tools and feeds results back
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from coq_print import execute_print_command
from verify import CoqProofVerifier

# =========================
# OpenAI configuration
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://yunwu.ai/v1"
NL_PROOF_MODEL = os.environ.get("NL_PROOF_MODEL", "claude-opus-4-6")
LEMMA_MODE_MAX_STEPS = int(os.environ.get("LEMMA_MODE_MAX_STEPS", "8"))



@dataclass
class TaskContext:
    theorem_id: str
    project: str
    file_relpath: str
    repo_path: str
    compile_args: List[str]
    theorem_statement: str
    print_setup_script: str
    pre_theorem_context: str
    header_context: str
    upper_context: str


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
        return [t.spec() for t in self._tools.values()]

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


class VerifyProofTool:
    name = "verify_proof"

    def __init__(
        self,
        verifier: CoqProofVerifier,
        theorem_id: str,
        prelude_supplier: Optional[Callable[[], str]] = None,
    ):
        self.verifier = verifier
        self.theorem_id = theorem_id
        self.prelude_supplier = prelude_supplier

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "验证证明内容。固定 theorem_id，不需要传 theorem_id。",
            "args_schema": {
                "type": "object",
                "required": ["proof"],
                "properties": {
                    "proof": {"type": "string", "description": "证明文本，可完整或未完成"},
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        proof = args.get("proof", "")
        if not isinstance(proof, str) or not proof.strip():
            return {"success": False, "error": "proof 不能为空字符串"}
        prelude = self.prelude_supplier() if self.prelude_supplier is not None else ""
        return self.verifier.verify_proof(self.theorem_id, proof, injected_prelude=prelude)


class PrintDefinitionTool:
    name = "print"

    def __init__(self, repo_path: str, compile_args: List[str], setup_script: str = ""):
        self.repo_path = repo_path
        self.compile_args = compile_args
        self.setup_script = setup_script

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "打印定义。固定 repo/compile_args，仅需传定义名。",
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
        cmd = f"Print {definition_name}."
        result = execute_print_command(
            query_command=cmd,
            setup_script=self.setup_script,
            compile_args=self.compile_args,
            cwd=self.repo_path,
        )
        result["definition_name"] = definition_name
        return result


class StepTacticTool:
    name = "step_tactic"

    def __init__(
        self,
        verifier: CoqProofVerifier,
        theorem_id: str,
        prelude_supplier: Optional[Callable[[], str]] = None,
    ):
        self.verifier = verifier
        self.theorem_id = theorem_id
        self.prelude_supplier = prelude_supplier

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "单步追加 tactic 并返回当前证明状态与更新后的证明前缀。",
            "args_schema": {
                "type": "object",
                "required": ["proof_prefix", "tactic"],
                "properties": {
                    "proof_prefix": {"type": "string", "description": "当前证明前缀，例如 Proof. intros x."},
                    "tactic": {"type": "string", "description": "单步 tactic（单行），例如 destruct H."},
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
        prelude = self.prelude_supplier() if self.prelude_supplier is not None else ""
        result = self.verifier.verify_proof(self.theorem_id, next_proof, injected_prelude=prelude)
        result["current_proof"] = next_proof
        result["step_appended"] = tactic.strip()
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
        return [t.lower() for t in self._TOKEN_RE.findall(text)]

    def _build_index(self) -> None:
        docs: List[Dict[str, Any]] = []
        for root, _, files in os.walk(self.repo_path):
            for fn in files:
                if not fn.endswith(".v"):
                    continue
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, self.repo_path).replace("\\", "/")
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception:
                    continue
                i = 0
                while i < len(lines):
                    m = self._DECL_RE.match(lines[i])
                    if not m:
                        i += 1
                        continue
                    kind = m.group(1)
                    name = m.group(2) or "(anonymous)"
                    start = i
                    block = [lines[i].rstrip("\n")]
                    j = i + 1
                    while "." not in block[-1] and j < len(lines) and j - start <= 8:
                        block.append(lines[j].rstrip("\n"))
                        j += 1
                    text = " ".join(x.strip() for x in block if x.strip())
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
                    i = max(j, i + 1)

        df: Dict[str, int] = {}
        for d in docs:
            for tok in set(d["tokens"]):
                df[tok] = df.get(tok, 0) + 1
        avgdl = sum(d["len"] for d in docs) / len(docs) if docs else 1.0

        self._docs = docs
        self._df = df
        self._avgdl = avgdl
        self._built = True

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

        if not self._built:
            self._build_index()
        if not self._docs:
            return {"success": False, "error": "未找到可索引的 Coq 声明"}

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return {"success": False, "error": "query 解析后为空"}

        N = len(self._docs)
        k1 = 1.5
        b = 0.75
        results: List[Tuple[float, Dict[str, Any]]] = []
        for d in self._docs:
            if not self._in_scope(d["file"], scope):
                continue
            tf: Dict[str, int] = {}
            for t in d["tokens"]:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for q in q_tokens:
                f = tf.get(q, 0)
                if f == 0:
                    continue
                df = self._df.get(q, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                denom = f + k1 * (1 - b + b * (d["len"] / self._avgdl))
                score += idf * (f * (k1 + 1) / max(denom, 1e-9))
            if score <= 0:
                continue
            # proximity bias: in repo scope, prefer nearby files slightly.
            if scope == "repo":
                if d["file"] == self.current_file_relpath:
                    score *= 1.15
                elif self.current_dir_relpath and d["file"].startswith(self.current_dir_relpath + "/"):
                    score *= 1.08
            results.append((score, d))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:k]
        return {
            "success": True,
            "scope": scope,
            "query": query,
            "k": k,
            "hits": [
                {
                    "score": round(score, 4),
                    "kind": d["kind"],
                    "name": d["name"],
                    "file": d["file"],
                    "line": d["line"],
                    "text": d["text"],
                }
                for score, d in top
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
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
            return {
                "success": True,
                "model": self.model,
                "natural_language_proof": text,
            }
        except Exception as e:
            return {"success": False, "error": f"natural language proof generation failed: {e}", "model": self.model}


class ModelDriver(Protocol):
    def next(self, messages: List[Dict[str, str]]) -> str:
        ...


class OpenAIModelDriver:
    """Use OpenAI SDK with key from environment variable OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-5-nano"):
        self.model = model
        self.api_key = OPENAI_API_KEY
        self.base_url = OPENAI_BASE_URL
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is empty. Set it in your shell environment before running.")

    @staticmethod
    def _build_query(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        parts.append("You must return exactly one JSON object. No markdown. No explanations.")
        parts.append("Conversation history follows. Continue based on the latest user message.")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
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
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[{"role": "user", "content": query}],
        )
        return (resp.choices[0].message.content or "").strip()


class ProofTaskClient:
    def __init__(self, theorem_id: str, context_lines: int = 80, coqstoq_path: Optional[str] = None):
        self.verifier = CoqProofVerifier(coqstoq_path=coqstoq_path)
        self.task = self._build_task_context(theorem_id, context_lines)
        self.proven_lemma_blocks: List[str] = []
        self._proven_lemma_set: set[str] = set()
        self.registry = ToolRegistry()
        self._register_builtin_tools()

    def get_injected_lemma_prelude(self) -> str:
        return "\n\n".join(self.proven_lemma_blocks)

    def add_proven_lemma(self, lemma_decl: str, proof_content: str) -> None:
        declaration = lemma_decl.strip()
        proof_text = proof_content.strip()
        if not declaration or not proof_text:
            return
        block = f"{declaration}\n{proof_text}"
        key = " ".join(block.split())
        if key in self._proven_lemma_set:
            return
        self._proven_lemma_set.add(key)
        self.proven_lemma_blocks.append(block)

    def _build_task_context(self, theorem_id: str, m: int) -> TaskContext:
        split_name, index = self.verifier._parse_theorem_id(theorem_id)  # pylint: disable=protected-access
        theorem_def = self.verifier._load_theorem_definition(split_name, index)  # pylint: disable=protected-access
        if theorem_def is None:
            raise ValueError(f"Theorem not found: {theorem_id}")

        repo_path = os.path.join(
            self.verifier.coqstoq_path,
            theorem_def["project"]["split"]["dir_name"],
            theorem_def["project"]["dir_name"],
        )
        src_file = os.path.join(repo_path, theorem_def["path"])
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Source file not found: {src_file}")

        with open(src_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        theorem_start_line = theorem_def["theorem_start_pos"]["line"]
        theorem_statement = self.verifier._extract_theorem_statement(repo_path, theorem_def)  # pylint: disable=protected-access
        compile_args = theorem_def["project"].get("compile_args", [])
        print_setup_script = self._build_print_setup_script(theorem_def["path"])
        pre_theorem_context = "".join(lines[:theorem_start_line]).rstrip("\n")

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
        headers = [ln.rstrip("\n") for ln in lines[:theorem_start_line] if ln.lstrip().startswith(prefixes)]
        header_context = "\n".join(headers[-120:]) if headers else "(无显式头部依赖语句)"

        start = max(0, theorem_start_line - m)
        upper_context = "".join(lines[start:theorem_start_line]).rstrip("\n")
        if not upper_context.strip():
            upper_context = "(No additional local context before theorem)"

        return TaskContext(
            theorem_id=theorem_id,
            project=theorem_def["project"]["dir_name"],
            file_relpath=theorem_def["path"],
            repo_path=repo_path,
            compile_args=compile_args,
            theorem_statement=theorem_statement,
            print_setup_script=print_setup_script,
            pre_theorem_context=pre_theorem_context,
            header_context=header_context,
            upper_context=upper_context,
        )

    @staticmethod
    def _build_print_setup_script(file_relpath: str) -> str:
        """Run query after loading the target source file context."""
        escaped = file_relpath.replace("\\", "\\\\").replace('"', '\\"')
        return f'Load "{escaped}".\n'

    @staticmethod
    def build_lemma_declaration(lemma_name: str, lemma_statement: str) -> str:
        stmt = lemma_statement.strip()
        if not stmt:
            return ""
        if re.match(r"^\s*(Lemma|Theorem|Corollary|Proposition|Fact|Remark)\b", stmt):
            return stmt if stmt.endswith(".") else stmt + "."
        name = lemma_name.strip() or "aux_lemma"
        body = stmt.rstrip(".").strip()
        return f"Lemma {name} : {body}."

    def verify_lemma_proof(self, lemma_decl: str, proof_content: str) -> Dict[str, Any]:
        """Verify a lemma in theorem-pre context using coqtop."""
        if not lemma_decl.strip():
            return {"success": False, "state": "error", "error": "lemma declaration is empty"}
        if not isinstance(proof_content, str) or not proof_content.strip():
            return {"success": False, "state": "error", "error": "proof 不能为空字符串"}

        has_qed = "Qed." in proof_content
        escaped = self.task.file_relpath.replace("\\", "\\\\").replace('"', '\\"')
        script_lines = [
            f'Load "{escaped}".',
            lemma_decl.strip(),
            proof_content.strip(),
        ]
        if has_qed:
            script_lines.append("Quit.")
        else:
            script_lines.extend(["Show.", "Abort.", "Quit."])
        script = "\n".join(script_lines) + "\n"

        cmd = ["coqtop"] + self.task.compile_args + ["-quiet"]
        try:
            proc = subprocess.run(
                cmd,
                input=script,
                text=True,
                capture_output=True,
                cwd=self.task.repo_path,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "state": "error", "error": "lemma verification timeout"}
        except Exception as e:
            return {"success": False, "state": "error", "error": f"lemma verification failed: {e}"}

        raw = (proc.stdout or "") + (proc.stderr or "")
        has_error = "Error:" in raw
        if has_qed and not has_error and proc.returncode == 0:
            return {
                "success": True,
                "state": "proven",
                "proof_status": "lemma proven",
                "lemma_declaration": lemma_decl,
            }
        if has_qed:
            return {
                "success": False,
                "state": "failed",
                "error": raw[-1200:],
                "proof_status": "lemma failed",
                "lemma_declaration": lemma_decl,
            }
        return {
            "success": False,
            "state": "in_progress" if not has_error else "failed",
            "error": raw[-1200:] if has_error else None,
            "proof_status": "lemma in progress" if not has_error else "lemma step failed",
            "lemma_declaration": lemma_decl,
            "proof_state": raw[-1200:],
        }

    def _register_builtin_tools(self) -> None:
        self.registry.register(
            VerifyProofTool(
                self.verifier,
                self.task.theorem_id,
                prelude_supplier=self.get_injected_lemma_prelude,
            )
        )
        self.registry.register(
            PrintDefinitionTool(
                self.task.repo_path,
                self.task.compile_args,
                setup_script=self.task.print_setup_script,
            )
        )
        self.registry.register(
            StepTacticTool(
                self.verifier,
                self.task.theorem_id,
                prelude_supplier=self.get_injected_lemma_prelude,
            )
        )
        self.registry.register(BM25SearchTool(self.task.repo_path, self.task.file_relpath))
        self.registry.register(NaturalLanguageProofTool(self.task))

    def register_tool(self, tool: Tool) -> None:
        """可扩展入口：允许后续继续新增工具。"""
        self.registry.register(tool)

    def build_system_prompt(self) -> str:
        tool_specs = json.dumps(self.registry.specs(), ensure_ascii=False, indent=2)
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
1) Tool call
{{"action":"verify_proof","args":{{"proof":"Proof. ..."}}}}
{{"action":"print","args":{{"definition":"foo"}}}}
{{"action":"step_tactic","args":{{"proof_prefix":"Proof. ...","tactic":"intros x."}}}}
{{"action":"bm25_search","args":{{"query":"rewrite equality lemma","k":8,"scope":"current_dir"}}}}
{{"action":"natural_language_proof","args":{{"proof_prefix":"Proof. ...","question":"what is the key idea?"}}}}
{{"action":"enter_lemma_mode","args":{{"lemma_name":"aux1","lemma_statement":"forall x, ..."}}}}

Do not return markdown. Do not return any extra text.

Proof strategy guidance:
- You may first attempt a full proof in one shot up to `Qed.`.
- If that fails, switch to iterative exploration:
  - step-by-step (one small step each turn), or
  - multi-steps per turn (a few tactics each turn).
- Use `verify_proof` frequently to debug and refine.
- Use `print` whenever you need to inspect definitions/lemmas.
- Use `step_tactic` for single-step exploration; it returns current proof state and updated proof prefix.
- Use `bm25_search` to retrieve nearby related theorem/lemma/definition candidates before trying new tactics.
- Use `natural_language_proof` when you need a stronger model to generate a human-readable proof plan under current context.
- If a key missing lemma is needed, call `enter_lemma_mode`, then use the same tools to prove lemma first.

Context:
[Theorem Statement]
{self.task.theorem_statement}

[Header Context]
{self.task.header_context}

[Upper Context: previous lines]
{self.task.upper_context}
"""

    def build_initial_user_prompt(self) -> str:
        return "Start proving iteratively. Use `step_tactic` for single-step progress, `bm25_search` for related lemmas/definitions, `natural_language_proof` for strong-model proof planning, `print` for definitions, and `verify_proof` for complete-proof validation."


def parse_action(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSON parse failed: {e}"
    if not isinstance(obj, dict):
        return None, "Response must be a JSON object"
    allowed_top = {"action", "args"}
    extra_top = sorted(set(obj.keys()) - allowed_top)
    if extra_top:
        return None, f"unexpected top-level keys: {extra_top}"
    action = obj.get("action")
    if action not in {"verify_proof", "print", "step_tactic", "bm25_search", "natural_language_proof", "enter_lemma_mode"}:
        return None, "action must be one of: verify_proof / print / step_tactic / bm25_search / natural_language_proof / enter_lemma_mode"
    if action in {"verify_proof", "print", "step_tactic", "bm25_search", "natural_language_proof", "enter_lemma_mode"}:
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
        for k, v in args.items():
            if isinstance(v, str):
                normalized_args[k] = " ".join(v.split())
            else:
                normalized_args[k] = v
    return json.dumps({"action": action, "args": normalized_args}, ensure_ascii=False, sort_keys=True)


def _compact_tool_result(action: str, result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in (
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
    ):
        if k in result:
            out[k] = result[k]

    # For print tool, avoid keeping full definition body in long-term context.
    if action == "print":
        if result.get("success"):
            out["output"] = "Definition viewed; full body cleared from context."
        else:
            raw_err = result.get("error") or result.get("error_message") or ""
            out["output"] = str(raw_err)[:400]
    else:
        if "output" in result:
            out["output"] = str(result.get("output", ""))[:800]
    if action == "bm25_search":
        hits = result.get("hits")
        if isinstance(hits, list):
            compact_hits = []
            for h in hits[:10]:
                if not isinstance(h, dict):
                    continue
                compact_hits.append(
                    {
                        "score": h.get("score"),
                        "kind": h.get("kind"),
                        "name": h.get("name"),
                        "file": h.get("file"),
                        "line": h.get("line"),
                    }
                )
            out["hits"] = compact_hits
            out["hit_count"] = len(hits)
        if "query" in result:
            out["query"] = result.get("query")
        if "scope" in result:
            out["scope"] = result.get("scope")
    if action == "natural_language_proof":
        text = str(result.get("natural_language_proof", ""))
        out["natural_language_proof"] = text[:2200]
    return out


def run_loop(client: ProofTaskClient, driver: ModelDriver, max_steps: int) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": client.build_system_prompt()},
        {"role": "user", "content": client.build_initial_user_prompt()},
    ]
    action_cache: Dict[str, Dict[str, Any]] = {}
    lemma_mode: Optional[Dict[str, Any]] = None

    for step in range(1, max_steps + 1):
        raw = driver.next(messages)
        messages.append({"role": "assistant", "content": raw})

        action_obj, err = parse_action(raw)
        if action_obj is None:
            feedback = {"success": False, "error": err, "raw_model_output": raw}
            messages.append({"role": "user", "content": "Tool execution result:\n" + json.dumps(feedback, ensure_ascii=False)})
            continue

        action_sig = _normalize_action_signature(action_obj)
        if action_sig in action_cache:
            feedback = {
                "success": False,
                "error": "Duplicate action detected. Do not repeat identical action+args.",
                "cached_result": action_cache[action_sig],
            }
            messages.append({"role": "user", "content": "Tool execution result:\n" + json.dumps(feedback, ensure_ascii=False)})
            continue

        action_name = action_obj["action"]
        args = action_obj.get("args", {})

        if action_name == "enter_lemma_mode":
            lemma_name = str(args.get("lemma_name", "") or "aux_lemma")
            lemma_statement = str(args.get("lemma_statement", "") or "")
            lemma_decl = client.build_lemma_declaration(lemma_name, lemma_statement)
            if not lemma_decl:
                tool_result = {"success": False, "state": "error", "error": "invalid lemma declaration"}
            else:
                lemma_mode = {"declaration": lemma_decl, "name": lemma_name, "steps": 0}
                tool_result = {
                    "success": True,
                    "state": "in_progress",
                    "proof_status": "entered lemma mode",
                    "lemma_declaration": lemma_decl,
                    "max_steps": LEMMA_MODE_MAX_STEPS,
                }
        elif lemma_mode is not None and action_name in {"verify_proof", "step_tactic"}:
            if action_name == "verify_proof":
                proof_text = str(args.get("proof", ""))
            else:
                proof_prefix = str(args.get("proof_prefix", ""))
                tactic = str(args.get("tactic", ""))
                proof_text = proof_prefix.rstrip() + "\n" + tactic.strip()
            tool_result = client.verify_lemma_proof(lemma_mode["declaration"], proof_text)
            lemma_mode["steps"] += 1
            if tool_result.get("success") and tool_result.get("state") == "proven":
                client.add_proven_lemma(lemma_mode["declaration"], proof_text)
                tool_result["proof_status"] = "lemma proven; exit lemma mode"
                tool_result["injected_lemma_count"] = len(client.proven_lemma_blocks)
                lemma_mode = None
            elif lemma_mode is not None and lemma_mode["steps"] >= LEMMA_MODE_MAX_STEPS:
                # Drop lemma context and return to theorem mode.
                lemma_mode = None
                messages = [
                    {"role": "system", "content": client.build_system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "Lemma mode auto-abandoned after reaching step limit. "
                            "Discard lemma-related context and continue proving theorem."
                        ),
                    },
                ]
                action_cache.clear()
                tool_result = {
                    "success": False,
                    "state": "failed",
                    "proof_status": "lemma mode abandoned",
                    "error": f"lemma not proven within {LEMMA_MODE_MAX_STEPS} steps; context cleared",
                }
        else:
            tool_result, dispatch_err = client.registry.validate_and_dispatch(action_obj)
            if tool_result is None:
                tool_result = {"success": False, "error": dispatch_err}
            elif action_name in {"verify_proof", "step_tactic"}:
                if tool_result.get("success") and tool_result.get("state") == "proven":
                    return {
                        "success": True,
                        "steps_used": step,
                        "verification": tool_result,
                        "messages": messages,
                    }

        feedback = {"tool": action_name, "result": _compact_tool_result(action_name, tool_result)}
        if lemma_mode is not None:
            feedback["mode"] = {
                "name": "lemma",
                "lemma_declaration": lemma_mode.get("declaration"),
                "steps_used": lemma_mode.get("steps", 0),
                "steps_left": max(0, LEMMA_MODE_MAX_STEPS - int(lemma_mode.get("steps", 0))),
            }
        action_cache[action_sig] = feedback
        messages.append({"role": "user", "content": "Tool execution result:\n" + json.dumps(feedback, ensure_ascii=False)})

    return {"success": False, "error": f"Reached max steps ({max_steps}) without proven verification", "messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser("Extensible fixed-theorem proof-task client.")
    parser.add_argument("--theorem-id", required=True, help="e.g. test:39")
    parser.add_argument("--context-lines", type=int, default=80, help="Number of lines before theorem as local context")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--dump-system-prompt", action="store_true", help="Print system prompt only")
    args = parser.parse_args()

    client = ProofTaskClient(theorem_id=args.theorem_id, context_lines=args.context_lines)
    if args.dump_system_prompt:
        print(client.build_system_prompt())
        return

    driver: ModelDriver = OpenAIModelDriver(model=args.model)

    result = run_loop(client=client, driver=driver, max_steps=args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

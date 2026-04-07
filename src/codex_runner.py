#!/usr/bin/env python3
"""Thin Codex runner for theorem proving with the local Coq shell environment."""

from __future__ import annotations

import json
import os
import selectors
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from acprover_config import load_config
    from logging_utils import append_jsonl, render_readable_log, write_json, write_text
    from theorem_task import TheoremTask
except ModuleNotFoundError:
    from .acprover_config import load_config
    from .logging_utils import append_jsonl, render_readable_log, write_json, write_text
    from .theorem_task import TheoremTask


@dataclass
class CodexRunConfig:
    model: Optional[str] = None
    max_tokens: int = 32000
    timeout_seconds: int = 900
    reasoning_effort: str = "medium"
    full_auto: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowWorkspaceManifest:
    workspace_root: str
    workspace_repo: str
    target_file_path: str
    compile_target_path: str
    original_file_relpath: str
    cleanup_mode: str
    hidden_original_source: bool
    workspace_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _compile_command_hint(task: TheoremTask) -> str:
    compile_parts = ["$ACPROVER_COQC", *task.compile_args, "$ACPROVER_COMPILE_TARGET"]
    return shlex.join(compile_parts)


def build_codex_prompt(task: TheoremTask, config: CodexRunConfig, experience_block: str = "") -> str:
    repo_root = _repo_root()
    skill_path = repo_root / "skills" / "coq-proof-driver" / "SKILL.md"
    task_contract_path = repo_root / "docs" / "task-contract.md"
    workflow_path = repo_root / "docs" / "codex-rocq-workflow.md"
    logging_contract_path = repo_root / "docs" / "logging-contract.md"
    relevant_experience = ""
    if experience_block.strip():
        relevant_experience = f"""
{experience_block.rstrip()}
"""

    return f"""Read and follow these repository files before proving:
- {skill_path}
- {task_contract_path}
- {workflow_path}
- {logging_contract_path}

Task:
- theorem_id: {task.theorem_id}
- project: {task.project}
- file: {task.file_relpath}:{task.theorem_start_line}
- theorem_name: {task.theorem_name}

Current theorem declaration (the original proof is hidden and unavailable):
```coq
{task.theorem_declaration}
```
{relevant_experience}

Requirements:
- Use the local shell environment for repository exploration, file inspection, and editing only.
- The launcher has already prepared the environment, PATH, and Python import path. Do not spend time debugging opam switch setup, installation, cache directories, or launcher internals.
- The launcher has created an isolated shadow workspace for this run.
- In that workspace, every repository file except the current target source file is read-only.
- The original answer file for the current target theorem is unavailable. The target source file at `{task.file_relpath}` is your only writable temp file for this run.
- Do not inspect launcher implementation files unless a proof attempt requires a fact about the theorem source itself.
- Find any additional context yourself from the local repository using shell commands and the existing helpers under {repo_root / "src"}.
- The preferred theorem-oriented local tools are provided at `$ACPROVER_COQSTOQ_TOOLS`.
- Prefer these direct helper commands over re-implementing them manually:
  - `python3 "$ACPROVER_COQSTOQ_TOOLS" verify-proof --theorem-id "$ACPROVER_THEOREM_ID" --proof-file /tmp/proof.v`
  - `python3 "$ACPROVER_COQSTOQ_TOOLS" step-tactic --theorem-id "$ACPROVER_THEOREM_ID" --proof-prefix-file /tmp/prefix.v --tactic "intros x."`
  - `python3 "$ACPROVER_COQSTOQ_TOOLS" bm25-search --theorem-id "$ACPROVER_THEOREM_ID" --query "rewrite equality lemma" --scope current_dir`
- Use shell commands such as `rg`, `sed`, `find`, `ls`, and file reads to locate surrounding definitions, notations, imports, and related lemmas yourself.
- All interaction with Coq itself must go through `$ACPROVER_COQSTOQ_TOOLS`. Do not call `coqc`, `coqtop`, or `verify.py` directly unless the launcher instructions are changed.
- Prefer fast proof attempts over long reading passes. After a small number of targeted reads/searches, switch to `coqtop`, `coqc`, or `verify.py` and test a candidate proof.
- After a small number of targeted reads/searches, switch to `$ACPROVER_COQSTOQ_TOOLS step-tactic` or `$ACPROVER_COQSTOQ_TOOLS verify-proof` and test a candidate proof.
- For difficult theorems, decompose the proof into auxiliary lemmas early.
- To use an auxiliary lemma, insert it as an admitted stub before the target theorem in the writable target file, then prove that lemma in the same Codex session before returning to the main theorem.
- Keep auxiliary lemmas small and local. Prefer one or two lemmas that directly unblock the next proof step.
- Do not spend the session mining large neighboring proofs unless a specific lemma or proof pattern is already identified and immediately useful.
- If you still have not made a Coq proof attempt after several shell read/search commands, stop reading and try a concrete proof step.
- Attempt proofs by validating candidate proof scripts through `$ACPROVER_COQSTOQ_TOOLS`. Keep each externally visible proof attempt and tool result explicit.
- Prefer validating proof candidates with `$ACPROVER_COQSTOQ_TOOLS verify-proof` and single-step exploration with `$ACPROVER_COQSTOQ_TOOLS step-tactic`.
- The launcher exposes CoqStoq/project metadata via environment variables such as `ACPROVER_COQSTOQ_ROOT`, `ACPROVER_PROJECT_ROOT`, `ACPROVER_SOURCE_FILE`, `ACPROVER_SOURCE_FILE_RELPATH`, `ACPROVER_COMPILE_ARGS_JSON`, and `ACPROVER_COMPILE_ARGS_SHELL`. Use them directly instead of reverse-engineering the environment from launcher code.
- Do not search for or reconstruct the original proof from the source file, git history, or repository metadata.
- If the local Coq environment fails at the environment level, stop proof search quickly and report the failure instead of trying to repair the environment.
- As soon as a proof candidate has been validated successfully, immediately emit the final JSON response and stop. Do not spend extra time on cleanup, formatting-only checks, or post-proof file inspection.
- The launcher will automatically stop the run once it detects a successful compile of the target file with the target theorem finished by `Qed.` or `Defined.`.
- Keep your visible reasoning explicit enough for debugging. Every externally visible proof attempt and tool outcome should be understandable from the saved logs.
- Use roughly a {config.max_tokens}-token visible-output budget across the session. Be complete, but do not repeat yourself.
- Your final JSON must include:
  - `experience_reasoning`: a natural-language explanation of why the theorem is proved this way, including the key definitions and facts needed to understand the proof. Include relevant Coq code snippets for the statement and the crucial proof fragment.
  - `experience_result`: the final proof code only. If no final proof is available, return an empty string.
- Your final response must satisfy the provided JSON schema.
"""


def _build_output_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "final_status",
            "final_proof",
            "summary",
            "experience_reasoning",
            "experience_result",
        ],
        "properties": {
            "final_status": {
                "type": "string",
                "enum": ["proven", "failed", "timeout", "tool_error", "budget_exhausted"],
            },
            "final_proof": {
                "type": ["string", "null"],
                "description": "A complete Rocq/Coq proof script when available.",
            },
            "summary": {
                "type": "string",
                "description": "Concise debugging-oriented summary of what happened.",
            },
            "experience_reasoning": {
                "type": "string",
                "description": "Natural-language explanation of why the proof works, including the key definitions, intermediate facts, and relevant Coq code snippets.",
            },
            "experience_result": {
                "type": "string",
                "description": "Final proof code only. Return an empty string if no final proof is available.",
            },
        },
    }


def _extract_agent_messages(events: List[Dict[str, Any]]) -> List[str]:
    messages: List[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        if item.get("type") == "agent_message":
            messages.append(str(item.get("text", "")))
    return messages


def _extract_reasoning_messages(events: List[Dict[str, Any]]) -> List[str]:
    messages: List[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        if item.get("type") == "reasoning":
            messages.append(str(item.get("text", "")))
    return messages


def _write_text_preserve(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def _clone_repo_fast(repo_path: str, workspace_repo: Path) -> str:
    workspace_repo.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = ["cp", "-aT", "--reflink=auto", repo_path, str(workspace_repo)]
    completed = subprocess.run(clone_cmd, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        return "cp_reflink_auto"
    shutil.rmtree(workspace_repo, ignore_errors=True)
    shutil.copytree(repo_path, workspace_repo)
    return "copytree_fallback"


def _set_workspace_permissions(workspace_repo: Path, temp_v: Path) -> None:
    for current_root, dirnames, filenames in os.walk(workspace_repo):
        root_path = Path(current_root)
        if root_path == temp_v.parent:
            root_path.chmod(0o755)
        else:
            root_path.chmod(0o555)
        for filename in filenames:
            file_path = root_path / filename
            if file_path == temp_v:
                file_path.chmod(0o644)
            else:
                file_path.chmod(0o444)


def _prepare_shadow_workspace(
    task: TheoremTask,
    log_dir: Path,
    target_source_text: str,
) -> ShadowWorkspaceManifest:
    workspace_root = log_dir / "shadow_workspace"
    workspace_repo = workspace_root / Path(task.repo_path).name
    workspace_strategy = _clone_repo_fast(task.repo_path, workspace_repo)

    target_file_path = workspace_repo / task.file_relpath

    _write_text_preserve(target_file_path, target_source_text)
    write_text(log_dir / "temp_initial.v", target_file_path.read_text(encoding="utf-8"))
    _set_workspace_permissions(workspace_repo, target_file_path)

    manifest = ShadowWorkspaceManifest(
        workspace_root=str(workspace_root),
        workspace_repo=str(workspace_repo),
        target_file_path=str(target_file_path),
        compile_target_path=str(target_file_path),
        original_file_relpath=task.file_relpath,
        cleanup_mode="delete_workspace",
        hidden_original_source=True,
        workspace_strategy=workspace_strategy,
    )
    write_json(log_dir / "workspace_manifest.json", manifest.to_dict())
    return manifest


def _snapshot_temp_file(temp_v_path: Path) -> Optional[str]:
    if not temp_v_path.exists():
        return None
    return temp_v_path.read_text(encoding="utf-8")


def _classify_command(command: str) -> str:
    if "coqstoq_tools.py verify-proof" in command:
        return "verify_proof_tool"
    if "coqstoq_tools.py step-tactic" in command:
        return "step_tactic_tool"
    if "coqstoq_tools.py bm25-search" in command:
        return "bm25_search_tool"
    if "coqc" in command:
        return "compile_check"
    if "coqtop" in command:
        return "interactive_check"
    if "verify.py" in command or "import verify" in command:
        return "proof_attempt"
    if "rg " in command or command.startswith("rg"):
        return "context_search"
    if "sed " in command or command.startswith("sed"):
        return "context_read"
    return "command"


def _build_attempt(index: int, item: Dict[str, Any], temp_v_path: Path) -> Dict[str, Any]:
    return {
        "index": index,
        "kind": _classify_command(str(item.get("command", ""))),
        "command": str(item.get("command", "")),
        "proof_text": _snapshot_temp_file(temp_v_path),
        "stdout": str(item.get("aggregated_output", "")),
        "stderr": str(item.get("stderr", "")),
        "exit_code": item.get("exit_code"),
        "success": item.get("exit_code") == 0 and item.get("status") == "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": str(item.get("status", "")),
    }


def _is_successful_target_compile(command: str, task: TheoremTask, compile_target_path: str) -> bool:
    if "coqc" not in command:
        return False
    return compile_target_path in command or task.file_relpath in command


def _parse_tool_json_output(output: str) -> Optional[Dict[str, Any]]:
    text = output.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _is_successful_tool_proof(command: str, item: Dict[str, Any]) -> bool:
    if "coqstoq_tools.py verify-proof" not in command and "coqstoq_tools.py step-tactic" not in command:
        return False
    parsed = _parse_tool_json_output(str(item.get("aggregated_output", "")))
    if not parsed:
        return False
    return parsed.get("success") is True and parsed.get("state") == "proven"


def _extract_completed_theorem(task: TheoremTask, temp_v_path: Path) -> Optional[Dict[str, str]]:
    if not temp_v_path.exists():
        return None
    extracted = task.extract_theorem_block(temp_v_path.read_text(encoding="utf-8"))
    if extracted is None:
        return None
    if extracted.get("terminator") not in {"Qed", "Defined"}:
        return None
    return extracted


def _normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    input_tokens = usage.get("input_tokens") if usage else None
    cached_input_tokens = usage.get("cached_input_tokens") if usage else None
    output_tokens = usage.get("output_tokens") if usage else None
    reasoning_tokens = usage.get("reasoning_tokens") if usage else None
    numeric_total = 0
    has_any = False
    for value in (input_tokens, output_tokens, reasoning_tokens):
        if isinstance(value, int):
            numeric_total += value
            has_any = True
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": numeric_total if has_any else None,
        "source": "turn.completed" if usage else "unavailable",
    }


def _remove_workspace_tree(path: str) -> None:
    root_path = Path(path)
    if not root_path.exists():
        return
    for current_root, dirnames, filenames in os.walk(root_path, topdown=False):
        current_path = Path(current_root)
        current_path.chmod(0o755)
        for dirname in dirnames:
            (current_path / dirname).chmod(0o755)
        for filename in filenames:
            (current_path / filename).chmod(0o644)
    root_path.chmod(0o755)
    shutil.rmtree(root_path)


def _build_runtime_env(task: TheoremTask) -> Dict[str, str]:
    runtime_config = load_config()
    env = dict(os.environ)
    opam_switch = os.environ.get("ACPROVER_OPAM_SWITCH", runtime_config.opam_switch)
    opam_bin = Path.home() / ".opam" / opam_switch / "bin"
    path_parts = []
    if opam_bin.is_dir():
        path_parts.append(str(opam_bin))
    existing_path = env.get("PATH")
    if existing_path:
        path_parts.append(existing_path)
    env["PATH"] = ":".join(path_parts)
    repo_root = str(_repo_root())
    tools_path = str(Path(repo_root) / "src" / "coqstoq_tools.py")
    existing_pythonpath = env.get("PYTHONPATH")
    python_parts = [str(Path(repo_root) / "src")]
    if existing_pythonpath:
        python_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(python_parts)
    env["ACPROVER_REPO_ROOT"] = repo_root
    env["ACPROVER_COQSTOQ_TOOLS"] = tools_path
    env["ACPROVER_THEOREM_ID"] = task.theorem_id
    env["ACPROVER_THEOREM_NAME"] = task.theorem_name
    env["ACPROVER_COQSTOQ_ROOT"] = task.coqstoq_path
    env["ACPROVER_PROJECT_ROOT"] = task.repo_path
    env["ACPROVER_COQ_REPO"] = task.repo_path
    env["ACPROVER_SOURCE_FILE"] = task.source_file
    env["ACPROVER_SOURCE_FILE_RELPATH"] = task.file_relpath
    env["ACPROVER_THEOREM_START_LINE"] = str(task.theorem_start_line)
    env["ACPROVER_THEOREM_END_LINE"] = str(task.theorem_end_line)
    env["ACPROVER_THEOREM_END_COLUMN"] = str(task.theorem_end_column)
    env["ACPROVER_PROOF_END_LINE"] = str(task.proof_end_line)
    env["ACPROVER_PROOF_END_COLUMN"] = str(task.proof_end_column)
    env["ACPROVER_COMPILE_ARGS_JSON"] = json.dumps(task.compile_args, ensure_ascii=False)
    env["ACPROVER_COMPILE_ARGS_SHELL"] = shlex.join(task.compile_args)
    env["ACPROVER_OPAM_SWITCH"] = opam_switch
    coqc_path = opam_bin / "coqc"
    if coqc_path.is_file():
        env["ACPROVER_COQC"] = str(coqc_path)
    return env


def _run_codex_session(
    task: TheoremTask,
    config: CodexRunConfig,
    log_dir: Path,
    readable_path: Path,
    *,
    prompt: str,
    target_source_text: str,
    retrieved_experiences: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    codex = shutil.which("codex")
    if codex is None:
        raise FileNotFoundError("`codex` is not on PATH.")

    prompt_path = log_dir / "prompt.txt"
    task_path = log_dir / "task.json"
    schema_path = log_dir / "output_schema.json"
    final_message_path = log_dir / "final_message.json"
    events_path = log_dir / "events.jsonl"
    stderr_path = log_dir / "codex_stderr.log"
    command_path = log_dir / "codex_command.json"
    runtime_env_path = log_dir / "runtime_env.json"
    final_temp_snapshot_path = log_dir / "final_temp_snapshot.v"
    workspace_manifest = _prepare_shadow_workspace(task, log_dir, target_source_text)
    workspace_repo = workspace_manifest.workspace_repo
    temp_v_path = Path(workspace_manifest.target_file_path)

    write_text(prompt_path, prompt)
    write_json(task_path, task.to_dict())
    write_json(schema_path, _build_output_schema())

    command = [
        codex,
        "exec",
        "--json",
        "-C",
        workspace_repo,
        "--skip-git-repo-check",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(final_message_path),
        "-c",
        f"model_reasoning_effort={json.dumps(config.reasoning_effort)}",
    ]
    if config.model:
        command.extend(["--model", config.model])
    if config.full_auto:
        command.append("--full-auto")
    command.append("-")
    write_json(command_path, {"argv": command})

    env = _build_runtime_env(task)
    env["ACPROVER_WORKSPACE_REPO"] = workspace_manifest.workspace_repo
    env["ACPROVER_COQ_REPO"] = workspace_manifest.workspace_repo
    env["ACPROVER_TEMP_V"] = workspace_manifest.target_file_path
    env["ACPROVER_COMPILE_TARGET"] = workspace_manifest.compile_target_path
    write_json(
        runtime_env_path,
        {
            "PATH": env.get("PATH", ""),
            "PYTHONPATH": env.get("PYTHONPATH", ""),
            "ACPROVER_REPO_ROOT": env.get("ACPROVER_REPO_ROOT", ""),
            "ACPROVER_COQSTOQ_TOOLS": env.get("ACPROVER_COQSTOQ_TOOLS", ""),
            "ACPROVER_THEOREM_ID": env.get("ACPROVER_THEOREM_ID", ""),
            "ACPROVER_THEOREM_NAME": env.get("ACPROVER_THEOREM_NAME", ""),
            "ACPROVER_COQSTOQ_ROOT": env.get("ACPROVER_COQSTOQ_ROOT", ""),
            "ACPROVER_PROJECT_ROOT": env.get("ACPROVER_PROJECT_ROOT", ""),
            "ACPROVER_COQ_REPO": env.get("ACPROVER_COQ_REPO", ""),
            "ACPROVER_SOURCE_FILE": env.get("ACPROVER_SOURCE_FILE", ""),
            "ACPROVER_SOURCE_FILE_RELPATH": env.get("ACPROVER_SOURCE_FILE_RELPATH", ""),
            "ACPROVER_THEOREM_START_LINE": env.get("ACPROVER_THEOREM_START_LINE", ""),
            "ACPROVER_THEOREM_END_LINE": env.get("ACPROVER_THEOREM_END_LINE", ""),
            "ACPROVER_THEOREM_END_COLUMN": env.get("ACPROVER_THEOREM_END_COLUMN", ""),
            "ACPROVER_PROOF_END_LINE": env.get("ACPROVER_PROOF_END_LINE", ""),
            "ACPROVER_PROOF_END_COLUMN": env.get("ACPROVER_PROOF_END_COLUMN", ""),
            "ACPROVER_COMPILE_ARGS_JSON": env.get("ACPROVER_COMPILE_ARGS_JSON", ""),
            "ACPROVER_COMPILE_ARGS_SHELL": env.get("ACPROVER_COMPILE_ARGS_SHELL", ""),
            "ACPROVER_WORKSPACE_REPO": env.get("ACPROVER_WORKSPACE_REPO", ""),
            "ACPROVER_TEMP_V": env.get("ACPROVER_TEMP_V", ""),
            "ACPROVER_COMPILE_TARGET": env.get("ACPROVER_COMPILE_TARGET", ""),
            "ACPROVER_OPAM_SWITCH": env.get("ACPROVER_OPAM_SWITCH", ""),
            "ACPROVER_COQC": env.get("ACPROVER_COQC", ""),
        },
    )

    events: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []
    stderr_chunks: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    timed_out = False
    error_messages: List[str] = []
    return_code = 0
    final_temp_snapshot: Optional[str] = None
    compile_autostop = False
    compiled_theorem: Optional[Dict[str, str]] = None

    try:
        start = time.time()
        process = subprocess.Popen(
            command,
            cwd=workspace_repo,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        process.stdin.write(prompt)
        process.stdin.close()

        selector = selectors.DefaultSelector()
        assert process.stdout is not None
        assert process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        while selector.get_map():
            if time.time() - start > config.timeout_seconds:
                timed_out = True
                process.kill()
                break
            ready = selector.select(timeout=1.0)
            for key, _ in ready:
                stream = key.fileobj
                tag = key.data
                line = stream.readline()
                if line == "":
                    selector.unregister(stream)
                    continue
                if tag == "stdout":
                    stripped = line.rstrip("\n")
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        payload = {"type": "stdout.raw", "raw": stripped}
                    append_jsonl(events_path, payload)
                    events.append(payload)
                    if payload.get("type") == "item.completed":
                        item = payload.get("item", {})
                        item_type = item.get("type")
                        if item_type == "error" and item.get("message"):
                            error_messages.append(str(item["message"]))
                        if item_type == "command_execution":
                            attempts.append(_build_attempt(len(attempts) + 1, item, temp_v_path))
                            command_text = str(item.get("command", ""))
                            if (
                                item.get("exit_code") == 0
                                and item.get("status") == "completed"
                            ):
                                if _is_successful_target_compile(
                                    command_text,
                                    task,
                                    workspace_manifest.compile_target_path,
                                ) or _is_successful_tool_proof(command_text, item):
                                    compiled_theorem = _extract_completed_theorem(task, temp_v_path)
                                    if compiled_theorem is not None:
                                        compile_autostop = True
                                        process.kill()
                                        break
                    if payload.get("type") == "error" and payload.get("message"):
                        error_messages.append(str(payload["message"]))
                    if payload.get("type") == "turn.failed" and payload.get("message"):
                        error_messages.append(str(payload["message"]))
                    if payload.get("type") == "turn.completed":
                        usage = payload.get("usage")
                else:
                    stderr_chunks.append(line)

            if compile_autostop:
                break

            if process.poll() is not None and not selector.get_map():
                break

        return_code = process.wait()
        stderr_text = "".join(stderr_chunks)
        write_text(stderr_path, stderr_text)

        parsed_final: Dict[str, Any] = {}
        if final_message_path.exists():
            raw_final = final_message_path.read_text(encoding="utf-8").strip()
            if raw_final:
                try:
                    parsed_final = json.loads(raw_final)
                except json.JSONDecodeError:
                    parsed_final = {
                        "final_status": "failed",
                        "final_proof": None,
                        "summary": raw_final,
                    }

        final_temp_snapshot = _snapshot_temp_file(temp_v_path)
        if final_temp_snapshot is not None:
            write_text(final_temp_snapshot_path, final_temp_snapshot)

        agent_messages = _extract_agent_messages(events)
        reasoning_messages = _extract_reasoning_messages(events)
        if compile_autostop and compiled_theorem is not None:
            final_status = "proven"
            final_proof = compiled_theorem.get("block")
            summary = "Target file compiled successfully with a completed theorem proof; run stopped immediately."
            error = None
        else:
            final_status = "timeout" if timed_out else parsed_final.get("final_status", "failed")
            final_proof = parsed_final.get("final_proof")
            summary = parsed_final.get("summary") or (agent_messages[-1] if agent_messages else "")
            error = None if final_status == "proven" else (error_messages[-1] if error_messages else None)

        result = {
            "success": final_status == "proven",
            "theorem_id": task.theorem_id,
            "task": task.to_dict(),
            "final_status": final_status,
            "final_proof": final_proof,
            "summary": summary,
            "error": error,
            "timeout": timed_out,
            "return_code": return_code,
            "compile_autostop": compile_autostop,
            "token_usage": _normalize_usage(usage),
            "attempts": attempts,
            "attempt_count": len(attempts),
            "agent_messages": agent_messages,
            "reasoning_messages": reasoning_messages,
            "retrieved_experiences": retrieved_experiences or [],
            "config": config.to_dict(),
            "workspace_strategy": workspace_manifest.workspace_strategy,
            "compile_target": workspace_manifest.compile_target_path,
            "workspace_manifest_file": str(log_dir / "workspace_manifest.json"),
            "runtime_env_file": str(runtime_env_path),
            "command_file": str(command_path),
            "prompt_file": str(prompt_path),
            "task_file": str(task_path),
            "events_file": str(events_path),
            "stderr_file": str(stderr_path),
            "final_message_file": str(final_message_path),
            "temp_initial_file": str(log_dir / "temp_initial.v"),
            "final_temp_snapshot_file": str(final_temp_snapshot_path) if final_temp_snapshot is not None else None,
            "final_temp_snapshot": final_temp_snapshot,
            "log_dir": str(log_dir),
        }
        write_json(log_dir / "result.json", result)
        write_text(
            readable_path,
            render_readable_log(task, result, events, prompt=prompt, stderr_text=stderr_text),
        )
        return result
    finally:
        if final_temp_snapshot is None:
            final_temp_snapshot = _snapshot_temp_file(temp_v_path)
            if final_temp_snapshot is not None and not final_temp_snapshot_path.exists():
                write_text(final_temp_snapshot_path, final_temp_snapshot)
        workspace_root = Path(workspace_manifest.workspace_root)
        if workspace_root.exists():
            _remove_workspace_tree(workspace_manifest.workspace_root)


def run_codex_task(
    task: TheoremTask,
    config: CodexRunConfig,
    log_dir: Path,
    readable_path: Path,
    *,
    experience_block: str = "",
    retrieved_experiences: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    source_path = task.source_path()
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        original_text = handle.read()
    return _run_codex_session(
        task,
        config,
        log_dir,
        readable_path,
        prompt=build_codex_prompt(task, config, experience_block),
        target_source_text=task.build_masked_target_source(original_text),
        retrieved_experiences=retrieved_experiences,
    )

#!/usr/bin/env python3
"""Extract reusable theorem-proving experiences."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from theorem_task import TheoremTask
except ModuleNotFoundError:
    from .theorem_task import TheoremTask


PROOF_ATTEMPT_KINDS = {"verify_proof_tool", "step_tactic_tool", "proof_attempt"}
SUCCESS_STATUSES = {"proven"}
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
QUALIFIED_RE = re.compile(r"\b([A-Z][A-Za-z0-9_']*(?:\.[A-Za-z0-9_']+)+)")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "experience"


def _strip_code_fence(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).strip()


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _parse_tool_json(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_failed_proofs(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    failed: List[Dict[str, Any]] = []
    for attempt in result.get("attempts", []) or []:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("kind") not in PROOF_ATTEMPT_KINDS:
            continue
        proof_text = _strip_code_fence(attempt.get("proof_text"))
        if not proof_text or attempt.get("success"):
            continue
        stdout = str(attempt.get("stdout", ""))
        stderr = str(attempt.get("stderr", ""))
        parsed = _parse_tool_json(stdout)
        failure_kind = "tool_error"
        error_excerpt = (stderr or stdout)[-1200:]
        if parsed:
            state = parsed.get("state")
            if state == "in_progress":
                failure_kind = "in_progress"
            elif state == "failed":
                failure_kind = "compile_error"
            elif state == "error":
                failure_kind = "tool_error"
            if parsed.get("error_message"):
                error_excerpt = str(parsed.get("error_message"))
            elif parsed.get("error"):
                error_excerpt = str(parsed.get("error"))
        elif "timeout" in (stderr + stdout).lower():
            failure_kind = "timeout"
        failed.append(
            {
                "index": attempt.get("index"),
                "proof_text": proof_text,
                "failure_kind": failure_kind,
                "error_excerpt": error_excerpt,
                "tool_name": attempt.get("kind"),
            }
        )
    return failed


def infer_proof_shape_tags(text: str) -> List[str]:
    lowered = text.lower()
    tags: List[str] = []
    if "induction" in lowered or "elim" in lowered:
        tags.append("induction")
    if "rewrite" in lowered:
        tags.append("rewrite")
    if "transitivity" in lowered:
        tags.append("transitivity")
    if "contradiction" in lowered or "exfalso" in lowered:
        tags.append("contradiction")
    if "assert " in lowered or "\nhave " in lowered or "\nlemma " in lowered:
        tags.append("local_lemma")
    if "auto" in lowered or "eauto" in lowered:
        tags.append("automation")
    if "destruct" in lowered or "case" in lowered:
        tags.append("case_analysis")
    if "apply " in lowered:
        tags.append("apply_chain")
    return sorted(set(tags))


def extract_repair_chain(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    attempts = [a for a in (result.get("attempts", []) or []) if isinstance(a, dict)]
    chains: List[Dict[str, Any]] = []
    chain_index = 1
    for idx, attempt in enumerate(attempts):
        if attempt.get("kind") not in PROOF_ATTEMPT_KINDS or attempt.get("success"):
            continue
        failed_proof = _strip_code_fence(attempt.get("proof_text"))
        if not failed_proof:
            continue
        for later in attempts[idx + 1 :]:
            if later.get("kind") not in PROOF_ATTEMPT_KINDS:
                continue
            repaired_proof = _strip_code_fence(later.get("proof_text"))
            if not repaired_proof or repaired_proof == failed_proof:
                continue
            chains.append(
                {
                    "chain_index": chain_index,
                    "failed_attempt_index": attempt.get("index"),
                    "repaired_attempt_index": later.get("index"),
                    "repair_outcome": "proven" if later.get("success") else "still_failed",
                    "repair_pattern_tags": infer_proof_shape_tags(repaired_proof),
                }
            )
            chain_index += 1
            break
    return chains


def _extract_statement_body(declaration: str) -> str:
    if ":" not in declaration:
        return declaration.strip().rstrip(".")
    return declaration.split(":", 1)[1].strip().rstrip(".")


def _declaration_kind(declaration: str) -> str:
    match = re.match(r"^\s*(Lemma|Theorem|Corollary|Proposition|Fact|Remark|Definition)\b", declaration)
    return match.group(1) if match else "Theorem"


def _explain_statement(declaration: str) -> str:
    body = _extract_statement_body(declaration)
    if "<->" in body:
        left, right = [part.strip() for part in body.split("<->", 1)]
        return f"it states that {left} is equivalent to {right}"
    if "->" in body:
        parts = [part.strip() for part in body.split("->")]
        if len(parts) >= 2:
            assumptions = ", ".join(parts[:-1])
            conclusion = parts[-1]
            return f"it shows that if {assumptions}, then {conclusion}"
    if body.lower().startswith("exists"):
        return f"it establishes the existence claim {body}"
    return f"it establishes {body}"


def _parse_declared_imports(source_text: str) -> List[str]:
    imports: List[str] = []
    patterns = [
        re.compile(r"\bFrom\s+([A-Za-z0-9_'.]+)\s+Require\s+Import\s+([^.]*)\."),
        re.compile(r"\bFrom\s+([A-Za-z0-9_'.]+)\s+Require\s+Export\s+([^.]*)\."),
        re.compile(r"\bRequire\s+Import\s+([^.]*)\."),
        re.compile(r"\bRequire\s+Export\s+([^.]*)\."),
    ]
    normalized_text = source_text.replace("\n", " ")
    for pattern in patterns:
        for match in pattern.finditer(normalized_text):
            prefix = ""
            modules_text = ""
            if match.lastindex == 2:
                prefix = match.group(1)
                modules_text = match.group(2)
            elif match.lastindex == 1:
                modules_text = match.group(1)
            for module in modules_text.split():
                cleaned = module.strip()
                if not cleaned:
                    continue
                imports.append(f"{prefix}.{cleaned}" if prefix else cleaned)
    deduped: List[str] = []
    for item in imports:
        if item not in deduped:
            deduped.append(item)
    return deduped[:24]


def _parse_referenced_namespaces(texts: List[str]) -> List[str]:
    namespaces: List[str] = []
    for text in texts:
        for match in QUALIFIED_RE.findall(text):
            namespaces.append(match)
        for import_match in re.finditer(r"\bImport\s+([^.]*)\.", text.replace("\n", " ")):
            namespaces.extend(token.strip() for token in import_match.group(1).split() if token.strip())
    deduped: List[str] = []
    for item in namespaces:
        if item not in deduped:
            deduped.append(item)
    return deduped[:24]


def _extract_coq_libraries(task: TheoremTask, final_proof: str, oracle_proof: str) -> Dict[str, List[str]]:
    source_text = task.source_path().read_text(encoding="utf-8")
    return {
        "declared_imports": _parse_declared_imports(source_text),
        "referenced_namespaces": _parse_referenced_namespaces(
            [source_text, task.theorem_declaration, final_proof, oracle_proof]
        ),
    }


def _normalized_theorem_types(task: TheoremTask, proof_texts: List[str]) -> List[str]:
    declaration = task.theorem_declaration
    lowered_decl = declaration.lower()
    combined = " ".join(proof_texts).lower()
    theorem_types: List[str] = []
    if "<->" in declaration:
        theorem_types.append("iff")
    if "->" in declaration:
        theorem_types.append("implication")
        theorem_types.append("forward_rule")
    if "==" in declaration or re.search(r"[^<>=]=[^=]", declaration):
        theorem_types.append("equality")
    if any(symbol in declaration for symbol in ("<=", ">=", "<", ">")):
        theorem_types.append("order")
    if any(symbol in declaration for symbol in ("+", "-", "*", "/")) or re.search(r"\b\d+\b", declaration):
        theorem_types.append("arithmetic")
    if "exists" in lowered_decl:
        theorem_types.append("existential")
    if "reflect" in lowered_decl:
        theorem_types.extend(["reflection", "boolean_spec"])
    if "proper" in lowered_decl or "setoid" in lowered_decl or "morphism" in lowered_decl:
        theorem_types.append("setoid")
    if "rewrite" in combined:
        theorem_types.append("rewrite_rule")
    if "induction" in combined or "elim" in combined:
        theorem_types.append("induction")
    if "destruct" in combined or "case" in combined:
        theorem_types.append("case_analysis")
    if "assert " in combined or "\nhave " in combined:
        theorem_types.append("local_lemma_pattern")
    if not theorem_types:
        theorem_types.append("structural")
    deduped: List[str] = []
    for item in theorem_types:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _extract_oracle_proof(task: TheoremTask) -> Dict[str, Optional[str]]:
    source_text = task.source_path().read_text(encoding="utf-8")
    block = task.extract_theorem_block(source_text)
    if block is None:
        return {"oracle_block": None, "oracle_proof_text": None}
    proof_text = str(block.get("proof_text", "")).strip()
    return {
        "oracle_block": str(block.get("block", "")).strip() or None,
        "oracle_proof_text": proof_text or None,
    }


def _proof_strategy_line(proof_text: str) -> str:
    tags = infer_proof_shape_tags(proof_text)
    if not tags:
        return "The proof is short and does not expose a strong tactic signature."
    return "The proof relies on these recognizable proof moves: " + ", ".join(tags) + "."


def _semantic_explanation(
    task: TheoremTask,
    theorem_types: List[str],
    coq_libraries: Dict[str, List[str]],
    proof_text: str,
) -> str:
    theorem_kind = _declaration_kind(task.theorem_declaration)
    statement = _explain_statement(task.theorem_declaration)
    libraries = coq_libraries.get("declared_imports", [])[:4]
    lib_text = ""
    if libraries:
        lib_text = " It is proved in a context importing " + ", ".join(libraries) + "."
    type_text = ""
    if theorem_types:
        type_text = " The theorem is normalized as " + ", ".join(theorem_types[:4]) + "."
    strategy_text = ""
    strategy_tags = infer_proof_shape_tags(proof_text)
    if strategy_tags:
        strategy_text = " The successful proof shape is " + ", ".join(strategy_tags[:4]) + "."
    return (
        f"{theorem_kind} {task.theorem_name} from {task.project}/{task.file_relpath}: "
        f"{statement}.{lib_text}{type_text}{strategy_text}"
    ).strip()


def _complete_proof_block(task: TheoremTask, proof_text: str) -> str:
    stripped = proof_text.strip()
    if not stripped:
        return ""
    if task.theorem_declaration in stripped:
        return stripped
    return task.theorem_declaration.rstrip() + "\n" + stripped


def _summarize_reasoning(
    task: TheoremTask,
    result: Dict[str, Any],
    theorem_types: List[str],
    coq_libraries: Dict[str, List[str]],
    final_proof: str,
    oracle_proof: str,
) -> str:
    summary = str(result.get("summary", "")).strip() or "No run summary was recorded."
    proof_basis = final_proof if final_proof else oracle_proof
    lines = [
        "# Reasoning",
        "",
        f"- Theorem: `{task.theorem_name}`",
        f"- Source theorem id: `{task.theorem_id}`",
        f"- Project file: `{task.project}/{task.file_relpath}`",
        f"- Normalized theorem types: {', '.join(theorem_types)}",
        "",
        "## Why this proof shape fits",
        "",
        summary,
        "",
        _proof_strategy_line(proof_basis),
        "",
        "## Coq library context",
        "",
        f"- Declared imports: {', '.join(coq_libraries.get('declared_imports', [])[:8]) or 'none detected'}",
        f"- Referenced namespaces: {', '.join(coq_libraries.get('referenced_namespaces', [])[:8]) or 'none detected'}",
    ]
    if final_proof:
        lines.extend(
            [
                "",
                "## Outcome-aware reasoning",
                "",
                "The final proof produced during the run is the authoritative successful script for this experience.",
            ]
        )
    elif oracle_proof:
        lines.extend(
            [
                "",
                "## Outcome-aware reasoning",
                "",
                "The direct run did not prove the theorem. The reasoning here uses the oracle proof to explain the right proof structure for future retrieval.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _diff_strategy_summary(failed_proof: str, oracle_proof: str) -> str:
    failed_tags = set(infer_proof_shape_tags(failed_proof))
    oracle_tags = set(infer_proof_shape_tags(oracle_proof))
    missing = [tag for tag in sorted(oracle_tags - failed_tags)]
    if "case_analysis" in missing:
        return "The failed attempt never performs the case split used by the successful proof."
    if "rewrite" in missing:
        return "The failed attempt misses a rewrite step that the successful proof relies on."
    if "apply_chain" in missing:
        return "The failed attempt does not invoke the key lemma/application chain present in the successful proof."
    if "induction" in missing:
        return "The failed attempt avoids the induction step required by the successful proof."
    if missing:
        return "The failed attempt is missing these proof moves from the successful script: " + ", ".join(missing) + "."
    if failed_proof.strip() != oracle_proof.strip():
        return "The failed attempt has a different proof structure from the successful proof and does not align with the goal shape."
    return "The tool output indicates the proof still failed despite matching part of the expected structure."


def _oracle_fix_strategy(failed_proof: str, oracle_proof: str) -> str:
    oracle_tags = infer_proof_shape_tags(oracle_proof)
    if oracle_tags:
        return "Modify the attempt so it follows the oracle proof shape: " + ", ".join(oracle_tags) + "."
    if oracle_proof.strip():
        compact = " ".join(line.strip() for line in oracle_proof.splitlines() if line.strip())
        return f"Replace the failed script with the proof structure from `oracle_proof.v`, starting with: `{compact[:180]}`."
    return "No oracle proof was available for a concrete fix."


def _render_issues(
    failed_proofs: List[Dict[str, Any]],
    oracle_proof_text: str,
    theorem_types: List[str],
    coq_libraries: Dict[str, List[str]],
    run_summary: str,
) -> str:
    lines = ["# Issues", ""]
    if not failed_proofs:
        if oracle_proof_text:
            lines.extend(
                [
                    "No concrete failed proof script was captured in the run logs, but the run still failed before proving the theorem.",
                    "",
                    f"Run summary: {run_summary or 'No summary recorded.'}",
                    "",
                    "Why it is wrong:",
                    "The direct run never reached the proof structure that actually closes the goal. Use `oracle_proof.v` as the authoritative correction and treat the missing proof move as a retrieval target.",
                    "",
                    "How to modify it:",
                    _oracle_fix_strategy("", oracle_proof_text),
                    "",
                    f"Related theorem types: {', '.join(theorem_types)}",
                ]
            )
        else:
            lines.extend(
                [
                    "No blocking proof issues were recorded during this run.",
                    "",
                    f"Related theorem types: {', '.join(theorem_types)}",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    for index, failed in enumerate(failed_proofs, start=1):
        proof_text = str(failed.get("proof_text", "")).strip()
        error_excerpt = str(failed.get("error_excerpt", "")).strip() or "No explicit tool error was recorded."
        lines.extend(
            [
                f"## Issue {index}",
                "",
                f"- Failed attempt index: `{failed.get('index')}`",
                f"- Failure kind: `{failed.get('failure_kind')}`",
                f"- Related theorem types: {', '.join(theorem_types)}",
                f"- Related Coq libraries: {', '.join(coq_libraries.get('declared_imports', [])[:6]) or 'none detected'}",
                "",
                "### Failed proof",
                "",
                "```coq",
                proof_text or "(* no failed proof text captured *)",
                "```",
                "",
                "### Why it is wrong",
                "",
                error_excerpt,
                "",
                _diff_strategy_summary(proof_text, oracle_proof_text),
                "",
                "### How to modify it",
                "",
                _oracle_fix_strategy(proof_text, oracle_proof_text),
            ]
        )
        if oracle_proof_text:
            lines.extend(
                [
                    "",
                    "### Oracle comparison",
                    "",
                    "Consult `oracle_proof.v` for the authoritative corrected script used in this postmortem experience.",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_result(
    task: TheoremTask,
    result: Dict[str, Any],
    final_proof: str,
    oracle_block: str,
) -> str:
    status = str(result.get("final_status", "failed"))
    lines = [
        "# Result",
        "",
        f"- Theorem: `{task.theorem_name}`",
        f"- Source theorem id: `{task.theorem_id}`",
        f"- Final status: `{status}`",
        f"- Run summary: {str(result.get('summary', '')).strip() or 'No summary recorded.'}",
        "",
    ]
    if final_proof:
        lines.extend(
            [
                "The run produced a successful proof. The saved `final_proof.v` is the exact reusable artifact for future attempts.",
            ]
        )
    elif oracle_block:
        lines.extend(
            [
                "The run did not finish the proof. The experience was completed in postmortem mode using the hidden correct proof, saved to `oracle_proof.v`.",
            ]
        )
    else:
        lines.extend(
            [
                "The run did not finish the proof and no oracle proof could be extracted from the source file.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_experience_bundle(
    task: TheoremTask,
    result: Dict[str, Any],
    log_dir: Path,
) -> Dict[str, Any]:
    failed_proofs = extract_failed_proofs(result)
    repair_chain = extract_repair_chain(result)
    final_proof = _strip_code_fence(result.get("final_proof"))
    oracle = _extract_oracle_proof(task) if str(result.get("final_status", "")) not in SUCCESS_STATUSES else {
        "oracle_block": None,
        "oracle_proof_text": None,
    }
    oracle_block = str(oracle.get("oracle_block") or "")
    oracle_proof_text = str(oracle.get("oracle_proof_text") or "")
    proof_texts = [text for text in [final_proof, oracle_proof_text] if text]
    proof_texts.extend(item.get("proof_text", "") for item in failed_proofs if item.get("proof_text"))
    theorem_types = _normalized_theorem_types(task, proof_texts)
    coq_libraries = _extract_coq_libraries(task, final_proof, oracle_proof_text)
    semantic_explanation = _semantic_explanation(
        task,
        theorem_types,
        coq_libraries,
        final_proof or oracle_proof_text,
    )

    record_id = f"{_slug(task.theorem_id)}_{_slug(str(result.get('final_status', 'unknown')))}_{_slug(log_dir.name)}"

    reasoning_md = _summarize_reasoning(
        task,
        result,
        theorem_types,
        coq_libraries,
        final_proof,
        oracle_proof_text,
    )
    issues_md = _render_issues(
        failed_proofs,
        oracle_proof_text,
        theorem_types,
        coq_libraries,
        str(result.get("summary", "")).strip(),
    )
    result_md = _render_result(task, result, final_proof, oracle_block)

    return {
        "record_id": record_id,
        "source_theorem_id": task.theorem_id,
        "project": task.project,
        "file_relpath": task.file_relpath,
        "source_file_path": str(task.source_path()),
        "semantic_explanation": semantic_explanation,
        "normalized_theorem_types": theorem_types,
        "coq_libraries": coq_libraries,
        "reasoning_md": reasoning_md,
        "issues_md": issues_md,
        "result_md": result_md,
        "final_proof_text": _complete_proof_block(task, final_proof),
        "oracle_proof_text": oracle_block,
        "attempts": result.get("attempts", []) or [],
        "failed_proofs": failed_proofs,
        "repair_chain": repair_chain,
        "artifacts": {
            "log_dir": str(log_dir),
            "result_file": str(log_dir / "result.json"),
            "events_file": str(log_dir / "events.jsonl"),
            "readable_file": str(log_dir / "readable"),
            "task_file": str(log_dir / "task.json"),
            "prompt_file": str(result.get("prompt_file", "")),
        },
    }

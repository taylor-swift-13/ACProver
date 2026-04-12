#!/usr/bin/env python3
"""Build retrieval records from selected Coq standard-library modules."""

from __future__ import annotations

import argparse
import json
import openai
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from acprover_config import load_config
    from experience_store import experience_domain_root, refresh_experience_indexes
    from logging_utils import write_json, write_text
except ModuleNotFoundError:
    from .acprover_config import load_config
    from .experience_store import experience_domain_root, refresh_experience_indexes
    from .logging_utils import write_json, write_text


DECL_RE = re.compile(
    r"^\s*(Lemma|Theorem|Corollary|Proposition|Fact|Remark|Definition|Fixpoint|Inductive|Record)\s+([A-Za-z0-9_']+)\b"
)
NOTATION_RE = re.compile(r'^\s*Notation\s+(.+?)\s*:=')
END_RE = re.compile(r"^\s*(Qed|Defined|Admitted)\.")
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
DECL_HEAD_RE = re.compile(
    r"^\s*(Lemma|Theorem|Corollary|Proposition|Fact|Remark)\s+[A-Za-z0-9_']+\b(?:[^:\n]|\n(?!\s*Proof\b))*?:",
    re.MULTILINE,
)
REQUIRE_EXPORT_RE = re.compile(r"^\s*Require\s+Export\s+(.+)\.\s*$")
REQUIRE_IMPORT_RE = re.compile(r"^\s*Require\s+Import\s+(.+)\.\s*$")
EXPORT_RE = re.compile(r"^\s*Export\s+(.+)\.\s*$")
DECLARE_ML_RE = re.compile(r'^\s*Declare\s+ML\s+Module\s+"([^"]+)"\s*\.')
LTAC_RE = re.compile(r"^\s*Ltac\s+([A-Za-z0-9_']+)\b")
PROOF_ITEM_KINDS = {"Lemma", "Theorem", "Corollary", "Proposition", "Fact", "Remark"}
ITEM_KIND_MAP = {
    "Lemma": "lemma",
    "Theorem": "theorem",
    "Corollary": "corollary",
    "Proposition": "proposition",
    "Fact": "fact",
    "Remark": "remark",
    "Definition": "definition",
    "Fixpoint": "fixpoint",
    "Inductive": "inductive",
    "Record": "record",
    "Notation": "notation",
    "Module": "module",
    "Ltac": "tactic",
}


@dataclass
class StdlibRecord:
    record_id: str
    module_path: str
    item_kind: str
    item_name: str
    semantic_explanation: str
    normalized_theorem_types: List[str]
    context: str
    proof: str
    related: List[Dict[str, str]]
    detail_md: str
    reasoning_md: str


@dataclass
class StdlibBuildOptions:
    limit: Optional[int] = None


def _normalize_code_block(text: str) -> str:
    return text.rstrip() + "\n"


def _normalize_item_name(kind: str, name: str) -> str:
    if kind != "Notation":
        return name
    stripped = name.strip()
    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
        stripped = stripped[1:-1]
    replacements = {
        "[": "_lbrack_",
        "]": "_rbrack_",
        "(": "_lparen_",
        ")": "_rparen_",
        "{": "_lbrace_",
        "}": "_rbrace_",
        ";": "_semi_",
        ":": "_colon_",
        ",": "_comma_",
        ".": "_dot_",
        "'": "_quote_",
        "`": "_bquote_",
        "/": "_slash_",
        "\\": "_bslash_",
        "|": "_bar_",
    }
    safe = stripped
    for source, target in replacements.items():
        safe = safe.replace(source, target)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"[^A-Za-z0-9_+*<>=@!$%^&?-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_") or "notation"


def _slug_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_'.-]+", "_", text).strip("_")
    return safe or "item"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _stdlib_record_dir(record_id: str) -> Path:
    safe = record_id.replace("::", "__").replace("/", "_")
    return experience_domain_root("stdlib") / safe


def _run_in_conda(argv: List[str]) -> str:
    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("`conda` is required to locate the Coq standard library.")
    config = load_config()
    command = [conda, "run", "-n", config.vector_conda_env, *argv]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "command failed")
    return completed.stdout.strip()


def detect_stdlib_root() -> Path:
    root = Path(_run_in_conda(["coqc", "-where"])).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Coq stdlib root not found: {root}")
    return root


def module_to_source_path(module_path: str, stdlib_root: Path) -> Path:
    if not module_path.startswith("Coq."):
        raise ValueError(f"unsupported module path: {module_path}")
    relative = Path("theories") / Path(*module_path.split(".")[1:])
    source = (stdlib_root / relative).with_suffix(".v")
    if not source.exists():
        raise FileNotFoundError(f"source file for {module_path} not found: {source}")
    return source


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _extract_supporting_context(source_text: str, declaration: str) -> str:
    blocks: List[str] = []
    if "Add " in declaration:
        add_block = _extract_named_block(source_text, "Add")
        if add_block:
            blocks.append(add_block)
    return "\n\n".join(blocks).strip()


def _build_llm_prompt(
    *,
    kind: str,
    name: str,
    declaration: str,
    proof_text: str,
    module_path: str,
    supporting_context: str,
) -> str:
    proof_block = proof_text.strip()
    context_block = supporting_context.strip()
    return f"""You are generating theorem-retrieval artifacts for a Coq theorem database.

Target theorem:
- module_path: {module_path}
- theorem_name: {name}
- theorem_kind: {kind}

Theorem statement:
```coq
{declaration.rstrip()}
```

Saved proof:
```coq
{proof_block}
```

Supporting definitions or context:
```coq
{context_block}
```

Output JSON with exactly these fields:
- semantic_explanation
- detail_md
- reasoning_md

Requirements:
- semantic_explanation:
  - pure natural language
  - short
  - use 1 sentence only
  - target 12 to 24 words
  - never exceed 32 words
  - explain the theorem itself
  - no markdown code fences
  - avoid raw Coq syntax unless unavoidable
  - do not explain the proof
  - do not restate every quantifier or every parameter
  - do not start with phrases like "The lemma states that", "This theorem says that", "The theorem states that", or similar wrappers
  - start directly with the mathematical content
- detail_md:
  - detailed
  - explain the theorem itself
  - explain what the statement says
  - explain what the conclusion is asserting
  - explain how the theorem is used
  - include relevant Coq code blocks
  - do not focus on proof tactics
- reasoning_md:
  - detailed
  - explain the key definitions needed by the proof
  - explain why the theorem is proved this way
  - explain why the proof shape fits the statement
  - include relevant Coq code blocks when useful

Keep the explanation concrete and polished. Avoid generic filler."""


def _generate_llm_artifacts(
    *,
    kind: str,
    name: str,
    declaration: str,
    proof_text: str,
    module_path: str,
    source_text: str,
) -> Dict[str, str]:
    config = load_config()
    client = openai.OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )
    prompt = _build_llm_prompt(
        kind=kind,
        name=name,
        declaration=declaration,
        proof_text=proof_text,
        module_path=module_path,
        supporting_context=_extract_supporting_context(source_text, declaration),
    )
    messages = [
        {"role": "system", "content": "You generate precise theorem-database artifacts in valid JSON."},
        {"role": "user", "content": prompt + "\n\nReturn JSON only."},
    ]
    last_error = "unknown generation failure"
    for attempt in range(3):
        response = client.chat.completions.create(
            model=config.semantic_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=config.semantic_temperature,
        )
        raw = response.choices[0].message.content or ""
        payload = _parse_llm_json_payload(raw)
        if not isinstance(payload, dict):
            last_error = "LLM generation returned non-object JSON"
        else:
            semantic_explanation = str(payload.get("semantic_explanation", "")).strip()
            detail_md = str(payload.get("detail_md", "")).strip()
            reasoning_md = str(payload.get("reasoning_md", "")).strip()
            if not semantic_explanation or not detail_md or not reasoning_md:
                last_error = "LLM generation returned incomplete fields"
            else:
                return {
                    "semantic_explanation": semantic_explanation,
                    "detail_md": detail_md,
                    "reasoning_md": reasoning_md,
                }
        if attempt < 2:
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "The previous JSON was invalid or incomplete. "
                        "Fix it and return valid JSON only with non-empty fields "
                        "`semantic_explanation`, `detail_md`, and `reasoning_md`."
                    ),
                }
            ]
    raise RuntimeError(last_error)


def _parse_llm_json_payload(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise RuntimeError("LLM generation returned empty content")
    candidates = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidates.append("\n".join(lines[1:-1]).strip())
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1].strip())

    errors: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
            try:
                payload = json.loads(sanitized)
            except json.JSONDecodeError as inner_exc:
                errors.append(str(inner_exc))
                continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("LLM generation returned invalid JSON: " + " | ".join(errors[:4]))


def _generate_artifacts(
    *,
    kind: str,
    name: str,
    declaration: str,
    proof_text: str,
    module_path: str,
    source_text: str,
) -> Dict[str, str]:
    return _generate_llm_artifacts(
        kind=kind,
        name=name,
        declaration=declaration,
        proof_text=proof_text,
        module_path=module_path,
        source_text=source_text,
    )


def _extract_statement_body(declaration: str) -> str:
    stripped = declaration.strip().rstrip(".")
    if stripped.startswith("Module "):
        return stripped
    if stripped.startswith("Ltac "):
        return stripped
    if stripped.startswith("Definition ") or stripped.startswith("Fixpoint "):
        if ":=" in stripped:
            return stripped.split(":=", 1)[1].strip()
        return stripped
    if stripped.startswith("Inductive ") or stripped.startswith("Record ") or stripped.startswith("Notation "):
        return stripped
    match = DECL_HEAD_RE.match(declaration)
    if match is None:
        if ":" not in declaration:
            return declaration.strip().rstrip(".")
        return declaration.split(":", 1)[1].strip().rstrip(".")
    body = declaration[match.end() :]
    if not body.strip():
        return declaration.strip().rstrip(".")
    return body.strip().rstrip(".")


def _humanize_body(body: str) -> str:
    text = " ".join(body.split())
    text = text.replace("++", " append ")
    text = text.replace("[]", " empty list ")
    text = text.replace("::", " cons ")
    text = text.replace("<>", " is not equal to ")
    text = text.replace("<->", " iff ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _explain_statement(declaration: str) -> str:
    body = _extract_statement_body(declaration)
    humanized = _humanize_body(body)
    if "<->" in body:
        left, right = [part.strip() for part in humanized.split(" iff ", 1)]
        return f"it states that {left} is equivalent to {right}"
    if "->" in body:
        parts = [part.strip() for part in humanized.split("->")]
        if len(parts) >= 2:
            assumptions = ", ".join(parts[:-1])
            conclusion = parts[-1]
            return f"it shows that if {assumptions}, then {conclusion}"
    if body.lower().startswith("forall "):
        return f"it gives a universally quantified fact about {humanized}"
    return f"it establishes {humanized}"


def _semantic_sentence(kind: str, name: str, declaration: str) -> str:
    body = _extract_statement_body(declaration)
    if kind == "Module":
        return f"`{name}` re-exports or assembles a group of standard-library components."
    if kind == "Ltac":
        return f"`{name}` defines a tactic used to automate proof steps in this module."
    if kind == "Definition":
        return f"`{name}` defines {body}.".replace("`" + name + "` defines " + name + " ", f"`{name}` defines ")
    if kind == "Fixpoint":
        return f"`{name}` is a recursive definition over {body}."
    if kind == "Inductive":
        return f"`{name}` introduces an inductive type or relation in this module."
    if kind == "Record":
        return f"`{name}` introduces a record type in this module."
    if kind == "Notation":
        return f"`{name}` introduces notation used by later list developments."
    if name == "app_nil_r":
        return "Appending the empty list to the right of a list leaves the list unchanged."
    if name == "app_nil_l":
        return "Appending a list to the empty list on the left leaves the list unchanged."
    if name == "Add_app":
        return "If a list is split into a prefix and a suffix, inserting one element at that boundary yields the prefix followed by the new element and then the suffix."
    if body.startswith("Add "):
        return "This theorem describes how inserting one element into a list changes the resulting list."
    if "++ [] = " in body or " append empty list = " in _humanize_body(body):
        return "This theorem says that appending the empty list does not change a list."
    sentence = _explain_statement(declaration)
    sentence = sentence.removeprefix("it establishes ").removeprefix("it states that ").removeprefix("it shows that ")
    sentence = sentence.removeprefix("it gives a universally quantified fact about ")
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _proof_shape_tags(proof_text: str) -> List[str]:
    lowered = proof_text.lower()
    tags: List[str] = []
    if "induction" in lowered or "elim" in lowered:
        tags.append("induction")
    if "rewrite" in lowered:
        tags.append("rewrite")
    if "simpl" in lowered or "cbn" in lowered:
        tags.append("simplification")
    if "destruct" in lowered or "case" in lowered:
        tags.append("case_analysis")
    if "apply " in lowered:
        tags.append("apply")
    if "reflexivity" in lowered or "easy" in lowered or "trivial" in lowered:
        tags.append("closure")
    return sorted(set(tags))


def _normalized_theorem_types(kind: str, declaration: str, proof_text: str) -> List[str]:
    if kind not in PROOF_ITEM_KINDS:
        return []
    body = _extract_statement_body(declaration)
    lowered_body = body.lower()
    lowered_proof = proof_text.lower()
    theorem_types: List[str] = []
    if "<->" in body:
        theorem_types.append("iff")
    if "->" in body:
        theorem_types.append("implication")
    if "=" in body:
        theorem_types.append("equality")
    if any(token in lowered_body for token in ["<=", ">=", "<", ">"]):
        theorem_types.append("order")
    if "exists" in lowered_body:
        theorem_types.append("existential")
    if "forall" in lowered_body:
        theorem_types.append("structural")
    if "add " in lowered_body or "nodup" in lowered_body or "in " in lowered_body:
        theorem_types.append("structural")
    if "induction" in lowered_proof or "elim" in lowered_proof:
        theorem_types.append("induction")
    if "rewrite" in lowered_proof:
        theorem_types.append("rewrite_rule")
    if not theorem_types:
        theorem_types.append("structural")
    return sorted(set(theorem_types))


def _read_file_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _extract_named_block(source_text: str, symbol: str) -> str:
    pattern = re.compile(
        rf"(?ms)^\s*(Inductive|Definition|Fixpoint|Lemma|Theorem|Record)\s+{re.escape(symbol)}\b.*?\.\s*$"
    )
    match = pattern.search(source_text)
    if not match:
        return ""
    return match.group(0).strip()


def _line_starts_proof(line: str) -> bool:
    return bool(re.match(r"^\s*Proof\b", line))


def _collect_declarations(source_path: Path) -> List[Dict[str, str]]:
    lines = _read_file_lines(source_path)
    records: List[Dict[str, str]] = []
    index = 0
    while index < len(lines):
        match = DECL_RE.match(lines[index])
        notation_match = NOTATION_RE.match(lines[index]) if not match else None
        if not match and not notation_match:
            index += 1
            continue
        if match:
            kind = match.group(1)
            name = _normalize_item_name(kind, match.group(2))
        else:
            kind = "Notation"
            name = _normalize_item_name(kind, notation_match.group(1))
        declaration_lines = [lines[index].rstrip("\n")]
        if "." in lines[index]:
            next_index = index
        else:
            next_index = index + 1
            while next_index < len(lines):
                declaration_lines.append(lines[next_index].rstrip("\n"))
                if "." in lines[next_index]:
                    break
                next_index += 1
        declaration = "\n".join(line for line in declaration_lines).strip()
        proof_lines: List[str] = []
        cursor = next_index + 1
        if kind in PROOF_ITEM_KINDS:
            while cursor < len(lines) and not _line_starts_proof(lines[cursor]):
                if lines[cursor].strip():
                    break
                cursor += 1
            if cursor < len(lines) and _line_starts_proof(lines[cursor]):
                while cursor < len(lines):
                    proof_lines.append(lines[cursor].rstrip("\n"))
                    if END_RE.match(lines[cursor]):
                        cursor += 1
                        break
                    cursor += 1
        proof_text = "\n".join(line for line in proof_lines).strip()
        records.append(
            {
                "kind": kind,
                "name": name,
                "item_kind": ITEM_KIND_MAP[kind],
                "declaration": declaration,
                "proof_text": proof_text,
            }
        )
        index = max(cursor, index + 1)
    if records:
        return records
    source_text = source_path.read_text(encoding="utf-8")
    module_name = source_path.stem
    require_exports: List[str] = []
    require_imports: List[str] = []
    exports: List[str] = []
    ml_modules: List[str] = []
    tactics: List[str] = []
    for line in source_text.splitlines():
        match = REQUIRE_EXPORT_RE.match(line)
        if match:
            require_exports.extend(tok.strip() for tok in match.group(1).split() if tok.strip())
            continue
        match = REQUIRE_IMPORT_RE.match(line)
        if match:
            require_imports.extend(tok.strip() for tok in match.group(1).split() if tok.strip())
            continue
        match = EXPORT_RE.match(line)
        if match:
            exports.extend(tok.strip() for tok in match.group(1).split() if tok.strip())
            continue
        match = DECLARE_ML_RE.match(line)
        if match:
            ml_modules.append(match.group(1).strip())
            continue
        match = LTAC_RE.match(line)
        if match:
            tactics.append(match.group(1))
    aggregate_lines = []
    if require_exports:
        aggregate_lines.append("Require Export " + " ".join(require_exports) + ".")
    if require_imports:
        aggregate_lines.append("Require Import " + " ".join(require_imports) + ".")
    if exports:
        aggregate_lines.append("Export " + " ".join(exports) + ".")
    if ml_modules:
        aggregate_lines.extend([f'Declare ML Module "{name}".' for name in ml_modules])
    if aggregate_lines:
        records.append(
            {
                "kind": "Module",
                "name": _slug_name(module_name),
                "item_kind": ITEM_KIND_MAP["Module"],
                "declaration": "\n".join(aggregate_lines),
                "proof_text": "",
            }
        )
    for tactic_name in tactics:
        records.append(
            {
                "kind": "Ltac",
                "name": _slug_name(tactic_name),
                "item_kind": ITEM_KIND_MAP["Ltac"],
                "declaration": f"Ltac {tactic_name}.",
                "proof_text": "",
            }
        )
    return records


def _extract_related_items(
    item: Dict[str, str],
    module_path: str,
    all_items: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    text = " ".join(part for part in [item.get("declaration", ""), item.get("proof_text", "")] if part).lower()
    related: List[Dict[str, str]] = []
    for candidate in all_items:
        if candidate["name"] == item["name"]:
            continue
        candidate_name = candidate["name"]
        if len(candidate_name) < 2:
            continue
        if re.search(rf"\b{re.escape(candidate_name.lower())}\b", text):
            related.append(
                {
                    "kind": candidate["item_kind"],
                    "id": f"{module_path}::{candidate_name}",
                }
            )
    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in related:
        key = (entry["kind"], entry["id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped[:24]


def _semantic_explanation(kind: str, name: str, declaration: str) -> str:
    return _semantic_sentence(kind, name, declaration)


def _detail_body(kind: str, name: str, declaration: str) -> List[str]:
    body = _extract_statement_body(declaration)
    lines = [
        f"`{name}` is a standard-library {kind.lower()} in this module.",
        "",
        "## Statement",
        "",
        "```coq",
        declaration.rstrip(),
        "```",
        "",
        "## What This Item Does",
        "",
    ]

    if name == "Add_app":
        lines.extend(
            [
                "This lemma says that if you split a list into `l1 ++ l2`, then adding one element `a` exactly at that split point produces `l1 ++ a :: l2`.",
                "So it does not talk about an arbitrary append fact; it gives a concrete witness for the `Add` relation.",
                "",
                "In other words:",
                "- the source list is `l1 ++ l2`",
                "- the target list is `l1 ++ a :: l2`",
                "- the theorem certifies that the target list is obtained by inserting `a` into the source list",
                "",
                "## How To Use It",
                "",
                "Use this lemma when you need to prove an `Add` goal for a list that is already written as a concatenation.",
                "It is especially useful when working with permutation-style reasoning, `NoDup`, or proofs that insert one element into a list at a chosen boundary.",
            ]
        )
        return lines

    if name == "app_nil_r":
        lines.extend(
            [
                "This theorem says that appending the empty list on the right does not change a list.",
                "It is the standard right-identity law for list append.",
                "",
                "## How To Use It",
                "",
                "Use it to simplify expressions of the form `l ++ []`.",
                "In practice it is most often used with `rewrite app_nil_r` to normalize a goal or hypothesis after a proof step introduces a trailing empty list.",
            ]
        )
        return lines

    if kind in PROOF_ITEM_KINDS:
        lines.extend(
            [
                _explain_statement(declaration).capitalize() + ".",
                "",
                "## How To Use It",
                "",
                "Use this theorem when your goal or hypotheses match the statement shape above.",
                "It is typically applied directly, rewritten with, or specialized to concrete arguments.",
            ]
        )
    elif kind in {"Definition", "Fixpoint"}:
        lines.extend(
            [
                f"This {kind.lower()} introduces a reusable term named `{name}`.",
                "",
                "## How To Use It",
                "",
                "Use this item when later statements or proofs refer to the defined symbol directly.",
            ]
        )
    elif kind in {"Inductive", "Record"}:
        lines.extend(
            [
                f"This {kind.lower()} introduces a reusable type or relation named `{name}`.",
                "",
                "## How To Use It",
                "",
                "Use this item when later statements reason by constructors, inversion, or induction over this object.",
            ]
        )
    elif kind == "Module":
        lines.extend(
            [
                "This module-level item groups together re-exported components from other standard-library files.",
                "",
                "## How To Use It",
                "",
                "Use this record to understand which lower-level libraries this module assembles and re-exports.",
            ]
        )
    elif kind == "Ltac":
        lines.extend(
            [
                "This item defines a tactic script used for proof automation.",
                "",
                "## How To Use It",
                "",
                "Use this record when later proofs invoke the tactic by name.",
            ]
        )
    else:
        lines.extend(
            [
                f"This {kind.lower()} introduces a reusable notation named `{name}`.",
                "",
                "## How To Use It",
                "",
                "Use this item to understand later statements that rely on this notation.",
            ]
        )
    if "Add " in body:
        lines.extend(
            [
                "",
                "If you are not used to `Add`, read it as: the second list is obtained from the first list by inserting one occurrence of the element.",
            ]
        )
    return lines


def _detail_md(kind: str, name: str, declaration: str, module_path: str, source_text: str) -> str:
    lines = ["# Detail", ""]
    lines.extend(_detail_body(kind, name, declaration))
    lines[1] = f"`{module_path}::{name}`"
    return "\n".join(lines).rstrip() + "\n"


def _reasoning_md(kind: str, name: str, declaration: str, proof_text: str, module_path: str, source_text: str) -> str:
    if kind not in PROOF_ITEM_KINDS or not proof_text.strip():
        return ""
    tags = _proof_shape_tags(proof_text)
    if tags:
        why = "The saved standard-library proof appears to rely on " + ", ".join(f"`{tag}`" for tag in tags) + "."
    else:
        why = "The saved standard-library proof is short or opaque enough that no strong tactic signature is visible from text alone."
    lines = [
        "# Reasoning",
        "",
        f"`{module_path}::{name}` is proved this way because the statement shape is simple and reusable in later list proofs.",
        "",
        "## Statement",
        "",
        "```coq",
        declaration.rstrip(),
        "```",
        "",
        "## Why This Proof Shape Fits",
        "",
        why,
        "This lemma is usually cited as a small structural fact, not as a complex derived theorem.",
    ]
    if "Add " in declaration:
        lines.extend(
            [
                "",
                "## Key Definition",
                "",
                "The proof depends on the inductive relation `Add`, which expresses that one list is obtained from another by inserting one element at some position.",
            ]
        )
    if "Add " in declaration:
        add_block = _extract_named_block(source_text, "Add")
        if add_block:
            lines.extend(["", "```coq", add_block, "```"])
    if proof_text.strip():
        lines.extend(
            [
                "",
                "## Saved Proof",
                "",
                "```coq",
                proof_text.rstrip(),
                "```",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_records_for_module(
    module_path: str,
    stdlib_root: Optional[Path] = None,
    *,
    options: Optional[StdlibBuildOptions] = None,
) -> List[StdlibRecord]:
    options = options or StdlibBuildOptions()
    stdlib_root = stdlib_root or detect_stdlib_root()
    source_path = module_to_source_path(module_path, stdlib_root)
    source_text = source_path.read_text(encoding="utf-8")
    records: List[StdlibRecord] = []
    items = _collect_declarations(source_path)
    for item in items[: options.limit]:
        record_id = f"{module_path}::{item['name']}"
        generated = _generate_artifacts(
            kind=item["kind"],
            name=item["name"],
            declaration=item["declaration"],
            proof_text=item["proof_text"],
            module_path=module_path,
            source_text=source_text,
        )
        records.append(
            StdlibRecord(
                record_id=record_id,
                module_path=module_path,
                item_kind=item["item_kind"],
                item_name=item["name"],
                semantic_explanation=generated["semantic_explanation"],
                normalized_theorem_types=_normalized_theorem_types(
                    item["kind"],
                    item["declaration"],
                    item["proof_text"],
                ),
                context=item["declaration"].rstrip(),
                proof=(
                    (item["declaration"].rstrip() + "\n" + item["proof_text"].rstrip()).strip()
                    if item["proof_text"].strip()
                    else ""
                ),
                related=_extract_related_items(item, module_path, items),
                detail_md=_normalize_code_block(generated["detail_md"]),
                reasoning_md=_normalize_code_block(generated["reasoning_md"]),
            )
        )
    return records


def write_records(records: List[StdlibRecord], rebuild_indexes: bool = True) -> Dict[str, Any]:
    written: List[str] = []
    domain_root = experience_domain_root("stdlib")
    for record in records:
        record_dir = _stdlib_record_dir(record.record_id)
        record_dir.mkdir(parents=True, exist_ok=True)
        detail_path = record_dir / "detail.md"
        reasoning_path = record_dir / "reasoning.md"
        metadata_path = record_dir / "metadata.json"
        write_text(detail_path, _normalize_code_block(record.detail_md))
        write_text(reasoning_path, _normalize_code_block(record.reasoning_md))
        write_json(
            metadata_path,
            {
                "record_id": record.record_id,
                "module_path": record.module_path,
                "item_kind": record.item_kind,
                "item_name": record.item_name,
                "semantic_explanation": record.semantic_explanation,
                "normalized_theorem_types": record.normalized_theorem_types,
                "context": record.context,
                "proof": record.proof,
                "related": record.related,
                "detail_path": str(detail_path),
                "reasoning_path": str(reasoning_path),
            },
        )
        written.append(str(metadata_path))
    refresh = refresh_experience_indexes(domain_root) if rebuild_indexes else {}
    return {
        "success": True,
        "record_count": len(records),
        "metadata_paths": written,
        "refresh": refresh,
    }


def build_and_write(
    module_path: str,
    rebuild_indexes: bool = True,
    *,
    options: Optional[StdlibBuildOptions] = None,
) -> Dict[str, Any]:
    options = options or StdlibBuildOptions()
    stdlib_root = detect_stdlib_root()
    source_path = module_to_source_path(module_path, stdlib_root)
    source_text = source_path.read_text(encoding="utf-8")
    domain_root = experience_domain_root("stdlib")
    written: List[str] = []
    record_count = 0

    items = _collect_declarations(source_path)
    for item in items[: options.limit]:
        record_id = f"{module_path}::{item['name']}"
        record_dir = _stdlib_record_dir(record_id)
        metadata_path = record_dir / "metadata.json"
        if metadata_path.exists():
            written.append(str(metadata_path))
            record_count += 1
            continue
        generated = _generate_artifacts(
            kind=item["kind"],
            name=item["name"],
            declaration=item["declaration"],
            proof_text=item["proof_text"],
            module_path=module_path,
            source_text=source_text,
        )

        record = StdlibRecord(
            record_id=record_id,
            module_path=module_path,
            item_kind=item["item_kind"],
            item_name=item["name"],
            semantic_explanation=generated["semantic_explanation"],
            normalized_theorem_types=_normalized_theorem_types(
                item["kind"],
                item["declaration"],
                item["proof_text"],
            ),
            context=item["declaration"].rstrip(),
            proof=(
                (item["declaration"].rstrip() + "\n" + item["proof_text"].rstrip()).strip()
                if item["proof_text"].strip()
                else ""
            ),
            related=_extract_related_items(item, module_path, items),
            detail_md=_normalize_code_block(generated["detail_md"]),
            reasoning_md=_normalize_code_block(generated["reasoning_md"]),
        )

        record_dir.mkdir(parents=True, exist_ok=True)
        detail_path = record_dir / "detail.md"
        reasoning_path = record_dir / "reasoning.md"
        write_text(detail_path, _normalize_code_block(record.detail_md))
        write_text(reasoning_path, _normalize_code_block(record.reasoning_md))
        write_json(
            metadata_path,
            {
                "record_id": record.record_id,
                "module_path": record.module_path,
                "item_kind": record.item_kind,
                "item_name": record.item_name,
                "semantic_explanation": record.semantic_explanation,
                "normalized_theorem_types": record.normalized_theorem_types,
                "context": record.context,
                "proof": record.proof,
                "related": record.related,
                "detail_path": str(detail_path),
                "reasoning_path": str(reasoning_path),
            },
        )
        written.append(str(metadata_path))
        record_count += 1

    refresh = refresh_experience_indexes(domain_root) if rebuild_indexes else {}
    return {
        "success": True,
        "record_count": record_count,
        "metadata_paths": written,
        "refresh": refresh,
        "module_path": module_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Build standard-library retrieval records.")
    parser.add_argument("--module-path", default="Coq.Lists.List")
    parser.add_argument("--no-rebuild-indexes", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_and_write(
        args.module_path,
        rebuild_indexes=not args.no_rebuild_indexes,
        options=StdlibBuildOptions(limit=args.limit),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

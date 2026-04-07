#!/usr/bin/env python3
"""Build retrieval records from selected Coq standard-library modules."""

from __future__ import annotations

import argparse
import json
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


DECL_RE = re.compile(r"^\s*(Lemma|Theorem|Corollary|Proposition|Fact|Remark)\s+([A-Za-z0-9_']+)\b")
END_RE = re.compile(r"^\s*(Qed|Defined|Admitted)\.")
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


@dataclass
class StdlibRecord:
    record_id: str
    module_path: str
    semantic_explanation: str
    normalized_theorem_types: List[str]
    context: str
    proof: str
    detail_md: str
    reasoning_md: str


def _normalize_code_block(text: str) -> str:
    return text.rstrip() + "\n"


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


def _extract_statement_body(declaration: str) -> str:
    if ":" not in declaration:
        return declaration.strip().rstrip(".")
    return declaration.rsplit(":", 1)[1].strip().rstrip(".")


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
        rf"(?ms)^\s*(Inductive|Definition|Fixpoint|Lemma|Theorem)\s+{re.escape(symbol)}\b.*?\.\s*$"
    )
    match = pattern.search(source_text)
    if not match:
        return ""
    return match.group(0).strip()


def _collect_declarations(source_path: Path) -> List[Dict[str, str]]:
    lines = _read_file_lines(source_path)
    records: List[Dict[str, str]] = []
    index = 0
    while index < len(lines):
        match = DECL_RE.match(lines[index])
        if not match:
            index += 1
            continue
        kind = match.group(1)
        name = match.group(2)
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
                "declaration": declaration,
                "proof_text": proof_text,
            }
        )
        index = max(cursor, index + 1)
    return records


def _semantic_explanation(kind: str, name: str, declaration: str) -> str:
    return _semantic_sentence(kind, name, declaration)


def _detail_body(kind: str, name: str, declaration: str) -> List[str]:
    body = _extract_statement_body(declaration)
    lines = [
        f"`{name}` is a standard-library {kind.lower()} about lists.",
        "",
        "## Statement",
        "",
        "```coq",
        declaration.rstrip(),
        "```",
        "",
        "## What This Theorem Does",
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


def build_records_for_module(module_path: str, stdlib_root: Optional[Path] = None) -> List[StdlibRecord]:
    stdlib_root = stdlib_root or detect_stdlib_root()
    source_path = module_to_source_path(module_path, stdlib_root)
    source_text = source_path.read_text(encoding="utf-8")
    records: List[StdlibRecord] = []
    for item in _collect_declarations(source_path):
        record_id = f"{module_path}::{item['name']}"
        records.append(
            StdlibRecord(
                record_id=record_id,
                module_path=module_path,
                semantic_explanation=_semantic_explanation(item["kind"], item["name"], item["declaration"]),
                normalized_theorem_types=_normalized_theorem_types(
                    item["kind"],
                    item["declaration"],
                    item["proof_text"],
                ),
                context=item["declaration"].rstrip(),
                proof=(item["declaration"].rstrip() + "\n" + item["proof_text"].rstrip()).strip(),
                detail_md=_detail_md(item["kind"], item["name"], item["declaration"], module_path, source_text),
                reasoning_md=_reasoning_md(
                    item["kind"],
                    item["name"],
                    item["declaration"],
                    item["proof_text"],
                    module_path,
                    source_text,
                ),
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
                "semantic_explanation": record.semantic_explanation,
                "normalized_theorem_types": record.normalized_theorem_types,
                "context": record.context,
                "proof": record.proof,
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


def build_and_write(module_path: str, rebuild_indexes: bool = True) -> Dict[str, Any]:
    records = build_records_for_module(module_path)
    result = write_records(records, rebuild_indexes=rebuild_indexes)
    result["module_path"] = module_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Build standard-library retrieval records.")
    parser.add_argument("--module-path", default="Coq.Lists.List")
    parser.add_argument("--no-rebuild-indexes", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_and_write(args.module_path, rebuild_indexes=not args.no_rebuild_indexes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

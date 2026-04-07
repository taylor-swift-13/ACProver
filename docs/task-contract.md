# Task Contract

This repository gives Codex a minimal theorem task rather than a large preloaded proof context.

## Input fields

Each task contains:

- `theorem_id`
- `project`
- `file_relpath`
- `repo_path`
- `compile_args`
- `theorem_name`
- `theorem_start_line`
- `theorem_declaration`

## Proof-hiding rule

- The original proof of the current target theorem is hidden.
- The theorem declaration remains visible.
- During a proving run, the launcher creates an isolated shadow workspace.
- In that workspace, the target source file itself is replaced with a writable masked temp file at the original relative path.
- Other repository files remain readable, but the original answer-bearing target file is not exposed to Codex.
- Other source material is not rewritten or globally hidden.

## Context strategy

Codex is expected to discover relevant context itself from the local repository and Coq environment instead of relying on a large preloaded source snippet.

The intended proving style is attempt-driven rather than reading-driven:

- do a small amount of targeted context lookup
- move quickly into `$ACPROVER_COQSTOQ_TOOLS step-tactic` or `$ACPROVER_COQSTOQ_TOOLS verify-proof`
- use failing proof attempts to guide the next search
- avoid spending long stretches reading large adjacent proofs before the first real proof attempt
- for difficult theorems, decompose early into auxiliary lemmas in the same session instead of continuing to read neighboring large proofs

## Environment boundary

- Environment preparation is handled by the launcher before Codex starts.
- Codex should focus on theorem proving, not on installing tools, debugging opam setup, or inspecting launcher internals.
- If proof tools are unavailable because of an environment failure, Codex should stop quickly and report that failure.
- Machine-specific launcher defaults are configured in `config/acprover.local.json`.

Preferred discovery tools:

- `rg`
- `sed`
- `find`
- `python3 "$ACPROVER_COQSTOQ_TOOLS" bm25-search ...`

Preferred proof tools:

- `python3 "$ACPROVER_COQSTOQ_TOOLS" verify-proof ...`
- `python3 "$ACPROVER_COQSTOQ_TOOLS" step-tactic ...`

Shell is still allowed, and should be used to find relevant definitions and lemmas in the repository, but Coq interaction should go through `$ACPROVER_COQSTOQ_TOOLS` instead of direct `coqc`, `coqtop`, or `verify.py` calls.

## Non-goals

- No local JSON action protocol
- No local theorem/lemma orchestrator
- No custom proof kernel in Python

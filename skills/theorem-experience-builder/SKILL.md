---
name: theorem-experience-builder
description: Use when building theorem records and vector indexes for standard-library or CoqStoq sources in this repository.
---

# Theorem Experience Builder

Use this skill when the task is to construct or refresh the theorem retrieval database.

## Scope

- The repository is in retrieval-only mode.
- Do not run proving.
- Build records from either:
  - Coq standard-library modules
  - existing CoqStoq-derived records

## Metadata requirements

### Standard-library records

`metadata.json` must contain:

- `record_id`
- `module_path`
- `semantic_explanation`
- `normalized_theorem_types`
- `context`
- `proof`
- `detail_path`
- `reasoning_path`

### CoqStoq records

`metadata.json` must contain:

- `record_id`
- `project`
- `file_path`
- `module_path`
- `semantic_explanation`
- `normalized_theorem_types`
- `context`
- `proof`
- `detail_path`
- `reasoning_path`

Do not add extra metadata fields unless the user asks for them.
Do not create extra artifact files for `context` or `proof`; they must be stored inline in `metadata.json`.

## Field requirements

- `semantic_explanation` must be pure natural language.
- `semantic_explanation` must briefly explain the theorem itself.
- `semantic_explanation` must not contain Markdown code fences.
- `semantic_explanation` should avoid raw Coq syntax unless a symbol is unavoidable.
- `context` stores the theorem statement code.
- `proof` stores the theorem together with its proof code.

## Markdown artifact requirements

- `detail.md` must be detailed.
- `detail.md` records the theorem itself.
- `detail.md` must explain:
  - the statement
  - what the conclusion is saying
  - how the theorem is used
- `detail.md` must include relevant Coq code blocks.

- `reasoning.md` must be detailed.
- `reasoning.md` records the important definitions needed by the proof and why the theorem is proved this way.
- `reasoning.md` must explain:
  - which definitions, relations, predicates, or constructions matter
  - why this proof shape is natural for the statement
  - how the proof depends on the surrounding definitions
- `reasoning.md` must include relevant Coq code blocks when useful.

- Do not generate `result.md`.

## Quality bar

- Do not produce shallow template text when a concrete explanation can be derived from the statement and proof.
- Prefer polished, specific explanations over generic phrases like “this is a standard-library lemma”.
- In `detail.md`, focus on the theorem itself rather than tactic narration.
- In `reasoning.md`, focus on definitions and proof rationale rather than a flat list of tactics.
- If the proof is short and structural, explain which structure of the statement makes that proof shape natural.

## Primary commands

Build standard-library records:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py build-stdlib-index --module-path Coq.Lists.List
```

Refresh CoqStoq indexes from existing CoqStoq records:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py build-coqstoq-index
```

## What this skill is for

Use this skill for:

- defining the metadata schema for theorem records
- generating `detail.md` and `reasoning.md`
- rebuilding metadata indexes and FAISS indexes
- keeping standard-library and CoqStoq record construction separate

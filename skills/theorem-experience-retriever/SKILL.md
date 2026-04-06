---
name: theorem-experience-retriever
description: Use when querying many theorem experiences in this repository from natural-language descriptions, then deciding what to read based on metadata before opening reasoning/issues/result/proof files.
---

# Theorem Experience Retriever

Use this skill when the task is retrieval-heavy rather than proof-heavy.

## Primary interface

Query by natural-language description:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py query-experience --description "prove an order lemma from equality in mathcomp" -k 10
```

The command returns JSON hits with:

- `record_id`
- `source_theorem_id`
- `project`
- `file_relpath`
- `source_file_path`
- `semantic_explanation`
- `normalized_theorem_types`
- `coq_libraries`
- `reasoning_path`
- `issues_path`
- `result_path`
- `final_proof_path`
- `oracle_proof_path`
- `score`

## Retrieval workflow

1. Start from one or more short natural-language theorem descriptions.
2. Run `query-experience` for each description.
3. Merge hits by `record_id`; prefer higher `score`.
4. Use metadata first:
   - compare `normalized_theorem_types`
   - compare `coq_libraries`
   - compare `project` and `file_relpath`
   - note `source_file_path` for direct fallback into the original CoqStoq source file
5. Only after metadata triage, open the saved files you actually need:
   - `reasoning_path`
   - `issues_path`
   - `result_path`
   - `final_proof_path`
   - `oracle_proof_path`

Do not start by scanning every `.md` or `.v` file under `experience/`.

## When metadata is not enough

If the retrieved metadata and saved artifacts still do not give enough context, go back to the original project file named by `source_file_path`.

Use the original source file to:

- inspect nearby definitions
- inspect notation and imports
- inspect adjacent lemmas in the same file
- locate names that appear in `coq_libraries`

Use this only after metadata triage. Do not start from the whole project tree when the metadata already narrows the target file.

## Batch querying

For many related theorem descriptions, vary the query wording slightly instead of sending one long paragraph.

Good pattern:

- one query for the logical shape
- one query for the math domain
- one query for the expected proof pattern

Example query set:

- `prove an order lemma from equality`
- `mathcomp real order implication lemma`
- `short ssreflect proof by case analysis`

After querying, dedupe by `record_id` and rank with metadata before reading files.

## Metadata-first reading policy

Use these rules to decide which file to open:

- Open `issues_path` first when the current task is failing or similar attempts already broke.
- Open `reasoning_path` first when the current task is still planning the proof shape.
- Open `result_path` first when you need a concise lesson before deeper reading.
- Open `final_proof_path` only when you already think the hit is structurally relevant.
- Open `oracle_proof_path` when you want postmortem fixes for failed runs.

## What this skill is for

Use this skill for:

- querying many theorem experiences
- building a shortlist from natural-language problem descriptions
- metadata-driven selection before reading artifacts
- retrieving failure fixes from `issues.md` and `oracle_proof.v`

Do not use this skill as a replacement for proving. Once the shortlist is ready, switch back to the proving workflow.

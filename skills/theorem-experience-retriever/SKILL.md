---
name: theorem-experience-retriever
description: Use when querying existing standard-library or CoqStoq theorem records in this repository from natural-language descriptions or metadata SQL.
---

# Theorem Experience Retriever

Use this skill when the task is retrieval-heavy rather than construction-heavy.

If the task is to build or refresh the database itself, use `theorem-experience-builder` instead.

## Primary interface

Rebuild standard-library indexes from existing records:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py build-stdlib-from-existing
```

Query standard-library records by natural-language description:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py query-stdlib --description "append with empty list on the right" -k 10
```

Convenience wrapper for standard-library natural-language retrieval:

```bash
python3 /home/yangfp/ACProver/scripts/query_experience.py --domain stdlib --description "append with empty list on the right" -k 10
```

Query standard-library metadata by SQL:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py query-stdlib-sql --sql "select record_id, module_path from records where module_path = 'Coq.Lists.List' limit 10"
```

Convenience wrapper for standard-library SQL retrieval:

```bash
python3 /home/yangfp/ACProver/scripts/query_experience.py --domain stdlib --sql "select record_id, item_kind, item_name from records where module_path = 'Coq.Lists.List' limit 10"
```

Query CoqStoq records by natural-language description:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py query-coqstoq --description "append with empty list on the right" -k 10
```

Query CoqStoq metadata by SQL:

```bash
python3 /home/yangfp/ACProver/src/coqstoq_tools.py query-coqstoq-sql --sql "select record_id, project, file_path from records limit 10"
```

The command returns JSON hits with:

- `record_id`
- `project` when the source is CoqStoq
- `file_path` when the source is CoqStoq
- `module_path`
- `semantic_explanation`
- `normalized_theorem_types`
- `context`
- `proof`
- `detail_path`
- `reasoning_path`
- `score`

## Indexing model

- The FAISS vector index embeds only `semantic_explanation`.
- `detail.md`, `reasoning.md`, `context`, `proof`, and `related` are inspected after retrieval; they are not part of the vector embedding.
- Metadata SQL queries run against the local SQLite index under `experience/`.

## Retrieval workflow

1. Start from one or more short natural-language theorem descriptions.
2. Choose the correct source first:
   - use `query-stdlib` for standard-library records
   - use `query-coqstoq` for CoqStoq-derived records
   - use `scripts/query_experience.py` when you want one wrapper for both natural-language and SQL retrieval
   - use `query-stdlib-sql` or `query-coqstoq-sql` when you need direct metadata filtering with SQL
3. Merge hits by `record_id`; prefer higher `score`.
4. Use metadata first:
   - compare `module_path`
   - compare `semantic_explanation`
   - compare `normalized_theorem_types`
   - inspect `context`
   - inspect `proof`
5. Only after metadata triage, open the saved files you actually need:
   - `detail_path`
   - `reasoning_path`

Do not start by scanning every `.md` file under `experience/`.

## Expansion policy

- The retrieved JSON hits are the starting point, not a hard boundary.
- The model should first consume the returned JSON metadata, then decide whether to expand.
- The model may freely continue reading based on any metadata field it finds useful.
- The model may run additional SQL queries over any metadata fields of interest.
- The model may open any `detail_path` or `reasoning_path` that becomes relevant during this expansion.
- For CoqStoq records, the model may also continue reading from the original project context via `project` and `file_path` when those fields are available.

## Metadata-first reading policy

Use these rules to decide which file to open:

- Read `context` in metadata when you need the theorem statement code immediately.
- Read `proof` in metadata when you need the saved proof code immediately.
- Open `detail_path` first when you need to understand the theorem statement and how it is used.
- Open `reasoning_path` first when you need the supporting definitions and the explanation of why the proof works.

## What this skill is for

Use this skill for:

- querying standard-library theorem records
- querying CoqStoq theorem records
- rebuilding stdlib indexes after records already exist
- querying metadata with SQL over the local SQLite index
- building a shortlist from natural-language theorem descriptions
- metadata-driven selection before reading artifacts
- retrieving theorem explanations

Do not use this skill to define database schema or artifact-writing rules. That belongs to `theorem-experience-builder`.

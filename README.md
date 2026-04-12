# ACProver

ACProver is now a local retrieval workspace for Coq standard-library theorem records.

Current scope:

- proving is disabled
- standard-library theorem records are stored under `experience/`
- FAISS indexes `semantic_explanation`
- agents read `detail.md` and `reasoning.md`
- standard-library text generation uses `gpt-5-nano` by default
- local rule-based text generation is only a fallback when model generation fails

## Environment

The retrieval workflow expects:

- Coq 8.20 available from the configured environment
- `coq-py310` conda env for FAISS and index rebuilds

Quick check:

```bash
CONDA_NO_PLUGINS=true conda run -n coq-py310 coqc -where
```

Install FAISS if needed:

```bash
conda install -n coq-py310 numpy faiss-cpu -c conda-forge
```

## Main entry points

Build records for `Coq.Lists.List`:

```bash
python3 src/coqstoq_tools.py build-stdlib-index --module-path Coq.Lists.List
```

Query by natural language:

```bash
python3 src/coqstoq_tools.py query-stdlib --description "append with empty list on the right" -k 5
```

Rebuild stdlib indexes from existing records:

```bash
python3 src/coqstoq_tools.py build-stdlib-from-existing
```

Convenience query wrapper:

```bash
python3 scripts/query_experience.py --domain stdlib --description "append with empty list on the right" -k 5
python3 scripts/query_experience.py --domain stdlib --sql "select record_id, item_kind from records limit 10"
```

Detailed retrieval guide:

- `docs/stdlib-retrieval.md`

Legacy proving entrypoint:

```bash
python3 src/proof_task_client.py
```

This now returns a clear error because proving is disabled.

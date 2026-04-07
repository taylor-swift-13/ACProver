# ACProver

ACProver is now a local retrieval workspace for Coq standard-library theorem records.

Current scope:

- proving is disabled
- standard-library theorem records are stored under `experience/`
- FAISS indexes `semantic_explanation`
- agents read `detail.md` and `reasoning.md`

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
python3 src/coqstoq_tools.py query-experience --description "append with empty list on the right" -k 5
```

Legacy proving entrypoint:

```bash
python3 src/proof_task_client.py
```

This now returns a clear error because proving is disabled.

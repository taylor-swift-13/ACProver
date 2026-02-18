# ACProver

ACProver is a Coq proof verification workspace built on top of the CoqStoq dataset.

Repository layout:

- `src/`: verification tools, build scripts, and LLM proof client
- `CoqStoq/`: dataset projects, theorem metadata, and source repositories

## 1) Environment Setup

This project uses **conda + opam**.

### Create conda env

```bash
CONDA_NO_PLUGINS=true conda create --solver classic -y -n coq-py310 python=3.10
conda activate coq-py310
```

### Create opam switch

```bash
opam switch create coqswitch ocaml-base-compiler.4.14.2
opam repo add coq-released https://coq.inria.fr/opam/released
eval "$(opam env --switch=coqswitch)"
```

### Install Coq toolchain

```bash
opam install --switch=coqswitch -y \
  coq.8.18.0 \
  coq-mathcomp-ssreflect.2.2.0 \
  coq-mathcomp-algebra.2.2.0 \
  coq-mathcomp-field.2.2.0 \
  coq-mathcomp-solvable.2.2.0 \
  coq-mathcomp-finmap.2.1.0 \
  coq-mathcomp-real-closed.2.0.1 \
  coq-mathcomp-multinomials.2.2.0 \
  coq-fourcolor.1.3.1 \
  coq-bignums.9.0.0+coq8.18 \
  coq-paramcoq.1.1.3+coq8.18
```

### Activate project env vars

```bash
cd /home/yangfp/ACProver/src
source activate_coq_env.sh
coqc --version
```

Expected: `The Coq Proof Assistant, version 8.18.0`.

## 2) Build All CoqStoq Projects

```bash
cd src
conda activate coq-py310
source activate_coq_env.sh
export N_JOBS=4
bash build_coqstoq_complete.sh
```

Build status check:

```bash
python3 check_build_status.py
```

## 3) Verify a Proof

### CLI

```bash
cd src
python3 verify_proof.py test:39 "Proof. intros a b1 b2 l H; inversion H; auto. Qed."
```

### Python API

```python
from verify import verify_proof

proof = """Proof.
intros a b1 b2 l H; inversion H; auto.
Qed."""

result = verify_proof("test:39", proof)
print(result["state"], result["proof_status"])
```

Return states:

- `proven`
- `failed`
- `in_progress`
- `error`

## 4) Print / Check / Locate / Search

Use `coq_print.py` for external queries.

```bash
cd src
python3 coq_print.py "Print nat."
python3 coq_print.py "Check plus."
python3 coq_print.py "Print unique_key_in." --repo ../CoqStoq/test-repos/huffman --compile-args "-R theories Huffman"
```

Note: proof-state `Show.` is handled internally by `verify_proof` during `in_progress` verification and is not exposed as a separate script.

## 5) Proof Task Client

`proof_task_client.py` runs an LLM tool loop for a fixed theorem task.

- Fixed per task: `theorem_id`, `repo`, `compile_args`
- Model-provided parameters only:
  - `verify_proof`: `proof`
  - `print`: `definition`

Run:

```bash
cd /home/yangfp/ACProver/src
python3 proof_task_client.py --theorem-id test:39 --max-steps 20
```

System prompt preview:

```bash
python3 proof_task_client.py --theorem-id test:39 --dump-system-prompt
```

## 6) Common Issues

### `.vo` version mismatch (`bad version number`)

Rebuild all projects with the current Coq switch:

```bash
cd src
bash build_coqstoq_complete.sh
```

### Wrong opam switch / OCaml mismatch

```bash
eval "$(opam env --switch=coqswitch)"
coqc --version
```

### Missing `coqpyt` import

Always source project env vars before running tools:

```bash
cd src
source activate_coq_env.sh
```

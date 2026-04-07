# ACProver Tools (`src/`)

Use the repository root guide first:

- `../README.md`

Current primary runtime files:

- `proof_task_client.py`: thin Codex launcher
- `theorem_task.py`: theorem lookup and target-proof masking
- `codex_runner.py`: shadow-workspace setup and `codex exec` runner
- `logging_utils.py`: log writing and readable summaries

Local Coq helpers:

- `verify.py`: proof verification library
- `verify_proof.py`: CLI wrapper around `verify.py`

Maintenance scripts:

- `build_coqstoq_complete.sh`: clean + full rebuild
- `check_build_status.py`: build output report

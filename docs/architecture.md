# Architecture

RocSql now uses a thin launcher architecture:

- `config/acprover.local.json`
  - repository-local machine/runtime defaults
  - records the preferred opam switch
- `src/theorem_task.py`
  - resolves a `theorem_id`
  - extracts only the target theorem declaration
  - hides the target theorem's original proof
- `src/codex_runner.py`
  - launches `codex exec`
  - prepares the local Coq shell environment
  - creates an isolated shadow workspace where the target source file path itself becomes the writable masked temp file
  - saves the raw event stream and final result
- `src/logging_utils.py`
  - writes `task.json`, `prompt.txt`, `events.jsonl`, `codex_stderr.log`, `result.json`, and `readable`

The local code no longer runs a custom proof loop, custom tool registry, or lemma orchestrator. Codex is the proof engine. The local Coq shell environment and existing helper scripts are the proof surface, but Codex works inside an isolated shadow workspace rather than the original repository tree. For difficult theorems, Codex is expected to introduce and prove auxiliary lemmas sequentially in the same session.

## Hidden-proof rule

Only the current target theorem's original proof is hidden. Other theorems and lemmas in the repository remain unchanged.

## Logging model

Every run writes a log directory under `log/` unless `--readable-log-file` selects another location. The raw Codex JSON event stream is preserved in `events.jsonl` to keep all visible reasoning, tool activity, and proof attempts available for debugging.

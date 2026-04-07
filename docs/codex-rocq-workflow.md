# Codex + Local Coq Workflow

Use this workflow when proving a theorem in this repository.

1. Read the task metadata and the target theorem declaration.
2. Treat the original proof of that theorem as unavailable.
3. Assume the launcher has already prepared the local Coq environment.
4. Treat the current target source file path as the only writable temp file for the run; other repository files are read-only context.
5. Use local repository inspection plus shell commands such as `rg`, `sed`, `find`, and `ls` to locate surrounding definitions, lemmas, notation, and module structure.
6. Use `$ACPROVER_COQSTOQ_TOOLS step-tactic`, `$ACPROVER_COQSTOQ_TOOLS verify-proof`, and `$ACPROVER_COQSTOQ_TOOLS bm25-search` for Coq interaction.
7. Iterate with short visible proof attempts instead of spending time on environment debugging.
8. As soon as a proof candidate is validated, return the final JSON immediately. Do not continue with cleanup-only or proof-irrelevant work.
9. The launcher may stop the run automatically as soon as the target file compiles with the target theorem completed by `Qed.` or `Defined.`.

## Guidance

- Prefer tool-driven discovery over guessing.
- Prefer a small amount of targeted reading followed by a concrete proof attempt.
- For difficult theorems, introduce auxiliary lemmas early and prove them in the same session before returning to the main theorem.
- Do not spend long stretches reading neighboring large proofs. Only inspect a similar proof when you already know the exact pattern or lemma you want to reuse.
- If several `rg`/`sed` commands have happened and no `$ACPROVER_COQSTOQ_TOOLS step-tactic` or `$ACPROVER_COQSTOQ_TOOLS verify-proof` attempt has been made yet, switch immediately to a proof attempt.
- Do not spend the session debugging installation, opam bootstrap, or launcher implementation.
- Keep visible reasoning concise but explicit enough to explain proof attempts and failures.
- If a proof attempt fails, record what was tried and use the tool output to refine the next attempt.
- Do not search the source file for the hidden original proof.
- Use shell for reading and editing only; route Coq interaction through `$ACPROVER_COQSTOQ_TOOLS`.

## Output shape

The launcher asks Codex for a JSON object with:

- `final_status`
- `final_proof`
- `summary`

`final_proof` should be a complete compilable proof script when `final_status` is `proven`.

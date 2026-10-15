# Thor (SM110) self-checking probes

Reusable native CUDA probe sources staged for the **root orchestrator** to
compile and run on real Thor hardware under the shared GPU lock
(`flock /home/cilab/zbc/hmz-thor-validation.lock`). Authors do not run these;
they prepare the source, the exact commands, and the fixed expected outputs.

- `thor_probe.cu` — a positive, self-checking SIMT + TMA copy/reduce probe. It
  validates finite and **exact** FP32 element copies and per-block sums with no
  tolerance, printing `{"status":"passed",...}` and exiting `0` on success, or
  `{"status":"error",...}` and exiting `1` on any mismatch. Modes: `check`
  (correctness), `timing` (diagnostic event timing — not a speedup claim), and
  `profile` (captures exactly two named actions for NCU).
- `tcgen05_target_gate.cu` — a compile-only `tcgen05.fence` target gate. It is
  **not** a feature correctness test: it probes assembler target acceptance only.

Both files are byte-identical copies of the campaign harness already validated on
Thor; their sha256 digests are recorded in `probe-manifest.yaml` and match the
evidence registry (`../data/thor-evidence.yaml`) and the `probe-build-02` /
`validation-check` / `validation-profile` receipts.

`probe-manifest.yaml` lists every probe with its exact build/run/profile command,
expected exit code, and expected stdout. The positive probes must pass; the
negative probe (`tcgen05.fence` under the base `sm_110` target) must be rejected,
and a positive control compiling the *same* source under `sm_110a` / `sm_110f`
proves the rejection is target gating, not an incidental syntax error.

Scope note: only the SIMT and genuine TMA copy/reduce paths are runtime evidence.
tcgen05 MMA, TMEM, NVFP4, CLC and 2-CTA remain documented / compile-only /
pending in the evidence registry.

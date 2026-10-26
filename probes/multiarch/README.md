# Bounded multi-architecture validation probes

These original probes belong to this repository's license. They test small,
explicit contracts and are not optimized operators or benchmark suites.
The exact source SHA256, native build command, result and observed device
properties are in [the SM87 receipt](../../evidence/multiarch-2026-10-27/sm87.json),
[SM89 receipt](../../evidence/multiarch-2026-10-27/sm89.json) and
[SM110 receipt](../../evidence/multiarch-2026-10-27/sm110.json).

From this directory on the selected GPU, use its actual compiler and baseline
target. Example for SM87 (replace only with the verified target/compiler):

```bash
nvcc -O2 -lineinfo -arch=sm_87 validation.cu -o validation
./validation --info
./validation
nvcc -O2 -lineinfo -arch=sm_87 instruction-probe.cu -o instruction-probe
./instruction-probe
```

`validation.cu` checks a 262144-element copy and block sum plus an FP32
64×64×64 SIMT matrix multiply. `instruction-probe.cu` checks a 65536-element
4-byte ordinary async copy and BF16-input/FP32-accumulator WMMA at 16×16×16.
Matrix acceptance uses an absolute tolerance of 1e-5; the recorded inputs
produced zero error. All probes run device ordinal 0; select the intended
visible device before running in a multi-GPU environment.

These tests do not establish tcgen05, TMEM, TMA, WGMMA, two-CTA correctness or
performance gains. No timing claims are recorded. No power or clock settings
were changed during the recorded runs. Profiler collection/read evidence is
separate from ordinary execution; raw `.ncu-rep` files are external artifacts
identified by hashes, not included in this lightweight skill.

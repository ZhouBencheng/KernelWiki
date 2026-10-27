# Observed multi-architecture validation — 2026-10-27

These bounded local experiments establish compilation, selected correctness
checks, and profiler collection/readability on one configuration per target.
They record no optimization speedup or ordinary timing result. Device resources
below are observed SKU/configuration values, not architecture-wide constants.

The original CUDA probes and three machine-form experiment receipts are
historical material in Git snapshot `dd4c2b2` at `probes/multiarch/` and
`evidence/multiarch-2026-10-27/`. They are omitted from the installed skill;
this Markdown summary preserves the useful results. Raw `.ncu-rep` reports were
external artifacts and were never bundled. Their digests are retained below.

## Observed devices and toolchains

| Observation | Orin / SM87 | GeForce RTX 4090 / SM89 | NVIDIA Thor / SM110 |
| --- | --- | --- | --- |
| Runtime CC; native compiled target | 8.7; `sm_87` | 8.9; `sm_89` | 11.0; `sm_110` |
| SM count; integrated GPU | 14; yes | 128; no | 20; yes |
| Global memory, bytes | 32,188,633,088 | 25,386,352,640 | 131,920,637,952 |
| Shared memory per block / opt-in / per SM, bytes | 49,152 / 166,912 / 167,936 | 49,152 / 101,376 / 102,400 | 49,152 / 232,448 / 233,472 |
| L2, bytes | 4,194,304 | 75,497,472 | 33,554,432 |
| Max blocks per SM | 16 | 24 | 24 |
| nvcc | 12.6.85 | 12.4.131 | 13.2.86 |
| nvcc path | `/usr/local/cuda/bin/nvcc` | `/usr/local/cuda-12.4/bin/nvcc` | `/usr/local/cuda/bin/nvcc` |
| CUDA runtime / driver API integer | 12060 / 12060 | 12040 / 12040 | 13020 / 13020 |
| Driver | 540.4.0 | 550.163.01 | 595.78 |
| OS; Jetson L4T | Ubuntu 22.04.5 LTS; R36.4.7 | Ubuntu 22.04.3 LTS; not applicable | Ubuntu 24.04.4 LTS; R39.2.1 |
| Observed power mode; counter privilege | MAXN; `sudo -n` | not recorded; unprivileged | MAXN; `sudo -n` |
| NCU collector | 2025.2.1 | 2024.1.1 | 2026.1.1 |
| NCU Python report reader | 2026.1.1 on a separate aarch64 validation host | 2024.1.1 on the device host | 2026.1.1 on the device host |

All three reported warp size 32, 65,536 registers per SM, at most 1,536
threads per SM and 1,024 threads per block. No power or clock settings were
changed during the recorded runs; NCU used `--clock-control none`.

On the observed Orin installation, the NCU 2025.2.1 `_ncu_report.so` was
x86-64 despite the aarch64 host. The separate aarch64 2026.1.1 reader successfully
read its reports. This describes that installation, not every 2025.2.1 package.

## Probe contracts and results

Both original sources compiled with the listed nvcc using `-O2 -lineinfo` and
the listed native target. Each executable passed under a 15-second timeout on
visible device ordinal 0. No upstream implementation was copied into the probes.

| Case | Inputs, shape and numerical contract | Result on all three devices |
| --- | --- | --- |
| Vector copy | 262,144 FP32 elements; exact equality | passed |
| Block reduction | 262,144 FP32 elements; 256 threads/block; exact comparison with each host block sum | passed |
| SIMT matrix multiply | FP32 inputs, accumulation and output; M=N=K=64; row-major matrices; finite output and absolute error ≤1e-5 | passed; maximum observed absolute error 0 |
| Ordinary `cp.async` | 65,536 FP32 elements; 4-byte global-to-shared copies; 256 threads/block; exact equality | passed |
| One-warp WMMA | BF16 inputs, FP32 accumulation/output; M=N=K=16; A row-major, B column-major, C row-major; finite output and absolute error ≤1e-5 | passed; maximum observed absolute error 0 |

Copy/reduction inputs were small deterministic exactly representable values;
SIMT inputs used quarter-step values, and WMMA reference results used the
converted BF16 values. Zero error is a result for those inputs only, not a
universal precision guarantee.

Historical source digests (SHA-256):

| Source at Git snapshot `dd4c2b2` | SHA-256 |
| --- | --- |
| `probes/multiarch/validation.cu` | `41a915a82c2c293de188093a15f5f6c50b548e8aa655ebf7b0da126b23a6d330` |
| `probes/multiarch/instruction-probe.cu` | `1b29f305ce390d3208ece80b66b3c71664a1cce1f3c51d3560278a552e1f301e` |

## Profiler collection, reader and helper observations

Basic collection used NCU's `basic` set with imported source and captured the
three generic kernels. Detail collection added `SourceCounters` and `PmSampling`,
selected `regex:vector_copy`, and captured one action. Both collection and Python
report reading passed for every device. Scalar extraction succeeded with each
listed collector/reader pair.

| Target | Basic actions; metric counts per action | Detail actions; metric count | Detail nonempty PC-sampling metrics | Detail PM series instances |
| --- | --- | --- | --- | --- |
| SM87 | 3; 501 each | 1; 618 | 37 | tensor-pipe activity series: 414 |
| SM89 | 3; 491 each | 1; 606 | 37 | tensor-pipe activity series: 22 |
| SM110 | 3; 561 each | 1; 708 | 53 | 15 series: GPC cycles 290; three L1TEX series 201 each; eleven LTS/SM/SMSP/TPC series 64 each |

Basic reports had no PM-series instances and no nonempty PC-sampling metrics.
The observed Python reader API exposed `source_info` for all three targets;
`rule_results` and `timed_warp_samples` were present in 2026.1.1 and absent in
2024.1.1. These are API-availability and readability observations, not proof
that every helper produces data or that every metric exists on another version.
No independent general helper-suite execution was recorded in these receipts.

| External report | SHA-256 |
| --- | --- |
| SM87 basic | `40044ccb7506293aaf204803c5bd45f0e234ff49609c9815db3c7c17ec0c2c1c` |
| SM87 detail | `77c45e4f0e14770f56af79dd8522bf4f5a8ff9c8d6ca436d8826c1b9291b33ec` |
| SM89 basic | `0748a1e7fe8f10f13f6a742eeb0ce05b55100192689646a7af2f63657a21d20d` |
| SM89 detail | `4c8fd3f469a9f4f8126f4018a74f9cbb8721c7e819725752556843550a66d650` |
| SM110 basic | `5bf025afd00f0980b051b423b05ccdd284e078e1fe22b98354fcef4e32473cc1` |
| SM110 detail | `7cdee8f026314cda58964723832c5f1471a9c6c95a8d851d4b1e394c1fa24986` |

These experiments do not validate TMA, WGMMA, tcgen05, TMEM, two-CTA
cooperative kernels, every dtype or shape, or an optimization speedup. In
particular, the Thor basic path does not establish a specialized Blackwell
implementation. No new SM100/B200 device experiment was recorded here. Keep
source confidence separate from compile, correctness, collection and benefit
as described in [the target routing contract](../references/target-routing.md).

---
id: hw-orin
title: "Orin (Jetson AGX Orin, sm_87): device identity and feature scope"
type: hardware
architectures: [sm87]
targets: [sm_87]
evidence_status: executed
tags: [cuda-cpp, cp-async]
hardware_features: [cp-async]
confidence: source-reported
related: [hw-ampere-mma-async-copy, hw-tma, hw-tcgen05-mma, hw-tmem, hw-nvfp4, hw-clc]
sources: [doc-nvidia-jetson-orin, doc-nvidia-ampere-sm87-features]
aliases: [Orin, "Jetson Orin", "AGX Orin", "Jetson AGX Orin", sm_87, SM87]
---

# Orin (Jetson AGX Orin, sm_87)

Use `--architecture Orin` or `--architecture sm87` to retrieve Orin device
references. `Orin`, `Jetson Orin`, `AGX Orin`, `sm87` and `sm_87` all resolve to
this page. The identity is the baseline Ampere target `sm_87`; it is **not** a
generic `Ampere` alias, and it is distinct from desktop Ampere `sm_80`/`sm_86`.
There is no `sm_87a` or `sm_87f` target. The existing `Blackwell`/`B200`
(`sm100`) and `Thor` (`sm110`) aliases are unchanged.

For the supported-instruction detail (`cp.async`, warp BF16 MMA) and the
unsupported Hopper/Blackwell paths, see
[Ampere async-copy and warp MMA on sm_87](ampere-mma-async-copy.md).

## Device identity (measured, not transferred)

A root probe on 2026-10-13 read these attributes from the CUDA device
(each with CUDA status 0). They are device **limits and identity**, not live
telemetry and not a kernel's achieved occupancy:

| Attribute | Value |
|---|---|
| Device name / compute capability | `Orin`, CC **8.7** (`sm_87`) |
| SM count | **14** |
| Threads per SM / warp size | 1536 / 32 → ceiling **48 resident warps/SM** |
| Max blocks per SM | 16 |
| Registers per SM / per block | 65536 / 65536 |
| L2 cache | 4 MiB (4194304 B) |
| Shared memory per SM | 167936 B |
| Shared memory per block (opt-in / default) | 166912 B / 49152 B |
| Global memory (shared with CPU) | ~32 GB (32188628992 B) |
| Memory bus width | 256-bit |
| Integrated / managed memory | yes / yes |
| Device-attribute clock | 930 MHz (attribute, not telemetry) |
| nvcc / runtime / driver | 12.6.85 / 12060 / 12060 |

The table above is the **captured observation of this one board/configuration**,
not a constant for every `sm_87` device. Another Orin module — even one reporting
the same CC 8.7 — may expose a different SM count, cache or shared-memory
configuration, so any other deployment must read fresh device attributes rather
than reuse these numbers. CC and the generic device name alone do not identify a
resource configuration. CUDA exposes SM count, cache/memory capacities and
shared-memory limits as device properties
([`cudaDeviceProp`](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html));
query the selected runtime device and the kernel's occupancy, not a page
constant.

Orin is an **integrated** GPU: it shares physical memory with the CPU
(`integrated=1`, `managed=1`), so there is no discrete VRAM pool and discrete-GPU
bandwidth/residency constants do not apply. Do not carry B200 or Thor SM counts,
resident-warp counts, shared-memory capacities or bandwidth onto this board. For
this measured board specifically, size persistent-grid and occupancy constants to
*these* 14 SMs and 48-warp ceiling; for any other board, re-query as above. A
streaming-memory roofline needs the matching module SKU plus measured traffic;
the generic device name does not identify a unique SKU, and the clock value above
is a static attribute.

## Execution evidence (compile, exact correctness, and bounded event timing)

The same probe built and ran two SIMT copy/reduce kernels natively for
`sm_87` (`nvcc 12.6.85`, binary/PTX version 87). Three distinct runs are kept
separate:

- **Exact-correctness run** (`mode=check`): each kernel checked **1048576**
  copied elements and **4096** block sums for finite exact FP32 equality. This
  invocation records `event_repeats_per_trial = 0` (no timing in this run).
  - `orin_simt_copy_reduce` — 16 registers/thread.
  - `orin_cp_async_copy_reduce` — 18 registers/thread, using the ordinary
    `cp.async` global→shared path (see the feature page).
- **Diagnostic event-timing run** (`mode=timing`): a *separate* invocation with
  five warmup launches then five trials of `event_repeats_per_trial = 20`
  CUDA-event-timed launches each. Per-launch trial means (microseconds):
  `orin_simt_copy_reduce` 4229.7 / 3178.6 / 3147.8 / 3148.3 / 3156.4
  (median ≈ 3156); `orin_cp_async_copy_reduce` 3466.3 / 3436.3 / 3427.6 /
  3450.6 / 3430.7 (median ≈ 3436).
- **NCU replay** is a third, separate collection and is not these event means.

These event means are a **bounded synthetic repeated-copy measurement** of this
probe on this board; the first SIMT trial was higher than the later trials, but
the cause was not established (the probe did not change clock or power settings).
They do **not** establish an optimized operator, a streaming-memory bandwidth
result, or any model/FFN or end-to-end speedup — no such claim is made. The warp BF16 MMA capability is recorded separately on the
feature page from its own receipts; it is **not** inferred from these copy
kernels.

## Feature scope

`cp.async` and Ampere warp-level MMA are available on `sm_87`; TMA, WGMMA,
tcgen05, TMEM, CLC and native NVFP4 are **not** (see
[the feature page](ampere-mma-async-copy.md) for the per-feature primary
citations). Pages describing those Hopper/Blackwell features
([TMA](tma.md), [tcgen05](tcgen05-mma.md), [TMEM](tmem.md), [NVFP4](nvfp4.md),
[CLC](clc.md)) are cross-architecture references for terminology only; their
instructions, capacities and performance numbers are not `sm_87` behavior.

## Same-precision M=1 workloads: measure, do not transfer

This is a **measurement plan** — candidate routes to profile — **not** an Orin
performance result. The bounded probes on this page and the
[feature page](ampere-mma-async-copy.md) (the `cp.async`/SIMT copy-reduce exact
correctness + event timing, and the single-warp `16×16×16` BF16 MMA) establish
capability and exact correctness only; none of them ranks an M=1 FFN or linear
layer, and no shape, threshold, candidate or speedup from another campaign
(Blackwell/Thor) transfers to `sm_87`.

**Fix the real workload first.** "M=1" alone does not set N/K, layout,
bias/activation, input/weight contents, output dtype, accumulation or math mode.
Read those from the actual model; do **not** borrow a shape from another page.

**BF16 M=1 FFN — operand dtype and accumulator are separate choices.** Preserve
the requested BF16 operands/output; the accumulator precision is an independent
decision. Candidate routes to *measure* against the real baseline:

- a **SIMT GEMV/reduction** keeping BF16 input/stored-weight/output; if the
  original numerical contract allows it, widen BF16→FP32 for the multiply-add and
  reduction and round only at the original output boundary. This is a candidate
  and does **not** assume the original implementation's reduction tree.
- the **Ampere BF16 warp-MMA** path with an FP32 accumulator as a *separate*
  candidate. The single-warp `M=16` `16×16×16` probe does **not** show that an
  M=1 tile is well utilized or faster, and an FP32-D probe does **not** prove a
  BF16-output FFN passes; verify real M=1 tile utilization, pad/copy/reduce cost
  and the output contract.

**Full FP32 M=1 linear layer — no precision substitution.** Keep FP32 input,
stored weights and output, and fix the accumulation precision and math mode
*separately*. Do **not** substitute BF16/FP16/FP4, and do **not** pass off a TF32
multiply with an FP32 store as full FP32. BF16 warp-MMA support does **not**
establish a strict-FP32 replacement or any speedup. Preserve the original
bias/activation, rounding boundary and the original accepted error threshold — do
**not** import an external campaign's `atol`/`rtol`; FP32 dtype alone does not
guarantee bitwise equivalence once the reduction order changes, so if "preserve
precision" means bitwise, state and check that explicitly.

**What to measure (either route) before ranking anything:**

1. read live resources / toolchain / power-frequency-temperature and other load —
   the static device-attribute clock above is **not** telemetry;
2. independent exact correctness on the real inputs against the frozen original
   output and its accepted threshold; a different reduction or fusion order can
   change results;
3. ordinary CUDA-event timing of the original and each candidate (warmup, several
   trials, explicit cache/exec mode, including any auxiliary/split-K kernels),
   kept **separate** from NCU collection; FFN and end-to-end benefit are further
   gates;
4. a separate NCU collection to *explain*: per-action waves, registers/SMEM,
   active/eligible warps, issue, memory throughput/cache, tensor activity,
   source/PC stalls — each checked for availability; a missing / `[N/A]` /
   unsampled metric is **not** zero.

Do **not** rank an M=1 route from this page's copy/reduce event means or the BF16
capability probe. The same same-precision *principles* are also readable on the
Thor page (`hw-thor`, "Same-precision M=1 workloads"), but its shapes, thresholds,
candidates and timing are Thor-campaign specifics and are not Orin evidence.

## Evidence roots (external to the skill tree)

The receipts cited here live **outside** the delivered skill, under the Orin
controller root `evidence/device-probe-v1/` (identity, SASS, cp.async/SIMT
correctness, NCU section enumeration, the two-action detached report) and
`evidence/bf16-mma-v2/` (the BF16 warp-MMA probe). A delivered copy of this
skill cannot resolve those paths on its own; when a receipt is unavailable at
analysis time, treat that evidence as **unavailable** and say so rather than
substituting prose for the receipt. The in-corpus evidence registry that will
record these digests is tracked separately from this page.

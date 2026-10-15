---
id: hw-ampere-mma-async-copy
title: "Ampere async-copy and warp MMA on sm_87 (cp.async supported; TMA/WGMMA/tcgen05/TMEM/CLC/NVFP4 unsupported)"
type: hardware
architectures: [sm87]
targets: [sm_87]
evidence_status: executed
tags: [cuda-cpp, ptx, cp-async, mma]
hardware_features: [cp-async, mma]
confidence: source-reported
related: [hw-orin, hw-tma, hw-tcgen05-mma, hw-tmem, hw-nvfp4, hw-clc]
sources: [doc-nvidia-ampere-sm87-features, doc-nvidia-jetson-orin]
aliases: [cp.async, LDGSTS, "warp MMA", HMMA, wmma, "wmma.mma"]
---

# Ampere async-copy and warp MMA on sm_87

This page scopes the instruction families that **are** and **are not** available
on the Jetson AGX Orin baseline target `sm_87` (see
[the Orin device page](orin.md) for identity and resource limits). Each claim is
tied either to a primary NVIDIA document or to a bounded root probe receipt, and
the two kinds of evidence are kept distinct.

## cp.async is supported — and is NOT TMA

`cp.async` (ordinary asynchronous global→shared copy, SASS `LDGSTS`) was
introduced at `sm_80` and is available on `sm_87`. The PTX
[`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
instruction copies a small, fixed element size per thread without staging through
registers. It is a **different instruction** from the Hopper
[`cp.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy)
family, which has two distinct forms:

- **`cp.async.bulk`** — ordinary bulk copy, parameterized by source/destination
  addresses and a byte count.
- **`cp.async.bulk.tensor`** — the Tensor Memory Accelerator (TMA) tensor form,
  parameterized by a tensor-map descriptor and tile coordinates. The descriptor
  is *typically* built on the host (`cuTensorMapEncodeTiled`), but device-side
  encoding also exists, so a host-built descriptor is not a universal requirement.

Both bulk forms differ from Ampere `cp.async`, and **neither is available on
`sm_87`**. A kernel using `cp.async` is **not** using TMA; see
[the TMA page](tma.md) for the tensor bulk-copy engine, which is unavailable here.

**Receipt.** A root probe built and ran `orin_cp_async_copy_reduce` natively for
`sm_87` (`nvcc 12.6.85`, 18 registers/thread) and checked **1048576** copied
elements plus **4096** block sums for finite exact FP32 equality. A *separate*
diagnostic event-timing run (recorded on [the Orin device page](orin.md)) exists
for this kernel; it is a bounded synthetic repeated-copy measurement only and
establishes no operator, bandwidth or speedup claim for the ordinary async-copy
path.

## BF16 warp MMA: bounded capability, from its own receipt only

Ampere tensor cores provide warp-level matrix-multiply-accumulate. A separate
root probe (`evidence/bf16-mma-v2`), with an independent source-bound review,
established exactly one bounded operation on the measured Orin:

- **What ran:** a single warp `16×16×16` matrix-multiply-accumulate with **BF16**
  A/B operands and **FP32** C/accumulator/D output; grid `[1,1,1]`, block
  `[32,1,1]`; built native `arch=compute_87,code=sm_87` with `nvcc 12.6`,
  30 registers, zero spills.
- **Correctness:** **256** outputs checked at tolerance 0 (exact equality).
- **Static SASS:** exactly two `HMMA.16816.F32.BF16` instructions in the named
  `sm_87` function, bound to the binary bytes by cuobjdump.
- **Profiling:** the NCU opcode correlation records `HMMA = 2.0` (and, in the
  selective-modifier view, `HMMA.16816 = 2.0`) — two views of the same static
  instruction observation, not additive counts. The BF16/FP32 qualifiers come
  from the bound SASS, not from the metric. The run has only three sampled PC
  instances and **no sampled HMMA PC**, so it establishes no bottleneck, stall
  proportion, latency or instruction coverage.

**Claim boundary.** This promotes only the exercised BF16 warp-MMA path with FP32
accumulation/output at the `16×16×16` shape. It does **not** establish M=1
efficiency, timing, throughput, speedup, arbitrary BF16 reductions, model
precision/weights, FFN replacement, or end-to-end benefit — those remain separate
acceptance work. The capability is **not** inferred from the `cp.async` copy
probe above; the copy probe says nothing about matrix execution.

## Unsupported on sm_87 (Hopper/Blackwell features)

The following are introduced on later architectures and are **not available or
executed** on `sm_87`. The primary basis is that each feature's own NVIDIA
documentation scopes it to a later target (`sm_90`/`sm_90a` or `sm_100`); the
exact target string is per-instruction and is not a monotonic "and later"
promise.

| Feature | First available | Primary citation |
|---|---|---|
| **TMA** (tensor-map bulk copy, `cp.async.bulk.tensor`) | Hopper `sm_90` | [PTX ISA cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy); [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) |
| **WGMMA** (`wgmma.mma_async`, warpgroup async MMA) | Hopper, architecture-specific `sm_90a` | [PTX ISA warpgroup MMA — Target ISA Notes require `sm_90a` for `wgmma.mma_async`/`wgmma.fence`/`wgmma.commit_group`/`wgmma.wait_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions); [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) |
| **Thread-block clusters / CLC** (cluster launch control) | clusters `sm_90`; CLC Blackwell `sm_100` | clusters: [Hopper tuning guide — thread-block clusters](https://docs.nvidia.com/cuda/hopper-tuning-guide/) + [PTX ISA cluster dimension directives](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cluster-dimension-directives); CLC: [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) |
| **tcgen05** (5th-gen tensor-core MMA) | Blackwell `sm_100` | [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) |
| **TMEM** (tensor memory) | Blackwell `sm_100` | [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) |
| **Native NVFP4** (hardware FP4 E2M1 tensor-core path) | Blackwell `sm_100` | [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) |

No page in this corpus claims any of these features is available or executed on
`sm_87`, and matrix execution on `sm_87` is **not** inferred from the `cp.async`
copy probe. The cross-architecture pages — [TMA](tma.md), [tcgen05](tcgen05-mma.md),
[TMEM](tmem.md), [NVFP4](nvfp4.md), [CLC](clc.md) — remain terminology references
whose instructions and performance numbers do not transfer to Orin.

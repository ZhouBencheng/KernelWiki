---
id: doc-nvidia-ampere-sm87-features
title: "Ampere sm_87 feature set: cp.async and warp MMA present; TMA/WGMMA/tcgen05/TMEM/CLC/native-NVFP4 absent"
url: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
source_category: official-doc
architectures: [sm87]
targets: [sm_87]
tags: [cuda-cpp, ptx]
retrieved_at: 2026-10-13
---

# Ampere sm_87 feature set

The CUDA C++ Programming Guide "Compute Capabilities" section and the
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) record which
instruction families each compute capability provides. For the Ampere
generation (`sm_80`/`sm_86`/`sm_87`) the relevant distinctions for this corpus:

- **`cp.async` (ordinary asynchronous copy, a.k.a. LDGSTS)** — introduced at
  `sm_80` and available on `sm_87`. It copies global→shared memory without
  staging through registers. It is **not** a Hopper bulk copy: the PTX
  [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
  instruction is distinct from both Hopper bulk forms —
  [`cp.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy)
  (addresses + byte count) and `cp.async.bulk.tensor` (the TMA tensor form,
  driven by a tensor-map descriptor + tile coordinates; the descriptor is
  typically but not universally host-built). Neither Hopper bulk form is
  available on `sm_87`.
- **Warp-level MMA (`wmma` / `mma.sync`, SASS `HMMA`)** — Ampere tensor cores
  provide warp-level matrix-multiply-accumulate, including BF16 operands with
  FP32 accumulation. This is distinct from the Hopper warpgroup `wgmma` and the
  Blackwell `tcgen05` tensor-core paths.

Features **absent** on `sm_87` (introduced on later architectures). The gate is
the per-instruction PTX Target ISA Notes, an exact architecture/family suffix
target — not a numeric "and later" rule — so each item below names its own target:

- **TMA** (tensor-map bulk copy, `cp.async.bulk.tensor`) — Hopper `sm_90`
  ([PTX ISA bulk/bulk-tensor copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy)).
- **WGMMA** (`wgmma.mma_async`, asynchronous warpgroup MMA) — Hopper,
  architecture-specific target `sm_90a` (the PTX Target ISA Notes require
  `sm_90a` for `wgmma.mma_async` and its fence/commit/wait instructions):
  [PTX ISA warpgroup MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions).
- **Thread-block clusters** — `sm_90`
  ([PTX ISA cluster dimension directives](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cluster-dimension-directives));
  **cluster launch control (CLC)** — `sm_100`
  ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)).
- **tcgen05** (5th-generation tensor-core MMA) — available on the
  architecture/family-specific targets explicitly listed by the
  [PTX `tcgen05.mma` Target ISA Notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma); `sm_87` is not included.
- **TMEM** (tensor memory) — allocation is available on the
  architecture/family-specific targets explicitly listed by the
  [PTX `tcgen05.alloc` Target ISA Notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit); `sm_87` is not included.
- **Native NVFP4** (hardware FP4 E2M1 block-scaled tensor-core path; software
  FP4 storage/conversion/emulation is a separate matter) — `sm_100`, per the
  [PTX ISA block-scaled MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/).

The authoritative gate for each feature is its own PTX Target ISA Note cited
above; the [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/) and
[Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) tuning guides
give architecture-level overviews only and are not relied on for the
instruction-level target details. This per-instruction target gate — not a
general numeric Blackwell capability rule — is the primary basis for marking each
feature unsupported on Ampere `sm_87`. This retrieval does
not advance the corpus refresh cutoff.

---
id: hw-tcgen05-mma
title: "tcgen05.mma — Blackwell MMA Instruction"
type: hardware
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, mbarrier]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tmem, hw-2sm-cooperative, technique-warp-specialization]
sources: [pr-cutlass-2139, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-colfax-cutlass]
aliases: [UMMA, tcgen05, "tensor core gen 05"]
---

# tcgen05.mma -- Blackwell MMA Instruction

## Overview

`tcgen05.mma` is the Blackwell (SM100/SM100a) matrix-multiply-accumulate instruction that replaces Hopper's `wgmma.mma_async`. The name stands for **Tensor Core Generation 05**. NVIDIA also refers to the higher-level abstraction as **UMMA** (Unified Matrix Multiply-Accumulate) in the CUTLASS framework.

Key differences from `wgmma`:

| Property | wgmma (SM90) | tcgen05.mma (SM100) |
|---|---|---|
| Issuing scope | Warpgroup (4 warps, 128 threads) | Single thread |
| Accumulator storage | Registers (high pressure) | Tensor Memory (TMEM, 256KB/SM) |
| Operand A source | Registers or SMEM | Shared memory only |
| Operand B source | Shared memory | Shared memory only |
| Matrix load | ldmatrix to registers | Direct from SMEM (no ldmatrix) |
| Synchronization | Warpgroup-scoped barriers | Fully async, fence-based |

## Instruction Variants

tcgen05.mma has 7 variants organized by precision and scaling mode:

| Variant | A Type | B Type | Accumulator | Scale | MMA Shape (1SM) | Notes |
|---|---|---|---|---|---|---|
| `tcgen05.mma.kind::f16` | FP16/BF16 | FP16/BF16 | FP32 | None | m128n256k16 | Standard half-precision |
| `tcgen05.mma.kind::tf32` | TF32 | TF32 | FP32 | None | m128n256k8 | Single-precision approximation |
| `tcgen05.mma.kind::f8f6f4` | FP8/FP6/FP4 | FP8/FP6/FP4 | FP32 | None (unscaled) | m128n256k32 | Narrow precision, **unscaled** — block scaling is the separate `.kind::mxf8f6f4.block_scale` variant (see below) |
| `tcgen05.mma.kind::i8` | INT8 | INT8 | INT32 | None | m128n256k32 | Integer quantized inference |
| `tcgen05.mma.kind::mxf8` | MXFP8 | MXFP8 | FP32 | MX (E8M0) | m128n256k32 | Microscaling FP8 |
| `tcgen05.mma.kind::mxf4` | MXFP4 | MXFP4 | FP32 | MX (E8M0) | m128n256k64 | Microscaling FP4 |
| `tcgen05.mma.kind::mxf4nvf4` | NVFP4 | MXFP4 | FP32 | Mixed | m128n256k64 | Mixed NVFP4/MXFP4 |

## MMA Shapes: 1-SM vs 2-SM

### 1-SM (Single CTA) Shapes

In single-SM mode, a single CTA owns the full MMA operation:

- **BF16/FP16**: M=128, N=256, K=16
- **TF32**: M=128, N=256, K=8
- **FP8/FP6/FP4**: M=128, N=256, K=32
- **MXFP4/NVFP4**: M=128, N=256, K=64

### 2-SM (Cooperative) Shapes

In two-SM cooperative mode, two CTAs share a single MMA across paired SMs:

- **BF16/FP16**: M=256, N=256, K=16
- **TF32**: M=256, N=256, K=8
- **FP8/FP6/FP4**: M=256, N=256, K=32
- **MXFP4/NVFP4**: M=256, N=256, K=64

The M dimension doubles because each SM contributes 128 rows from its own TMEM partition.

## Single-Thread Issuance Model

Unlike `wgmma` which required coordinated issuance from a warpgroup (4 warps, 128 threads), `tcgen05.mma` is issued by a **single thread**. This is a fundamental architectural simplification:

1. **No warpgroup synchronization overhead** -- one elected thread (typically lane 0 of warp 0) issues the MMA.
2. **Fully asynchronous** -- the instruction returns immediately; the hardware pipeline executes the MMA in the background.
3. **Commit/mbarrier completion, fence-based ordering** -- completion is signalled by committing the MMA to an mbarrier and waiting on it; a fence (`tcgen05.fence::{before,after}_thread_sync`) only *orders* tcgen05 ops around a thread sync and is **not** itself a completion wait (see "Completion and Fences" below).

```cuda
// Single-thread MMA issuance pattern
__device__ void issue_mma(uint32_t tmem_addr, uint64_t smem_desc_a, uint64_t smem_desc_b) {
    // Only one thread issues the MMA
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 0;"
            :
            : "r"(tmem_addr),
              "l"(smem_desc_a),
              "l"(smem_desc_b),
              "r"(0)  // scale descriptor (unused for f16)
        );
    }
}
```

## PTX Examples

### Basic BF16 MMA (1-SM)

```ptx
// Issue a 128x256x16 BF16 MMA
// Operand A: shared memory descriptor
// Operand B: shared memory descriptor
// Accumulator: TMEM address
tcgen05.mma.cta_group::1.kind::f16 [tmem_addr], desc_a, desc_b, idesc, 0;
```

### 2-SM Cooperative BF16 MMA

```ptx
// Issue a 256x256x16 BF16 MMA across two paired CTAs
// cta_group::2 indicates cooperative mode
tcgen05.mma.cta_group::2.kind::f16 [tmem_addr], desc_a, desc_b, idesc, 0;
```

### Narrow precision: unscaled vs. block-scaled

The pinned PTX ISA separates **unscaled** `.kind::f8f6f4` from the **block-scaled**
`.kind::mxf8f6f4.block_scale` variant; they are different instructions. `f8f6f4`
takes the usual instruction descriptor (no scale operand). Block scaling is a
distinct kind that adds scale-matrix operands:

The operand forms below are **conceptual** (illustrating operand shape, not
standalone-compilable or Thor-runtime-tested); confirm the exact operands and
qualifier applicability against the pinned
[PTX 9.2 / CUDA 13.2 §9.7.16.10.9.1](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html)
before use.

```ptx
// Unscaled narrow precision: no scale operand (enable-input-d predicate last).
tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_addr], desc_a, desc_b, idesc, enable_input_d;

// Block-scaled narrow precision: a DIFFERENT kind. Per the spec it takes
// BRACKETED scale-memory address operands (not bare registers) and the
// enable-input-d predicate — e.g.:
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
    [tmem_addr], desc_a, desc_b, idesc, [scale_a_addr], [scale_b_addr], enable_input_d;
```

Target-qualifier restrictions (pinned §9.7.16.10.9.1):

- `.scale_vec::1X/2X/4X` is documented for **sm100a** only.
- `.block16` / `.block32` require an **f-target** (`sm100f` or `sm110f`).
- `scale-input-d` is documented for sm100a and the sm100 family — not blanket Thor.
- The spec's family-support note (pinned PTX ISA §9.7.16.10.9.1) is
  **target-variant-specific**. The family-specific (`f`) target list **excludes**
  `.kind::i8`; the architecture-specific (`a`) target list **includes** it.
  `.kind::i8` is integer MMA, not a block-scaled floating-point kind. Read the
  exact target and qualifier restrictions for the variant you compile for, rather
  than assuming a single Thor spelling.

Do not infer blanket Thor (`sm110`/`sm110f`) support from the general instruction
target list; a compile-target family is not a device identity.

### CUDA Inline PTX for BF16 MMA with Accumulation

```cuda
__device__ void mma_bf16_128x256x16(
    uint32_t tmem_addr,
    uint64_t desc_a,
    uint64_t desc_b
) {
    if (threadIdx.x == 0) {
        // First MMA: zero-initialize accumulator
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 0;"
            :
            : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(0)
        );
    }
}

__device__ void mma_bf16_accumulate(
    uint32_t tmem_addr,
    uint64_t desc_a,
    uint64_t desc_b
) {
    if (threadIdx.x == 0) {
        // Subsequent MMAs: accumulate into existing TMEM
        // The enable_accumulate flag controls whether to add to or overwrite TMEM
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 1;"  // last arg 1 = accumulate
            :
            : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(0)
        );
    }
}
```

## Completion and Fences

TMEM is not implicitly synchronized with MMA, and a fence alone is **not** a
completion wait. Two **separate** mechanisms are required (pinned
[PTX ISA 9.2 / CUDA 13.2](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html)):

1. **Completion** — `tcgen05.mma` is asynchronous. Signal its completion with
   `tcgen05.commit` to an mbarrier, then wait on that mbarrier before reading the
   accumulator. A fence plus `__syncthreads()` does **not** wait for the MMA to
   finish (§9.7.16.12.1 commit; §9.7.16.6.4.2 mbarrier completion).
2. **Ordering** — `tcgen05.fence::before_thread_sync` /
   `tcgen05.fence::after_thread_sync` order tcgen05 operations relative to a
   thread synchronization; they do not themselves establish completion
   (§9.7.16.11.1). The opcode is `tcgen05.fence`, **not** `tcgen05.mma.fence`,
   and there is no `before_cluster_sync` qualifier.

The two helpers below are **conceptual pseudocode** — not standalone-compilable
or Thor-runtime-tested. They assume a single elected issuing thread and an
mbarrier initialized for the MMA's completion arrival. Exact opcode qualifiers
must match the MMA's `cta_group` and the pinned spec.

### Completion: commit a committed MMA to an mbarrier (commit-only fragment)

```cuda
// COMMIT-ONLY fragment: this records the MMA's completion onto `mbar`. It does
// NOT itself wait — the consumer (below) waits on `mbar` before reading.
__device__ void commit_mma_completion(uint64_t* mbar) {
    // Exact qualifiers must match the MMA's cta_group; see pinned §9.7.16.12.1.
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "l"(mbar));
}
```

### Consumer: wait for completion, then order, then read

```cuda
// One internally consistent sequence for safely reading the accumulator.
// Responsibilities: `mbar` was committed to by commit_mma_completion; the caller
// supplies the TMEM address to load from.
__device__ void read_tmem_after_mma(uint64_t* mbar, uint32_t tmem_addr, float* out) {
    mbarrier_try_wait_loop(mbar);                 // 1. WAIT for MMA completion
    asm volatile("tcgen05.fence::after_thread_sync;");  // 2. consumer-side ordering
                                                        //    before the TMEM load
    tcgen05_ld(tmem_addr, out);                   // 3. now read (tcgen05.ld)
}
```

`tcgen05.fence::before_thread_sync` / `after_thread_sync` only *order* tcgen05 ops
around a thread sync (step 2 and its producer-side counterpart); neither waits for
completion — that is step 1's mbarrier wait.

### Correct Mainloop Fence Pattern

```cuda
__device__ void gemm_mainloop(/* params */) {
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // 1. Wait for operand data to arrive in SMEM
        wait_barrier(k_tile % NUM_STAGES);

        // 2. Issue MMA (single thread)
        if (threadIdx.x == 0) {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, 1;"
                :
                : "r"(tmem_addr),
                  "l"(make_smem_desc(smem_a, k_tile)),
                  "l"(make_smem_desc(smem_b, k_tile)),
                  "r"(0)
            );
        }

        // 3. Release SMEM buffer for next TMA load
        if (threadIdx.x == 0) {
            arrive_barrier((k_tile + 1) % NUM_STAGES);
        }
    }

    // 4. CRITICAL: wait for MMA COMPLETION first (commit the final MMA to an
    //    mbarrier and wait on it) — a fence + __syncthreads does not wait.
    wait_last_mma_via_mbarrier(mbar);

    // 5. Order the tcgen05 ops across the thread sync, THEN read TMEM. The
    //    consumer-side after_thread_sync must sit between the completion wait and
    //    the load (pinned §9.7.16.6.4.2).
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    read_tmem_accumulator(tmem_addr, output);
}
```

## Comparison with wgmma

### Programming Model Shift

```cuda
// ---- HOPPER (SM90): wgmma ----
// Requires warpgroup-scoped execution
// All 128 threads in a warpgroup participate
__device__ void hopper_mma() {
    // Load A matrix into registers via ldmatrix
    uint32_t a_frag[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(a_frag[0]), "=r"(a_frag[1]),
                   "=r"(a_frag[2]), "=r"(a_frag[3])
                 : "r"(smem_addr));

    // Issue wgmma -- warpgroup scope, register accumulator
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
                 "{%0, %1, ...}, "  // register accumulators (128+ registers!)
                 "{%N, %N+1, ...}, " // A operand in registers
                 "desc_b, ...;"
                 : "+f"(acc[0]), "+f"(acc[1]), ...
                 : ...);

    // Commit and wait
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // Accumulators now in registers -- high register pressure
}

// ---- BLACKWELL (SM100): tcgen05 ----
// Single-thread issuance, TMEM accumulator
__device__ void blackwell_mma() {
    // No ldmatrix needed -- reads directly from SMEM
    // No register allocation for accumulators

    // Single thread issues MMA
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 1;"
            :
            : "r"(tmem_addr),   // accumulator in TMEM (not registers!)
              "l"(desc_a),       // A from SMEM descriptor
              "l"(desc_b),       // B from SMEM descriptor
              "r"(0)
        );
    }

    // Wait for MMA completion FIRST (an explicit mbarrier wait — the fence does
    // not wait), then order across the sync before reading:
    wait_last_mma_via_mbarrier(mbar);
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    // Read from TMEM -- registers free for other work
}
```

### Performance Implications

- **Register pressure**: Hopper wgmma uses 128+ registers for accumulators in a large GEMM tile. Blackwell stores accumulators in TMEM, freeing those registers for data movement and epilogue logic.
- **Occupancy**: Lower register pressure enables higher CTA occupancy or larger tile sizes without spilling.
- **Warp specialization**: With tcgen05, a single MMA-producer warp can feed the tensor cores while other warps handle TMA loads, epilogue, or softmax -- a natural fit for FlashAttention-style kernels.

## SMEM Layout Requirements

tcgen05.mma requires **128-byte swizzled** shared memory layouts for both operands. Non-swizzled or 64-byte swizzled layouts will produce incorrect results silently.

```cuda
// Shared memory descriptor construction for tcgen05
// The descriptor encodes: base address, stride, swizzle mode, dimensions
__device__ uint64_t make_smem_desc(void* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    // Encode base address (bits 0-13)
    desc |= (uint64_t)(addr >> 4);
    // Encode leading dimension stride (bits 16-29)
    desc |= (uint64_t)((stride_bytes >> 4) & 0x3FFF) << 16;
    // Encode 128-byte swizzle mode (bits 62-63) -- MANDATORY for tcgen05
    desc |= (uint64_t)(3) << 62;  // 3 = 128-byte swizzle

    return desc;
}
```

## Performance Progression (from tcgen05-for-dummies)

| Optimization Stage | Throughput (TFLOPS) | % of cuBLAS |
|---|---|---|
| Naive tcgen05.mma | 255 | 17% |
| + 128B swizzled SMEM | 695 | 46% |
| + TMA pipelining | 940 | 62% |
| + Persistent kernel + CLC | 1476 | 98% |
| cuBLAS reference | 1507 | 100% |

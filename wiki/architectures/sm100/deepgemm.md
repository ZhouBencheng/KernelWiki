---
id: kernel-sm100-deepgemm
title: "SM100 implementation: DeepGEMM block-scaled GEMM"
type: kernel
architectures: [sm100]
applicable_to: [sm100]
tags: [gemm, fp8, fp4, fine-grained-quantization, block-scale, grouped-gemm, tcgen05, tmem]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm, grouped-gemm]
languages: [cuda-cpp]
related: [kernel-deepgemm, technique-sm100-fine-grained-quantization, hw-sm100-tcgen05-mma]
sources: [blog-deepgemm, pr-DeepGEMM-304]
performance_claims: []
---

# SM100 DeepGEMM block-scaled implementation

The retained SM100 kernel builds a block-scaled instruction descriptor,
accumulates in TMEM and consumes packed UE8M0 scale layouts. These details are
not an SM90 implementation.

This contiguous excerpt from the pinned `sm100_fp8_gemm_1d1d.cuh` passes the
instruction descriptor, scale-factor TMEM columns and operand descriptors
through DeepGEMM's typed MMA wrapper:

```cpp
const auto& runtime_instr_desc =
    make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);
b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
    b_desc_base_lo, 0, k * UMMA_K);
mma_t::fma(a_desc, b_desc,
           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
           k_block_idx > 0 or k > 0, runtime_instr_desc,
           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
           kTmemStartColOfSFB);
```

This is not a standalone kernel. Use the pinned artifact for its layouts,
scheduler, synchronization and epilogue, and validate the exact SM100 build and
numerical contract independently. No B200 execution or speedup is claimed by
this page.

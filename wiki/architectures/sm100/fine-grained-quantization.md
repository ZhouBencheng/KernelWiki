---
id: technique-sm100-fine-grained-quantization
title: "SM100 implementation: packed block-scale descriptors"
type: technique
architectures: [sm100]
applicable_to: [sm100]
tags: [fine-grained-quantization, fp8, fp4, block-scale, tcgen05, tmem]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-sm100-tcgen05-mma]
related: [technique-fine-grained-quantization, kernel-sm100-deepgemm]
sources: [blog-deepgemm, doc-ptx-isa-sm100, pr-DeepGEMM-304]
---

# SM100 packed block-scale descriptors

DeepGEMM's retained SM100 kernel constructs a block-scaled descriptor and
selects runtime scale-factor IDs before issuing its typed MMA wrapper:

```cpp
auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
    UMMA_M, UMMA_N, kMajorA, kMajorB>();
const auto& runtime_instr_desc =
    make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);
mma_t::fma(a_desc, b_desc,
           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
           k_block_idx > 0 or k > 0, runtime_instr_desc,
           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
           kTmemStartColOfSFB);
```

The fragment depends on the surrounding template. It establishes the SM100
relationship among type, layout, descriptor and scale locations; it is not an
SM90 scale recipe.

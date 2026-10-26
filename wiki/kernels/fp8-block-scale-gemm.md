---
id: kernel-fp8-block-scale-gemm
title: FP8 block-scale GEMM
type: kernel
architectures: [sm90, sm100]
applicable_to: [sm90, sm100]
tags: [gemm, fp8, block-scale, fine-grained-quantization]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp]
related: [kernel-deepgemm, technique-fine-grained-quantization, kernel-sm100-fp8-block-scale-gemm]
sources: [blog-deepgemm, pr-DeepGEMM-304]
performance_claims:
  - gpu: H800
    dtype: fp8
    shape: best reported benchmark; shape not specified in README news entry
    metric: TFLOPS
    value: 1550
    source_id: blog-deepgemm
    source_locator: https://github.com/deepseek-ai/DeepGEMM#news (2025-04-18 entry)
---

# FP8 block-scale GEMM

An FP8 block-scale GEMM multiplies low-precision operands while applying scale
metadata at a granularity finer than a full tensor. The exact granularity,
scale type, layout, promotion policy and accumulator behavior belong to the
selected implementation; they must not be combined across targets or libraries.

DeepGEMM's retained dispatcher makes that distinction explicit:

```cpp
if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
    const int gran_n = recipe.has_value() ? std::get<1>(recipe.value()) : std::get<0>(recipe_b.value());
    if (gran_n == 1) {
        sm90_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, major_sfb, compiled_dims);
    }
} else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
    sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                            major_a, major_b, compiled_dims);
} else {
    DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
}
```

This is host dispatch evidence, including the SM90 1d1d/1d2d choice and the
unsupported-combination rejection, not a shared device mainloop. Hopper's FP32
scale contract does not authorize the [SM100 CUTLASS block-scale recipe](../architectures/sm100/fp8-block-scale-gemm.md).

DeepGEMM's README reports up to 1550 TFLOPS on H800 without naming the shape in
that news entry. It remains an attributed maximum, not a portable expectation.

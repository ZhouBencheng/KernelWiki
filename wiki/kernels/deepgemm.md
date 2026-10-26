---
id: kernel-deepgemm
title: DeepGEMM — runtime-JIT tensor-core kernels
type: kernel
architectures: [sm90, sm100]
applicable_to: [sm90, sm100]
tags: [gemm, fp8, fp4, fine-grained-quantization, block-scale, grouped-gemm]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm, grouped-gemm]
languages: [cuda-cpp]
related: [technique-fine-grained-quantization, kernel-grouped-gemm, kernel-sm100-deepgemm]
sources: [blog-deepgemm, pr-DeepGEMM-304]
performance_claims:
  - gpu: H800
    dtype: fp8
    shape: best reported benchmark; shape not specified in README news entry
    metric: TFLOPS
    value: 1550
    source_id: blog-deepgemm
    source_locator: https://github.com/deepseek-ai/DeepGEMM#news (2025-04-18 entry)
artifact_dir: artifacts/kernels/deepgemm
---

# DeepGEMM — runtime-JIT tensor-core kernels

DeepGEMM supplies dense and grouped tensor-core kernels for SM90 and SM100. Its
current API dispatches different implementations and scale contracts: SM90 uses
FP32 scales, while SM100 can transform scales to packed UE8M0. Runtime CC,
scale dtype and requested layout are therefore compatibility inputs, not tuning
hints.

## Shared dispatch boundary

This contiguous excerpt from the retained `csrc/apis/gemm.hpp` shows the
architecture and scale-type split before entering either device implementation:

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

The dispatcher selects the SM90 1d1d or 1d2d scale layout from `gran_n`, uses a
separate SM100 implementation for integer scale metadata, and rejects unsupported
architecture/scale combinations. It does not make the SM100 implementation a
Hopper recipe. Use the [SM100 implementation page](../architectures/sm100/deepgemm.md)
only for an SM100 target or an explicit comparison.

## API and performance boundaries

The public API distinguishes contiguous and masked M-grouped layouts plus a
K-grouped weight-gradient API. DeepGEMM compiles kernels at runtime through its
JIT module, with NVCC documented as the default and NVRTC as an opt-in path.

The project README reports up to 1550 TFLOPS on H800 in its 2025-04-18 news
entry. The line does not specify a matrix shape, so it remains a
source-reported maximum with an explicit unknown-shape boundary, not a B200
result or a portable expectation.

The exact retained files and hashes are recorded in
`artifacts/kernels/deepgemm/full/PROVENANCE.yaml` and the PR-304 bundle. The
former derived “Nc=128” teaching file was removed because it presented
illustrative code as the library implementation.

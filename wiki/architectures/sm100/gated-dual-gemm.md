---
id: kernel-sm100-gated-dual-gemm
title: "SM100 implementation: gated NVFP4 GEMM with SwiGLU"
type: kernel
architectures: [sm100]
applicable_to: [sm100]
tags: [gated-dual-gemm, gemm, fused-kernel, kernel-fusion, nvfp4, tmem]
confidence: source-reported
reproducibility: snippet
kernel_types: [gated-dual-gemm, gemm, fused-kernel]
languages: [cuda-cpp, cute-dsl]
related: [kernel-gated-dual-gemm, kernel-nvfp4-gemm, technique-sm100-kernel-fusion]
sources: [contest-gpumode-p3, blog-tflops-gap-fp4-moe, pr-TensorRT-LLM-11897]
performance_claims: []
---

# SM100 gated NVFP4 GEMM with SwiGLU

TensorRT-LLM PR 11897 supplies a concrete SM100 boundary: an NVFP4 dense GEMM
fused with SwiGLU. This contiguous BF16-output call site is not a standalone
kernel:

```python
output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
    act_fp4, module.weight, act_sf, module.weight_scale, alpha,
    module.dtype)
```

The complete source defines shape, packing, scale and output contracts. GPU
Mode problem 3 defines a related workload, but the retained official evidence
does not support the former local latency or standalone speedup claims. This
page therefore carries no numeric performance claim and must not enter SM90
guidance.

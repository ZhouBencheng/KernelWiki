---
id: kernel-sm100-fused-moe
title: "SM100 implementation: fused shared-expert projection"
type: kernel
architectures: [sm100]
applicable_to: [sm100]
tags: [moe, fused-kernel, nvfp4, kernel-fusion, gated-dual-gemm]
confidence: source-reported
reproducibility: snippet
kernel_types: [moe, fused-kernel, gated-dual-gemm]
languages: [cuda-cpp, cute-dsl]
related: [kernel-fused-moe, kernel-sm100-gated-dual-gemm, technique-sm100-kernel-fusion]
sources: [contest-flashinfer-track-a, pr-TensorRT-LLM-11897]
performance_claims: []
---

# SM100 fused shared-expert projection

TensorRT-LLM PR 11897's shared-expert path invokes a dense NVFP4 GEMM fused with
SwiGLU:

```python
output = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
    act_fp4, module.weight, act_sf, module.weight_scale, alpha,
    module.dtype)
```

This contiguous upstream excerpt demonstrates one expert-projection fusion. It
does not place routing, dispatch, both projections and combine in the same
device kernel. The FlashInfer contest establishes a B200 fused-MoE track but no
retained latency table, so no numeric performance claim is made here.

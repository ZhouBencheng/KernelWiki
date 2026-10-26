---
id: kernel-fused-moe
title: Fused MoE — expert GEMM and adjacent operations
type: kernel
architectures: []
applicable_to: [portable]
tags: [moe, fused-kernel, kernel-fusion, grouped-gemm, gated-dual-gemm]
confidence: source-reported
reproducibility: concept
kernel_types: [moe, fused-kernel, grouped-gemm, gated-dual-gemm]
languages: [cuda-cpp, cute-dsl, triton]
related: [kernel-grouped-gemm, kernel-deepgemm, technique-fine-grained-quantization, kernel-sm100-fused-moe]
sources: [doc-cuda-portable-practices, blog-deepgemm]
performance_claims: []
---

# Fused MoE

“Fused MoE” covers kernels that combine an expert GEMM with adjacent data
preparation, activation, quantization or result-combination work. It does not
imply that routing, dispatch, both expert projections and combine always belong
in one device kernel.

A concrete boundary should enumerate routed-row preparation, grouped gate/up
GEMM, activation, intermediate quantization, down projection and weighted
combine. Correctness must cover empty experts, repeated expert IDs, padding,
routing weights and quantization scales. The profitable boundary depends on
routed-row distribution, dtype, intermediate traffic, code size and target
backend.

DeepGEMM documents grouped expert primitives, not a universal end-to-end fusion
speedup. Select the architecture implementation separately; the retained
[SM100 shared-expert example](../architectures/sm100/fused-moe.md) is evidence
for one narrower boundary.

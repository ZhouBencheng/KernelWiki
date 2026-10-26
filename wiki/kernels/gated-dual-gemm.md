---
id: kernel-gated-dual-gemm
title: Gated dual GEMM (gate-up plus activation)
type: kernel
architectures: []
applicable_to: [portable]
tags: [gated-dual-gemm, gemm, fused-kernel, kernel-fusion]
confidence: source-reported
reproducibility: concept
kernel_types: [gated-dual-gemm, gemm, fused-kernel]
languages: [cuda-cpp, cute-dsl]
related: [kernel-fused-moe, technique-kernel-fusion, technique-epilogue-fusion, kernel-sm100-gated-dual-gemm]
sources: [doc-cuda-portable-practices, blog-deepgemm]
performance_claims: []
---

# Gated dual GEMM

A gated dual GEMM evaluates two projections of the same input and combines them
through a gated activation, for example `SiLU(X·W_gate) * (X·W_up)`. Reusing
the input tile and fusing the elementwise combination can avoid materializing
both projections in global memory.

The implementation must state whether the projections share an input tile,
where each accumulator lives, how scales are applied, and how the epilogue
drains results without exceeding target resources. Fusing both projections is
useful only when saved traffic and launches outweigh added live state and
synchronization.

Preserve dtype, accumulation, activation and output-rounding semantics in the
reference. No performance claim is attached to this common concept. Select a
target-specific implementation separately; the retained
[SM100 NVFP4/SwiGLU call site](../architectures/sm100/gated-dual-gemm.md) is not
a Hopper recipe.

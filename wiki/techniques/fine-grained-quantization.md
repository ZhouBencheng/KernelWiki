---
id: technique-fine-grained-quantization
title: Fine-grained quantization scaling
type: technique
architectures: []
applicable_to: [portable]
tags: [fine-grained-quantization, fp8, fp4, block-scale]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [kernel-deepgemm, kernel-fp8-block-scale-gemm, technique-sm100-fine-grained-quantization]
sources: [doc-cuda-portable-practices, blog-deepgemm]
---

# Fine-grained quantization scaling

Fine-grained quantization associates scales with regions smaller than a full
tensor. It can limit outlier influence at the cost of scale storage, conversion,
layout constraints and additional loads. “Block scale” is not one universal
format: scale dtype, granularity, packing and promotion policy belong to the
selected implementation.

Preserve the source format and granularity, use the library's layout
transformer instead of deriving scale strides from prose, and validate the full
quantize–GEMM–dequantize path including tails and exceptional values. A Hopper
FP32 scale contract is not interchangeable with packed UE8M0 or per-16 FP8
metadata.

The [SM100 DeepGEMM implementation](../architectures/sm100/fine-grained-quantization.md)
retains its descriptor and scale-location recipe separately.

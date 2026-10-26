---
id: technique-kernel-fusion
title: "Kernel fusion"
type: technique
architectures: []
applicable_to: [portable]
tags: [kernel-fusion, fused-kernel]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-sm100-kernel-fusion]
sources: [doc-cuda-portable-practices]
---

# Kernel fusion

Kernel fusion keeps producer/consumer operations within one launch to avoid
selected intermediate stores and launches. The legal boundary follows the
operation graph, synchronization requirements and numerical contract.

Benefits compete with extra live state, registers, shared storage, code size
and synchronization. Compare launch count, intermediate bytes, occupancy and
end-to-end latency with the same inputs. A fused operation must preserve
required dtype, accumulation, rounding and activation semantics; an FP4 recipe
is not a same-precision replacement for a BF16 or FP32 operator.

A particular accelerator storage model is not a prerequisite for fusion. The
SM100 example contains an NVFP4 GEMM/SwiGLU call with its own shape and scaling
contract; those details are not part of this portable principle.

## Select an implementation

Read the selected architecture entry using `query.py --target <device>`.
The [SM100 implementation](../architectures/sm100/kernel-fusion.md) preserves the
original excerpts and performance evidence. Open it only for SM100 or an
explicit cross-architecture comparison, not as a recipe for another device.

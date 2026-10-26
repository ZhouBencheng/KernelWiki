---
id: pattern-compute-bound
title: Not reaching the target compute roofline
type: pattern
architectures: []
applicable_to: [portable]
tags: [pipeline-stages, warp-specialization]
symptoms: [compute-bound, low-tensor-core-utilization, pipeline-stalls]
candidate_techniques: [technique-pipeline-stages, technique-warp-specialization, technique-epilogue-fusion]
related: [pattern-low-sm-utilization, pattern-register-pressure, pattern-sm100-compute-bound]
sources: [doc-cuda-portable-practices]
---

# Not reaching the target compute roofline

Use this pattern when the compute path is limiting, measured memory bandwidth
is not saturated and achieved throughput is materially below a shape-, dtype-
and target-matched bound. Do not apply a universal utilization cutoff.

Likely causes include pipeline bubbles, non-matmul work, insufficient arithmetic
intensity for the chosen decomposition, and a serialized epilogue. Measure the
actual waiting role, preserve the numerical contract, and compare changes at a
fixed tile and launch configuration before combining them.

Instruction participation and special-function balance differ by target. The
[SM100 diagnosis](../architectures/sm100/pattern-compute-bound.md) contains
tcgen05, two-CTA and FlashAttention-4 examples and must not enter H100 guidance.

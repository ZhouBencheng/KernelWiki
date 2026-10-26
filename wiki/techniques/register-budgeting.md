---
id: technique-register-budgeting
title: Register budgeting
type: technique
architectures: []
applicable_to: [portable]
tags: [register-budgeting, register-reuse]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [pattern-memory-bound, pattern-register-pressure, technique-sm100-register-budgeting]
sources: [doc-cuda-portable-practices]
---

# Register budgeting

Registers are one input to occupancy alongside threads, shared memory, barrier
state, architectural limits and the compiled kernel. Occupancy is stepwise, not
simply inverse to registers per thread. A lower budget can increase active
warps, but it can also cause spills, recomputation or lost instruction-level
parallelism.

Compare builds with source and launch shape fixed. Record compiler-reported
registers and spill bytes, achieved occupancy, relevant stalls and kernel time.
A compiler cap that does not bind cannot explain a speedup, and a fixed register
count does not imply the same residency across GPUs.

Target-specific build flags and observations belong to the selected route. The
[SM100 example](../architectures/sm100/register-budgeting.md) is not a Hopper
resource constant.

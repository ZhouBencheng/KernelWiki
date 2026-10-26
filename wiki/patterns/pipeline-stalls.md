---
id: pattern-pipeline-stalls
title: Pipeline stalls
type: pattern
architectures: []
applicable_to: [portable]
tags: [pipeline-stages, warp-specialization]
symptoms: [pipeline-stalls, compute-bound, low-tensor-core-utilization]
candidate_techniques: [technique-pipeline-stages, technique-warp-specialization, technique-double-buffering]
related: [pattern-compute-bound, pattern-tail-effect, pattern-sm100-pipeline-stalls]
sources: [doc-cuda-portable-practices]
---

# Pipeline stalls

Use profiling to identify which producer or consumer role waits and whether the
cause is too little in-flight work, incorrect ownership, role imbalance,
memory-system backpressure or too little remaining grid work. Copy completion,
consumer completion and stage reuse are separate events.

Reconstruct stage ownership, verify participant counts and completion rules for
the selected operation, then sweep stage depth while watching storage,
registers, occupancy and total runtime. A lower wait counter is not itself a
speedup.

The [SM100 diagnosis](../architectures/sm100/pattern-pipeline-stalls.md) retains
TMA, tcgen05 and tutorial-specific evidence separately.

---
id: pattern-memory-bound
title: Memory-bandwidth bound
type: pattern
architectures: []
applicable_to: [portable]
tags: [vectorized-loads, cache-policy, shared-memory-optimization]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-swizzling, technique-pipeline-stages]
related: [pattern-compute-bound, pattern-sm100-memory-bound]
sources: [doc-cuda-portable-practices]
---

# Memory-bandwidth bound

Use a byte-count roofline and profiling to establish that data movement, not
instruction throughput, dominates. High measured bandwidth alone is
insufficient; compare useful work, transactions and cache behavior on the
selected GPU.

Likely causes include low arithmetic intensity, limited reuse, uncoalesced or
misaligned accesses, redundant loads and unsuitable cache behavior. Evaluate
vector width, cache policy, reuse and shared-memory layout independently where
possible, including tails. A kernel can move between bandwidth, latency and
instruction limits as shapes change.

B200 NVFP4 and TMA examples remain in the
[SM100 pattern](../architectures/sm100/pattern-memory-bound.md).

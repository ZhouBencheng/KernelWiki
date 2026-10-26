---
id: pattern-sm100-memory-bound
title: "SM100 diagnosis: memory-bound kernels"
type: pattern
architectures: [sm100]
applicable_to: [sm100]
tags: [vectorized-loads, cache-policy, shared-memory-optimization, tma]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-sm100-cache-policy, technique-sm100-swizzling, technique-sm100-pipeline-stages]
related: [pattern-memory-bound, kernel-nvfp4-gemv]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, doc-nvidia-tuning-guide]
---

# SM100 memory-bound diagnosis

On SM100, evaluate legal vector widths, per-stream cache hints, TMA reuse and
descriptor/layout swizzling as separate candidates. TMA multicast can avoid
redundant transfers only when a valid cluster shares the same tile.

The retained NVFP4 GEMV writeups combine access, decoding, cache policy,
conversion and ILP changes. Their end-to-end deltas do not establish a
vector-load-only or cache-policy-only speedup. Use the exact B200 workload and
record bytes, sectors, cache behavior, register pressure and latency.

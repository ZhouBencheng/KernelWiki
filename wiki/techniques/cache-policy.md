---
id: technique-cache-policy
title: Cache-policy differentiation
type: technique
architectures: []
applicable_to: [portable]
tags: [cache-policy, vectorized-loads]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-vectorized-loads, pattern-memory-bound, technique-sm100-cache-policy]
sources: [doc-cuda-portable-practices]
---

# Cache-policy differentiation

Cache operators and eviction hints express different reuse expectations for
different input streams. A candidate may avoid allocating one-pass data in a
near cache while giving a reused input a longer-lived hint, but the operators
remain hints with architecture- and workload-dependent effects.

Establish reuse distance and working-set size, change one policy while holding
address mapping and instruction width fixed, inspect generated instructions,
and measure cache sectors, hit rates and DRAM traffic. Repeat every supported
shape: a policy that helps a long streaming dimension can hurt a cache-resident
case.

Cache policy does not override alignment, coalescing, capacity or correctness.
Select the target implementation separately; the retained
[SM100 attention call site](../architectures/sm100/cache-policy.md) is evidence
for one stream, not a portable default.

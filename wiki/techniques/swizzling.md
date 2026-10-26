---
id: technique-swizzling
title: Shared-memory swizzling
type: technique
architectures: []
applicable_to: [portable]
tags: [swizzling, shared-memory-optimization]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-pipeline-stages, pattern-memory-bound, technique-sm100-swizzling]
sources: [doc-cuda-portable-practices]
---

# Shared-memory swizzling

A shared-memory swizzle permutes address bits so a cooperative access pattern
spreads requests across banks. The correct mapping depends on element size,
matrix major mode, instruction shape and transfer pattern; a larger swizzle span
is not automatically faster.

For descriptor-based operations, the encoded descriptor and physical layout
must agree. Validate the pair against the selected instruction, check alignment
and leading-stride constraints, and measure bank conflicts and throughput for
the actual access pattern.

Use the target implementation for concrete layout atoms. The retained
[SM100 CuTe allocation fragment](../architectures/sm100/swizzling.md) does not
define a Hopper WGMMA layout.

---
id: technique-persistent-kernels
title: "Persistent kernels and work scheduling"
type: technique
architectures: []
applicable_to: [portable]
tags: [persistent-kernel, tile-scheduling]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-sm100-persistent-kernels]
sources: [doc-cuda-portable-practices]
---

# Persistent kernels and work scheduling

A persistent kernel keeps a worker CTA alive for multiple logical tiles. A
static schedule can assign tile IDs in software; dynamic scheduling may use
software atomics or a target-specific hardware protocol. Persistence does not
require Cluster Launch Control (CLC).

Size the resident worker grid from the actual device, the compiled kernel's
resource use and workload concurrency. Do not copy a B200 SM count. Check that
every logical tile is processed exactly once, with no omissions, duplicates or
out-of-range work. Compare static mapping and dynamic acquisition only when
work imbalance or launch cost justifies the scheduling overhead.

Persistence cannot create independent work where the workload has none. Measure
locality, tail time and scheduling overhead using the same shape and numerical
contract. The SM100 variant preserves CLC cancellation semantics and its
source-backed example; it is not a fallback for devices without CLC.

## Select an implementation

Read the selected architecture entry using `query.py --target <device>`.
The [SM100 implementation](../architectures/sm100/persistent-kernels.md) preserves the
original excerpts and performance evidence. Open it only for SM100 or an
explicit cross-architecture comparison, not as a recipe for another device.

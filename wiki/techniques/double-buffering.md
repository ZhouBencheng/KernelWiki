---
id: technique-double-buffering
title: "Double and multi-buffering"
type: technique
architectures: []
applicable_to: [portable]
tags: [double-buffering, pipeline-stages]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-sm100-double-buffering]
sources: [doc-cuda-portable-practices]
---

# Double and multi-buffering

Buffering uses distinct storage regions for pipeline iterations so a producer
can fill one stage while a consumer uses another. It applies to ordinary shared
memory as well as specialized storage; TMEM is not a prerequisite.

- Do not overwrite a stage until every consumer has released it.
- Track ownership and phase changes exactly once per reuse.
- Extra stages consume storage and may reduce resident work.
- Validate short loops, tails, initialization and final drain before timing.
- Choose the number of stages from measured waits and resource limits; two is
  a candidate, not a universal optimum.

The copy and synchronization implementation is target-specific. Ampere/Ada can
use the ordinary asynchronous-copy path; a target with TMA may use a different
transaction-completion protocol. Consult the selected target entry before
choosing one. The SM100 variant contains TMEM allocation constraints and the
original upstream example, not a portable code template.

## Select an implementation

Read the selected architecture entry using `query.py --target <device>`.
The [SM100 implementation](../architectures/sm100/double-buffering.md) preserves the
original excerpts and performance evidence. Open it only for SM100 or an
explicit cross-architecture comparison, not as a recipe for another device.

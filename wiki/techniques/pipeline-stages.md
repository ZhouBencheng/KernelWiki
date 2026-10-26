---
id: technique-pipeline-stages
title: "Software pipelining and stage ownership"
type: technique
architectures: []
applicable_to: [portable]
tags: [pipeline-stages, double-buffering]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-sm100-pipeline-stages]
sources: [doc-cuda-portable-practices]
---

# Software pipelining and stage ownership

Software pipelining overlaps work on different iterations. The consumer must
wait for data completion, and the producer must wait for all consumers before
reusing storage. Copy completion and consumer completion are separate events.

Choose a stage count using bytes per stage, measured producer/consumer time,
loop length, synchronization cost, occupancy and register pressure. More
stages can improve overlap but increase storage and prologue/drain overhead.

Check zero/short/non-multiple loops and the numerical contract independently
of performance. A change in barrier phase or stage indexing can deadlock or
corrupt output. Use the selected architecture's completion primitives; ordinary
`cp.async` waits, TMA transaction barriers and TMEM lifetime rules are not
interchangeable. No fixed stage count or B200 speedup is a portable promise.

## Select an implementation

Read the selected architecture entry using `query.py --target <device>`.
The [SM100 implementation](../architectures/sm100/pipeline-stages.md) preserves the
original excerpts and performance evidence. Open it only for SM100 or an
explicit cross-architecture comparison, not as a recipe for another device.

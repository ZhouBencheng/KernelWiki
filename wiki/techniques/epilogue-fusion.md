---
id: technique-epilogue-fusion
title: "Epilogue fusion"
type: technique
architectures: []
applicable_to: [portable]
tags: [epilogue-fusion, kernel-fusion]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-sm100-epilogue-fusion]
sources: [doc-cuda-portable-practices]
---

# Epilogue fusion

An epilogue converts an accumulator to output and can combine scaling, bias,
a source tensor, activation or quantization before the store. Fusion may save
a launch and intermediate memory traffic. Overlap with another tile is a
separate scheduling property requiring explicit ownership and synchronization.

Validate alpha/beta or source-tensor semantics, activation edge cases, output
conversion and tails. Preserve the requested numerical contract, including
accumulation and rounding. Measure fusion savings separately from tile-shape,
mainloop or precision changes.

Registers, shared memory and specialized accumulator storage need different
implementations. The SM100 variant retains the CUTLASS epilogue builder and its
TMEM caveats; a target entry selects the appropriate implementation knowledge.

## Select an implementation

Read the selected architecture entry using `query.py --target <device>`.
The [SM100 implementation](../architectures/sm100/epilogue-fusion.md) preserves the
original excerpts and performance evidence. Open it only for SM100 or an
explicit cross-architecture comparison, not as a recipe for another device.

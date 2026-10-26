---
id: hw-async-copy-warp-mma
title: "Ordinary async copy and warp MMA"
type: hardware
architectures: [sm87, sm89, sm110]
applicable_to: [sm87, sm89, sm110]
tags: [cp-async, warp-mma, ldmatrix]
confidence: source-reported
related: [technique-pipeline-stages, technique-double-buffering, hw-target-sm87, hw-target-sm89]
sources: [doc-nvidia-ampere-ada-instructions, doc-cuda-portable-practices]
aliases: [cp.async, mma.sync]
---

# Ordinary async copy and warp MMA

On SM87, SM89 and SM110, ordinary `cp.async` is a candidate for global-to-shared
movement and `mma.sync` provides warp-level matrix operations. This page
shares the programming contract; device resources and tuning stay in each
architecture entry.

## Copy ownership

1. Prove source/destination alignment, legal copy size and tail handling.
2. Issue copies, commit a group, then wait for the relevant group before use.
3. A copy-group wait is not an all-thread consumer barrier. If other threads
   consume the data, add the appropriate CTA synchronization and ensure it is
   reached legally.
4. Do not overwrite shared storage until all consumers finish.

`cp.async` is not TMA: it has no tensor-map descriptor or cluster multicast.
Do not replace its completion protocol with a TMA transaction-byte barrier
recipe. Use a synchronous-copy baseline to test whether extra stages and
synchronization pay for themselves.

## Warp MMA

All participating lanes must obey the chosen instruction's collective
execution contract and fragment layout. Shape, operand packing, dtype and
accumulator type are part of the operation, not tuning hints. BF16 operands
with an FP32 accumulator are distinct from strict FP32 multiplication; TF32
must not silently replace a strict-FP32 workload.

Warp MMA capability does not make a small-M problem efficient. Compare a SIMT
GEMV/reduction and legal matrix tiling on the actual shape, counting packing,
padding and epilogue work. No speedup is asserted here. Follow the instruction
Target ISA Notes for the exact form and inspect generated code when proving
that a backend selected the intended instruction.

## Evidence scope

This is official-document-based guidance. It does not claim that every MMA
form or pipelined implementation was executed on the recorded devices. The target receipts record the exact 4-byte copy and 16×16×16 BF16/FP32
WMMA checks. These do not validate all MMA forms or a pipelined implementation.

---
id: hw-tmem
title: "Tensor Memory: shared ownership and lifetime contract"
type: hardware
architectures: [sm100, sm100a, sm103, sm110]
applicable_to: [sm100, sm103, sm110]
tags: [tmem, tcgen05]
confidence: source-reported
related: [hw-tcgen05-mma, hw-sm100-tmem, technique-double-buffering]
sources: [doc-nvidia-blackwell-shared-instructions, doc-ptx-isa-sm100]
aliases: [TMEM, "tensor memory", "Tensor Memory"]
---

# Tensor Memory: shared ownership and lifetime contract

TMEM is specialized storage for supported tcgen05 operations. It is not
ordinary shared memory and its address is not a generic-memory pointer.
Matrix accumulators and subsequent consumers must follow the documented TMEM
access path rather than ordinary load/store instructions.

## Shared rules

Allocation/deallocation and TMEM loads/stores have collective participation
contracts. Match operands and control flow across participating lanes. Make
the allocated address visible before another thread consumes it, wait for
asynchronous producers before reading results, and release every allocation
before kernel exit. Cross-CTA operations add peer lifetime requirements.

Use the exact allocation mode and target notes for legal sizes. Resource
budgets must include simultaneously live accumulators and intermediate state.
The SM100 view and original allocation reference are preserved in
[SM100 TMEM](../architectures/sm100/tmem.md); its geometry must not be treated
as a measured Thor physical capacity or a universal bytes-per-SM constant.

Double buffering is an ownership principle, not a demand to divide TMEM into
two fixed halves. Derive storage from the actual instruction/layout and verify
both correctness and residency. The shared page applies to documented SM100,
SM103 and SM110 forms; it does not assert native TMEM on SM120, Ada or Orin.

No allocation or tcgen05 kernel execution is established by a SIMT/warp-MMA
smoke test. Keep compiler acceptance, instruction correctness and benchmark
results separate when adding a target implementation.

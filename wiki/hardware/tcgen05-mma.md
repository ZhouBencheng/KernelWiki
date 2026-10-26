---
id: hw-tcgen05-mma
title: "tcgen05: shared execution contract"
type: hardware
architectures: [sm100, sm100a, sm103, sm110]
applicable_to: [sm100, sm103, sm110]
tags: [tcgen05, tmem, mbarrier]
confidence: source-reported
related: [hw-tmem, hw-sm100-tcgen05-mma, technique-pipeline-stages]
sources: [doc-nvidia-blackwell-shared-instructions, doc-ptx-isa-sm100]
aliases: [UMMA, tcgen05, "tensor core gen 05"]
---

# tcgen05: shared execution contract

A supported `tcgen05.mma` operation starts an asynchronous matrix multiply
with its accumulator in TMEM. Its issuing-thread model differs from collective
warp `mma.sync` and warpgroup `wgmma`. Allocation, loads and stores have their
own collective participation requirements; single-thread MMA issue does not
make all TMEM operations single-threaded.

## Select an operation, not just a family name

SM100, SM103 and SM110 have applicable tcgen05 target variants, but `kind`,
shape, sparsity, block scaling, CTA group and optional operands each constrain
legality. Select the exact PTX form and compiler feature target from its Target
ISA Notes. Runtime CC alone does not choose `a` or `f`. This route does not
apply to RTX 4090, Orin or SM120.

## Ownership and completion

- Construct operand layouts and descriptors that match the exact operation.
- Make producer writes visible under the required memory/proxy ordering.
- Issue the matrix operation and establish its documented completion path
  before reading or reusing the accumulator or operands.
- Synchronize cross-thread consumers separately from instruction completion.
- Release allocated TMEM before exit, respecting collective lifetimes.

Do not assume that arbitrary neighboring asynchronous operations are ordered.
The PTX pipelined-pairing rules and fences must match the actual sequence.
A successful ordinary warp-MMA probe does not validate this execution path.

The [SM100 instruction reference](../architectures/sm100/tcgen05-mma.md)
preserves the original examples, descriptor details and B200 source references.
Load it with the SM100 route or explicitly as comparison material. Thor kernels
need their own target-specific build, correctness and resource checks; this
shared contract carries no new tcgen05 runtime or performance claim.

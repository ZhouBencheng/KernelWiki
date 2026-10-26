---
id: pattern-sm100-moe-load-imbalance
title: "SM100 diagnosis: MoE expert load imbalance"
type: pattern
architectures: [sm100]
applicable_to: [sm100]
tags: [moe, grouped-gemm, tile-scheduling, clc]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-sm100-persistent-kernels, technique-sm100-kernel-fusion]
related: [pattern-moe-load-imbalance, kernel-grouped-gemm, kernel-sm100-fused-moe]
sources: [contest-gpumode-p4, contest-flashinfer-track-a, blog-deepgemm, doc-ptx-isa-sm100]
---

# SM100 MoE expert load imbalance

SM100 CLC can let a persistent cluster acquire the ID of same-grid work that
has not launched. It is one acquisition option, not an arbitrary expert queue,
and cannot eliminate the final shortage of independent work.

Compare static and CLC-assisted scheduling with identical routed rows, tile
shape, cluster shape and correctness checks. Preserve empty experts, padding,
scales, routing weights and output offsets. The acquisition and synchronization
cost must be justified by measured imbalance.

---
id: pattern-moe-load-imbalance
title: MoE expert load imbalance
type: pattern
architectures: []
applicable_to: [portable]
tags: [moe, grouped-gemm, tile-scheduling]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-persistent-kernels, technique-kernel-fusion]
related: [kernel-grouped-gemm, kernel-fused-moe, pattern-tail-effect, pattern-sm100-moe-load-imbalance]
sources: [doc-cuda-portable-practices, blog-deepgemm]
---

# MoE expert load imbalance

A grouped-GEMM trace may show unequal expert tile counts or runtimes, leaving
workers idle while other groups remain active. Confirm this from routed row
counts and per-tile timings; no fixed router distribution is assumed.

Common causes include unequal routed rows, static assignment, fixed-capacity
masked layouts and thin expert matrices that underfill the selected tile.
Compare schedulers with identical routed inputs, tile shapes and correctness
checks. Dynamic acquisition has its own overhead and cannot create work after
too few independent tiles remain.

The [SM100 route](../architectures/sm100/pattern-moe-load-imbalance.md) adds CLC
as one target-specific acquisition option.

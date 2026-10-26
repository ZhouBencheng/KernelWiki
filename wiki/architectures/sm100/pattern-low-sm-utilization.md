---
id: pattern-sm100-low-sm-utilization
title: "SM100 diagnosis: low SM utilization"
type: pattern
architectures: [sm100]
applicable_to: [sm100]
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-sm100-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-low-sm-utilization, pattern-sm100-tail-effect]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-cutlass-2161]
---

# SM100 low SM utilization

After establishing a grid, tail or imbalance problem, compare static persistence
with CLC only when the toolkit and exact target support the intended protocol.
CLC may redistribute IDs of same-grid clusters that have not launched; it does
not create arbitrary work or cancel completed output tiles.

Record SM count, resident CTAs, grid and cluster dimensions, per-tile duration
and acquisition overhead. Correctness and progress must not depend on a
particular overlap outcome.

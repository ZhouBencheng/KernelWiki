---
id: pattern-low-sm-utilization
title: Low SM utilization
type: pattern
architectures: []
applicable_to: [portable]
tags: [persistent-kernel, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling]
related: [pattern-tail-effect, pattern-compute-bound, pattern-sm100-low-sm-utilization]
sources: [doc-cuda-portable-practices]
---

# Low SM utilization

Use this pattern when profiling shows idle SMs during material portions of a
kernel and resource occupancy is not the immediate limiter. Check for a short
last wave, unequal tile duration, static assignment and a grid smaller than the
available resident slots.

Persistent work reuse or a different tile order can help only when work remains
to redistribute. Record actual SM count, residency, grid size and per-tile
duration. A larger grid can reduce wave quantization but may add overhead.

CLC is not a portable remedy; its protocol is confined to the
[SM100 pattern](../architectures/sm100/pattern-low-sm-utilization.md).

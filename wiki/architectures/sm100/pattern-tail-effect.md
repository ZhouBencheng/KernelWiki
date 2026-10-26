---
id: pattern-sm100-tail-effect
title: "SM100 example: B200 last-wave underutilization"
type: pattern
architectures: [sm100]
applicable_to: [sm100]
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-sm100-persistent-kernels, hw-clc, technique-tile-scheduling]
related: [pattern-tail-effect, pattern-sm100-low-sm-utilization]
sources: [doc-blackwell-microbenchmarking, doc-ptx-isa-sm100]
---

# B200 last-wave example

The locally verified B200 exposes 148 SMs. Under the simplifying assumption of
one resident CTA per SM and equal-duration tiles, a 150-CTA grid has a first
wave of 148 CTAs and a final wave of two, leaving 146 SMs without work during
that final wave.

This is an observed SKU example, not an SM100 architecture constant. Persistent
scheduling or CLC may redistribute work and reduce other overheads, but cannot
make 150 independent tile tasks occupy more than 150 tile-time slots.

---
id: pattern-tail-effect
title: Tail effect — last-wave underutilization
type: pattern
architectures: []
applicable_to: [portable]
tags: [persistent-kernel, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling]
related: [pattern-low-sm-utilization, pattern-sm100-tail-effect]
sources: [doc-cuda-portable-practices]
---

# Tail effect

When a static grid is not a multiple of simultaneously resident CTA slots, its
last wave cannot occupy every slot. Confirm the effect from actual grid size,
kernel residency and CTA-duration variance; `num_tiles % num_SMs` is only the
one-resident-CTA-per-SM special case.

Measure resident CTAs per SM, logical tile and grid counts, per-CTA duration,
time in the final wave and any locality cost introduced by an alternative
schedule. Persistence cannot make fewer independent tasks occupy more slots.

The observed B200 numeric example remains in the
[SM100 pattern](../architectures/sm100/pattern-tail-effect.md), not this portable
principle.

---
id: pattern-register-pressure
title: Register pressure and low occupancy
type: pattern
architectures: []
applicable_to: [portable]
tags: [register-reuse, register-budgeting]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [technique-register-budgeting, technique-warp-specialization]
related: [pattern-compute-bound, pattern-sm100-register-pressure]
sources: [doc-cuda-portable-practices]
---

# Register pressure

High live-register demand can reduce residency or cause local-memory spills.
Confirm both with compiler resource output and profiler evidence; occupancy is
not itself the performance objective.

Common responses include shorter live ranges, less unrolling, smaller tiles or
role-specific register budgets. Each may add instructions, reduce reuse or
increase synchronization. Compare at a fixed numerical contract and record
spills, achieved occupancy and runtime together.

TMEM accumulator guidance is target-specific and remains in the
[SM100 pattern](../architectures/sm100/pattern-register-pressure.md).

---
id: pattern-sm100-register-pressure
title: "SM100 diagnosis: register pressure and TMEM"
type: pattern
architectures: [sm100]
applicable_to: [sm100]
tags: [tmem, register-reuse, warp-specialization]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [hw-sm100-tmem, technique-sm100-warp-specialization, migration-register-to-tmem, technique-sm100-register-budgeting]
related: [pattern-register-pressure, hw-sm100-tmem]
sources: [doc-ptx-isa-sm100, pr-vllm-16032]
---

# SM100 register pressure and TMEM

TMEM accumulators move one class of state out of per-thread registers, but add
allocation, synchronization and TMEM-to-register epilogue work. Role-specific
register budgets can also change live state and synchronization.

On `sm_100a`/`sm_100f`, PTX exposes each CTA a TMEM view of 512 columns by 128 lanes of 32-bit cells.
Treat its required columns as part of the kernel design,
not a documented physical byte capacity per SM. Measure register allocation,
spills, TMEM columns, residency and runtime together.

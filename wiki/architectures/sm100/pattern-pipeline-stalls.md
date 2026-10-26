---
id: pattern-sm100-pipeline-stalls
title: "SM100 diagnosis: TMA and tcgen05 pipeline stalls"
type: pattern
architectures: [sm100]
applicable_to: [sm100]
tags: [pipeline-stages, warp-specialization, tma, tcgen05, mbarrier]
symptoms: [pipeline-stalls, compute-bound, low-tensor-core-utilization]
candidate_techniques: [technique-sm100-pipeline-stages, technique-sm100-warp-specialization, technique-sm100-double-buffering, technique-ping-pong-scheduling]
related: [pattern-pipeline-stalls, pattern-sm100-compute-bound]
sources: [blog-tcgen05-tutorial, blog-flash-attention-4, doc-nvidia-tuning-guide]
---

# SM100 TMA and tcgen05 pipeline stalls

TMA transaction completion, tcgen05 completion, TMEM lifetime and ordinary
shared-memory consumers have distinct ordering rules. Diagnose the exact role
and phase rather than applying a generic fence sequence.

The retained tcgen05 tutorial reports 939.61 TFLOPS (62.4% of its cuBLAS
result) at the pipelining step, 1208.83 TFLOPS (80.2%) after warp
specialization, 1302.29 TFLOPS (86.4%) after 2-SM MMA and 1475.93 TFLOPS
(98.0%) after static persistence. These steps change multiple mechanisms and
are not an isolated stage-count or cross-device speedup.

Sweep stage depth and role allocation while tracking shared memory, TMEM,
registers, occupancy and end-to-end runtime. Phase bugs can be intermittent;
instrument ownership during correctness work.

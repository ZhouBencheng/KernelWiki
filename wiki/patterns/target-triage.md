---
id: pattern-target-triage
title: "Device-aware kernel triage"
type: pattern
architectures: []
applicable_to: [portable]
tags: [data-reuse, register-budgeting, tile-scheduling]
symptoms: [memory-bound, low-sm-utilization, register-pressure]
candidate_techniques: [technique-double-buffering, technique-pipeline-stages, technique-persistent-kernels, technique-kernel-fusion]
related: [technique-epilogue-fusion]
sources: [doc-cuda-portable-practices]
---

# Device-aware kernel triage

Fix the operation's shape, layouts, input/output dtype, accumulation mode and
error tolerance before comparing candidates. Record the selected device and
toolchain. A marketing architecture name is not a complete target.

| Observation to establish | Candidate investigation | Constraint |
|---|---|---|
| Too little independent work | Tile mapping and software persistence | Device SM count and kernel occupancy, not a B200 constant |
| Excess traffic or poor coalescing | Layout, reuse and legal vector widths | Alignment, tails and measured transaction efficiency |
| Low overlap | Buffering or pipelining | Select copy and completion primitives for this target |
| High register/shared-storage pressure | Smaller live state or tiling | Check spilling and occupancy together |
| Repeated intermediate materialization | Kernel or epilogue fusion | Preserve numerical and synchronization contracts |

These are hypotheses, not automatic diagnoses. Obtain available profiler
metrics and preserve units, sampling scope and missing-data states. If a
metric is absent, identify the evidence gap instead of substituting zero.
Measure ordinary kernel latency separately from profiler replay.

Open only the selected target entry and relevant shared principles. A raw PR
or blog found with `--architecture` is evidence to inspect, not permission to
run its code on the selected GPU. An unknown target gets this common workflow
until its instruction and tool support are established.

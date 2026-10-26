---
id: technique-warp-specialization
title: Warp specialization
type: technique
architectures: []
applicable_to: [portable]
tags: [warp-specialization, pipeline-stages]
confidence: source-reported
reproducibility: concept
prerequisites: []
related: [technique-persistent-kernels, technique-pipeline-stages, technique-sm100-warp-specialization]
sources: [doc-cuda-portable-practices]
---

# Warp specialization

Warp specialization assigns different warps to producer, matrix, scheduling,
reduction or epilogue work so independent pipeline phases can overlap. The
number of warps and their roles are kernel configuration choices, not an
architecture-wide canonical layout.

Name one owner for every pipeline transition, keep collective instructions on
their required converged participants, and budget registers and shared storage
per role. Measure overlap: extra roles and barriers can cost more than they save
when phases are short or imbalanced.

The exact role split follows the selected target's instruction participation
and accumulator model. Read the
[SM100 tcgen05/TMEM implementation](../architectures/sm100/warp-specialization.md)
only on that route or for explicit comparison; Hopper WGMMA code needs its own
warpgroup participation and register-accumulator contract.

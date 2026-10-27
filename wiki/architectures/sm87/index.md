---
id: hw-target-sm87
title: "Jetson AGX Orin / Ampere: target guidance"
type: hardware
architectures: [sm87]
applicable_to: [sm87]
tags: [cp-async, warp-mma, cuda-cpp]
confidence: source-reported
related: [pattern-target-triage, technique-double-buffering, technique-pipeline-stages, technique-persistent-kernels]
sources: [doc-nvidia-gpu-targets, doc-nvidia-ampere-ada-instructions, doc-cuda-portable-practices]
aliases: []
---

# Jetson AGX Orin / Ampere

Runtime compute capability: **8.7**. Select this route with
`python3 scripts/query.py --target sm87`. Selection indicates relevant guidance,
not a promise that every referenced instruction or kernel was executed.

## Programming route

Use native `sm_87` for baseline builds when supported by the installed compiler.
Orin is not SM80 or SM86 despite sharing the Ampere name. The shared
[ordinary async-copy and warp-MMA guide](../../hardware/async-copy-warp-mma.md)
explains `cp.async` and `mma.sync`, including BF16 versus accumulation precision.

TMA, WGMMA, tcgen05, TMEM and CLC are not SM87 execution paths. Do not load
SM100 snippets as Orin implementation advice. Keep software tile scheduling,
shared-memory buffering and SIMT alternatives available when their overhead
is lower for the actual shape.

## Device and platform

Query the selected module's SM count, shared-memory limits, memory size and
occupancy. Orin is an integrated Jetson platform; CPU/GPU memory traffic and
power configuration matter. Do not substitute A100 resource limits from a
generic Ampere table or B200 bandwidth in a roofline.

Record Jetson system/JetPack, driver, CUDA compiler and the NCU collector and
reader versions separately. Query supported collection sections and metrics.
A successful report read does not mean its missing counters were measured.

## Observed validation: 2026-10-27

The [validation summary](../../../evidence/multiarch-validation-2026-10-27.md) records
one device reporting 14 SMs, CC 8.7, compiled natively for
`sm_87` using nvcc 12.6.85. These are observations of
one SKU/configuration, not architecture constants.

The historical probe runs summarized there passed vector copy,
block reduction, FP32 64×64×64 SIMT matrix multiplication, a 4-byte ordinary
`cp.async` copy and one-warp BF16-input/FP32-accumulator WMMA 16×16×16. Their
checked inputs yielded zero observed matrix error. This validates only these
operations, shapes and contracts; it does not validate tcgen05, TMEM, TMA,
WGMMA, two-CTA kernels, all dtypes/shapes or an optimization speedup.

The summary separately records profiler collection/read results, reader API
availability and limitations. Report hashes identify external raw reports; the
reports are not bundled here. Use the observed collector/reader pair when
interpreting those results.

## Evidence levels

Keep official/source confidence separate from device validation. Record
compilation, numerical correctness, profiler collection and measured benefit
independently, with the exact source, build command and environment. A missing
receipt is an evidence gap. See [the validation contract](../../../references/target-routing.md).

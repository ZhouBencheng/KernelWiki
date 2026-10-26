---
id: hw-target-sm110
title: "Jetson AGX Thor / Blackwell: target guidance"
type: hardware
architectures: [sm110]
applicable_to: [sm110]
tags: [tcgen05, tmem, tma, cuda-cpp]
confidence: source-reported
related: [pattern-target-triage, technique-double-buffering, technique-pipeline-stages, technique-persistent-kernels]
sources: [doc-nvidia-gpu-targets, doc-nvidia-feature-targets, doc-ptx-isa-sm100]
aliases: []
---

# Jetson AGX Thor / Blackwell

Runtime compute capability: **11.0**. Select this route with
`python3 scripts/query.py --target sm110`. Selection indicates relevant guidance,
not a promise that every referenced instruction or kernel was executed.

## Target selection

Jetson Thor (T5000/T4000) is CC 11.0 / SM110, not SM120. Keep observed runtime
CC separate from native or virtual compilation targets. Select `sm_110a` or
`sm_110f` only after checking the exact instruction and the actual compiler or
framework assembler, not merely because the device reports CC 11.0.

The shared [ordinary copy/warp-MMA route](../../hardware/async-copy-warp-mma.md)
is also available as a measured basic path; it does not require tcgen05.

## Shared mechanisms, separate implementations

PTX lists SM110 targets for tcgen05/TMEM forms, but not every form, shape,
scaling variant or two-CTA recipe follows from that family name. The
[tcgen05](../../hardware/tcgen05-mma.md) and [TMEM](../../hardware/tmem.md)
pages provide the shared mechanism contract and can be loaded with `--target thor`.
Their separately linked SM100 implementations and performance are comparison
material, not Thor execution evidence. Use the PTX
Target ISA Notes and the selected library's dispatch for the exact operation.

Read [TMA](../../hardware/tma.md) for the shared descriptor/completion contract,
then begin with the common buffering/pipelining principles. Verify an intended
Thor native path by building, checking numerical results and inspecting the
produced instructions before tuning it. A basic SIMT probe cannot validate
TMEM, NVFP4, tcgen05 or two-CTA behavior.

## Platform constraints

Query actual device resources rather than copying B200's SM count, register
occupancy, memory capacity or bandwidth. Record Jetson system, CUDA/driver,
compiler target and NCU collector/reader versions. Collection and clock
control depend on the platform and installed profiler. Preserve actual power
and clock conditions with any performance record.

## Observed validation: 2026-10-27

The [sanitized receipt](../../../evidence/multiarch-2026-10-27/sm110.json) records
one device reporting 20 SMs, CC 11.0, compiled natively for
`sm_110` using nvcc 13.2.86. These are observations of
one SKU/configuration, not architecture constants.

The [retained probes](../../../probes/multiarch/README.md) passed vector copy,
block reduction, FP32 64×64×64 SIMT matrix multiplication, a 4-byte ordinary
`cp.async` copy and one-warp BF16-input/FP32-accumulator WMMA 16×16×16. Their
checked inputs yielded zero observed matrix error. This validates only these
operations, shapes and contracts; it does not validate tcgen05, TMEM, TMA,
WGMMA, two-CTA kernels, all dtypes/shapes or an optimization speedup.

The receipt separately records profiler collection/read results and limitations.
Original report hashes identify external raw reports; the reports are not
bundled here. Read the receipt's actual collector/reader pair rather than
assuming the installed collector provides a usable local Python reader.

## Evidence levels

Keep official/source confidence separate from device validation. Record
compilation, numerical correctness, profiler collection and measured benefit
independently, with the exact source, build command and environment. A missing
receipt is an evidence gap. See [the validation contract](../../../references/target-routing.md).

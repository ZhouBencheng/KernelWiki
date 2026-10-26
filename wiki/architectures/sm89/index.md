---
id: hw-target-sm89
title: "GeForce RTX 4090 / Ada: target guidance"
type: hardware
architectures: [sm89]
applicable_to: [sm89]
tags: [cp-async, warp-mma, cuda-cpp]
confidence: source-reported
related: [pattern-target-triage, technique-double-buffering, technique-pipeline-stages, technique-persistent-kernels]
sources: [doc-nvidia-gpu-targets, doc-nvidia-ada-tuning, doc-nvidia-ampere-ada-instructions]
aliases: []
---

# GeForce RTX 4090 / Ada

Runtime compute capability: **8.9**. Select this route with
`python3 scripts/query.py --target sm89`. Selection indicates relevant guidance,
not a promise that every referenced instruction or kernel was executed.

## Programming route

Use native `sm_89` for baseline builds when supported by the selected compiler.
Start with the shared [ordinary async-copy and warp-MMA guide](../../hardware/async-copy-warp-mma.md).
TMA, WGMMA, tcgen05, TMEM and CLC are not SM89 execution paths. An FP8 or
Tensor Core marketing label does not authorize a Hopper/Blackwell instruction.
Verify dtype and shape support of the actual instruction and library path.

## Resources and tuning

Ada permits at most 48 resident warps and 24 blocks per SM; it is not the
B200/Hopper 64-warp occupancy model. The official tuning guide lists 100 KB
shared-memory capacity per SM and a separate per-block opt-in limit. Query
runtime properties and the compiled kernel's resource use before choosing
block size or persistent worker count. RTX 4090 SKU resources and live clocks
are not constants for every SM89 GPU.

Measure bandwidth and cache effects on the actual workload. Compare aligned
vector loads, shared-memory reuse and software scheduling without borrowing
HBM bandwidth, TMEM layouts or NVFP4 recipes from B200. Preserve precision
when evaluating SIMT versus matrix instructions, especially for small M.

## Observed validation: 2026-10-27

The [sanitized receipt](../../../evidence/multiarch-2026-10-27/sm89.json) records
one device reporting 128 SMs, CC 8.9, compiled natively for
`sm_89` using nvcc 12.4.131. These are observations of
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

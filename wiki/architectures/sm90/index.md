---
id: hw-target-sm90
title: "Hopper / H100 and H200: target guidance"
type: hardware
architectures: [sm90]
applicable_to: [sm90]
tags: [tma, wgmma, cuda-cpp]
confidence: source-reported
related: [pattern-target-triage, technique-double-buffering, technique-pipeline-stages, technique-persistent-kernels, technique-warp-specialization, kernel-deepgemm, lang-triton]
sources: [doc-nvidia-gpu-targets, doc-nvidia-feature-targets, doc-ptx-isa-sm100]
aliases: []
---

# Hopper / H100 and H200

Runtime compute capability: **9.0**. Select this route with
`python3 scripts/query.py --target sm90`. Selection indicates relevant guidance,
not a promise that every referenced instruction or kernel was executed.

## Programming route

Hopper has TMA and warpgroup MMA paths. Select exact instruction/shape support
and distinguish a baseline `sm_90` target from architecture-specific `sm_90a`.
SM100 tcgen05/TMEM/CLC code is not a Hopper fallback.

Read [TMA](../../hardware/tma.md) for descriptor and completion concepts.
Retained source knowledge remains searchable with `--architecture sm90`;
that search reports evidence scope, not runnable compatibility. The
[SM90-to-SM100 migration](../sm100/wgmma-to-tcgen05.md) is a comparison,
not a recipe that replaces Hopper's instructions in place.

The selected route also exposes the shared DeepGEMM dispatcher and H800 result,
the Hopper Triton implementation note, and portable warp-specialization
principles. Their linked SM100 peers remain blocked by `get_page.py --target h100`.

## Validation

Query actual H100/H200 device resources and preserve dtype, shape and software
versions with measurements. No new Hopper runtime verification is claimed by
this multi-architecture change. Existing cited upstream evidence is retained.

## Evidence levels

Keep official/source confidence separate from device validation. Record
compilation, numerical correctness, profiler collection and measured benefit
independently, with the exact source, build command and environment. A missing
receipt is an evidence gap. See [the validation contract](../../../references/target-routing.md).

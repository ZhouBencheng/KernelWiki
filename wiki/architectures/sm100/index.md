---
id: hw-target-sm100
title: "Blackwell / B200 and GB200: target guidance"
type: hardware
architectures: [sm100]
applicable_to: [sm100]
tags: [tcgen05, tmem, tma, clc, cuda-cpp]
confidence: source-reported
related: [pattern-target-triage, technique-double-buffering, technique-pipeline-stages, technique-persistent-kernels, kernel-sm100-deepgemm, technique-sm100-warp-specialization, lang-sm100-triton]
sources: [doc-nvidia-gpu-targets, doc-nvidia-feature-targets, doc-ptx-isa-sm100]
aliases: []
---

# Blackwell / B200 and GB200

Runtime compute capability: **10.0**. Select this route with
`python3 scripts/query.py --target sm100`. Selection indicates relevant guidance,
not a promise that every referenced instruction or kernel was executed.

## Retained SM100 implementation knowledge

The B200 corpus remains the main source of detailed kernel implementations.
Read [PTX for SM100](ptx.md) for the compilation and instruction examples.
The shared principle and SM100 realization are now separate:

| Principle | SM100 implementation / retained evidence |
|---|---|
| [Buffering](../../techniques/double-buffering.md) | [TMEM/storage ownership](double-buffering.md) |
| [Pipelining](../../techniques/pipeline-stages.md) | [TMA/MMA stages](pipeline-stages.md) |
| [Persistence](../../techniques/persistent-kernels.md) | [CLC schedule](persistent-kernels.md) |
| [Fusion](../../techniques/kernel-fusion.md) | [NVFP4 GEMM/SwiGLU](kernel-fusion.md) |
| [Epilogue fusion](../../techniques/epilogue-fusion.md) | [SM100 CUTLASS builder](epilogue-fusion.md) |
| [Fine-grained scaling](../../techniques/fine-grained-quantization.md) | [Packed scale descriptors](fine-grained-quantization.md) |
| [Warp specialization](../../techniques/warp-specialization.md) | [tcgen05/TMEM role split](warp-specialization.md) |

Target-specific kernel and language pages, including
[DeepGEMM](deepgemm.md), [Triton](triton.md), [NVFP4 GEMM](nvfp4-gemm.md)
and [FlashAttention-4](flash-attention-4.md), live beside this entry. Their
shared IDs remain useful for principles or cross-target dispatch boundaries.

Shared descriptions of [tcgen05](../../hardware/tcgen05-mma.md),
[TMEM](../../hardware/tmem.md), [TMA](../../hardware/tma.md) and
[CLC](../../hardware/clc.md) retain their cited sources. Exact instructions,
shapes, dtype and `a`/`f` target requirements must still be checked. This
route selects relevant knowledge; it does not certify every snippet as runnable.

## Evidence continuity

Original page IDs, source PRs, pinned assets and benchmark claims remain
traceable. GPU/shape/dtype/software conditions stay attached to each original
claim. No B200 execution or new B200 speedup is claimed for this refactor.

## Evidence levels

Keep official/source confidence separate from device validation. Record
compilation, numerical correctness, profiler collection and measured benefit
independently, with the exact source, build command and environment. A missing
receipt is an evidence gap. See [the validation contract](../../../references/target-routing.md).

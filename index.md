# Device-aware GPU Kernel Optimization Knowledge Base

Start with [device-aware routing](references/target-routing.md); this wider corpus index also includes explicitly scoped comparison material.

> Shared guidance and a retained NVIDIA Blackwell/Hopper implementation corpus,
> with explicit routes for SM87, SM89, SM90, SM100, SM103, and SM110.
> Optimized for LLM agent retrieval. See [CLAUDE.md](CLAUDE.md) for schema and conventions.
> **For Claude Code agents**: this repository is a Claude Code skill — see [SKILL.md](SKILL.md).

## Recommended Query Tools (for LLM agents)

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for 10 worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Browse exact, family-only, or unknown architecture evidence | [queries/by-architecture.md](queries/by-architecture.md) |
| Fix a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |
| Write a specific kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Use a specific language/DSL | [queries/by-language.md](queries/by-language.md) |

## Hardware Features

- [hw-tcgen05-mma](wiki/hardware/tcgen05-mma.md) — Shared fifth-generation MMA execution contract
- [hw-tmem](wiki/hardware/tmem.md) — Shared Tensor Memory ownership and lifetime contract
- [hw-clc](wiki/hardware/clc.md) — Cluster Launch Control (dynamic tile scheduling)
- [hw-tma](wiki/hardware/tma.md) — Tensor Memory Accelerator (async bulk loads)
- [hw-2sm-cooperative](wiki/architectures/sm100/2sm-cooperative.md) — SM100 two-CTA cooperative MMA
- [hw-nvfp4](wiki/architectures/sm100/nvfp4.md) — SM100 NVFP4 and block-scaled narrow precision
- [hw-pdl-gdc](wiki/hardware/pdl-gdc.md) — Programmatic Dependent Launch / Grid Dependency Control

## Optimization Techniques

- [technique-warp-specialization](wiki/techniques/warp-specialization.md) — Warp role assignment
- [technique-persistent-kernels](wiki/techniques/persistent-kernels.md) — Persistent kernels and work scheduling
- [technique-swizzling](wiki/techniques/swizzling.md) — Shared memory swizzling
- [technique-pipeline-stages](wiki/techniques/pipeline-stages.md) — Software pipelining
- [technique-epilogue-fusion](wiki/techniques/epilogue-fusion.md) — Fusing epilogue with mainloop
- [technique-tile-scheduling](wiki/techniques/tile-scheduling.md) — Tile scheduling strategies
- [technique-double-buffering](wiki/techniques/double-buffering.md) — Double/multi-buffering
- [technique-software-exp](wiki/architectures/sm100/software-exp.md) — FlashAttention-4 software exponential
- [technique-fine-grained-quantization](wiki/techniques/fine-grained-quantization.md) - Fine-grained FP8/FP4 quantization
- [technique-vectorized-loads](wiki/techniques/vectorized-loads.md) — Wide vectorized loads and cache policies

## Kernel Case Studies

- [kernel-flash-attention-4](wiki/architectures/sm100/flash-attention-4.md) — FlashAttention-4 (up to 1613 TFLOPS on B200 in the paper's benchmark sweep)
- [kernel-deepgemm](wiki/kernels/deepgemm.md) — DeepGEMM FP8 GEMM (1550 TFLOPS on H800)
- [kernel-flashmla](wiki/kernels/flashmla.md) — FlashMLA sparse/dense MLA decoding
- [kernel-nsa](wiki/kernels/nsa.md) — Native Sparse Attention (9x fwd speedup)
- [kernel-gated-delta-net](wiki/architectures/sm90/gated-delta-net.md) — H200 Gated Delta Net projection fusion
- [kernel-nvfp4-gemm](wiki/architectures/sm100/nvfp4-gemm.md) — NVFP4 GEMM from GPU Mode hackathon
- [kernel-nvfp4-gemv](wiki/architectures/sm100/nvfp4-gemv.md) — NVFP4 batched GEMV optimization
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md) — Grouped GEMM for MoE
- [kernel-fused-moe](wiki/kernels/fused-moe.md) — Portable fused-MoE boundary; target implementations are separate

## Problem → Solution Patterns

- [pattern-low-sm-utilization](wiki/patterns/low-sm-utilization.md) — SM utilization is low
- [pattern-memory-bound](wiki/patterns/memory-bound.md) — Kernel is memory bandwidth limited
- [pattern-register-pressure](wiki/patterns/register-pressure.md) — Too many registers → low occupancy
- [pattern-compute-bound](wiki/patterns/compute-bound.md) — Not reaching peak FLOPS
- [pattern-tail-effect](wiki/patterns/tail-effect.md) — Last wave underutilizes GPU

## Languages & DSLs

- [lang-cute-dsl](wiki/architectures/sm100/cute-dsl.md) — CuTe DSL for SM100
- [lang-cuda-cpp](wiki/architectures/sm100/cuda-cpp.md) — CUDA C++ with SM100 inline PTX
- [lang-ptx](wiki/architectures/sm100/ptx.md) — PTX instructions for SM100
- [lang-triton](wiki/architectures/sm90/triton.md) — Triton on Hopper
- [lang-sm100-triton](wiki/architectures/sm100/triton.md) — Triton 3.6 Blackwell backend

## Migration Guides

- [migration-wgmma-to-tcgen05](wiki/architectures/sm100/wgmma-to-tcgen05.md) — Hopper wgmma → Blackwell tcgen05
- [migration-register-to-tmem](wiki/architectures/sm100/register-to-tmem.md) — Register accumulators → TMEM

## Source Repositories

| Repository | Focus |
|---|---|
| [NVIDIA/cutlass](queries/by-repo.md#nvidiacutlass) | CUTLASS 4.x Blackwell support |
| [sgl-project/sglang](queries/by-repo.md#sgl-projectsglang) | SGLang Blackwell integration |
| [vllm-project/vllm](queries/by-repo.md#vllm-projectvllm) | vLLM Blackwell support |
| [flashinfer-ai/flashinfer](queries/by-repo.md#flashinfer-aiflashinfer) | FlashInfer Blackwell kernels |
| [pytorch/pytorch](queries/by-repo.md#pytorchpytorch) | PyTorch/Inductor Blackwell |

## Competitions

- [GPU Mode NVFP4 Hackathon](sources/contests/gpu-mode-nvfp4/) — 4 NVFP4 kernel challenges on B200
- [FlashInfer MLSys 2026](sources/contests/flashinfer-mlsys26/) — MoE, Sparse Attention, GatedDeltaNet

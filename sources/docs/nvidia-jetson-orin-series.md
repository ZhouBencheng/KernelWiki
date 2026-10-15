---
id: doc-nvidia-jetson-orin
title: "Jetson AGX Orin: Ampere (GA10B) integrated GPU, compute capability 8.7"
url: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
source_category: official-doc
architectures: [sm87]
targets: [sm_87]
tags: [cuda-cpp]
retrieved_at: 2026-10-13
---

# Jetson AGX Orin: Ampere iGPU, compute capability 8.7

The NVIDIA Jetson AGX Orin series uses the Ampere-generation **GA10B** integrated
GPU. The CUDA device reports **compute capability 8.7** (`sm_87`). This is an
Ampere-family capability distinct from the desktop Ampere CC 8.0 (`sm_80`, GA100)
and CC 8.6 (`sm_86`, GA10x): per-capability feature sets and resource limits are
not interchangeable, so an `sm_80`/`sm_86` tuning note is not automatically valid
for `sm_87`.

Orin's GPU is an **integrated** device sharing physical memory with the CPU
(`integrated = 1`, `managed = 1`). There is no separate discrete-GPU VRAM pool;
memory-bandwidth and residency assumptions carried from a discrete board do not
transfer. The compiler target is the baseline `sm_87`; there is no
architecture-specific `sm_87a` or family `sm_87f` target.

The product page lists module configurations and marketing specifications. A
marketing memory-bandwidth headline is not a measured figure for any particular
run, and the exact module SKU is not identified by a generic CUDA device name.
This retrieval does not advance the corpus refresh cutoff.

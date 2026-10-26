---
id: doc-nvidia-ada-tuning
title: "NVIDIA Ada tuning guide"
url: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
source_category: official-doc
architectures: [sm89]
tags: [cuda-cpp, register-budgeting, shared-memory-optimization]
retrieved_at: 2026-10-27
---

# NVIDIA Ada tuning guide

Ada's occupancy limits include 48 resident warps per SM, 24 blocks per SM,
a 64K 32-bit register file and 100 KB shared-memory capacity per SM. Per-block
opt-in shared memory is a separate limit. The guide emphasizes coalescing,
reuse and resource-aware launch configuration; the optimal launch remains
kernel-dependent. These limits do not establish RTX 4090 SKU SM count, live
clock, measured bandwidth or a B200-equivalent tile choice.

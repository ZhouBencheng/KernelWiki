---
id: doc-nvidia-thor-targets
title: "CUDA compiler targets for Thor: baseline and architecture-specific"
url: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-compiler-driver-nvcc/index.html
source_category: official-doc
architectures: [sm110, sm110a]
targets: [sm_110, sm_110a, sm_110f]
tags: [cuda-cpp, ptx]
retrieved_at: 2026-10-11
---

# CUDA compiler targets for Thor

The archived CUDA 13.0 NVCC manual, section 5.7.3.1, lists
`compute_110`, `compute_110a`, `sm_110` and `sm_110a` as supported targets.
This is compiler documentation, not a test of an installed toolkit.

The [CUDA 13.0 Programming Guide, sections 20.1.1–20.1.3](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#feature-availability)
distinguishes baseline, family-specific (`f`) and architecture-specific (`a`)
feature sets. Architecture-specific code requires an explicit target and is
restricted to its exact compute capability. A device reporting CC 11.0 does
not tell the caller which feature set a binary was compiled for.

Upstream usage is recorded separately in
[FlashInfer PR 1608](../prs/flashinfer/PR-1608.md): its `flashinfer/jit/core.py`
diff adds `-gencode=arch=compute_110a,code=sm_110a` for the Thor path.
This initial integration does not establish support for every backend.

This focused source retrieval does not advance the corpus refresh cutoff.

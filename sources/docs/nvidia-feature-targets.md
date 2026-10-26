---
id: doc-nvidia-feature-targets
title: "CUDA feature-set compiler targets"
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
source_category: official-doc
architectures: [sm90, sm100, sm103, sm110, sm120]
tags: [cuda-cpp, ptx]
retrieved_at: 2026-10-27
---

# CUDA feature-set compiler targets

The compiler distinguishes baseline, family-specific (`f`) and
architecture-specific (`a`) feature targets. Runtime compute capability is a
device identity, not a suffix selection. A higher SM number is not permission
to reuse an architecture-specific binary or instruction form. Check the exact
instruction's supported targets and the compiler/assembler actually used by
the selected library, then build and execute the kernel separately.

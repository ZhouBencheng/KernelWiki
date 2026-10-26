---
id: technique-sm100-swizzling
title: "SM100 implementation: CuTe shared-memory swizzle coupling"
type: technique
architectures: [sm100]
applicable_to: [sm100]
tags: [swizzling, shared-memory-optimization, tma, tcgen05]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tma, hw-sm100-tcgen05-mma]
related: [technique-swizzling, technique-sm100-pipeline-stages]
sources: [doc-cutlass-cute-dsl, blog-tcgen05-tutorial, doc-ptx-isa-sm100]
---

# SM100 CuTe shared-memory swizzle coupling

The official CuTe DSL dense-GEMM documentation materializes a logical layout
and its selected swizzle together:

```python
sA = smem.allocate_tensor(
    a_smem_layout.outer, swizzle=a_smem_layout.inner, dtype=ab_dtype
)
sB = smem.allocate_tensor(
    b_smem_layout.outer, swizzle=b_smem_layout.inner, dtype=ab_dtype
)
```

The helper chooses supported atoms from the actual major dimension and
divisibility constraints. A 128-byte swizzle is neither universally mandatory
nor universally optimal for tcgen05 inputs; validate the exact descriptor and
physical layout.

---
id: technique-sm100-cache-policy
title: "SM100 implementation: explicit attention cache operation"
type: technique
architectures: [sm100]
applicable_to: [sm100]
tags: [cache-policy, vectorized-loads]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [technique-cache-policy, kernel-nvfp4-gemv]
sources: [pr-flashinfer-1039]
---

# SM100 explicit attention cache operation

FlashInfer PR 1039 contains a concrete cache-operation choice in an SM100
attention load path. This contiguous excerpt performs guarded 16-byte
`cp.async` copies with CUTLASS's `CacheOperation::Always` policy:

```cpp
Vec* dst_ptr = &dst(i);
const Vec* src_ptr = &src(i);
bool guard = elem_less(cc, limitQ);
cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
    dst_ptr, src_ptr, guard);
```

The source does not establish that this policy is best for another stream,
kernel or target. Measure it against the same mapping with a different policy.

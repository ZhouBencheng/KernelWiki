---
id: technique-vectorized-loads
title: "Wide Vectorized Loads and Cache Policies"
type: technique
architectures: []
applicable_to: [portable]
tags: [vectorized-loads, cache-policy, register-budgeting]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [pattern-memory-bound, technique-cache-policy]
sources: [pr-cccl-3517]
---

# Wide Vectorized Loads and Cache Policies

Memory-bound kernels can reduce load-instruction overhead and improve transaction formation by moving aligned contiguous words per thread. The legal and useful width depends on pointer alignment, per-lane address mapping, register pressure, and how the compiler lowers the operation.

## Evaluation procedure

1. Confirm from profiling and a byte-count roofline that the kernel is bandwidth-limited.
2. Prove alignment and contiguous per-thread access for every supported shape, including tails.
3. Compare scalar and candidate vector widths while holding work mapping and cache policy fixed.
4. Inspect generated instructions and measure memory sectors, transactions, replay, and registers.
5. Only then test cache operators separately for streaming and reused inputs.

Wider source syntax does not guarantee a single wider hardware transaction, and an excessive width can reduce occupancy or complicate tails. Treat register limits and cache operators as independent tuning dimensions rather than part of a universal recipe.

CCCL PR 3517 shows the alignment guard required before its CUB block-load path
reinterprets a scalar pointer as a vector pointer. This is a contiguous excerpt
from the captured upstream patch:

```cpp
using vector_t = typename CubVector<device_word_t, vector_size>::Type;
if (reinterpret_cast<uintptr_t>(block_src_ptr) % alignof(vector_t) == 0)
{
  vector_t vec_items[vectors_per_thread];
  const vector_t* vec_ptr = reinterpret_cast<const vector_t*>(block_src_ptr)
                            + linear_tid * vectors_per_thread;
}
```

The complete upstream branch includes the vector-load loop and fallback path;
the excerpt alone is not a replacement for `BlockLoad`. B200 NVFP4 timing
evidence remains with the target-specific `kernel-nvfp4-gemv` page.

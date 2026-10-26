---
id: lang-triton
title: Triton on Hopper
type: language
tags: [triton, gated-delta-net, kernel-fusion]
related: [kernel-gated-delta-net, lang-sm100-triton]
sources: [pr-sglang-21019]
reproducibility: snippet
architectures: [sm90]
applicable_to: [sm90]
confidence: source-reported
---

# Triton on Hopper

Triton is used for Hopper kernels beyond matrix multiplication, including
layout-sensitive projection fusion. Target selection still requires checking
the generated code and numerical contract; a `tl.dot` surface call does not by
itself prove WGMMA lowering or performance.

SGLang PR 21019 reports its projection fusion on H200. This contiguous excerpt
from the retained Triton file shows the interleaved-input stores:

```python
tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
```

The complete file supports both the interleaved Qwen3-Next layout and the
contiguous Qwen3.5 layout. Layout identity is a correctness condition, not a
performance-only choice. For Blackwell backend features and the retained SM100
attention example, use the [SM100 Triton page](../sm100/triton.md).

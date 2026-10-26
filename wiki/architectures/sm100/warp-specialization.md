---
id: technique-sm100-warp-specialization
title: "SM100 implementation: tcgen05 warp specialization"
type: technique
architectures: [sm100]
applicable_to: [sm100]
tags: [warp-specialization, tcgen05, tmem]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-sm100-tmem, hw-sm100-tcgen05-mma]
related: [technique-warp-specialization, technique-sm100-pipeline-stages]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, pr-flashinfer-1039, blog-colfax-cutlass]
artifact_dir: artifacts/kernels/warp-specialization
---

# SM100 tcgen05 warp specialization

On SM100, a `tcgen05.mma` operation is initiated by one thread, while TMEM
allocation and TMEM load/store operations remain warp-collective. This differs
from Hopper WGMMA's four-warp warpgroup issue model; port the complete
participation and accumulator contract rather than copying role counts.

This contiguous excerpt from FlashInfer PR 1039's retained SM100 attention
mainloop shows ownership transfer around one QK MMA step:

```cpp
q_index = pipeline_q_consumer_state.index();
pipeline_q.consumer_wait(pipeline_q_consumer_state);
++pipeline_q_consumer_state;
pipeline_s0.producer_acquire(pipeline_s0_producer_state);
gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS0);
pipeline_s0.producer_commit(pipeline_s0_producer_state);
```

The excerpt does not imply a universal warp count; in particular, there is no
architectural canonical 16-warp CTA. Budget registers, shared memory, TMEM
columns and resident CTAs for the actual role split.

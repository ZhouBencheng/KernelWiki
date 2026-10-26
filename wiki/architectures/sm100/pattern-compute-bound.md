---
id: pattern-sm100-compute-bound
title: "SM100 diagnosis: compute path below roofline"
type: pattern
architectures: [sm100]
applicable_to: [sm100]
tags: [tcgen05, 2sm-cooperative, pipeline-stages, warp-specialization]
symptoms: [compute-bound, low-tensor-core-utilization, pipeline-stalls]
candidate_techniques: [hw-2sm-cooperative, technique-sm100-pipeline-stages, technique-sm100-warp-specialization, technique-sm100-epilogue-fusion, technique-software-exp]
related: [pattern-compute-bound, kernel-flash-attention-4]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-flash-attention-4]
---

# SM100 compute-path diagnosis

After confirming a compute-bound SM100 path, distinguish tcgen05 waits,
non-matmul work, insufficient tile intensity and serialized epilogues. Two-CTA
2-SM MMA, deeper stages, specialized roles and software exponential are separate
candidates with different correctness and resource contracts.

FlashAttention-4 reports that Blackwell tensor throughput grew faster than the
special-function throughput in its comparison. Its software exponential is a
source-reported response for that attention workload, not a general instruction
replacement. 2-SM/two-CTA execution likewise requires a compatible cluster launch
and operand layout.

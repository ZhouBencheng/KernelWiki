---
id: lang-sm100-triton
title: "SM100 implementation: Triton 3.6 Blackwell backend"
type: language
tags: [triton, attention, tmem, tcgen05]
related: [lang-triton, kernel-sparse-mla, hw-sm100-tcgen05-mma]
sources: [doc-triton-3.6-blackwell, pr-vllm-34597, pr-sglang-22079]
reproducibility: snippet
architectures: [sm100]
applicable_to: [sm100]
confidence: verified
evidence_basis:
  - evidence_type: official-doc
    source_id: doc-triton-3.6-blackwell
  - evidence_type: upstream-code
    source_id: pr-vllm-34597
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Triton 3.6 on SM100

Triton 3.6 includes compiler infrastructure for Blackwell TMEM and `tcgen05`,
plus Gluon multi-CTA and scaled-MMA work. This does not mean every `tl.dot`
kernel lowers identically or performs like a specialized CuTe/CUDA kernel.

The following non-contiguous call sites come from the retained vLLM PR 34597
artifact. Consult the pinned full file for control flow and shapes:

```python
qk = tl.dot(q, k.to(q.dtype))
qk += tl.dot(qpe, kpe.to(qpe.dtype))
acc += tl.dot(p.to(v.dtype), v)
```

Pinned file: [`triton_decode_attention.py`](../../../artifacts/prs/vllm/PR-34597/key-files/vllm/v1/attention/ops/triton_decode_attention.py).

The release notes establish backend features; the downstream source establishes
one SM100 use. Neither proves a particular `tcgen05.mma` form for every call.
The version claim remains pinned in [`data/version-claims.yaml`](../../../data/version-claims.yaml).

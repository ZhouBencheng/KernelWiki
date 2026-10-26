---
id: kernel-sm100-fp8-block-scale-gemm
title: "SM100 implementation: FP8 block-scale GEMM"
type: kernel
architectures: [sm100]
applicable_to: [sm100]
tags: [gemm, fp8, block-scale, fine-grained-quantization, tcgen05]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp]
related: [kernel-fp8-block-scale-gemm, technique-sm100-fine-grained-quantization, hw-sm100-tcgen05-mma]
sources: [pr-cutlass-2139, doc-cutlass-changelog-sm100]
performance_claims: []
---

# SM100 FP8 block-scale GEMM

CUTLASS PR 2139 wires SM100 scale layouts into its collective builder. This
contiguous configuration fragment is target-specific:

```cpp
using ScaleConfig = decltype(
    cutlass::detail::sm100_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100
>::CollectiveOp;
```

Use the retained PR artifact as the executable source. The scale layouts,
alignment, descriptor and schedule are part of this SM100 implementation and
must not enter the H100 route.

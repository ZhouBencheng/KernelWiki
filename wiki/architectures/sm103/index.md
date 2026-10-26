---
id: hw-target-sm103
title: "Blackwell / B300 and GB300: target guidance"
type: hardware
architectures: [sm103]
applicable_to: [sm103]
tags: [cuda-cpp, ptx, tcgen05]
confidence: source-reported
related: [pattern-target-triage, technique-double-buffering, technique-pipeline-stages, technique-persistent-kernels]
sources: [doc-nvidia-gpu-targets, doc-nvidia-feature-targets, doc-ptx-isa-sm100]
aliases: []
---

# Blackwell / B300 and GB300

Runtime compute capability: **10.3**. Select this route with
`python3 scripts/query.py --target sm103`. Selection indicates relevant guidance,
not a promise that every referenced instruction or kernel was executed.

## Select the exact implementation

B300/GB300 is CC 10.3, not B200's CC 10.0. Check the selected compiler and
library for the exact `sm_103` feature target and instruction form. A shared
Blackwell mechanism does not transfer B200's SKU resources or benchmarks.

Use `--architecture sm103` to inspect retained exact-target source evidence.
A B200 implementation may be a useful comparison, but it must not silently
become the selected recipe. There is no new B300 kernel execution evidence in
this change; common principles and official target facts are available while
specific implementations are validated.

## Evidence levels

Keep official/source confidence separate from device validation. Record
compilation, numerical correctness, profiler collection and measured benefit
independently, with the exact source, build command and environment. A missing
receipt is an evidence gap. See [the validation contract](../../../references/target-routing.md).

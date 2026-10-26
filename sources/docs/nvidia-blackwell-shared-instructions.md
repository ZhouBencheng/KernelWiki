---
id: doc-nvidia-blackwell-shared-instructions
title: "PTX tcgen05 and TMEM target contracts"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm100, sm103, sm110]
tags: [ptx, tcgen05, tmem]
retrieved_at: 2026-10-27
---

# PTX tcgen05 and TMEM target contracts

The current PTX Target ISA Notes list SM100 and SM110 family/architecture
variants for tcgen05 allocation and floating matrix operations. Feature
qualifiers have narrower lists: dtype, shape, block-scaling and optional
operands must be checked separately. The shared contract concerns TMEM-backed
accumulators, collective allocation and explicit completion/lifetime rules.
It does not establish a common physical memory capacity, a compatible B200
binary, or a speedup. SM120 is not in these tcgen05/TMEM target lists.

The SM100 512-column/128-lane view is specifically described for SM100a/SM100f;
keep it labeled instead of inferring another GPU's physical bytes per SM.

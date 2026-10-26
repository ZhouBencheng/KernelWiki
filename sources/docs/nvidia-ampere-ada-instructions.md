---
id: doc-nvidia-ampere-ada-instructions
title: "PTX ordinary asynchronous copy and warp MMA target scope"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm87, sm89, sm110]
tags: [ptx, cp-async, warp-mma]
retrieved_at: 2026-10-27
---

# PTX ordinary asynchronous copy and warp MMA target scope

Ordinary `cp.async` copies global memory to shared memory and requires SM80 or
higher. Its commit/wait-group and alignment rules differ from bulk/tensor-copy
instructions. `mma.sync` is a warp-collective instruction family; BF16 and TF32
forms have SM80-or-higher target requirements, while exact shapes, data packing
and accumulator types remain instruction-specific. These facts support the
SM87, SM89 and SM110 routes, not every operation under the label Tensor Core.
TMA/bulk tensor copies, WGMMA and tcgen05 have separate target requirements.
Read the corresponding Target ISA Notes before selecting a form.

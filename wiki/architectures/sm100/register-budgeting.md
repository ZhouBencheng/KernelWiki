---
id: technique-sm100-register-budgeting
title: "SM100 implementation: register-cap experiment"
type: technique
architectures: [sm100]
applicable_to: [sm100]
tags: [register-budgeting, register-reuse]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [technique-register-budgeting, kernel-nvfp4-gemv]
sources: [blog-amandeep-nvfp4, blog-yue-nvfp4, blog-simon-nvfp4-gemv]
---

# SM100 register-cap experiment

A reproducible SM100 comparison keeps source and launch shape fixed and records
resource output for each build, for example:

```bash
nvcc -arch=sm_100a -Xptxas=-v kernel.cu -o kernel-default
nvcc -arch=sm_100a -Xptxas=-v --maxrregcount=80 kernel.cu -o kernel-r80
nvcc -arch=sm_100a -Xptxas=-v --maxrregcount=64 kernel.cu -o kernel-r64
```

Amandeep Singh reports that changing the cap from 80 to 64 did not affect the
tested kernel because its allocation was already below the cap. That is a
control, not evidence that either cap is generally optimal. Record spills,
occupancy and time on the actual build. The former fixed claim that 32 registers
guarantees four blocks per SM remains rejected: residency depends on
the complete compiled resource and launch profile.

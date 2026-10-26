---
id: doc-cuda-portable-practices
title: "CUDA C++ Best Practices: portable optimization principles"
url: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
source_category: official-doc
architectures: []
tags: [cuda-cpp, data-reuse, shared-memory-optimization]
retrieved_at: 2026-10-27
---

# CUDA C++ Best Practices: portable optimization principles

The official guide supports profiling first, coalesced global accesses, reducing
redundant transfers, resource-aware launch configuration and correctness checks.
Shared-memory reuse, asynchronous copies and concurrent work have implementation
constraints: a portable principle does not make every instruction portable.
Use the actual device resources and selected instruction documentation. A
benchmark on one GPU and shape does not establish a benefit elsewhere.

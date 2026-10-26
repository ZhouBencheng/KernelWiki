---
id: technique-tile-scheduling
title: "Tile Scheduling Strategies"
type: technique
architectures: []
applicable_to: [portable]
tags: [tile-scheduling, persistent-kernel]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [technique-persistent-kernels, pattern-low-sm-utilization, technique-sm100-persistent-kernels]
sources: [doc-cuda-portable-practices]
---

# Tile scheduling strategies

Tile scheduling maps logical work tiles to CTA or cluster IDs and determines the order in which a persistent kernel processes additional work. It affects operand reuse, load balance, and the size of the last partially occupied wave.

## Static raster

A row-major mapping is easy to validate and is an appropriate baseline:

```cpp
struct TileCoord { int m; int n; };

__device__ TileCoord row_major_tile(int linear_id, int tiles_n) {
  return {linear_id / tiles_n, linear_id % tiles_n};
}
```

A blocked or swizzled raster may improve locality when nearby tiles reuse A or B data, but the useful order depends on matrix layout, tile shape, cache residency, cluster shape, and traversal direction. It cannot be assigned a fixed cache-hit improvement without a benchmark.

## Persistent scheduling

A software persistent scheduler repeatedly maps acquired IDs to work. The
launch grid and the program's ID-to-tile mapping still define the work domain;
target-specific acquisition mechanisms must preserve exactly-once coverage.
The SM100 CLC realization remains in the target-specific persistent-kernel page.

## Selection checks

- Compare schedules with identical tile shapes and launch/resource settings.
- Report the exact GPU, SM count, grid size, raster, cluster shape, and matrix dimensions.
- Separate locality effects from tail effects and load imbalance.
- Do not claim that CLC removes the final shortage of parallel work; when fewer independent tiles remain than SMs, some SMs must be idle.

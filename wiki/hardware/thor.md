---
id: hw-thor
title: "Thor: target selection and reuse of Blackwell guidance"
type: hardware
architectures: [sm110, sm110a]
targets: [sm_110, sm_110a, sm_110f]
evidence_status: executed
tags: [cuda-cpp, ptx, tcgen05, tmem, nvfp4]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-thor-targets
    evidence_type: official-doc
  - source_id: pr-flashinfer-1608
    evidence_type: upstream-code
related: [hw-tcgen05-mma, hw-tmem, hw-nvfp4]
sources: [doc-nvidia-thor-targets, pr-flashinfer-1608, pr-flashinfer-2028, pr-flashinfer-2190, pr-flashinfer-2538, pr-flashinfer-2543, pr-sglang-18787]
aliases: [Thor, NVIDIA Thor, sm_110, sm_110a]
version_sensitive: vs-cuda-thor-targets
---

# Thor: target selection and reuse of Blackwell guidance

Use `--architecture Thor` or `--architecture sm110` to retrieve Thor device
references. Use `--architecture sm110a` for pages with explicit coverage of
the architecture-specific target. The aliases remain distinct; the existing
`Blackwell` and `B200` aliases still resolve to `sm100`.

## Select and check the target

The CUDA 13.0 compiler manual lists both `sm_110` and `sm_110a`.
The `a` suffix selects architecture-specific features; CC 11.0 alone does not
select it. Check the installed compiler before building:

```bash
nvcc --version
nvcc --list-gpu-arch
nvcc --list-gpu-code
# Baseline cubin, using your kernel's self-checking harness:
nvcc -gencode arch=compute_110,code=sm_110 check.cu -o check-sm110
# Separate architecture-specific cubin:
nvcc -gencode arch=compute_110a,code=sm_110a check.cu -o check-sm110a
```

Run each binary and check its results. A target list may omit suffixed targets;
compilation provides the direct check. Target acceptance and a simple kernel
pass do not validate every instruction available under that target.

## Reuse concepts, validate the kernel

Start with the existing [tcgen05](tcgen05-mma.md), [TMEM](tmem.md) and
[NVFP4](nvfp4.md) explanations for terminology. Those pages describe SM100/B200;
their hardware capacities, instruction variants, snippets and performance
results are not Thor validation. Confirm the exact instruction and library
path, then test correctness on Thor before tuning. Do not transfer B200 tile
sizes or throughput expectations without measurements.

The linked PRs are historical integration evidence:

- FlashInfer 1608 introduces Thor targets; 2028 uses an `11.0f` wheel target,
  which is not evidence for `sm110a`.
- FlashInfer 2190 adjusts SM100 configuration reuse on SM110. PR 2538 changes
  backend requirement decorators and a dtype test skip; inspect the diff
  before treating its summary as a backend support matrix.
- FlashInfer 2543 addresses a bundled `ptxas` target rejection. Check the
  assembler actually selected by the framework; a system `nvcc` version alone
  does not identify it. This historical workaround is not a blanket setup step.
- SGLang 18787 expands a capability gate to include major version 11;
  that gate is not an NVFP4 correctness test.

## Hardware evidence, separate from source confidence

On 2026-10-11, a NVIDIA Thor reported CC 11.0 with driver 595.78.
CUDA 13.2 `nvcc V13.2.86` compiled separate native `sm_110` and `sm_110a`
binaries. Both checked 257 integer outputs successfully. Device code reported
`__CUDA_ARCH__=1100` for both; `__CUDA_ARCH_SPECIFIC__` was absent for the
baseline target and 1100 for the `a` target. GPU execution used a shared
advisory lock. No timing or speedup claim was measured. This earlier integer
self-check covered only the base and `a` targets; the root's later three-target
`probe-build-02` float copy/reduce build — which adds `sm_110f` — is the separate
"Three native targets" evidence below, not a revision of these counts.

This is only target-compilation and basic execution evidence. Thor execution
of tcgen05, TMEM, NVFP4 and the linked framework kernels remains **pending**.
`confidence: verified` above means official-document plus upstream-code
evidence, not that those features or every linked snippet ran on Thor.

## Three native targets and per-target evidence scope

The root built and checked **three** native targets from the same source —
`sm_110`, `sm_110f`, and `sm_110a` — not only base and `a`. Receipts:
`probe-build-02/build-manifest.json` (exact `nvcc` argv, source/binary hashes) and
`validation-check/sm_110-lineinfo-check.stdout.txt`,
`validation-check/sm_110f-lineinfo-check.stdout.txt`,
`validation-check/sm_110a-lineinfo-check.stdout.txt`. All three report
`__CUDA_ARCH__=1100`, and the compile-time macros separate them:

- **base `sm_110`** defines *neither* the family nor the architecture-specific macro;
- **family `sm_110f`** defines *only* the family macro;
- **arch `sm_110a`** defines *both* the family and the architecture macros (each 1100).

So an `a` target implies the family (`a`-plus-family); it is not interchangeable
with `f`, and none of this follows from CC 11.0. Read the page-wide
`evidence_status: executed` as *per-target compile + basic self-check* evidence
(the integer/SIMT/TMA probe), scoped to each probe — not as execution of tcgen05,
TMEM or NVFP4 on any target.

## Evidence roots (external to the skill tree)

The receipts cited on this page live **outside** the delivered skill; `data/thor-evidence.yaml`
is the in-corpus registry that records them. Their actual paths (campaign root
`/home/cilab/zbc/thor-skills-sm110-20261012`):

- **Probe build/check/profile:** `probe-build-02/build-manifest.json`,
  `validation-check/{sm_110,sm_110f,sm_110a}-lineinfo-check.stdout.txt`,
  `validation-profile/manifest.json` — the compile/self-check/NCU receipts.
- **External immutable FFN campaign (not relative to this skill):**
  `/home/cilab/zbc/embodiinfer-operators/profile/embodiinfer-thor-m1-ffn-20261011/REPORT.md`
  and its `reports/` directory.

A delivered copy of this skill cannot resolve these paths on its own; when the
external receipt is not available at analysis time, treat that evidence as
**unavailable** and say so — do not substitute prose for the receipt.

## Same-precision M=1 workloads: measure, do not transfer

Thor-targeted M=1 FFN/GEMV questions must preserve the *requested* precision. The
NVFP4 GEMV blog examples that surface for these queries (`blog-simon-nvfp4-gemv`,
`blog-amandeep-nvfp4`) use FP4 weights, FP8 scales and FP16 output — they do **not**
satisfy a BF16 or FP32 input/weight/output contract, and their tiles and speedups do
not transfer. Keep them as explicitly limited conceptual references.

**BF16 M=1 FFN (dtype vs. accumulator).** Preserve BF16 operands and output; the
accumulator precision is a *separate* choice. The selected historical candidate for
this exact shape **(M, N, K) = (1, 3584, 18944)** expands BF16 operands to FP32, uses
an explicit balanced FP32 reduction, and stores BF16 — external immutable campaign
`embodiinfer-thor-m1-ffn-20261011/REPORT.md`, sections "1. Approved replacement and
usage" and "4. Candidates and frozen numerical acceptance". Preserve the original
math-mode settings and validate against the frozen original output and numerical
contract; this candidate description does **not** establish the original
implementation's reduction tree — the original is `F.linear` with a main plus an
auxiliary split-K reduction (two NCU actions). A same-precision SIMT reduction is a
candidate to *measure*, preserving the correctness contract **without** assuming the
candidate and original reduction orders are identical. Compare the actual original
and candidate on this shape with **separate** correctness, ordinary event timing, and
NCU evidence (include the original auxiliary kernels, including its split-K
reduction); real model integration is a further gate. Do not promote a tile or
speedup from the quantized examples, and make no speedup claim without that timing.

**Original FP32 M=1 (no precision substitution).** When the contract is FP32,
preserve FP32 input, stored weights and output *separately* from the accumulation
and math-mode controls (gate/up `(M,N,K)=(1,11008,2048)`, down `(1,2048,11008)`). A BF16,
TF32 or NVFP4 result — or a TF32-accumulator description — does **not** establish a
replacement for the FP32 contract, and Tensor-Core availability alone is not a
speedup. **Do not substitute TF32 or NVFP4 for an FP32 request.** Evaluate a
same-contract candidate against the real original inputs/outputs, reject numerical
failures without loosening acceptance, and independently time the actual eager
execution path before promotion; retain the original if the benefit is unproven. The
original `highest` / `allow_tf32=False` setting, the eager-versus-graph boundary, and
the frozen FP32 acceptance (`atol=rtol=1e-5`) are recorded in the same external
`embodiinfer-thor-m1-ffn-20261011/REPORT.md`, sections "2. Environment, models and
authentic shapes", "4. Candidates and frozen numerical acceptance" and "5. Independent
operator timing and unselected paths"; they are not inferable from the BF16 record or
the TMA probe.

### Measured FFN gain does not establish model gain

The 2026-10-13 original-FP32 ActiveVLN experiment retained vendor up/down and
fused the gate GEMV with SiLU/multiply, reducing the complete FFN from five
kernels to three. For gate/up `(1,11008,2048)` and down `(1,2048,11008)`, a fixed
batch of 16 eager calls reusing the same weights measured 1.100115 ms versus
1.082108 ms per FFN: **1.64% lower latency**, B/C ratio 1.01664065 and descriptive
interval [1.01354657, 1.02029359]. This is a same-weight queue/cache experiment,
not the 36-layer model. The real initial-request model experiment subsequently
measured primary CPU-RGB-to-CPU-action B/C 1.00028814 with interval
[0.99515433, 1.00389403]; one process pair was slower. The predefined E2E gain
gate failed, so this candidate was **not integrated** despite passing model
correctness. Keep the scopes and all outcomes separate.

Its 40-register native gate also reported
`sm__warps_active.avg.pct_of_peak_sustained_active=110.96731945718507%` in NCU.
Preserve that raw anomaly and exclude it from physical achieved-occupancy
conclusions; do not clamp it or invent a denominator/cause. This does not make
every `pct_of_peak` value above 100 invalid: interpret each metric's definition.
The gate's 96.3366% ordinary PC long-scoreboard share is a **sample** fraction,
not a duration fraction; unavailable DRAM measurements are not zero and cannot
establish saturation. See the companion NCU skill's `references/09-common-issues.md`.

Evidence references for these observations are the exact receipt paths and SHA256 values
in the **2026-10-13 operator observations** comment block of
[`data/thor-evidence.yaml`](../../data/thor-evidence.yaml). That block specifies
the external campaign roots, performance/NCU reviews and E2E rejection evidence.
These workload observations do not advance any pending hardware-feature status.

## Device resources: measure this board, not B200 constants

Do not carry B200 SM counts, resident-warp counts, or bandwidth onto Thor. The
checked device reports (receipt `validation-check/sm_110-lineinfo-check.stdout.txt`,
each attribute with CUDA status 0): **20 SMs**; **1536 threads/SM**, which with warp
size 32 implies a ceiling of **48 resident warps/SM**; **32 MiB L2**; **228 KiB
shared memory/SM**; and separately **227 KiB opt-in shared memory/block**. These are
device *limits*, distinct from a kernel's achieved occupancy and its actual
register/shared-memory/block allocations.

A persistent-grid constant like "132 CTAs / 64 resident warps" cannot be justified
by an assumed B200 SM count — 64 warps/SM already exceeds this device's 48-warp
ceiling, and 132 CTAs is not categorically illegal but must be sized to *this* device
and the real kernel's constraints. Clock values in the receipt are device
**attributes**, not live telemetry, and the probe's own allocation is not a capacity.

The check records only a generic device name, **not** a unique board SKU, so the
board identity and live state stay unknown here. A streaming-memory roofline needs
the matching SKU plus the workload's traffic/cache assumptions and independent
measurements — the cache-reusing copy/reduce probe timing cannot establish it.
NVIDIA's [Jetson Thor product specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
list 273 GB/s for the AGX Thor Developer Kit / T5000 / T4000 configurations, but with
the SKU unidentified that headline is not measured bandwidth for this run.

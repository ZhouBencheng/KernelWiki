---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Worked Query Examples

Concrete patterns showing how to translate a user question into a navigation path and synthesize an answer.

---

## Example 1: "How do I write a fast GEMM kernel on B200?"

**Navigation path**:
1. `queries/by-kernel-type.md` → find `gemm` row → lists kernel pages
2. Read the shared boundary at `wiki/kernels/fp8-block-scale-gemm.md`, then the selected implementations under `wiki/architectures/sm100/`
3. Follow `sources:` to concrete PRs: `pr-cutlass-2139` (blockwise+groupwise GEMM), `pr-vllm-13798` (FP8 GEMM)
4. Read the common principles, then `wiki/architectures/sm100/warp-specialization.md` and `wiki/architectures/sm100/persistent-kernels.md`
5. For progression, cite the tcgen05 tutorial (`sources/blogs/tcgen05-tutorial.md`): Naive 17% → 128B-swizzled 3D TMA 46% → Pipelining 62% → Warp specialization 80% → 2-SM 86% → persistent static scheduling 98%

**Command**:
```bash
python3 scripts/query.py --target b200 --type kernel --tag gemm
```

---

## Example 2: "My kernel has low SM utilization on B200"

**Navigation path**:
1. `queries/by-problem.md` → find "low-sm-utilization" row
2. Common pattern: `wiki/patterns/low-sm-utilization.md`; SM100 diagnosis: `wiki/architectures/sm100/pattern-low-sm-utilization.md`
3. Candidate techniques: `technique-persistent-kernels`, `technique-tile-scheduling`, `hw-clc`
4. Read the CLC page: `wiki/hardware/clc.md` for code example
5. Cite: the tcgen05 tutorial showed 86% → 98% with static persistent scheduling. It presents CLC only as a possible follow-on exercise; use `wiki/hardware/clc.md` for the separate CLC mechanism.

**Command**:
```bash
python3 scripts/query.py --target b200 --symptom low-sm-utilization
python3 scripts/get_page.py pattern-sm100-low-sm-utilization --target b200
```

---

## Example 3: "Where is tcgen05.mma used in CUTLASS?"

**Navigation path**:
1. `queries/by-hardware-feature.md` → find `tcgen05` row → lists all relevant pages
2. `queries/by-repo.md` → NVIDIA/cutlass section
3. Cross-reference via tag filter

**Command**:
```bash
python3 scripts/query.py --tag tcgen05 --repo cutlass --limit 30
python3 scripts/grep_wiki.py "tcgen05\\.mma" --only sources
```

Tip: the `--tag` filter also accepts aliases, so `--tag UMMA` resolves to `tcgen05`.

---

## Example 4: "Show me the FlashAttention-4 implementation details"

**Navigation path**:
1. Direct: `wiki/architectures/sm100/flash-attention-4.md`
2. Performance: up to 1613 TFLOPS on B200 BF16 (71%) in the paper's sweep
3. Techniques used: ping-pong scheduling, software exp, 2-CTA backward
4. Follow sources → `sources/docs/flash-attention-4.md` (paper), `sources/blogs/flash-attention-4.md` (author blog)

**Command**:
```bash
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
```

---

## Example 5: "What's the difference between Hopper wgmma and Blackwell tcgen05?"

**Navigation path**:
1. `wiki/architectures/sm100/wgmma-to-tcgen05.md` — dedicated target-specific migration guide
2. `wiki/hardware/tcgen05-mma.md` — canonical reference for the new instruction
3. Contrast with Hopper behavior implicit in the migration page

**Command**:
```bash
python3 scripts/get_page.py migration-wgmma-to-tcgen05
python3 scripts/get_page.py hw-tcgen05-mma
```

---

## Example 6: "How did teams in the GPU Mode NVFP4 Hackathon win?"

**Navigation path**:
1. `sources/contests/gpu-mode-nvfp4/` (4 problems)
2. Each problem page records the organizer task definition, speed-of-light values when published, and a dated leaderboard snapshot; no page declares a per-submission listing
3. Read participant blogs: `sources/blogs/yue-nvfp4-hackathon.md`, `sources/blogs/amandeep-nvfp4-attempts.md`, `sources/blogs/simon-nvfp4-gemv.md`
4. Techniques: PTX-level control, cache policy differentiation, register budgeting

**Command**:
```bash
python3 scripts/query.py --type contest --tag nvfp4
python3 scripts/get_page.py contest-gpumode-p1
```

---

## Example 7: "Which Triton evidence applies to GatedDeltaNet on H100 versus Blackwell?"

**Navigation path**:
1. `wiki/architectures/sm90/gated-delta-net.md` and `wiki/architectures/sm90/triton.md` — retained H200 implementation evidence
2. `wiki/architectures/sm100/triton.md` — Blackwell compiler/backend evidence, not proof that the H200 GDN kernel is an SM100 recipe
3. Use `--architecture` separately when researching raw GDN sources

**Command**:
```bash
python3 scripts/query.py --target h100 "gated delta net" --language triton
python3 scripts/query.py --target b200 --language triton
```

---

## Example 8: "What are the memory-bound kernel optimization tricks on B200?"

**Navigation path**:
1. `wiki/patterns/memory-bound.md` — portable diagnosis; `wiki/architectures/sm100/pattern-memory-bound.md` — B200-specific candidates
2. `wiki/techniques/vectorized-loads.md` — wide loads + cache policies (covers `evict_first` / `no_allocate`)
3. `wiki/architectures/sm100/nvfp4-gemv.md` records the author's 2000 → 443 → 39 → 27 → 22.392 μs aggregate sequence and keeps the combined-edit attribution with the B200 implementation

**Command**:
```bash
python3 scripts/query.py --target b200 --symptom memory-bound
```

---

## Example 9: "Find all FlashInfer PRs for FP8 MoE"

**Command**:
```bash
python3 scripts/query.py --repo flashinfer --tag moe --limit 30
python3 scripts/query.py --repo flashinfer --tag fp8 --limit 30
python3 scripts/grep_wiki.py "fp8" "moe" --only sources --files-only
```

---

## Example 10: "What PTX instructions are unique to SM100?"

**Navigation path**:
1. `wiki/architectures/sm100/ptx.md` — direct reference
2. Cross-reference: tcgen05.alloc/mma/ld/st/dealloc/fence, clusterlaunchcontrol.try_cancel, cp.async.bulk.tensor multicast

**Command**:
```bash
python3 scripts/get_page.py lang-ptx --body-only
python3 scripts/grep_wiki.py "tcgen05" --only wiki --context 0
```

---

## Synthesis Pattern

For most questions, a high-quality answer follows this shape:

```
1. Topic framing (1-2 sentences, cite wiki page)
   → Pull from wiki/<section>/<topic>.md ## Overview

2. Technical mechanism (cite hardware + technique pages)
   → wiki/hardware/*.md for hw details
   → wiki/techniques/*.md for optimization patterns

3. Concrete code snippet (when the selected implementation page provides one)
   → A snippet's fence is structurally validated; compilation and execution
     require separate evidence

4. Performance evidence (cite performance_claims)
   → Always report gpu, dtype, shape, metric, value, source_id

5. References (cite source IDs)
   → PR: pr-cutlass-2139 → sources/prs/cutlass/PR-2139.md
   → Blog/doc: blog-* / doc-*
```

## Anti-Patterns (Don't Do These)

- Don't recommend techniques without citing `sources:` — the wiki exists precisely for this.
- Don't quote performance without the full 6-field `performance_claims` record.
- Don't conflate `sm90` and `sm100` patterns — always check the `architectures:` field.
- Don't cite `verified` claims without checking the page actually has `evidence_basis` entries that name both an official doc and an upstream code source.
- Don't recommend DeepEP/DualPipe/EPLB — they're explicitly out of scope (kernel-only KB).
